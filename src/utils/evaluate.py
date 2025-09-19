import torch
import numpy as np
import pandas as pd
import re
import string
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
import pickle
import os
# from pykeen.training.callbacks import EvaluationTrainingCallback
import math
from collections import Counter

def lcs(X, Y):
    """Compute the length of the longest common subsequence between two sequences."""
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    return dp[m][n]

def rouge_l_score(candidate, reference):
    """
    Calculate ROUGE-L score between a candidate and a reference string.
    
    Args:
        candidate (str): Generated text.
        reference (str): Ground truth text.
        
    Returns:
        dict: {'precision': ..., 'recall': ..., 'f1': ...}
    """
    candidate_tokens = candidate.split()
    reference_tokens = reference.split()

    lcs_length = lcs(candidate_tokens, reference_tokens)

    precision = lcs_length / len(candidate_tokens) if candidate_tokens else 0.0
    recall = lcs_length / len(reference_tokens) if reference_tokens else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return {'precision': precision, 'recall': recall, 'f1': f1}

def compute_bleu4(candidate, reference, max_n=4):
    # Tokenize
    cand_tokens = candidate.split()
    ref_tokens = reference.split()

    precisions = []
    smooth = 1e-6  # small smoothing factor

    gram_list = list(range(1, max_n+1))

    for n in gram_list:  # 1-gram ~ max_n-gram
        cand_ngrams = Counter([tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens)-n+1)])
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])

        match_count = sum(min(count, ref_ngrams[ngram]) for ngram, count in cand_ngrams.items())
        total_count = sum(cand_ngrams.values())

        if total_count == 0:
            precisions.append(smooth)  # smoothing 적용
        else:
            precisions.append((match_count + smooth) / (total_count + smooth))

    # Brevity Penalty (BP)
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    if cand_len > ref_len:
        bp = 1
    elif cand_len == 0:
        bp = 0
    else:
        bp = math.exp(1 - ref_len / cand_len)

    # BLEU score
    score = bp * math.exp(sum(math.log(p) for p in precisions) / len(gram_list))

    return score



def get_rank(all_rep):
    score_matrix = get_cossim_matrix(all_rep)
    return np.argsort(-1 * score_matrix, axis=1)
                      
def get_cossim_matrix(all_rep, chunk_size=10):

    if type(all_rep) == torch.Tensor:
        score_matrix = torch.zeros(all_rep.size(0), all_rep.size(0)).to(all_rep.device)
        for idx1 in range(0, all_rep.size(0), chunk_size):
            for idx2 in range(0, all_rep.size(0), chunk_size):
                rep1 = all_rep[idx1:idx1+chunk_size]
                rep2 = all_rep[idx2:idx2+chunk_size]
                score_matrix[idx1:idx1+chunk_size, idx2:idx2+chunk_size] = \
                    F.cosine_similarity(rep1.unsqueeze(1), rep2.unsqueeze(0), dim=2)
        score_matrix = score_matrix.detach().cpu().numpy()

    elif type(all_rep) == np.ndarray:
        all_rep = all_rep / np.linalg.norm(all_rep, axis=1, keepdims=True)
        score_matrix = np.dot(all_rep, all_rep.T)

    return score_matrix

def get_macro_f1_score(quantized_rep, pos_idx, args):
    quant_score_matrix = get_cossim_matrix(quantized_rep, args)
    f1_score_list = []
    for idx, row in enumerate(quant_score_matrix):
        predicted = set(np.where(row > args.thr)[0])
        solutions = set(pos_idx[idx])
        correct = len(predicted.intersection(solutions))
        recall = correct / len(solutions)
        precision = correct / len(predicted)
        f1_score = 2 * recall * precision / (recall + precision)
        f1_score_list.append(f1_score)
    return np.mean(f1_score_list)

def convert_rank_name(true_rank, predicted_rank):
    # to ensure that true_rank is always sorted.
    N, B = true_rank.shape
    new_predicted_rank = np.zeros_like(predicted_rank)
    for idx, arr in enumerate(true_rank):
        rule = dict(zip(arr, range(B)))
        rule = np.vectorize(rule.get)
        new_predicted_rank[idx] = rule(predicted_rank[idx])
    return new_predicted_rank

def precision_at_k(predicted_rank, k):
    return (predicted_rank[:, :k] < k).mean(axis=1)

def get_mAP(cossim_matrix, quantized_rep, thr= 5):
    true_rank = np.argsort(-1 * cossim_matrix, axis=1)
    pred_cossim_matrix = get_cossim_matrix(quantized_rep)
    predicted_rank = np.argsort(-1 * pred_cossim_matrix, axis=1)
    predicted_rank = convert_rank_name(true_rank, predicted_rank)
    precision_table = np.zeros((predicted_rank.shape[0], thr))
    for ridx, rel in enumerate(predicted_rank):
        for cidx, order in enumerate(np.where(rel < thr)[0]):
            precision_table[ridx, cidx] = (predicted_rank[ridx, :order+1] < thr).mean() 
    AP = precision_table.mean(axis=1)
    mAP = AP.mean(axis=0)
    return mAP

def get_filter(triples_factory):
    triples = triples_factory.mapped_triples
    
    rank_filter = dict()
    for head, rel, tail in triples:
        head, rel, tail = head.item(), rel.item(), tail.item()
        if (head, rel) not in rank_filter.keys():
            rank_filter[(head, rel)] = [tail]
        else:
            rank_filter[(head, rel)].append(tail)

        if (tail, rel) not in rank_filter.keys():
            rank_filter[(tail, rel)] = [head]
        else:
            rank_filter[(tail, rel)].append(head)

    return rank_filter

def get_mrr(score, rank_filter, triples_factory):
    triples = triples_factory.mapped_triples
    rank_list = []
    for head, rel, tail in triples:
        head, rel, tail = head.item(), rel.item(), tail.item()
        target_score = score[(head, rel)][tail]

        score_list = np.copy(score[(head, rel)])
        score_list[rank_filter[(head, rel)]] = np.inf
        rank = (np.sum(score_list < target_score) + 1).item()
        rank_list.append(1/rank)

        score_list = np.copy(score[(tail, rel)])
        score_list[rank_filter[(tail, rel)]] = np.inf
        rank = (np.sum(score_list < target_score) + 1).item()
        rank_list.append(1/rank)
    return np.mean(rank_list).item()


# class CustomEvaluationTrainingCallback(EvaluationTrainingCallback):
#     def __init__(self, evaluation_triples, frequency, triple_num, args):
#         super().__init__(
#             evaluation_triples=evaluation_triples, 
#             frequency=frequency
#             )
#         self.best_mrr = 0.
#         self.args = args
#         self.ckpts_path = args.ckpts_path
#         self.sampled_indices = np.arange(triple_num)
    
#     def post_epoch(self, epoch, epoch_loss, **kwargs):
#         if epoch % self.frequency != 1:
#             return

#         np.random.shuffle(self.sampled_indices)
#         sampled_indices = self.sampled_indices[:10000]
#         validation_triples = self.evaluation_triples[sampled_indices]
#         result = self.evaluator.evaluate(
#             model=self.model,
#             mapped_triples=validation_triples,
#             device=self.training_loop.device,
#             batch_size=50,
#             tqdm_kwargs=dict(ncols=80),
#             **self.kwargs,
#         )

#         mrr = result.to_flat_dict()["both.realistic.inverse_harmonic_mean_rank"]
#         if mrr > self.best_mrr:
#             self.best_mrr = mrr
#             ckpts = dict(model=self.model, best_mrr=self.best_mrr, args=self.args)
#             with open(os.path.join(self.ckpts_path, "best_ReconHashEncoder.pkl"), 'wb') as wf:
#                 pickle.dump(ckpts, wf)
#         metric_dict = {"both.realistic.inverse_harmonic_mean_rank": mrr,
#                        "best_mrr": self.best_mrr}
#         self.result_tracker.log_metrics(metrics=metric_dict, step=epoch, prefix=self.prefix)
#         print(f"config: {self.args.config}, mrr: {mrr}, best_mrr: {self.best_mrr}")
        
def get_accuracy_gqa(path):
    df = pd.read_json(path, lines=True)
    # compute accuracy
    correct = 0
    for pred, label in zip(df["pred"], df["label"]):
        if label in pred:
            correct += 1
    return correct / len(df)


def get_accuracy_expla_graphs(path):
    df = pd.read_json(path, lines=True)
    # compute accuracy
    correct = 0
    for pred, label in zip(df["pred"], df["label"]):
        matches = re.findall(r"support|Support|Counter|counter", pred.strip())
        if len(matches) > 0 and matches[0].lower() == label:
            correct += 1

    return correct / len(df)


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


def eval_acc(prediction, answer):
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def get_accuracy_webqsp(path):
    df = pd.read_json(path, lines=True)

    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []

    for prediction, answer in zip(df.pred.tolist(), df.label.tolist()):

        prediction = prediction.replace("|", "\n")
        answer = answer.split("|")

        prediction = prediction.split("\n")
        f1_score, precision_score, recall_score = eval_f1(prediction, answer)
        f1_list.append(f1_score)
        precission_list.append(precision_score)
        recall_list.append(recall_score)
        prediction_str = " ".join(prediction)
        acc = eval_acc(prediction_str, answer)
        hit = eval_hit(prediction_str, answer)
        acc_list.append(acc)
        hit_list.append(hit)

    acc = sum(acc_list) * 100 / len(acc_list)
    hit = sum(hit_list) * 100 / len(hit_list)
    f1 = sum(f1_list) * 100 / len(f1_list)
    pre = sum(precission_list) * 100 / len(precission_list)
    recall = sum(recall_list) * 100 / len(recall_list)

    print(f"Accuracy: {acc:.4f}")
    print(f"Hit: {hit:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    return hit


eval_funcs = {
    "expla_graphs": get_accuracy_expla_graphs,
    "scene_graphs": get_accuracy_gqa,
    "scene_graphs_baseline": get_accuracy_gqa,
    "webqsp": get_accuracy_webqsp,
    "webqsp_baseline": get_accuracy_webqsp,
}
        

    

    



# def split_questions(query_num, dest_path, train_ratio=0.4):
#     train_indices, temp_data = train_test_split(np.arange(query_num), test_size=train_ratio, random_state=42)
#     val_indices, test_indices = train_test_split(temp_data, test_size=0.5, random_state=42)
#     print(f"train: {len(train_indices)} valid: {len(val_indices)} test: {len(test_indices)}")
#     np.savetxt(os.path.join(dest_path, "splits", "train.txt"), train_indices, fmt='%d')
#     np.savetxt(os.path.join(dest_path, "splits", "valid.txt"), val_indices, fmt='%d')
#     np.savetxt(os.path.join(dest_path, "splits", "test.txt"), test_indices, fmt='%d')


# def get_acc(logits, label):
#     logits = logits.reshape(-1, logits.size(2))
#     _, predicted = torch.max(logits, dim=1)
#     correct = (predicted == label).sum().item()
#     acc = correct / label.size(0)
#     return acc

# def get_mAPs(args, model, queries, valid_img_rep, text_rep, sep_idx):
#     # zero-hop queries
#     solutions = queries[0]
#     node_rep = model.get_node_rep(valid_img_rep, text_rep)
#     zero_reps = [node_rep[sep_idx[i]:sep_idx[i+1], :] for i in range(args.img_node_num)]
#     mAP = calcul_mAP(zero_reps, node_rep, solutions)

#     # one-hop queries
#     hr_queries, hr_solutions, tr_queries, tr_solutions = queries[1]
#     hr_reps, tr_reps = model.ask(hr_queries, tr_queries, node_rep, sep_idx)
#     hr_mAP = calcul_mAP(hr_reps, node_rep, hr_solutions)
#     tr_mAP = calcul_mAP(tr_reps, node_rep, tr_solutions)
#     one_hop_mAP = (hr_mAP + tr_mAP) / 2

#     # two-hop queries

#     return mAP, one_hop_mAP

# def calcul_mAP(q_rep_list, node_rep, solutions, batch_size=10000, metric=1):

#     ap_list = []
#     for idx, q_rep in enumerate(tqdm(q_rep_list, desc="calculate mAP", ncols=100)):
#         distance = []
#         for i in range(0, node_rep.shape[0], batch_size):
#             distance.append(torch.cdist(q_rep, node_rep[i : i+ batch_size], p=metric))
#         distance = torch.cat(distance, dim=1)
#         sol_idx = torch.Tensor(solutions[idx].astype(np.int64)).cuda()
        
#         order = torch.argsort(distance, dim=1).cuda()
#         rel_ = sol_idx.reshape(1, -1, 1) == order.unsqueeze(dim=1)
#         rel_ = rel_.any(dim=1).to(torch.int64)
#         rtd_rel_num = rel_.cumsum(dim=1)
#         rel_ = rel_.to(torch.float32)
#         rtd_num = torch.arange(1, order.shape[1] + 1).to(torch.float32)
#         rtd_num = rtd_num.reshape(1, -1).cuda()
#         precision = rtd_rel_num / rtd_num
#         average_precision = precision * rel_ 
#         average_precision = average_precision.sum(dim=1)/ sol_idx.size(0)
#         average_precision = average_precision.mean(dim=0, keepdim=True)
#         ap_list.append(average_precision)
#     ap_list = torch.cat(ap_list, dim=0)
#     return torch.mean(ap_list, dim=0).item()

# def get_queries(triples, sep_idx, args):

#     hr_to_sol, tr_to_sol = dict(), dict()
#     for head, tail, rel in triples:
#         hr = (head, rel)
#         if hr not in hr_to_sol.keys():
#             hr_to_sol[hr] = [tail]
#         else:
#             hr_to_sol[hr].append(tail)
        
#         tr = (tail, rel)
#         if tr not in tr_to_sol.keys():
#             tr_to_sol[tr] = [head]
#         else:
#             tr_to_sol[tr].append(head)

#     # 0-hop queries
#     sol_idx_array = [np.array(range(sep_idx[i].item(), sep_idx[i+1].item())) for i in range(args.img_node_num)]
#     sol_idx_array = np.array(sol_idx_array, dtype=object)
    
#     # 1-hop queries
#     one_hop_num = args.one_hop_num
#     if one_hop_num // 2 > len(hr_to_sol):
#         one_hop_num = len(hr_to_sol) * 2
#     if one_hop_num // 2 > len(tr_to_sol):
#         one_hop_num = len(tr_to_sol) * 2

#     hr_queries = list(hr_to_sol.keys())
#     hr_sol_cls_list = list(hr_to_sol.values())
#     tr_queries = list(tr_to_sol.keys())
#     tr_sol_cls_list = list(tr_to_sol.values())

#     hr_idx = np.random.choice(len(hr_queries), one_hop_num//2 , replace=False)
#     tr_idx = np.random.choice(len(tr_queries), one_hop_num//2 , replace=False)

#     hr_queries = np.array(hr_queries)[hr_idx, :]
#     hr_sol_cls_list = np.array(hr_sol_cls_list, dtype=object)[hr_idx]
#     tr_queries = np.array(tr_queries)[tr_idx, :]
#     tr_sol_cls_list = np.array(tr_sol_cls_list, dtype=object)[tr_idx]

#     hr_sol_idx_array = []
#     for idx in range(len(hr_queries)):
#         sol_idx = []
#         for sol in hr_sol_cls_list[idx]:
#             sol_idx += list(range(sep_idx[sol].item(), sep_idx[sol+1].item()))
#         hr_sol_idx_array.append(np.array(sol_idx))
#     hr_sol_idx_array = np.array(hr_sol_idx_array, dtype=object)

#     tr_sol_idx_array = []
#     for idx in range(len(tr_queries)):
#         sol_idx = []
#         for sol in tr_sol_cls_list[idx]:
#             sol_idx += list(range(sep_idx[sol].item(), sep_idx[sol+1].item()))
#         tr_sol_idx_array.append(np.array(sol_idx))
#     tr_sol_idx_array = np.array(tr_sol_idx_array, dtype=object)
    
#     print(f"1-hop queries have been created.")

#     return sol_idx_array, \
#         (hr_queries, hr_sol_idx_array, tr_queries, tr_sol_idx_array)


    
# ht_to_rel, query_to_sol = dict(), dict()
# for head, tail, rel in triples:
#     # ht_to_rel
#     if head not in ht_to_rel.keys():
#         ht_to_rel[head] = [rel]
#     else:
#         ht_to_rel[head].append(rel)
#     if tail not in ht_to_rel.keys():
#         ht_to_rel[tail] = [-1-rel]
#     else:
#         ht_to_rel[tail].append(-1-rel)

#     # query_to_sol
#     query = (head, rel)
#     if query not in query_to_sol.keys():
#         query_to_sol[query] = [tail]
#     else:
#         query_to_sol[query].append(tail)
#     query = (tail, -1-rel)
#     if query not in query_to_sol.keys():
#         query_to_sol[query] = [head]
#     else:
#         query_to_sol[query].append(head)


import torch
from sklearn.metrics import f1_score

def get_f1_score(labels, predictions, average='binary'):
    '''
    labels : (1, N) tensor
    predictions : (1, N) tensor
    '''
    # NumPy 배열로 변환
    labels_np = labels.cpu().numpy().flatten()
    predictions_np = predictions.cpu().numpy().flatten()
    
    # F1 score 계산
    f1 = f1_score(labels_np, predictions_np, average=average)
    
    return f1