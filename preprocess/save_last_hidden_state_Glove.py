import json
from tqdm import tqdm
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
import os
import numpy as np


# 문장 하나를 (L, 300) 임베딩 시퀀스로 변환
def sentence_to_glove_tensor(sentence, glove, tokenizer, dim=300):
    tokens = tokenizer(sentence)
    vectors = []
    for token in tokens:
        if token in glove.stoi:
            vectors.append(glove[token])
        else:
            vectors.append(torch.zeros(dim))  # OOV → zero vector
    return torch.stack(vectors) if vectors else torch.zeros((1, dim))

# 문장 리스트를 (B, max_words_num, 300) 텐서와 attention mask로 변환
def batch_to_tensor(sentences, glove, tokenizer, max_len, dim=300):
    tensor_list = [sentence_to_glove_tensor(sent, glove, tokenizer, dim) for sent in sentences]
    lengths = [t.size(0) for t in tensor_list]

    # 패딩
    padded_tensor = pad_sequence(tensor_list, batch_first=True)  # (B, L, D)
    padded_tensor = torch.cat([padded_tensor, torch.zeros(padded_tensor.size(0), max_len-padded_tensor.size(1), padded_tensor.size(2))], dim=1)

    # attention mask 생성 (1 for real token, 0 for padding)
    attention_mask = torch.zeros((len(sentences), max_len), dtype=torch.long)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1

    return padded_tensor, attention_mask

def save_last_hidden_state(glove, tokenizer):
    path = f"./data/preprocessed_data/webqsp/last_hidden_state/GloVe_300dim/ori"
    if not os.path.exists(path):
        os.makedirs(os.path.join(path, "queries"))
        os.makedirs(os.path.join(path, "rel"))
        os.makedirs(os.path.join(path, "rel_inv"))

    with open("./data/preprocessed_data/webqsp/_domains/total/train.json", "r") as rf:
        question_list = [json.loads(line)["question"] for line in tqdm(rf.readlines(), ncols=100)]
    query_rep_train, query_mask_train = batch_to_tensor(question_list, glove, tokenizer, max_len=15)
    print(f"query_rep_train size: {query_rep_train.size()}")
    print(f"query_mask_train size: {query_mask_train.size()}")
    if not os.path.exists(os.path.join(path, "queries", "total", "train")):
        os.makedirs(os.path.join(path, "queries", "total", "train"))
    for i in tqdm(range(len(question_list)), ncols=100):
        torch.save(query_rep_train[i], os.path.join(path, "queries", "total", "train", f"{i}.pt"))
        torch.save(query_mask_train[i], os.path.join(path, "queries", "total", "train", f"{i}_mask.pt"))

    with open("./data/preprocessed_data/webqsp/_domains/total/dev.json", "r") as rf:
        question_list = [json.loads(line)["question"] for line in tqdm(rf.readlines(), ncols=100)]
    query_rep_dev, query_mask_dev = batch_to_tensor(question_list, glove, tokenizer, max_len=15)
    print(f"query_rep_dev size: {query_rep_dev.size()}")
    print(f"query_mask_dev size: {query_mask_dev.size()}")
    if not os.path.exists(os.path.join(path, "queries", "total", "dev")):
        os.makedirs(os.path.join(path, "queries", "total", "dev"))
    for i in tqdm(range(len(question_list)), ncols=100):
        torch.save(query_rep_dev[i], os.path.join(path, "queries", "total", "dev", f"{i}.pt"))
        torch.save(query_mask_dev[i], os.path.join(path, "queries", "total", "dev", f"{i}_mask.pt"))

    with open("./data/preprocessed_data/webqsp/_domains/total/test.json", "r") as rf:
        question_list = [json.loads(line)["question"] for line in tqdm(rf.readlines(), ncols=100)]
    query_rep_test, query_mask_test = batch_to_tensor(question_list, glove, tokenizer, max_len=15)
    print(f"query_rep_test size: {query_rep_test.size()}")
    print(f"query_mask_test size: {query_mask_test.size()}")
    if not os.path.exists(os.path.join(path, "queries", "total", "test")):
        os.makedirs(os.path.join(path, "queries", "total", "test"))
    for i in tqdm(range(len(question_list)), ncols=100):
        torch.save(query_rep_test[i], os.path.join(path, "queries", "total", "test", f"{i}.pt"))
        torch.save(query_mask_test[i], os.path.join(path, "queries", "total", "test", f"{i}_mask.pt"))

    with open("./data/preprocessed_data/webqsp/rel2idx.pkl", "rb") as rf:
        rel2idx = pickle.load(rf)
    rel_list = []
    rel_inv_list = []
    for rel in rel2idx:
        try:
            rel = rel.strip()
            fields = rel.split(".")
            words = fields[-2].split('_') + fields[-1].split('_')
        except:
            words = ['unk']
        rel_list.append(" ".join(words))
        rel_inv_list.append(" ".join(words[::-1]))

    rel_rep, rel_mask = batch_to_tensor(rel_list, glove, tokenizer, max_len=12)
    print(f"rel_rep size: {rel_rep.size()}")
    print(f"rel_mask size: {rel_mask.size()}")
    torch.save(rel_rep.cpu(), os.path.join(path, "rel", "rel_all.pt"))
    torch.save(rel_mask.cpu(), os.path.join(path, "rel", "rel_mask.pt"))
    rel_pad = torch.zeros((1, rel_rep.size(1), rel_rep.size(2)))
    torch.save(rel_pad, os.path.join(path, "rel", "rel_pad.pt"))
    rel_pad_mask = torch.zeros((1, rel_rep.size(1)))
    torch.save(rel_pad_mask, os.path.join(path, "rel", "rel_pad_mask.pt"))
    for idx in tqdm(range(len(rel_list)), ncols=100):
        np.save(os.path.join(path, "rel", f"rel_{idx}.npy"), rel_rep[idx].cpu().numpy())

    rel_inv_rep, rel_inv_mask = batch_to_tensor(rel_inv_list, glove, tokenizer, max_len=12)
    print(f"rel_inv_rep size: {rel_inv_rep.size()}")
    print(f"rel_inv_mask size: {rel_inv_mask.size()}")
    torch.save(rel_inv_rep.cpu(), os.path.join(path, "rel_inv", "rel_inv_all.pt"))
    torch.save(rel_inv_mask.cpu(), os.path.join(path, "rel_inv", "rel_inv_mask.pt"))
    rel_inv_pad = torch.zeros((1, rel_inv_rep.size(1), rel_inv_rep.size(2)))
    torch.save(rel_inv_pad, os.path.join(path, "rel_inv", "rel_inv_pad.pt"))
    rel_inv_pad_mask = torch.zeros((1, rel_inv_rep.size(1)))
    torch.save(rel_inv_pad_mask, os.path.join(path, "rel_inv", "rel_inv_pad_mask.pt"))
    for idx in tqdm(range(len(rel_inv_list)), ncols=100):
        np.save(os.path.join(path, "rel_inv", f"rel_inv_{idx}.npy"), rel_inv_rep[idx].cpu().numpy())
            
if __name__ == "__main__":
    # GloVe 300차원 로드
    glove = GloVe(name='6B', dim=300)
    tokenizer = get_tokenizer("basic_english")
    save_last_hidden_state(glove, tokenizer)


