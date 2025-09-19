import os
from datetime import datetime
import torch
import torch.utils.data as data
from utils.evaluate import get_cossim_matrix
import math
import pickle
import numpy as np
# from pykeen.triples import TriplesFactory
# from pykeen.sampling import BasicNegativeSampler
from abc import *
from tqdm import tqdm
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas as pd
import datasets
from scipy.sparse import load_npz
from torch.utils.data._utils.collate import default_collate


class triple_dataset(data.Dataset):
    def __init__(self, args, meta_data, triples, qid2q, qid2a, ent2idx, rel2idx):
        super().__init__()
        
        self.meta_data = meta_data
        self.ent2idx = ent2idx
        self.rel2idx = rel2idx
        self.qid2q = qid2q
        self.qid2a = qid2a
        self.triples = triples
        self.dataset = args.dataset
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        
        split, qid, _order = self.meta_data[idx]
        pos_triple = self.triples[qid]
        with open(f"./data/preprocessed_data/{self.dataset}/nx_format_graph/{split}/{_order}.gpickle", "rb") as rf:
            graph = pickle.load(rf)
            



class infoNCE_dataset(data.Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.noise_level = args.noise_level
        with open(f"{args.data_path}/{args.domain}/{split}/ent_set_{split}.pkl", "rb") as rf:
            self.ent_list = list(pickle.load(rf))
            self.ent_num = len(self.ent_list)
        with open(f"{args.data_path}/{args.domain}/{split}/rel_set_{split}.pkl", "rb") as rf:
            self.rel_list = list(pickle.load(rf))
            self.rel_num = len(self.rel_list)
        self.length = self.ent_num + self.rel_num
        self.total_list = np.array(self.ent_list + self.rel_list)
        self.dataset = args.dataset
        self.embed_path = f"{args.data_path}/embeds/{args.model_name}"

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        rep = self.load_rep(idx)
        return self.add_random_noise(rep)

    def load_rep(self, idx):
        if idx < self.ent_num:
            idx = self.total_list[idx]
            rep = np.load(os.path.join(self.embed_path, f"nodes/{idx - idx % 50000}/{idx}.npy"))
        else:
            idx = self.total_list[idx]
            rep = np.load(os.path.join(self.embed_path, f"edges/{idx}.npy"))
        return rep
    
    def add_random_noise(self, rep):
        rep1 = rep + np.random.randn(rep.shape[0]) * self.noise_level
        rep2 = rep + np.random.randn(rep.shape[0]) * self.noise_level
        return rep1, rep2

    def sample_rep(self, sample_num):
        sampled_indices = np.random.choice(self.length, sample_num, replace=False)
        sampled_rep = []
        for idx in sampled_indices:
            rep = self.load_rep(idx)
            sampled_rep.append(rep)

        return np.stack(sampled_rep)

    

class Triple_dataset(data.Dataset):
    def __init__(self, args, neg_idx, pos_idx, all_rep):
        super().__init__()

        self.pos_num = args.pos_num
        self.neg_num = args.neg_num
        self.pos_idx = pos_idx
        self.neg_idx = neg_idx
        self.all_rep = all_rep

        if args.dataset == "expla_graphs":
            self.length, self.ent_num = 7279 + 28, 7279

        self.select_pos_neg_pairs()

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        anchor = self.all_rep[idx:idx+1]
        pos = self.all_rep[self.sampled_pos_idx[idx]]
        neg = self.all_rep[self.sampled_neg_idx[idx]]

        return anchor, pos, neg
    
    def select_pos_neg_pairs(self):
        self.sampled_pos_idx = []
        for arr in self.pos_idx:
            arr = np.random.choice(arr, self.pos_num, replace=True)
            self.sampled_pos_idx.append(arr)
        self.sampled_pos_idx = np.stack(self.sampled_pos_idx)
        
        self.sampled_neg_idx = []
        for arr in self.neg_idx:
            arr = np.random.choice(arr, self.neg_num, replace=True)
            self.sampled_neg_idx.append(arr)
        self.sampled_neg_idx = np.stack(self.sampled_neg_idx)


class LogRatio_dataset(data.Dataset):
    def __init__(self, args, query_list: list, rel_list: list, split: str):
        super().__init__()
        self.args = args
        self.rel_num = len(rel_list)
        self.data_list = query_list
        self.index_list = np.arange(len(self.data_list))
        self.data_path = args.data_path
        self.split = split
        
    def __len__(self):
        return math.ceil(len(self.data_list) / self.args.batch_size)
    
    def __getitem__(self, index):

        batch_indices = self.index_list[index*self.args.batch_size: (index+1)*self.args.batch_size]

        rep_list = []
        for index in batch_indices:
            data_index = self.data_list[index]
            # if index < self.rel_num:
            #     rep = np.load(os.path.join(
            #         self.data_path, "last_hidden_state", self.args.model_name, "ori",
            #         "rel", f"rel_{data_index}.npy"))
            # elif self.rel_num <= index < 2*self.rel_num:
            #     rep = np.load(os.path.join(
            #         self.data_path, "last_hidden_state", self.args.model_name, "ori",
            #         "rel_inv", f"rel_inv_{data_index}.npy"))
            # else:
            rep = torch.load(os.path.join(
                self.data_path, "last_hidden_state", self.args.model_name, "ori",
                "queries", self.args.domain, self.split, f"{data_index}.pt"), map_location='cpu').numpy()
            rep_list.append(rep)
        rep_list = np.concatenate(rep_list, axis=0)
        
        return rep_list

    def collate_fn(self, batch_data):
        batch_rep = batch_data[0]
        batch_rep = torch.from_numpy(batch_rep)
        batch_rep = batch_rep.to(torch.float32)
        return batch_rep

        # if self.all_rep is None:
        #     idx_in_batch = self.idx_list[index*self.batch_size: (index+1)*self.batch_size]
        #     batch_rep = load_batch_rep(self.args, idx_in_batch)
        #     batch_data = (batch_rep, None)

        # else:
        #     batch_rep = self.all_rep[index*self.batch_size: (index+1)*self.batch_size, :]
        #     batch_true_cossim = self.true_cossim[index*self.batch_size: \
        #                                         (index+1)*self.batch_size, :]
        #     batch_true_cossim = batch_true_cossim[:, index*self.batch_size: \
        #                                             (index+1)*self.batch_size]
        #     batch_data = (batch_rep, batch_true_cossim)
        # return batch_data

    # 미리 저장해두고 불러오는 방식이 빠르지 않음
    # def load_batch_true_cossim(self, idx_in_batch):
    #     batch_true_cossim = np.zeros((idx_in_batch.shape[0], idx_in_batch.shape[0]))
    #     for idx1 in idx_in_batch:
    #         for idx2 in idx_in_batch:
    #             cossim = np.load(os.path.join(self.data_path, "cossim", f"{idx1}", f"{idx2}.npy"))
    #             batch_true_cossim[idx1, idx2] = cossim
    #     return batch_true_cossim
    

    
    # def post_epoch(self):
    #     randperm = torch.randperm(self.length)
    #     if self.all_rep is None:
    #         self.idx_list = self.idx_list[randperm]
    #     else:
    #         self.all_rep = self.all_rep[randperm, :]
    #         self.true_cossim = self.true_cossim[randperm, :]
    #         self.true_cossim = self.true_cossim[:, randperm]

# class query_dataset(data.Dataset):
#     def __init__(self, args):
#         super().__init__()
#         self.length = args.query_num
#         self.data_path = args.data_path

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         rep = torch.load(os.path.join(self.data_path, "queries", f"{idx}.pt"))
#         return rep
    
class ent_rel_dataset(data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.length = args.ent_num + args.rel_num
        self.ent_num = args.ent_num
        self.data_path = args.data_path
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx < self.ent_num:
            rep = torch.load(os.path.join(self.data_path, "nodes", f"{idx}.pt"))
        else:
            rep = torch.load(os.path.join(self.data_path, "edges", f"{idx-self.ent_num}.pt"))
        return rep

    
class ReconDataset(data.IterableDataset):
    def __init__(self, triples_factory, tokenizer, args):
        super().__init__()

        self.mapped_triples = triples_factory.mapped_triples
        self.iter_len = math.ceil(len(self.mapped_triples) / args.batch_size)
        self.triple_len = len(self.mapped_triples)
        self.hash_info = list(tokenizer.hash.keys())
        if "close_anchors" in self.hash_info:
            self.ancs = tokenizer.hash["close_anchors"]["anchor"]
            ancs2id = np.vectorize(tokenizer.token2id["close_anchors"]["anchor"].get)
            self.ancs = ancs2id(self.ancs)

            self.dists = tokenizer.hash["close_anchors"]["distance"]
            dists2id = np.vectorize(tokenizer.token2id["close_anchors"]["distance"].get)
            self.dists = dists2id(self.dists)

        if "connected_relations" in self.hash_info:
            self.rels = tokenizer.hash["connected_relations"]
            rel2id = np.vectorize(tokenizer.token2id["connected_relations"].get)
            self.rels = rel2id(self.rels)
        
        self.batch_size = args.batch_size
        self.negative_sampler = BasicNegativeSampler(
            mapped_triples=self.mapped_triples,
            num_negs_per_pos=args.neg_num,
            corruption_scheme=('head', 'tail'),
            num_entities = triples_factory.num_entities,
            num_relations = triples_factory.num_relations
        )

    def __iter__(self):
        for iter_cnt in range(self.iter_len):
            positive_batch = self.mapped_triples[iter_cnt*self.batch_size: (iter_cnt+1)*self.batch_size]
            negative_batch, _ = self.negative_sampler.sample(positive_batch)
            pos_ent, pos_rel_hash = positive_batch[:, [0, 2]], positive_batch[:, 1]
            neg_ent, neg_rel_hash = negative_batch[:, :, [0, 2]], negative_batch[:, :, 1]

            pos_ent_hash = dict.fromkeys(self.hash_info)
            neg_ent_hash = dict.fromkeys(self.hash_info)
            if "close_anchors" in self.hash_info:
                positive_ancs = self.ancs[pos_ent]
                positive_dists = self.dists[pos_ent]
                negative_ancs = self.ancs[neg_ent]
                negative_dists = self.dists[neg_ent]
                pos_ent_hash["close_anchors"] = [positive_ancs, positive_dists]
                neg_ent_hash["close_anchors"] = [negative_ancs, negative_dists]

            if "connected_relations" in self.hash_info:
                positive_rels = self.rels[pos_ent]
                negative_rels = self.rels[neg_ent]
                pos_ent_hash["connected_relations"] = [positive_rels]
                neg_ent_hash["connected_relations"] = [negative_rels]

            yield pos_ent_hash, pos_rel_hash, neg_ent_hash, neg_rel_hash

    def __len__(self) -> int:
        return math.ceil(len(self.mapped_triples)/ self.batch_size)

    def shuffle(self, epoch):
        randperm = torch.randperm(self.triple_len)
        self.mapped_triples = self.mapped_triples[randperm]
    
    def worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        if worker_info is not None:  
            per_worker = math.ceil(dataset.triple_len / worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min((worker_id+1)*per_worker, dataset.triple_len)
            dataset.mapped_triples = dataset.mapped_triples[iter_start:iter_end]
            dataset.iter_len = math.ceil(len(dataset.mapped_triples) / dataset.batch_size)
            dataset.triple_len = len(dataset.mapped_triples)

    def collate_fn(self, data):
        pos_ent_hash, pos_rel_hash, neg_ent_hash, neg_rel_hash = data[0]
        for key in pos_ent_hash.keys():
            pos_ent_hash[key] = [torch.from_numpy(x).unsqueeze(dim=1) 
                                 for x in pos_ent_hash[key]]
        pos_rel_hash = pos_rel_hash.unsqueeze(dim=1)
        for key in neg_ent_hash.keys():
            neg_ent_hash[key] = [torch.from_numpy(x) for x in neg_ent_hash[key]]

        return pos_ent_hash, pos_rel_hash, neg_ent_hash, neg_rel_hash
    
    def get_all_ent_hash(self):
        all_ent_hash = dict.fromkeys(self.hash_info)
        if "close_anchors" in self.hash_info:
            all_ent_hash["close_anchors"] = [torch.from_numpy(self.ancs), 
                                             torch.from_numpy(self.dists)]
        if "connected_relations" in self.hash_info:
            all_ent_hash["connected_relations"] = [torch.from_numpy(self.rels)]

        return all_ent_hash


class expla_graph_dataset(data.Dataset):
    def __init__(self, args, model):
        super().__init__()
        all_rep = torch.load(
            os.path.join(f"./../logs/ckpts/{args.load_SH}", "SearchEncoder_rep.pt"))
        all_rep = all_rep.to(model.device)
        assert all_rep.size(0) == args.ent_num + args.rel_num

        with open(os.path.join(args.data_path, "nodes", "ent2idx.pkl"), 'rb') as rf:
            self.ent2idx = pickle.load(rf)
            self.idx2ent = {v: k for k, v in self.ent2idx.items()}
            self.idx2ent = np.vectorize(self.idx2ent.get)
        with open(os.path.join(args.data_path, "edges", "rel2idx.pkl"), 'rb') as rf:
            self.rel2idx = pickle.load(rf)
            self.idx2rel = {v: k for k, v in self.rel2idx.items()}
            self.idx2rel = np.vectorize(self.idx2rel.get)

        query_rep = torch.load(os.path.join(f"./../logs/ckpts/{args.load_SH}", "SearchEncoder_query_rep.pt"),
                               map_location=model.device)
        self.query_num = query_rep.size(0)
        
        self.data_path = args.data_path
        self.path = os.path.join(args.data_path, "graphs", args.config)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.ent_num, self.rel_num = args.ent_num, args.rel_num
        ent_rep, rel_rep = all_rep[:self.ent_num], all_rep[self.ent_num:self.ent_num+self.rel_num]
        q_to_sim_ent, q_to_sim_rel = self.find_cossim_ent_and_rel(args, query_rep, ent_rep, rel_rep)
        valid_triples = self.find_valid_triples(q_to_sim_ent, q_to_sim_rel, model, args)
        self.retrieve_subgraph(valid_triples, ent_rep, rel_rep)

        self.text = pd.read_csv("./../data/original_data/expla_graphs/train_dev.tsv", sep="\t")
        self.prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
        del all_rep
        del query_rep
        del ent_rep
        del rel_rep
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text.iloc[idx]
        question = f'Argument 1: {text.arg1}\nArgument 2: {text.arg2}\n{self.prompt}'
        nodes = pd.read_csv(os.path.join(self.path, f"{idx}_nodes.csv"))
        edges = pd.read_csv(os.path.join(self.path, f"{idx}_edges.csv"))
        edges = edges[["src_id", "edge_attr", "dest_id"]]
        desc = nodes.to_csv(index=False)+'\n'+edges.to_csv(index=False)
        graph = torch.load(os.path.join(self.path, f"{idx}.pt"), map_location='cpu')

        return {
            'id': idx,
            'label': text['label'],
            'desc': desc,
            'graph': graph.detach(),
            'question': question,
        }


    def find_cossim_ent_and_rel(self, args, query_rep, ent_rep, rel_rep):
        
        q_to_sim_ent = F.cosine_similarity(query_rep.unsqueeze(dim=1), 
                                           ent_rep.unsqueeze(dim=0), dim=2)
        q_to_sim_ent = torch.argsort(q_to_sim_ent, dim=1, descending=True)
        q_to_sim_ent = q_to_sim_ent[:, :args.ent_ret_num]
            
        q_to_sim_rel = F.cosine_similarity(query_rep.unsqueeze(dim=1), 
                                           rel_rep.unsqueeze(dim=0), dim=2)
        q_to_sim_rel = torch.argsort(q_to_sim_rel, dim=1, descending=True)
        q_to_sim_rel = q_to_sim_rel[:, :args.rel_ret_num]

        return q_to_sim_ent, q_to_sim_rel

    def find_valid_triples(self, q_to_sim_ent, q_to_sim_rel, model, args):
        valid_triples = dict()
        for q_idx in range(q_to_sim_ent.size(0)):
            sim_ents, sim_rels = q_to_sim_ent[q_idx], q_to_sim_rel[q_idx]
            ht_pairs = torch.cartesian_prod(sim_ents, sim_ents)
            ht_pairs = ht_pairs[ht_pairs[:, 0] < ht_pairs[:, 1]] # exclude repeated pairs
            ht_len = ht_pairs.shape[0]
            ht_pairs = torch.repeat_interleave(ht_pairs, len(sim_rels), dim=0)
            relations = sim_rels.repeat(ht_len)

            scores = model(ht_pairs[:, 0], relations, ht_pairs[:, 1], mode=None)
            scores += model(ht_pairs[:, 1], relations + self.rel_num, ht_pairs[:, 0], mode=None)
            scores = scores/2
            if args.thr is not None: # threshold based selection
                scores_idx = scores > args.thr
            elif args.triple_ret_num is not None: # top-k based selection
                scores_idx = torch.argsort(scores, descending=True)[:args.triple_ret_num]
            ht_pairs, relations = ht_pairs[scores_idx], relations[scores_idx]
            ht_pairs = ht_pairs.detach().cpu().numpy()
            relations = relations.detach().cpu().numpy()
            valid_triples[q_idx] = (ht_pairs, relations)
        return valid_triples
    
    def retrieve_subgraph(self, valid_triples, ent_rep, rel_rep):

        for q_idx, (ht_pairs, relations) in tqdm(valid_triples.items(), desc="retrieve subgraph", ncols=80):
            assert len(ht_pairs) != 0

            # save graph
            nodes = np.unique(ht_pairs)
            node_rep = ent_rep[nodes, :].detach().cpu()
            node_rep = torch.sign(node_rep)
            node_convert = dict(zip(nodes, np.arange(len(nodes))))
            edge_rep = rel_rep[relations].detach().cpu()
            edge_rep = torch.sign(edge_rep)
            node_convert = np.vectorize(node_convert.get)
            edge_index = node_convert(ht_pairs).transpose(1, 0)
            edge_index = torch.from_numpy(edge_index).to(torch.int64)

            graph = Data(x=node_rep, 
                         edge_index=edge_index, 
                         edge_attr=edge_rep,
                         num_nodes=node_rep.size(0))
            torch.save(graph, os.path.join(self.path, f"{q_idx}.pt"))

            # save nodes and edges
            nodes_name = self.idx2ent(nodes)
            src_dest_name = self.idx2ent(ht_pairs)
            edges_name = self.idx2rel(relations)

            nodes_df = pd.DataFrame({"node_id": nodes, "node_attr": nodes_name})
            edges_df = pd.DataFrame({"src_id": ht_pairs[:, 0], 
                                     "src_attr": src_dest_name[:, 0], 
                                     "edge_attr": edges_name, 
                                     "dest_id": ht_pairs[:, 1],
                                     "dest_attr": src_dest_name[:, 1]})
            nodes_df.to_csv(os.path.join(self.path, f"{q_idx}_nodes.csv"), index=False)
            edges_df.to_csv(os.path.join(self.path, f"{q_idx}_edges.csv"), index=False)

    def eval_valid_triples(self, args):
        pass
        # triples = load_triples_factory(args).mapped_triples.numpy()
        # correct = set()
        # for head, rel, tail in triples:
        #     correct.add((head, rel, tail))
        # acc_list = []
        # for q_idx, (ht_pairs, relations) in valid_triples.items():
        #     total_num, correct_num = ht_pairs.shape[0], 0
        #     for idx in range(total_num):
        #         head, tail = ht_pairs[idx]
        #         rel = relations[idx]
        #         if (head, rel, tail) in correct:
        #             correct_num += 1
        #     acc_list.append(correct_num / total_num)
        # print(f"average accuracy: {np.mean(acc_list)}")

    def get_idx_split(self):
        with open("./../data/original_data/expla_graphs/split/train_indices.txt", "rt") as rf:
            train_indices = [int(line.strip()) for line in rf]
        with open("./../data/original_data/expla_graphs/split/val_indices.txt", "rt") as rf:
            dev_indices = [int(line.strip()) for line in rf]
        with open("./../data/original_data/expla_graphs/split/test_indices.txt", "rt") as rf:
            test_indices = [int(line.strip()) for line in rf]
        
        return {"train": train_indices, "val": dev_indices, "test": test_indices}
        
def collate_fn(data):
    batch = {}
    for k in data[0].keys():
        batch[k] = [d[k] for d in data]
    if 'graph' in batch:
        batch['graph'] = Batch.from_data_list(batch['graph'])
    return batch

def RAG_dataset(args, **kwargs):
    if args.dataset == "expla_graphs":
        model = kwargs["model"]
        dataset = expla_graph_dataset(args, model)
    
    return dataset

load_dataset = {
    "InfoNCELoss": infoNCE_dataset,
    "TripletLoss": Triple_dataset,
    "LogRatioLoss": LogRatio_dataset,
    "NSSALoss": Triple_dataset,
}

def load_batch_rep(args, idx_in_batch):
    batch_rep = []
    for idx in idx_in_batch:
        if idx < args.ent_num:
            rep = torch.load(os.path.join(args.data_path, "nodes", f"{idx}.pt")).numpy()
            batch_rep.append(rep)
        elif idx >= args.ent_num:
            idx = idx - args.ent_num
            rep = torch.load(os.path.join(args.data_path, "edges", f"{idx}.pt")).numpy()
            batch_rep.append(rep)
    return np.stack(batch_rep)

# true cossim을 미리 저장하는 방식이 그리 효율적이지 않음
# def save_true_cossim(args, rep_num=500):
#     length = args.ent_num + args.rel_num
#     for idx in range(length):
#         if not os.path.exists(os.path.join(args.data_path, "cossim", f"{idx}")):
#             os.makedirs(os.path.join(args.data_path, "cossim", f"{idx}"))
            
#     idx_list = np.arange(length)
#     num = math.ceil(length / rep_num)
#     for batch_idx1 in range(0, length, rep_num):
#         for batch_idx2 in tqdm(range(0, length, rep_num), desc="saving true cossim", ncols=90):
#             rep1 = load_batch_rep(args, idx_list[batch_idx1: batch_idx1+rep_num])
#             rep1 = torch.from_numpy(rep1).cuda()
#             rep2 = load_batch_rep(args, idx_list[batch_idx2: batch_idx2+rep_num])
#             rep2 = torch.from_numpy(rep2).cuda()
#             cossim_matrix = F.cosine_similarity(rep1.unsqueeze(1), rep2.unsqueeze(0), dim=2)
#             cossim_matrix = cossim_matrix.detach().cpu().numpy()
#             for idx1 in range(rep_num):
#                 for idx2 in range(rep_num):
#                     path = os.path.join(args.data_path, "cossim", 
#                                         f"{batch_idx1 + idx1}", f"{batch_idx2 + idx2}.npy")
#                     np.save(path, cossim_matrix[idx1, idx2])
#         print(f"{batch_idx1} / {num}")

def get_idx(score_matrix, args):
    if args.thr_cossim is not None and args.thr_num is not None:
        raise ValueError("thr_cossim and thr_num cannot be used together")
    elif args.thr_cossim is None and args.thr_num is None:
        raise ValueError("please input threshold")
    
    pos_idx, neg_idx, _score_matrix = [], [], score_matrix.copy()
    if args.thr_cossim is not None:
        np.fill_diagonal(_score_matrix, -2) # exclude itself
        thr = args.thr_cossim
        for row in _score_matrix:
            indices = np.where(row > thr)[0]
            if len(indices) == 0:
                pos_idx.append([np.argsort(-1 * row)[0]]) 
            else:
                pos_idx.append(indices.tolist())

            indices = np.where((-1 <= row) * (row <= thr) == True)[0]
            if len(indices) == 0:
                neg_idx.append([np.argsort(row)[1]]) # exclue itself
            else:
                neg_idx.append(indices.tolist())

        print("pos_num statistics")
        pos_num_list = np.array([len(n) for n in pos_idx])
        print(f"min: {pos_num_list.min()},"
            f"Q1: {np.percentile(pos_num_list, 25)},"
            f"median: {np.median(pos_num_list)},"
            f"Q3: {np.percentile(pos_num_list, 75)},"
            f"max: {pos_num_list.max()}")
        
    elif args.thr_num is not None:
        thr = args.thr_num
        rank = np.argsort(-1 * _score_matrix, axis=1)
        pos_idx = rank[:, 1:thr+1]
        neg_idx = rank[:, thr+1:200]

    return pos_idx, neg_idx

def load_all_rep(args):
    with open(os.path.join(args.data_path, "nodes", "ent2idx.pkl"), 'rb') as rf:
        ent2idx = pickle.load(rf)
    with open(os.path.join(args.data_path, "edges", "rel2idx.pkl"), 'rb') as rf:
        rel2idx = pickle.load(rf)
    args.ent_num, args.rel_num = len(ent2idx), len(rel2idx)

    # small dataset
    if args.ent_num < 200000:
        all_rep = []
        for idx in tqdm(ent2idx.values(), desc="load node reps", ncols=80):
            rep = torch.load(os.path.join(args.data_path, "nodes", f"{idx}.pt")).numpy()
            all_rep.append(rep)
        for idx in tqdm(rel2idx.values(), desc="load edge reps", ncols=80):
            rep = torch.load(os.path.join(args.data_path, "edges", f"{idx}.pt")).numpy()
            all_rep.append(rep)
        loaded_data = (args, np.stack(all_rep))
    # large dataset
    else:
        loaded_data = (args, None)
    
    return loaded_data

def sample_rep(args, sample_num):
    length = args.ent_num + args.rel_num
    sampled_indices = np.arange(length)
    np.random.shuffle(sampled_indices)
    sampled_indices = sampled_indices[:sample_num]
    sampled_rep = []
    for idx in sampled_indices:
        if idx < args.ent_num:
            rep = torch.load(os.path.join(args.data_path, "nodes", f"{idx}.pt")).numpy()
            sampled_rep.append(rep)
        elif idx >= args.ent_num:
            idx = idx - args.ent_num
            rep = torch.load(os.path.join(args.data_path, "edges", f"{idx}.pt")).numpy()
            sampled_rep.append(rep)

    return np.stack(sampled_rep)


def load_triples_factory(args):
    if args.dataset == "expla_graphs" or args.dataset == "webqsp":
        all_triples = np.load(os.path.join(args.data_path, "all_triples.npy"))
        with open(os.path.join(args.data_path, "nodes", "ent2idx.pkl"), 'rb') as rf:
            ent2idx = pickle.load(rf)
        with open(os.path.join(args.data_path, "edges", "rel2idx.pkl"), 'rb') as rf:
            rel2idx = pickle.load(rf)
        triples_factory = TriplesFactory(mapped_triples=all_triples,
                                         entity_to_id=ent2idx, 
                                         relation_to_id=rel2idx)
                                                              
    if args.create_inverse_triples is True:
        triples_factory = TriplesFactory(
            mapped_triples=triples_factory.mapped_triples,
            entity_to_id=triples_factory.entity_to_id,
            relation_to_id=triples_factory.relation_to_id,
            create_inverse_triples=True
        )
        
    return triples_factory


class retrieval_dataset(data.Dataset):
    def __init__(self, args, domain, split):
        super().__init__()
        if args.dataset == "webqsp":
            self.dataset = datasets.load_dataset("rmanluo/RoG-webqsp", split=split)
        elif args.dataset == "cwq":
            self.dataset = datasets.load_dataset("rmanluo/RoG-cwq", split=split)
        
        self.path = f"./data/preprocessed_data/{args.dataset}/{domain}/{split}"
        self.hiddens_path = f"{self.path}/query_hidden_states/{args.llm}"
        self.embeds_path = f"./data/preprocessed_data/{args.dataset}/embeds/{args.llm}"
        self.max_token = args.max_q_tokens
        self.max_lin_graph_len = args.max_graph_tokens
        self.mask_len = args.max_q_tokens + args.max_graph_tokens + 1
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        with open(f"{self.path}/graphs/{index}.pkl", "rb") as rf:
            graph = pickle.load(rf)
        if graph.number_of_nodes() == 0:
            return (None, None, None, None, None)

        with open(f"{self.path}/linearized_graphs/{index}.pkl", "rb") as rf:
            linearized_graph = pickle.load(rf)
        lin_graph_len = len(linearized_graph)
        
        label = np.load(f"{self.path}/label/{index}.npy")

        inputs = np.load(f"{self.path}/inputs/{index}.npy")
        query_hidden_state = np.load(f"{self.hiddens_path}/{index}.npy")
        hidden_len = query_hidden_state.shape[0]

        mask = np.zeros((self.mask_len, self.mask_len))
        mask[:hidden_len, :hidden_len] = 1
        mask[self.max_token:self.max_token+lin_graph_len+1, :hidden_len] = 1
        adj_mask = load_npz(f"{self.path}/mask/{index}.npz").toarray()
        mask[self.max_token+1:self.max_token+1+lin_graph_len, self.max_token+1:self.max_token+1+lin_graph_len] = adj_mask
        mask[self.max_token, self.max_token] = 1

        ent_idx = [False]*(self.max_token+1) + \
            [True if idx % 2 == 0 else False for idx in range(lin_graph_len)] + \
                [False]*(self.max_lin_graph_len-lin_graph_len)
        ent_idx = np.array(ent_idx)

        return label, inputs, query_hidden_state, mask, ent_idx

    def collate_fn(self, data):
        label, inputs, query_hidden_state, mask, ent_idx = data[0]
        if type(label) != np.ndarray:
            return (None, None, None, None, None)
        else:
            return default_collate(data)

        
        
    

    