import os
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default=3)
parser.add_argument("--dataset", type=str, default="webqsp")
parser.add_argument("--model_name", type=str, default="relbert")
parser.add_argument("--dim_list", type=list, default=[128, 2])
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import json
from tqdm import tqdm
import numpy as np
import pickle

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.module.selection_module import Selection_Module
from src.utils.dataset import triple_dataset

rep_dim_dict = {
    "qwen2:0.5b": 896,
    "gemma2:2b": 2304,
    "phi3.5": 3072,
    "llama3.1": 4096,
    "sbert": 384,
    "relbert": 768,
    "gte-large": 1024,
    "gte-qwen2:1.5b": 1536
}

def get_qid2q_qid2a(split):
    with open(f"./data/preprocessed_data/{args.dataset}/_domains/total/RoG-{args.dataset}_total_{split}.json", "r") as rf:
        qid2q, qid2a = {}, {}
        for idx, line in enumerate(tqdm(rf.readlines())):
            data = json.loads(line)
            qid = data["id"]
            qid2q[qid] = data["question_entity"]
            qid2a[qid] = data["answer_entity"]
    return qid2q, qid2a


def main():

    with open("./data/preprocessed_data/webqsp/ent2idx.pkl", "rb") as rf:
        ent2idx = pickle.load(rf)
    with open("./data/preprocessed_data/webqsp/rel2idx.pkl", "rb") as rf:
        rel2idx = pickle.load(rf)

    train_meta_data = np.loadtxt(f"./data/preprocessed_data/{args.dataset}/_domains/total/train.txt", dtype=str)
    train_triples = torch.load("./data/original_data/SubgraphRAG/gpt_labeled_webqsp_cleaned.pt")
    train_qid2q, train_qid2a = get_qid2q_qid2a("train")
    train_dataset = triple_dataset(args, train_meta_data, train_triples, train_qid2q, train_qid2a, ent2idx, rel2idx)

    val_meta_data = np.loadtxt(f"./data/preprocessed_data/{args.dataset}/_domains/total/dev.txt", dtype=str)
    val_triples = torch.load("./data/original_data/SubgraphRAG/gpt_labeled_webqsp_cleaned.pt")
    val_qid2q, val_qid2a = get_qid2q_qid2a("validation")
    val_dataset = triple_dataset(args, val_meta_data, val_triples, val_qid2q, val_qid2a, ent2idx, rel2idx)

    test_meta_data = np.loadtxt(f"./data/preprocessed_data/{args.dataset}/_domains/total/test.txt", dtype=str)
    test_triples = torch.load("./data/original_data/SubgraphRAG/gpt_labeled_webqsp_cleaned.pt")
    test_qid2q, test_qid2a = get_qid2q_qid2a("test")
    test_dataset = triple_dataset(args, test_meta_data, test_triples, test_qid2q, test_qid2a, ent2idx, rel2idx)



    # rep_dim = rep_dim_dict[args.model_name]
    # model = Selection_Module(args, rep_dim=256)




if __name__ == "__main__":
    main()
