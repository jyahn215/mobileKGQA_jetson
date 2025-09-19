import json
import os
import numpy as np
from tqdm import tqdm
import re
import argparse


def sparql2hop(sparql_query: str, question) -> int:
    where_match = re.search(r'WHERE\s*{(.*?)}', sparql_query, re.DOTALL)
    if not where_match:
        raise ValueError("Invalid SPARQL query: WHERE clause not found")
    where_clause = where_match.group(1)
    
    triple_pattern = re.compile(r'([\w:.?]+)\s+([\w:.?]+)\s+([\w:.?]+)\s*\.\n?')
    triples = triple_pattern.findall(where_clause)

    predicates = set(pred for _, pred, _ in triples)

    return len(predicates)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cwq")
    args = parser.parse_args()

    question_list = np.loadtxt(f"./data/preprocessed_data/{args.dataset}/_domains/total/total.txt", dtype=str, delimiter=" ")
    question_list = question_list[:, 1]
    qid2hop = dict.fromkeys(question_list)

    if args.dataset == "webqsp":
        path_list = ["./data/original_data/webqsp_orig/data/WebQSP.train.json",
                     "./data/original_data/webqsp_orig/data/WebQSP.test.json"]
        
        for path in path_list:
            with open(path, "r") as rf:
                data = json.load(rf)
                for question_data in tqdm(data["Questions"], ncols=100):
                    qid = question_data["QuestionId"]
                    if qid not in qid2hop:
                        continue
                    hop = question_data["Parses"][0]["InferentialChain"]
                    if hop == None:
                        sparql = question_data["Parses"][0]["Sparql"]
                        assert sparql.startswith("#MANUAL SPARQL")
                        continue
                    else:
                        qid2hop[qid] = len(hop)
        
    elif args.dataset == "cwq":
        path_list = ["./data/original_data/cwq_orig/ComplexWebQuestions_train.json",
                     "./data/original_data/cwq_orig/ComplexWebQuestions_dev.json",
                     "./data/original_data/cwq_orig/ComplexWebQuestions_test.json"]
        
        for path in path_list:
            with open(path, "r") as rf:
                data = json.load(rf)
                for question_data in tqdm(data, ncols=100):
                    qid = question_data["ID"]
                    if qid not in qid2hop:
                        continue

                    sparql = question_data["sparql"]
                    if sparql.startswith("#MANUAL SPARQL"):
                        continue
                    else:
                        qid2hop[qid] = sparql2hop(sparql, question_data["question"])
    
    unique, counts = np.unique(np.array(list(qid2hop.values()), dtype=str), return_counts=True)
    print(f"Number of questions: {len(qid2hop)}")
    for idx in range(len(unique)):
        print(f"{unique[idx]}: {counts[idx]} / {counts[idx]/len(qid2hop)*100:.2f}%")
    
    with open(f"./data/preprocessed_data/{args.dataset}/qid2triple_num.json", "w") as wf:
        json.dump(qid2hop, wf, indent=4)