import os
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import datasets
import random
import networkx as nx
import csv
import pandas as pd
import time
import json
from tqdm import tqdm
import argparse
import glob
import json
import os
import re
import string
from sklearn.metrics import precision_score



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

def eval_hit1(prediction, answer):
    for a in answer:
        if match(prediction[0], a):
            return 1
    return 0

def eval_hit(prediction, answer):
    for a in answer:
        for pred in prediction:
            if match(pred, a):
                return 1
    return 0

def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = ' '.join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


# from node2vec import Node2Vec
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.module.ollama_LLM import OLLAMA_LLM
from src.utils.graph_utils import bfs_with_rule, build_graph_as_nx_format




def find_all_relation_paths(G, start_node, relations):
    paths = []  # 모든 가능한 경로 저장 리스트

    def dfs(current_node, relation_idx, path_text):
        """ 깊이 우선 탐색(DFS)으로 가능한 모든 경로 탐색 """
        if relation_idx >= len(relations):  # 모든 관계를 탐색 완료하면 경로 저장
            paths.append(path_text)
            return
        
        relation = relations[relation_idx]  # 현재 탐색할 relation
        
        # 현재 노드에서 나가는 모든 엣지를 확인
        for neighbor in G.neighbors(current_node):
            edge_data = G.get_edge_data(current_node, neighbor)
            
            if edge_data and edge_data.get("relation") == relation:
                # 중복 방지를 위해 path에 있는 노드 제외
                dfs(neighbor, relation_idx + 1, f"{path_text} -> {relation} -> {neighbor}")

    # DFS 실행 (경로 탐색 시작)
    dfs(start_node, 0, str(start_node))

    return paths  # 모든 가능한 경로 리스트 반환

def find_all_shortest_paths(G, source, target):
    paths_with_relations = []

    try:
        # 모든 최단 경로 찾기
        shortest_paths = list(nx.all_shortest_paths(G, source=source, target=target))
        
        for path in shortest_paths:
            formatted_path = [str(path[0])]  # 시작 노드 추가

            # 노드 간 relation 추가
            for i in range(len(path) - 1):
                v1, v2 = path[i], path[i + 1]
                edge_data = G.get_edge_data(v1, v2)

                if edge_data and "relation" in edge_data:
                    relation = edge_data["relation"]
                    formatted_path.append(f"-> {relation} -> {v2}")  # 중복 없이 출력
                else:
                    formatted_path.append(f"-> ??? -> {v2}")  # relation이 없을 경우

            paths_with_relations.append(" ".join(formatted_path))

    except nx.NetworkXNoPath:
        return ["No valid path found"]  # 경로가 없으면 에러 처리

    return paths_with_relations


def read_prediction(args):
    idx2qinfo = dict()
    qinfo2idx = dict()
    with open(f"./data/preprocessed_data/{args.dataset}/_domains/total/total.txt", "rt") as rf:
        for data in tqdm(rf.readlines()):
            split, qid, idx = data.split(" ")
            idx = idx[:-1] if idx.endswith("\n") else idx
            idx2qinfo[(split, idx)] = qid
            qinfo2idx[qid] = (split, idx)

    print(f"read predictions from ./ckpts/mobileKGQA/{args.config}_test_{args.test_domain}.info")
    with open(f"./ckpts/mobileKGQA/{args.config}_test_{args.test_domain}.info", "rb") as rf:
        cands_dict = dict()
        for line in tqdm(rf.readlines()):
            data = json.loads(line)
            cands_dict[data["qid"]] = data["cand"]

    return idx2qinfo, qinfo2idx, cands_dict


def LLM_RAG(args, idx2qinfo, qinfo2idx, cands_dict):
    model = OLLAMA_LLM(
        model_name=args.model_name,
        port=args.port,
        max_token=args.max_token,
        seed=args.seed
    )

    # with open(f"./prompt/plan_selection.txt", "rt") as rf:
    #     plan_selection_prompt = rf.read()
    with open(f"./prompt/answer_generation.txt", "rt") as rf:
        answer_generation_prompt = rf.read()

    qid2data = dict()
    with open(f"./data/preprocessed_data/{args.dataset}/_domains/total/RoG-{args.dataset}_total_train.json", "rt") as rf:
        for line in tqdm(rf.readlines()):
            data = json.loads(line)
            qid = data["id"]
            qid2data[qid] = {"q_entity": data["q_entity"], "a_entity": data["a_entity"], "question": data["question"]}
    with open(f"./data/preprocessed_data/{args.dataset}/_domains/total/RoG-{args.dataset}_total_validation.json", "rt") as rf:
        for line in tqdm(rf.readlines()):
            data = json.loads(line)
            qid = data["id"]
            qid2data[qid] = {"q_entity": data["q_entity"], "a_entity": data["a_entity"], "question": data["question"]}
    with open(f"./data/preprocessed_data/{args.dataset}/_domains/total/RoG-{args.dataset}_total_test.json", "rt") as rf:
        for line in tqdm(rf.readlines()):
            data = json.loads(line)
            qid = data["id"]
            qid2data[qid] = {"q_entity": data["q_entity"], "a_entity": data["a_entity"], "question": data["question"]}

    f1_list = []
    hit_list = []
    # wf = open(f"logs_{args.config}_{args.model_name}.txt", "wt")
    for qid, _cands in tqdm(cands_dict.items()):
        split, _idx = qinfo2idx[qid]
        question_entities = qid2data[qid]["q_entity"]
        answer_entities = qid2data[qid]["a_entity"]
        question = qid2data[qid]["question"]

        split = "dev" if split == "validation" else split
        with open(f"./data/preprocessed_data/{args.dataset}/nx_format_graph/{split}/{_idx}.gpickle", "rb") as rf:
            nx_graph = pickle.load(rf)
        if args.shortest_path:
            reasoning_path_list = []
            for question_entity in question_entities:
                for predicted_entity, _ in _cands:
                    try:
                        reasoning_path_list += find_all_shortest_paths(nx_graph, question_entity, predicted_entity)
                    except Exception as e:
                        print(e)
                        continue
            answer_generation_prompt_input = answer_generation_prompt.format(
                question=question, Paths="\n".join(reasoning_path_list))
            answer_generation_message = [{"role": "user", "content": answer_generation_prompt_input}]

            response = model.generate_chat_response(answer_generation_message)
            response = response.content

            try:
                response = response.split(",")
            except:
                response = [response]
            response = [r.strip() for r in response]
            hit = eval_hit(response, answer_entities)
            f1 = eval_f1(response, answer_entities)
            hit_list.append(hit)
            f1_list.append(f1[0])

            # wf.write(f"{qid}\t{f1}\t{hit}\n")

    print(f"Hit: {sum(hit_list) / len(hit_list)}, F1: {sum(f1_list) / len(f1_list)}, config: {args.config}, model: {args.model_name} test_domain: {args.test_domain}")


            







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cwq")
    parser.add_argument("--config", type=str)
    parser.add_argument("--test_domain", type=str)
    parser.add_argument("--model_name", type=str, default="gemma2:2b")
    parser.add_argument("--max_token", type=int, default=500)
    parser.add_argument("--port", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shortest_path", default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.shortest_path = True

    idx2qinfo, qinfo2idx, cands_dict = read_prediction(args)
    answers = LLM_RAG(args, idx2qinfo, qinfo2idx, cands_dict)


    
