import json
import numpy as np
from tqdm import tqdm
import re
import pickle
import itertools
from collections import defaultdict
from transformers import AutoTokenizer
import random
random.seed(10)


def check_validity(tokenizer, word, thr=0.5):
    return len(tokenizer.tokenize(word)) / len(word) < thr

def remove_encrypted_bindings(bindings, tokenizer):
    valid_bindings = []
    for _binding in bindings:
        valid = True
        for key, value in _binding.items():
            if not check_validity(tokenizer, value):
                valid = False
                break
        if valid:
            valid_bindings.append(_binding)
    return valid_bindings

def sparql2triples(sparql_query):
    where_match = re.search(r'WHERE\s*{(.*?)}', sparql_query, re.DOTALL)
    if not where_match:
        raise ValueError("Invalid SPARQL query: WHERE clause not found")
    where_clause = where_match.group(1)
    
    triple_pattern = re.compile(r'([\w:.?]+)\s+([\w:.?]+)\s+([\w:.?]+)\s*\.\n?')
    triples = triple_pattern.findall(where_clause)

    cleaned_triples = []
    for _triple in triples:
        h, r, t = _triple
        if h.startswith("ns:"):
            h = h[3:]
        if r.startswith("ns:"):
            r = r[3:]
        if t.startswith("ns:"):
            t = t[3:]
        cleaned_triples.append((h, r, t))

    return cleaned_triples

def get_converted_triples():
    qid2triples = dict()

    path_list = ["./data/original_data/webqsp_orig/data/WebQSP.train.json",
                 "./data/original_data/webqsp_orig/data/WebQSP.test.json"]
    
    cnt = [0]*7
    for path in path_list:
        with open(path, "r") as rf:
            data = json.load(rf)
            for question_data in tqdm(data["Questions"], ncols=100):
                qid = question_data["QuestionId"]
                sparql = question_data["Parses"][0]["Sparql"]
                if sparql.startswith("#MANUAL"):
                    continue
                else:
                    sparql = re.sub(r'#.*', '', sparql)

                    constraints = question_data["Parses"][0]["Constraints"]
                    mid_dict = dict()
                    for constraint in constraints:
                        mid_dict[constraint["Argument"]] = constraint["EntityName"]
                    mid_dict[question_data["Parses"][0]["TopicEntityMid"]] = question_data["Parses"][0]["TopicEntityName"]
                    
                    triples = sparql2triples(sparql)
                    cleaned_triples = []
                    try:
                        for h, r, t in triples:
                            _triple = (h, r, t)
                            if not h.startswith("?"):
                                assert h in mid_dict
                                _triple = (mid_dict[h], r, t)
                            if not t.startswith("?"):
                                assert t in mid_dict
                                _triple = (h, r, mid_dict[t])
                            cleaned_triples.append(_triple)
                    except:
                        continue
                    qid2triples[qid] = cleaned_triples
                    cnt[len(cleaned_triples)-1] += 1

    # print(f"cnt: {cnt}")
    # print(f"total: {sum(cnt)}")
    # for idx, i in enumerate(cnt):
    #     print(f"triple_num {idx+1}:  {round(100*i/sum(cnt), 2)}%")
    # print("")
    
    return qid2triples

def sample_matching_triples(graph, triple_patterns, tokenizer):
    # 1. triple pattern별 후보 찾기
    var_to_candidates = defaultdict(list)
    
    var_set = set()
    for h, r, t in triple_patterns:
        if h.startswith("?"):
            var_set.add(h)
        if t.startswith("?"):
            var_set.add(t)
            
    for h, r, t in triple_patterns:
        if not h.startswith('?') and not t.startswith('?'):
            # 둘 다 확정된 경우
            if any(data.get('relation') == r for _, _, data in graph.edges(h, data=True) if _ == t):
                continue
            else:
                return [], None  # 만족하는 경우 없음
        elif h.startswith('?') and not t.startswith('?'):
            # head가 변수
            for src, dst, data in graph.edges(data=True):
                if data.get('relation') == r and dst == t:
                    var_to_candidates[h].append(src)
        elif not h.startswith('?') and t.startswith('?'):
            # tail이 변수
            for src, dst, data in graph.edges(data=True):
                if data.get('relation') == r and src == h:
                    var_to_candidates[t].append(dst)
        else:
            # 둘 다 변수
            for src, dst, data in graph.edges(data=True):
                if data.get('relation') == r:
                    var_to_candidates[h].append(src)
                    var_to_candidates[t].append(dst)

    if not var_to_candidates:
        return [], None
    elif len(var_to_candidates) != len(var_set):
        return [], None

    # 2. 변수들의 모든 가능한 조합 생성
    vars_sorted = sorted(var_to_candidates.keys())
    candidates_list = [var_to_candidates[var] for var in vars_sorted]

    all_combinations = list(itertools.product(*candidates_list))

    # 3. 조합 검증
    valid_bindings = []

    for combo in all_combinations:
        binding = dict(zip(vars_sorted, combo))
        
        # 매핑 후 triple_patterns 만족하는지 확인
        is_valid = True
        for h, r, t in triple_patterns:
            h = binding[h] if h.startswith('?') else h
            t = binding[t] if t.startswith('?') else t
            if not graph.has_edge(h, t):
                is_valid = False
                # print(f"no edge exist: {h} {t}")
                break
            else:
                if graph[h][t]['relation'] != r:
                    is_valid = False
                    # print(f"relation not match: {h} {r} {t}")
                    break

        if is_valid:
            valid_bindings.append(binding)
    if len(valid_bindings) == 0:
        return [], None
    # else:
    #     valid_bindings = remove_encrypted_bindings(valid_bindings, tokenizer)
    #     if len(valid_bindings) == 0:
    #         return [], None
    
    binding = random.choice(valid_bindings)
    try:
        answer = binding["?x"]
    except:
        answer = None
    chosen_triples = []
    for _triple in triple_patterns:
        h, r, t = _triple
        h = binding[h] if h.startswith('?') else h
        t = binding[t] if t.startswith('?') else t
        chosen_triples.append((h, r, t))
    
    return chosen_triples, answer

def save_sampled_triples_and_answer(qid2triples, question_list, tokenizer):
    qid2answer, qid2question = dict(), dict()
    with open(f"./data/preprocessed_data/webqsp/_domains/total/train.json", "r") as rf:
        for line in tqdm(rf.readlines(), ncols=100):
            data = json.loads(line)
            qid2answer[data["id"]] = [answer["text"] for answer in data["answers"]]
            qid2question[data["id"]] = data["question"]
    with open(f"./data/preprocessed_data/webqsp/_domains/total/dev.json", "r") as rf:
        for line in tqdm(rf.readlines(), ncols=100):
            data = json.loads(line)
            qid2answer[data["id"]] = [answer["text"] for answer in data["answers"]]
            qid2question[data["id"]] = data["question"]
    with open(f"./data/preprocessed_data/webqsp/_domains/total/test.json", "r") as rf:
        for line in tqdm(rf.readlines(), ncols=100):
            data = json.loads(line)
            qid2answer[data["id"]] = [answer["text"] for answer in data["answers"]]
            qid2question[data["id"]] = data["question"]

    qid2sampled_triples = dict()
    cnt = [0]*5
    for split, qid, idx in tqdm(question_list, desc="save triples and questions", ncols=100):
        if qid in qid2triples:
            with open(f"./data/preprocessed_data/webqsp/nx_format_graph/{split}/{idx}.gpickle", "rb") as rf:
                graph = pickle.load(rf)
            
            # rdf_graph = convert_nx_graph_to_rdf_graph(graph)
            triples = qid2triples[qid]
            if triples[0][0] not in graph:
                continue
            
            chosen_triples, answer = sample_matching_triples(graph, triples, tokenizer)
            if len(chosen_triples) != 0 and answer in qid2answer[qid]:
                cnt[len(chosen_triples)-1] += 1
                qid2sampled_triples[qid] = {"chosen_triples": chosen_triples, 
                                            "a_entity": answer, 
                                            "q_entity": triples[0][0], 
                                            "question": qid2question[qid]}
    with open(f"./data/preprocessed_data/webqsp/qid2sampled_triples.json", "w") as wf:
        json.dump(qid2sampled_triples, wf, indent=4)
    print("valid cnt: ", cnt)
    print("total cnt: ", sum(cnt))

if __name__ == "__main__":
    question_list = np.loadtxt(f"./data/preprocessed_data/webqsp/_domains/total/total.txt", dtype=str, delimiter=" ")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", cache_dir="./ckpts/llm")
    qid2triples = get_converted_triples()
    save_sampled_triples_and_answer(qid2triples, question_list, tokenizer)

    
                        
