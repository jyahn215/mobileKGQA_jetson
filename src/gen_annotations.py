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
import networkx as nx
import networkx as nx
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.module.ollama_LLM import OLLAMA_LLM
from src.utils.graph_utils import bfs_with_rule, build_graph_as_nx_format, sample_valid_path



def save_graph(dataset_name):
    for split in ["train", "dev", "test"]:
        path = f"./data/preprocessed_data/{dataset_name}/nx_format_graph/{split}"
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print(f"already exists: {path}")
            continue

        print(f"loading {split} dataset")
        st = time.time()
        _split = "validation" if split == "dev" else split
        dataset = datasets.load_dataset(f"rmanluo/RoG-{dataset_name}", split=_split)
        end = time.time()
        print(f"loading time: {end-st}")

        for idx, data in enumerate(tqdm(dataset, desc="saving graph as nx format")):
            graph = build_graph_as_nx_format(data["graph"])
            with open(os.path.join(path, f"{idx}.gpickle"), "wb") as wf:
                pickle.dump(graph, wf)


# def filter_paths(args, group_num=1):
#     sampled_paths = f"./data/preprocessed_data/{args.dataset}/_domains/{args.domain}/sampled_paths_{args.prob_str}_{args.domain}_{args.seed}.csv"
#     filtered_paths = f"./data/preprocessed_data/{args.dataset}/_domains/{args.domain}/filtered_paths_{args.prob_str}_{args.domain}_{args.seed}.csv"

#     # filter encrypted entities
#     if not os.path.exists(filtered_paths):
#         with open(filtered_paths, "w") as wf:
#             writer = csv.writer(wf)
#             sampled_paths_df = pd.read_csv(sampled_paths, header=None)

#             for idx in tqdm(range(0, len(sampled_paths_df), group_num), desc="filtering paths"):
#                 group = sampled_paths_df.iloc[idx:idx + group_num]

#                 ent_list = []
#                 for _, row in group.iterrows():
#                     triples = []
#                     for _triple in row[3:]:
#                         if pd.isna(_triple):
#                             break
#                         else:
#                             triples.append(_triple.strip())

#                     ent_list.append(triples[0].split(" -> ")[0])
#                     for path in triples:
#                         ent_list.append(path.split(" -> ")[2])

#                 valid = True
#                 for ent in ent_list:
#                     if check_validity(ent) is False:
#                         valid = False
#                         break
#                 if valid is True:
#                     writer.writerow(group.iloc[0, :].tolist())
#     else:
#         print(f"already exists: {filtered_paths}")


def masking(sentence, a_entity, place_holder, prompt_masking, model):
    lower_sentence, lower_a_entity = sentence.lower(), a_entity.lower()
    lower_sentence = lower_sentence.replace("'s ", "s' ") if "'s " in lower_sentence and "s' " in lower_a_entity else lower_sentence
    lower_sentence = lower_sentence.replace("s' ", "'s ") if "s' " in lower_sentence and "'s " in lower_a_entity else lower_sentence

    if len(place_holder) >= 2:
        place_holder = "*" + place_holder if place_holder[0] != "*" else place_holder
        place_holder = place_holder + "*" if place_holder[-1] != "*" else place_holder
    if lower_a_entity not in lower_sentence:
        
        prompt = prompt_masking.format(merged=lower_sentence, a_entity=lower_a_entity, place_holder=place_holder)
        message = [{"role": "user", "content": prompt}]
        response = model.generate_chat_response(message)
        masked = response.content

    else:
        patterns = [
            f"a {lower_a_entity}",
            f"an {lower_a_entity}",
            f"the {lower_a_entity}",
            lower_a_entity
        ]

        masked = lower_sentence
        for _pattern in patterns:
            masked = masked.replace(_pattern, place_holder)
    masked = masked.lower().strip()
    # print(masked)
    # print("-------------------------------------------------")

    return masked

    
def generate_question_mobileKGQA(triples, a_entity, q_entity, model):
    with open("./prompt/annotation_generation_verbalization.txt", "rt") as rf:
        prompt_verbalization = rf.read()
    with open("./prompt/annotation_generation_merge.txt", "rt") as rf:
        prompt_merge = rf.read()
    with open("./prompt/annotation_generation_placeholder.txt", "rt") as rf:
        prompt_placeholder = rf.read()
    with open("./prompt/annotation_generation_masking.txt", "rt") as rf:
        prompt_masking = rf.read()
    with open("./prompt/annotation_generation_question.txt", "rt") as rf:
        prompt_question = rf.read()
    with open("./prompt/annotation_generation_refine.txt", "rt") as rf:
        prompt_refine = rf.read()

    triple_list = []
    for h, r, t in triples:
        triple_list.append(f"({h}, {r}, {t})\n")

    sentence_list = []
    for triple in triple_list:
        prompt = prompt_verbalization.format(triple=triple)
        message = [{"role": "user", "content": prompt}]
        response = model.generate_chat_response(message)
        sentence = response.content.strip()
        sentence_list.append(sentence)
    sentences_as_text = ""
    for sentence in sentence_list:
        sentences_as_text += f"- {sentence}\n"
    # print(triple_list)
    # print(sentences_as_text)
    # print("-------------------------------------------------")
    
    
    if len(triple_list) > 1:
        prompt = prompt_merge.format(sentences=sentences_as_text, a_entity=a_entity)
        message = [{"role": "user", "content": prompt}]
        response = model.generate_chat_response(message)
        merged = response.content.strip()
        # print(sentences_as_text)
        # print(merged)
        # print("-------------------------------------------------")
    else:
        merged = sentence_list[0]
        # print(sentences_as_text)
        # print(merged)
        # print("-------------------------------------------------")

    prompt = prompt_placeholder.format(merged=merged, a_entity=a_entity)
    message = [{"role": "user", "content": prompt}]
    response = model.generate_chat_response(message)
    place_holder = response.content.lower().strip()
    # print(place_holder)
    # print("-------------------------------------------------")
    masked = masking(merged, a_entity, place_holder, prompt_masking, model)
    # print(masked)
    # print("-------------------------------------------------")

    prompt = prompt_question.format(masked=masked, q_entity=q_entity)
    message = [{"role": "user", "content": prompt}]
    response = model.generate_chat_response(message)
    question = response.content
    question = question.strip()

    lower_question, lower_q_entity = question.lower(), q_entity.lower()
    lower_question = lower_question.replace("'s ", "s' ") if "'s " in lower_question and "s' " in lower_q_entity else lower_question
    lower_question = lower_question.replace("s' ", "'s ") if "s' " in lower_question and "'s " in lower_q_entity else lower_question
    if lower_q_entity not in lower_question:
        prompt = prompt_refine.format(masked=masked, q_entity=lower_q_entity, question=lower_question)
        message = [{"role": "user", "content": prompt}]
        response = model.generate_chat_response(message)
        lower_question = response.content.strip().lower()
    # print(lower_question)
    # print("-------------------------------------------------")
    
    return lower_question


def generate_question_RLM(triples: list[list[str]], a_entity, q_entity, model):
    '''
    reasoning_path = [(e1, r1, e2), (e2, r2, e3), (e2, r3, e4) ...]
    '''
    with open("./prompt/annotation_generation_RLM.txt", "rt") as rf:
        prompt_generation_RLM = rf.read()

    # reasoning_path_as_text for ReasoningLM
    reasoning_path_as_text = []
    for h, r, t in triples:
        triple = ", ".join([h, r, t])
        triple = f"({triple})"
        reasoning_path_as_text.append(triple)
    reasoning_path_as_text = ", ".join(reasoning_path_as_text) + "."

    prompt = prompt_generation_RLM.format(
        triples=reasoning_path_as_text,
        a_entity=a_entity,
        q_entity=q_entity
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": prompt}
    ]
    response = model.generate_chat_response(messages)
    question = response.content
    question = question.strip()

    return question

def parse_CoT(response):
    response = response.lower()
    if "question" not in response:
        return response
    question = response.split("question")[-1]
    question = question[1:] if question.startswith(":") else question
    question = question.strip()
    question = question.split("?")[0] if "?" in question else question
    return question

def generate_question_CoT(triples, a_entity, q_entity, model):
    with open("./prompt/annotation_generation_CoT.txt", "rt") as rf:
        prompt_generation_CoT = rf.read()

    triples_as_text = ""
    for h, r, t in triples:
        triples_as_text += f"({h}, {r}, {t})\n"

    prompt = prompt_generation_CoT.format(
        triples=triples_as_text,
        a_entity=a_entity,
        q_entity=q_entity
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": prompt}
    ]
    response = model.generate_chat_response(messages)
    question = response.content
    question = parse_CoT(question)
    return question 


def generate_question(triples, a_entity, q_entity, model):
    if args.method == "mobileKGQA":
        return generate_question_mobileKGQA(triples, a_entity, q_entity, model)
    elif args.method == "RLM":
        return generate_question_RLM(triples, a_entity, q_entity, model)
    elif args.method == "CoT":
        return generate_question_CoT(triples, a_entity, q_entity, model)
    else:
        raise ValueError("Invalid method")

def find_a_entity(graph, reasoning_path, a_entity, q_entity, args):

    with open(f"./data/preprocessed_data/{args.dataset}/ent2idx.pkl", "rb") as rf:
        ent2idx = pickle.load(rf)

    if not os.path.exists(f"./data/preprocessed_data/{args.dataset}/_domains/{args.domain}/answers_{args.domain}_LD.csv"):
        generated_question_path = f"./data/preprocessed_data/{args.dataset}/_domains/{args.domain}/merged_{args.domain}_LD.csv"
        generated_question_df = pd.read_csv(generated_question_path, header=None)
        with open(f"./data/preprocessed_data/{args.dataset}/_domains/{args.domain}/merged_{args.domain}_LD.json", "w") as wf:

            for _, row in tqdm(generated_question_df.iterrows()):
                try:
                    qid, split, idx, q_entity, a_entity, question = row[0], row[1], row[2], row[3], row[4], row[5]
                    q_entity = q_entity.strip()
                    a_entity = a_entity.strip()

                    triples = []
                    for _triple in row[6:]:
                        if pd.isna(_triple):
                            break
                        else:
                            triples.append(_triple.strip())

                    ent_list = []
                    ent_list.append(triples[0].split(" - ")[0])
                    for path in triples:
                        ent_list.append(path.split(" - ")[2])

                    rel_list = []
                    for path in triples:
                        rel_list.append(path.split(" - ")[1])
                
                    q_idx = ent_list.index(q_entity)
                    a_idx = ent_list.index(a_entity)
                    if q_idx < a_idx:
                        rel_list = rel_list[q_idx:a_idx]
                    elif q_idx > a_idx:
                        rel_list = rel_list[a_idx:q_idx]
                    else:
                        continue

                    with open(f"./data/preprocessed_data/{args.dataset}/nx_format_graph/{split}/{idx}.gpickle", "rb") as rf:
                        graph = pickle.load(rf)

                    assert q_entity in graph
                    assert a_entity in graph
                    answer_list = bfs_with_rule(graph, q_entity, rel_list)

                    data_to_write = data_dict[qid]
                    data_to_write["entities"] = [ent2idx[q_entity]]
                    data_to_write["question"] = question

                    answer_list_dump = []
                    for answer in answer_list:
                        answer_list_dump.append({"kb_id": None, "text": answer})
                    data_to_write["answers"] = answer_list_dump

                    wf.write(json.dumps(data_to_write) + "\n")

                except Exception as e:
                    print(e)
                    continue

# def setup_adaptation_data(args):
#     qid2adap_data = dict()
#     with open(f"./data/preprocessed_data/{args.dataset}/_domains/{args.domain}/answers_{args.domain}_{args.seed}.csv", "r") as rf:
#         reader = csv.reader(rf)
#         for row_data in tqdm(reader):
#             qid, question, answers = row_data[0], row_data[1], row_data[2]
#             answers = answers.strip("[]").replace("'", "").split(", ")
#             answers = [{"kb_id": None, "text": _answer} for _answer in answers]

#             if qid not in qid2adap_data:
#                 qid2adap_data[qid] = [{
#                     "question": question,
#                     "answers": answers
#                 }]
#             else:
#                 qid2adap_data[qid].append({
#                     "question": question,
#                     "answers": answers
#                 })

    # with open(f"./data/preprocessed_data/{args.dataset}/_domains/{args.domain}/adapt_data_{args.domain}_{args.seed}.json", "w") as wf:
    #     with open(f"./data/preprocessed_data/{args.dataset}/_domains/{args.domain}/train.json", "r") as rf:
    #         data_list = [json.loads(line) for line in rf.readlines()]

    #         for data in tqdm(data_list):
    #             if data["id"] in qid2adap_data.keys(): # data generation might have failed.
    #                 for adap_data in qid2adap_data[data["id"]]:
    #                     data["question"] = adap_data["question"]
    #                     data["answers"] = adap_data["answers"]
    #                     wf.write(json.dumps(data) + "\n")

def find_answer(graph, triples, q_entity):

    rel_list = []
    for _triple in triples:
        rel_list.append(_triple[1])
    answer_list = bfs_with_rule(graph, q_entity, rel_list)
    answer_list = list(set(answer_list)) # remove duplicates

    return answer_list
    

def generate_annotations(args):

    save_graph(args.dataset)
    domain_meta_data = np.loadtxt(f"./data/preprocessed_data/{args.dataset}/_domains/{args.domain}/train.txt", dtype=str)
    length_list = []
    for i in range(len(args.gen_num)):
        length_list += [i+1] * args.gen_num[i]
    np.random.shuffle(length_list)

    # load ollama model
    model = OLLAMA_LLM(
        model_name=args.model_name, 
        port=args.port, 
        max_token=args.max_token, 
        seed=args.seed
        )
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", cache_dir="./ckpts/llm")

    # load data_dict for adaptation
    data_dict = dict()
    with open(f"./data/preprocessed_data/{args.dataset}/_domains/total/train.json", "r") as rf:
        for line in tqdm(rf.readlines()):
            data = json.loads(line)
            data_dict[data["id"]] = data
    with open(f"./data/preprocessed_data/{args.dataset}/_domains/total/dev.json", "r") as rf:
        for line in tqdm(rf.readlines()):
            data = json.loads(line)
            data_dict[data["id"]] = data
    with open(f"./data/preprocessed_data/{args.dataset}/_domains/total/test.json", "r") as rf:
        for line in tqdm(rf.readlines()):
            data = json.loads(line)
            data_dict[data["id"]] = data

    # setup_adaptation_data
    gen_num = "+".join([str(x) for x in args.gen_num])
    gen_data_dir = f"gen_{args.domain}_{args.method}_{gen_num}_{args.seed}"
    if not os.path.exists(f"./data/preprocessed_data/{args.dataset}/_domains/{gen_data_dir}"):
        os.makedirs(f"./data/preprocessed_data/{args.dataset}/_domains/{gen_data_dir}")
    gen_data_path = f"./data/preprocessed_data/{args.dataset}/_domains/{gen_data_dir}/train.json"
    print("genearte an annotation file for adaptation in ", gen_data_path)
    assert not os.path.exists(gen_data_path), f"already exists: {gen_data_path}"
    wf = open(gen_data_path, "w")

    with open(f"./data/preprocessed_data/{args.dataset}/ent2idx.pkl", "rb") as rf:
        ent2idx = pickle.load(rf)
    
    tqdm_object = tqdm(length_list)
    cnt_list = [0] * len(args.gen_num)
    for length in tqdm_object:
        while True:
            np.random.shuffle(domain_meta_data)
            split, query_id, idx = domain_meta_data[0]

            split = "dev" if split == "validation" else split

            with open(f"./data/preprocessed_data/{args.dataset}/nx_format_graph/{split}/{idx}.gpickle", "rb") as rf:
                graph = pickle.load(rf)

            # sample path
            reasoning_path = sample_valid_path(graph, length, tokenizer, args)
            if reasoning_path is None:
                continue
            assert len(reasoning_path) % 2 == 1
            triples = []
            for idx in range(0, len(reasoning_path)-2, 2):
                triples.append([reasoning_path[idx], reasoning_path[idx+1], reasoning_path[idx+2]])

            q_entity, a_entity = triples[0][0], triples[-1][-1]
            assert q_entity in graph

            # generate question
            question = generate_question(triples, a_entity, q_entity, model)
            if question is None: # resample graph
                continue

            # find more answers and write to the file
            answer_list = find_answer(graph, triples, q_entity)
            assert a_entity in answer_list

            # write to the file
            # print(triples)
            # print(question))
            # print(answer_list)
            # print("-------------------------------------------------")

            data_to_write = data_dict[query_id]
            data_to_write["entities"] = [ent2idx[q_entity]]
            data_to_write["question"] = question
            answer_list_dump = []
            for answer in answer_list:
                answer_list_dump.append({"kb_id": None, "text": answer})
            data_to_write["answers"] = answer_list_dump
            wf.write(json.dumps(data_to_write) + "\n")
            wf.flush()
            break

        cnt_list[length-1] += 1
        tqdm_object.set_postfix_str(f"plan: {args.gen_num}, generated: {cnt_list}")
    wf.close()


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='webqsp')
    parser.add_argument("--domain", type=str, default="total")
    parser.add_argument("--gen_num", type=int, nargs="+", default=[500, 500])
    parser.add_argument("--model_name", type=str, default="gemma2:2b")
    parser.add_argument("--max_token", type=int, default=1000)
    parser.add_argument("--port", type=int, default=11000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", type=str, choices=["mobileKGQA", "CoT", "RLM"])
    args, _ = parser.parse_known_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    generate_annotations(args)