import json
from openai import OpenAI
import pickle
import networkx as nx 
from tqdm import tqdm
import random
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cwq")
parser.add_argument("--part", type=int)
parser.add_argument("--part_num", type=int)
args = parser.parse_args()

random.seed(42)

api_key = open("./api_key.txt", "r").read().strip()
client = OpenAI(api_key=api_key)


sys_query_prompt = '''
Based on the triplets retrieved from a knowledge graph, please select all the relevant triplets for answering the question. Please return formatted triplets as a list, each prefixed with \"evidence:\"."
'''


user_query_prompt = '''
Triplets:
{triples}

Question:
{question}
'''


for _split in ["train", "validation"]:
    with open(f"./data/preprocessed_data/{args.dataset}/_domains/total/RoG-{args.dataset}_total_{_split}.json", "r") as rf:
        wf = open(f"./data/preprocessed_data/{args.dataset}/_domains/total/{args.dataset}_rel_triples_{_split}_{args.part}_{args.part_num}.jsonl", "w")
        for idx, line in enumerate(tqdm(rf.readlines())):
            if idx % args.part_num == args.part:
                raw_data = dict()
                data = json.loads(line)
                raw_data["qid"] = data["id"]
                
                with open(f"./data/preprocessed_data/{args.dataset}/nx_format_graph/{_split}/{idx}.gpickle", "rb") as f:
                    graph = pickle.load(f)

                cand_triples = dict()
                for answer in data["a_entity"]:
                    if answer not in graph:
                        continue
                    for question_entity in data["q_entity"]:
                        if question_entity not in graph:
                            continue
                        try:
                            for ent_list in nx.all_shortest_paths(graph, source=answer, target=question_entity):
                                if len(ent_list) > 1:
                                    for idx in range(len(ent_list)-1):
                                        h, t = ent_list[idx], ent_list[idx+1]
                                        cand_triples[(h, graph[h][t]["relation"], t)] = None 
                        except Exception as e:
                            print(e)
                            continue
                
                cand_triples = list(cand_triples.keys())
                triples = ""
                for h, r, t in cand_triples:
                    triples += f"({h}, {r}, {t})\n"
                question = data["question"]
                user_query = user_query_prompt.format(triples=triples, question=question)
                raw_data["user_query"] = user_query
                messages = [
                    {"role": "system", "content": sys_query_prompt},
                    {"role": "user", "content": user_query}
                ]
                success = False
                for _ in range(20):
                    try:
                        response = client.chat.completions.create(
                            model='gpt-4o-2024-11-20',
                            messages=messages,
                            seed=42
                        )
                        success = True
                        break
                    except Exception as e:
                        print(e)
                        time.sleep(30)
                if success:
                    response = response.choices[0].message.content.strip()
                    raw_data["response"] = response
                    raw_data["question"] = question
                    wf.write(json.dumps(raw_data) + "\n")
                    wf.flush()
        wf.close()

