import os
import random
import argparse
import json
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-drop_rate", type=float, default=0.5)
parser.add_argument("-domain", type=str, default="total")
parser.add_argument("-dataset", type=str, default="webqsp")
parser.add_argument("-seed", type=int, default=0)
args = parser.parse_args()

random.seed(args.seed)

src_path = f"./data/preprocessed_data/{args.dataset}/_domains/{args.domain}"
dst_path = f"./data/preprocessed_data/{args.dataset}/_domains/{args.domain}_drop_{args.drop_rate}_{args.seed}"

os.makedirs(dst_path, exist_ok=True)

NLP_file = open(os.path.join(src_path, f"RoG-webqsp_total_train.json"), "r")
converted_file = open(os.path.join(src_path, f"train.json"), "r")
dropped_NLP_file = open(os.path.join(dst_path, f"RoG-webqsp_{args.domain}_drop_{args.drop_rate}_{args.seed}_test.json"), "w")
dropped_converted_file = open(os.path.join(dst_path, f"train.json"), "w")

for idx in tqdm(range(2826)):
    NLP_line = NLP_file.readline()
    converted_line = converted_file.readline()

    NLP_data = json.loads(NLP_line)
    converted_data = json.loads(converted_line)

    sample_num = round(len(NLP_data["graph"]) * (1 - args.drop_rate))
    indices = random.sample(range(len(NLP_data["graph"])), k=sample_num)

    NLP_data["graph"] = [NLP_data["graph"][i] for i in indices]
    converted_data["subgraph"]["tuples"] = [converted_data["subgraph"]["tuples"][i] for i in indices]

    dropped_NLP_file.write(json.dumps(NLP_data) + "\n")
    dropped_converted_file.write(json.dumps(converted_data) + "\n")
    dropped_NLP_file.flush()
    dropped_converted_file.flush()


NLP_file.close()
converted_file.close()
dropped_NLP_file.close()
dropped_converted_file.close()

os.system(f"cp {os.path.join(dst_path, 'train.json')} {os.path.join(dst_path, 'test.json')}")

    



