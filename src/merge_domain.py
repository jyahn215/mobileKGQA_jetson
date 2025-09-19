import argparse
import os
from tqdm import tqdm
import json
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="cwq")
    parser.add_argument('--domain_list', nargs='+', type=str)
    parser.add_argument('--sample_num', nargs="+", type=int)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    random.seed(args.seed)

    merged_data_name = "^".join([str(d) for d in args.domain_list])
    merged_data_name += "@"
    merged_data_name += "^".join([str(s) for s in args.sample_num])
    merged_data_path = f"./data/preprocessed_data/{args.dataset_name}/_domains/{merged_data_name}"

    if os.path.exists(merged_data_path):
        print(f"Data for domain {merged_data_name} already exists")
    else:
        print(f"Create merged domain {merged_data_name}")
        os.makedirs(merged_data_path)

    sample_data_list = []
    for idx, domain in enumerate(args.domain_list):
        sample_num = args.sample_num[idx]
        with open(f"./data/preprocessed_data/{args.dataset_name}/_domains/{domain}/train.json", "r") as f1:
            data_list = [json.loads(line) for line in tqdm(f1.readlines())]
            sample_num = min(sample_num, len(data_list))
            print(f"Processing domain {domain} with sample number {sample_num}")
            sample_data_list += random.sample(data_list, sample_num)
            del data_list

    with open(os.path.join(merged_data_path, "train.json"), "w") as f:
        for data in sample_data_list:
            f.write(json.dumps(data) + "\n")
    print("Merged train.json")




    






