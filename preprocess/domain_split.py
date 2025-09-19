import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import datasets
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
from langchain_community.embeddings import OllamaEmbeddings


def save_query_rep(args):
    total_query_rep_path = f"./data/preprocessed_data/{args.dataset}/embeds/total_query_rep_{args.model_name}_{args.dataset}.npy"
    total_meta_data_path = f"./data/preprocessed_data/{args.dataset}/_domains/total/total.txt"

    print(f"query_rep_path: {total_query_rep_path}")
    print(f"meta_data_path: {total_meta_data_path}")
    if not os.path.exists(total_query_rep_path) or not os.path.exists(total_meta_data_path):
        print("start generating embeddings")

        embedding_model = OllamaEmbeddings(
            model=args.model_name,
            base_url=f"http://localhost:{args.port}"
        )
        
        if not os.path.exists(f"./data/preprocessed_data/{args.dataset}/embeds"):
            os.makedirs(f"./data/preprocessed_data/{args.dataset}/embeds")

        total_domain_meta_data, qid = [], 0
        for split in ["train", "dev", "test"]:
            domain_meta_data, cnt = [], 0
            with open(f"./data/preprocessed_data/{args.dataset}/_domains/total/{split}.json", "r") as rf:
                # ollama embedding does not support batch processing yet
                for line in tqdm(rf.readlines()):
                    data = json.loads(line)
                    domain_meta_data.append([split, data["id"], cnt])
                    embeds = embedding_model.embed_query(data["question"])
                    np.save(f"./data/preprocessed_data/{args.dataset}/embeds/{args.model_name}/{qid}.npy", embeds)
                    qid += 1
                    cnt += 1
            np.savetxt(f"./data/preprocessed_data/{args.dataset}/_domains/total/{split}.txt", domain_meta_data, fmt="%s")
            total_domain_meta_data.extend(domain_meta_data)
        total_domain_meta_data = np.array(total_domain_meta_data) 
        np.savetxt(f"./data/preprocessed_data/{args.dataset}/_domains/total/total.txt", total_domain_meta_data, fmt="%s")

        embeds = []
        for _qid in tqdm(range(len(total_domain_meta_data))):
            embeds.append(np.load(f"./data/preprocessed_data/{args.dataset}/embeds/{args.model_name}/{_qid}.npy"))
            os.system(f"rm -rf ./data/preprocessed_data/{args.dataset}/embeds/{args.model_name}/{_qid}.npy")
        embeds = np.stack(embeds)
        np.save(f"./data/preprocessed_data/{args.dataset}/embeds/total_query_rep_{args.model_name}.npy", embeds)

    else:
        print(f"{total_query_rep_path} and {total_meta_data_path} already exist")
        embeds = np.load(total_query_rep_path)
        total_domain_meta_data = np.loadtxt(f"./data/preprocessed_data/{args.dataset}/_domains/total/total.txt", dtype=str)

    return embeds, total_domain_meta_data

def load_query_rep(args):
    if not os.path.exists(f"./data/preprocessed_data/{args.dataset}/embeds/total_query_rep_{args.model_name}.npy"):
        if args.dataset == "webqsp":
            hg_dataset = "rmanluo/RoG-webqsp"
        elif args.dataset == "cwq":
            hg_dataset = "rmanluo/RoG-cwq"

        rep_list, idx2query = [], dict()
        for split in ["train", "validation", "test"]:
            path = f"./data/preprocessed_data/{args.dataset}/embeds/{split}"
            dataset = datasets.load_dataset(hg_dataset, split=split)
            for idx, data in enumerate(tqdm(dataset)):
                query_id = data["id"]
                idx2query[len(rep_list)] = (split, query_id, idx)
                rep = np.load(os.path.join(path, f"{idx}.npy"))
                rep_list.append(rep)
        rep = np.stack(rep_list)
        # np.save(f"./data/preprocessed_data/{args.dataset}/embeds/total_query_rep.npy", rep)
        with open(f"./data/preprocessed_data/{args.dataset}/idx2query.pkl", "wb") as f:
            pickle.dump(idx2query, f)
    else:
        rep = np.load(f"./data/preprocessed_data/{args.dataset}/embeds/total_query_rep.npy")
        with open(f"./data/preprocessed_data/{args.dataset}/idx2query.pkl", "rb") as f:
            idx2query = pickle.load(f)

    return rep, idx2query

def split_domain(args, embedding, meta_data_array, cluster_num=3):
    print("clustering started")
    kmeans = KMeans(n_clusters=cluster_num, random_state=args.seed, max_iter=3000)
    labels = kmeans.fit_predict(embedding)
    print("clustering ended")

    domain1_array = meta_data_array[labels == 0]
    d1_num = len(domain1_array)
    domain2_array = meta_data_array[labels == 1]
    d2_num = len(domain2_array)
    domain3_array = meta_data_array[labels == 2]
    d3_num = len(domain3_array)
    print(f"D1: {d1_num}, D2: {d2_num}, D3: {d3_num}")
    
    # ratio (5:1:4)
    np.random.shuffle(domain1_array)
    domain1_train = domain1_array[:d1_num // 2]
    domain1_valid = domain1_array[d1_num // 2:d1_num // 2 + int(d1_num * 0.1)]
    domain1_test = domain1_array[d1_num // 2 + int(d1_num * 0.1):]
    np.random.shuffle(domain2_array)
    domain2_train = domain2_array[:d2_num // 2]
    domain2_valid = domain2_array[d2_num // 2: d2_num // 2 + int(d2_num * 0.1)]
    domain2_test = domain2_array[d2_num // 2 + int(d2_num * 0.1):]
    np.random.shuffle(domain3_array)
    domain3_train = domain3_array[:d3_num // 2]
    domain3_valid = domain3_array[d3_num // 2: d3_num // 2 + int(d3_num * 0.1)]
    domain3_test = domain3_array[d3_num // 2 + int(d3_num * 0.1):]

    print(f"D1_train: {len(domain1_train)}, D1_valid: {len(domain1_valid)}, D1_test:{len(domain1_test)}\n"
    f"D2_train: {len(domain2_train)}, D2_valid: {len(domain2_valid)}, D2_test: {len(domain2_test)}\n"
    f"D3_train: {len(domain3_train)}, D3_valid: {len(domain3_valid)}, D3_test: {len(domain3_test)}")

    np.savetxt(f"./data/preprocessed_data/{args.dataset}/_domains/domain1/train.txt", domain1_train, fmt="%s")
    np.savetxt(f"./data/preprocessed_data/{args.dataset}/_domains/domain1/dev.txt", domain1_valid, fmt="%s")
    np.savetxt(f"./data/preprocessed_data/{args.dataset}/_domains/domain1/test.txt", domain1_test, fmt="%s")
    np.savetxt(f"./data/preprocessed_data/{args.dataset}/_domains/domain2/train.txt", domain2_train, fmt="%s")
    np.savetxt(f"./data/preprocessed_data/{args.dataset}/_domains/domain2/dev.txt", domain2_valid, fmt="%s")
    np.savetxt(f"./data/preprocessed_data/{args.dataset}/_domains/domain2/test.txt", domain2_test, fmt="%s")
    np.savetxt(f"./data/preprocessed_data/{args.dataset}/_domains/domain3/train.txt", domain3_train, fmt="%s")
    np.savetxt(f"./data/preprocessed_data/{args.dataset}/_domains/domain3/dev.txt", domain3_valid, fmt="%s")
    np.savetxt(f"./data/preprocessed_data/{args.dataset}/_domains/domain3/test.txt", domain3_test, fmt="%s")

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(embedding)
    print("pca ended")

    plt.figure(figsize=(5, 5))
    colors = ['red', 'green', 'blue']
    for i in range(3):
        cluster_point = data_pca[labels == i]
        plt.scatter(cluster_point[:, 0], cluster_point[:, 1], color=colors[i], label=f"D{i+1}", s=5)

    # plt.title(f"domain split with PCA visualization ({args.dataset})", fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(f"./asset/domain_split_{args.dataset}.png")

    return domain1_train[:, 1], domain1_valid[:, 1], domain1_test[:, 1], \
        domain2_train[:, 1], domain2_valid[:, 1], domain2_test[:, 1], \
            domain3_train[:, 1], domain3_valid[:, 1], domain3_test[:, 1]

def save_data(split_result, args):
    d1_train = open(f"./data/preprocessed_data/{args.dataset}/_domains/domain1/train.json", "w")
    d1_valid = open(f"./data/preprocessed_data/{args.dataset}/_domains/domain1/dev.json", "w")
    d1_test = open(f"./data/preprocessed_data/{args.dataset}/_domains/domain1/test.json", "w")
    d2_train = open(f"./data/preprocessed_data/{args.dataset}/_domains/domain2/train.json", "w")
    d2_valid = open(f"./data/preprocessed_data/{args.dataset}/_domains/domain2/dev.json", "w")
    d2_test = open(f"./data/preprocessed_data/{args.dataset}/_domains/domain2/test.json", "w")
    d3_train = open(f"./data/preprocessed_data/{args.dataset}/_domains/domain3/train.json", "w")
    d3_valid = open(f"./data/preprocessed_data/{args.dataset}/_domains/domain3/dev.json", "w")
    d3_test = open(f"./data/preprocessed_data/{args.dataset}/_domains/domain3/test.json", "w")

    (d1_train_list, d1_valid_list, d1_test_list, 
    d2_train_list, d2_valid_list, d2_test_list, 
    d3_train_list, d3_valid_list, d3_test_list)= split_result

    for split in ["train", "dev", "test"]:
        with open(f"./data/preprocessed_data/{args.dataset}/_domains/total/{split}.json", "r") as f:
            data = f.readlines()
            for line in tqdm(data, desc=f"Processing {split}"):
                line = json.loads(line)
                query_id = line["id"]
                if query_id in d1_train_list:
                    d1_train.write(json.dumps(line) + "\n")
                elif query_id in d1_valid_list:
                    d1_valid.write(json.dumps(line) + "\n")
                elif query_id in d1_test_list:
                    d1_test.write(json.dumps(line) + "\n")
                elif query_id in d2_train_list:
                    d2_train.write(json.dumps(line) + "\n")
                elif query_id in d2_valid_list:
                    d2_valid.write(json.dumps(line) + "\n")
                elif query_id in d2_test_list:
                    d2_test.write(json.dumps(line) + "\n")
                elif query_id in d3_train_list:
                    d3_train.write(json.dumps(line) + "\n")
                elif query_id in d3_valid_list:
                    d3_valid.write(json.dumps(line) + "\n")
                elif query_id in d3_test_list:
                    d3_test.write(json.dumps(line) + "\n")
    
    d1_train.close()
    d1_valid.close()
    d1_test.close()
    d2_train.close()
    d2_valid.close()
    d2_test.close()
    d3_train.close()
    d3_valid.close()
    d3_test.close()

def calcul_entities_and_relations_in_each_domain(args):
    for domain in ["total", "domain1", "domain2", "domain3"]:
        ent_set, rel_set, question_cnt, triple_set = set(), set(), 0, set()
        for split in ["train", "dev", "test"]:
            with open(f"./data/preprocessed_data/{args.dataset}/_domains/{domain}/{split}.json", "r") as rf:
                for line in tqdm(rf.readlines()):
                    data = json.loads(line)
                    for head, rel, tail in data["subgraph"]["tuples"]:
                        ent_set.add(head)
                        ent_set.add(tail)
                        rel_set.add(rel)
                        triple_set.add((head, rel, tail))
                    question_cnt += 1
        if domain != "total":
            with open(f"./data/preprocessed_data/{args.dataset}/_domains/{domain}/rel_in_domain.json", "w") as f:
                json.dump(list(rel_set), f, indent=4)
        print(f"{domain} questions: {question_cnt}, entities: {len(ent_set)}, relations: {len(rel_set)}, triples: {len(triple_set)}")




if __name__ == "__main__":

    def save_query_meta_data(args):
        total_meta_data_path = f"./data/preprocessed_data/{args.dataset}/_domains/total/total.txt"
        print(f"meta_data_path: {total_meta_data_path}")
        if not os.path.exists(total_meta_data_path):
            total_domain_meta_data, qid = [], 0
            for split in ["train", "dev", "test"]:
                domain_meta_data, cnt = [], 0
                with open(f"./data/preprocessed_data/{args.dataset}/_domains/total/{split}.json", "r") as rf:
                    for line in tqdm(rf.readlines()):
                        data = json.loads(line)
                        domain_meta_data.append([split, data["id"], cnt])
                        qid += 1
                        cnt += 1
                np.savetxt(f"./data/preprocessed_data/{args.dataset}/_domains/total/{split}.txt", domain_meta_data, fmt="%s")
                total_domain_meta_data.extend(domain_meta_data)
            total_domain_meta_data = np.array(total_domain_meta_data) 
            np.savetxt(f"./data/preprocessed_data/{args.dataset}/_domains/total/total.txt", total_domain_meta_data, fmt="%s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="webqsp")
    parser.add_argument("--port", type=int, default=13000)
    parser.add_argument("--model_name", type=str, default="llama3.1")
    args = parser.parse_args()

    if not os.path.exists(f"./data/preprocessed_data/{args.dataset}/_domains/domain1"):
        os.makedirs(f"./data/preprocessed_data/{args.dataset}/_domains/domain1")
    if not os.path.exists(f"./data/preprocessed_data/{args.dataset}/_domains/domain2"):
        os.makedirs(f"./data/preprocessed_data/{args.dataset}/_domains/domain2")
    if not os.path.exists(f"./data/preprocessed_data/{args.dataset}/_domains/domain3"):
        os.makedirs(f"./data/preprocessed_data/{args.dataset}/_domains/domain3")

    args.seed = 54
    np.random.seed(args.seed)

    save_query_meta_data(args)
    rep, meta_data_array = save_query_rep(args)
    split_result = split_domain(args, rep, meta_data_array)
    save_data(split_result, args)
    calcul_entities_and_relations_in_each_domain(args)