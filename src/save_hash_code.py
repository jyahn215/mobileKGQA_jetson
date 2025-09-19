import os
from module.hashing_module import Hashing_Module
import torch
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-domain", type=str)
parser.add_argument("-split", type=str, default="train")
parser.add_argument("-model_name", type=str, default="relbert")
parser.add_argument("-dataset_name", type=str, default="webqsp")
parser.add_argument("-bit", type=str, default="256")
args = parser.parse_args()

best_config = {
    "relbert":{
        "webqsp": {
            "64":  "SH-05-12-06-54-17-448509",
            "128": "SH-05-12-06-50-03-993706",
            "256": "SH-05-12-06-43-59-673807",
            "512": "SH-05-12-07-03-32-220491"
            },
        "cwq": {
            "256": "SH-05-12-08-02-45-707169",
            "64": "SH-05-12-08-07-15-770050",
            "128": "SH-05-12-08-03-36-770123",
            "512": "SH-05-12-07-58-52-466091"
        }
    },
    "gte-large":{
        "webqsp": {
            "256": "SH-05-12-10-50-58-312680"
            },
        "cwq": {
            "256": "SH-05-12-10-54-22-882763"
        }
    },
    "gte-qwen2:1.5b":{
        "webqsp": {
            "64": "SH-05-12-10-40-10-051973",
            "128": "SH-05-12-10-38-03-626568",
            "256": "SH-05-12-10-36-16-815785",
            "512": "SH-05-12-10-34-11-354090"
            },
        "cwq": {
            "256": "SH-05-12-10-25-56-598732",
            "64": "SH-05-12-10-31-41-804748",
            "128": "SH-05-12-10-30-23-438145",
            "512": "SH-05-12-10-24-30-145170"
        }
    }
}


def save_hash_code():
    
    model_name = args.model_name
    dataset_name = args.dataset_name
    bit = args.bit
    config_name = best_config[model_name][dataset_name][bit]
    domain = args.domain
    split = args.split

    src_path = f"./data/preprocessed_data/{dataset_name}/last_hidden_state/{model_name}/ori/"
    path = f"./data/preprocessed_data/{dataset_name}/last_hidden_state/{model_name}/{bit}bit/"
    print("save hash codes for model:", model_name, "dataset:", dataset_name, "bit:", bit)
    print("path: ", path)

    ckpts = torch.load(f"./ckpts/hashing/{config_name}/SearchEncoder.pt")
    state_dict = ckpts["model_state_dict"]
    assert int(bit) == state_dict["model.0.weight"].shape[0]
    rep_dim = ckpts["args"].dim_list[0]
    assert rep_dim == state_dict["model.0.weight"].shape[1]
    ckpts["args"].dim_list = ckpts["args"].dim_list[1:]
    model = Hashing_Module(ckpts["args"], rep_dim)
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    if not os.path.exists(path):
        os.makedirs(os.path.join(path, "queries"))
        os.makedirs(os.path.join(path, "rel"))
        os.makedirs(os.path.join(path, "rel_inv"))

    if not os.path.exists(os.path.join(path, "rel", "rel_all.pt")):
        rel_all = torch.load(os.path.join(src_path, "rel", "rel_all.pt")).to(torch.float32)
        length = rel_all.size(1)
        rel_all = rel_all.reshape(-1, rel_all.size(-1)).cuda()
        rel_all_hash = torch.sign(model(rel_all)).cpu().detach()
        rel_all_hash = rel_all_hash.reshape(-1, length, int(bit))
        rel_all_hash = rel_all_hash.to(torch.int8)
        torch.save(rel_all_hash, f"{path}/rel/rel_all.pt")

        rel_pad_hash = torch.zeros((1, rel_all_hash.size(1), int(bit))).to(torch.int8)
        torch.save(rel_pad_hash, f"{path}/rel/rel_pad.pt")

        rel_mask = torch.load(os.path.join(src_path, "rel", "rel_mask.pt")).to(torch.float32)
        torch.save(rel_mask, f"{path}/rel/rel_mask.pt")

        rel_inv_mask = torch.load(os.path.join(src_path, "rel", "rel_pad_mask.pt")).to(torch.float32)
        torch.save(rel_inv_mask, f"{path}/rel/rel_pad_mask.pt")

        print("rel hash done")

    if not os.path.exists(os.path.join(path, "rel_inv", "rel_inv_all.pt")):
        rel_inv_all = torch.load(os.path.join(src_path, "rel_inv", "rel_inv_all.pt")).to(torch.float32)
        length = rel_inv_all.size(1)
        rel_inv_all = rel_inv_all.reshape(-1, rel_inv_all.size(-1)).cuda()
        rel_inv_all_hash = torch.sign(model(rel_inv_all)).cpu().detach()
        rel_inv_all_hash = rel_inv_all_hash.reshape(-1, length, int(bit))
        rel_inv_all_hash = rel_inv_all_hash.to(torch.int8)
        torch.save(rel_inv_all_hash, f"{path}/rel_inv/rel_inv_all.pt")

        rel_inv_pad_hash = torch.zeros((1, rel_inv_all_hash.size(1), int(bit))).to(torch.int8)
        torch.save(rel_inv_pad_hash, f"{path}/rel_inv/rel_inv_pad.pt")

        rel_inv_mask = torch.load(os.path.join(src_path, "rel_inv", "rel_inv_mask.pt")).to(torch.float32)
        torch.save(rel_inv_mask, f"{path}/rel_inv/rel_inv_mask.pt")

        rel_inv_pad_mask = torch.load(os.path.join(src_path, "rel_inv", "rel_inv_pad_mask.pt")).to(torch.float32)
        torch.save(rel_inv_pad_mask, f"{path}/rel_inv/rel_inv_pad_mask.pt")

        print("rel_inv hash done")

    if not os.path.exists(os.path.join(path, "queries", domain, split)):
        os.makedirs(os.path.join(path, "queries", domain, split))
        print(f"{os.path.join(path, 'queries', domain, split)} created")

        query_num = len(os.listdir(os.path.join(src_path, "queries", domain, split))) // 2
        for i in tqdm(range(query_num)):
            query_rep = torch.load(os.path.join(src_path, "queries", domain, split, f"{i}.pt")).to(torch.float32)
            query_rep = query_rep.to("cuda") if not query_rep.is_cuda else query_rep
            query_rep_hash = torch.sign(model(query_rep)).cpu().detach()
            query_rep_hash = query_rep_hash.to(torch.int8)
            torch.save(query_rep_hash, os.path.join(path, "queries", domain, split, f"{i}.pt"))

            query_mask = torch.load(os.path.join(src_path, "queries", domain, split, f"{i}_mask.pt")).to(torch.float32)
            torch.save(query_mask, os.path.join(path, "queries", domain, split, f"{i}_mask.pt"))
    else:
        print(f"{os.path.join(path, 'queries', domain, split)} already exists")
    


if __name__ == "__main__":
    save_hash_code()