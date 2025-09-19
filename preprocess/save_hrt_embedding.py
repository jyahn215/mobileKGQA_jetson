import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default=1)
parser.add_argument("--dataset", type=str, default="cwq")
parser.add_argument("--model_name", type=str, default="relbert")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from torch import Tensor
import pickle
import numpy as np
from tqdm import tqdm


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_node_embeddings(args, last_hidden, attention_mask):
    if args.model_name in ["relbert", "gte-large"]:
        return last_hidden[:, 0, :]
    elif args.model_name in ["gte-qwen2:1.5b"]:
        return last_token_pool(last_hidden, attention_mask) 
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

def load_model_and_tokenizer(args):
    if args.model_name == "relbert":
        tokenizer = AutoTokenizer.from_pretrained("./ckpts/pretrained_lms/sr-simbert")
        model = AutoModel.from_pretrained("./ckpts/pretrained_lms/sr-simbert").cuda()
    elif args.model_name == "gte-large":
        tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", cache_dir="./ckpts/llm")
        model = AutoModel.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", cache_dir="./ckpts/llm", trust_remote_code=True).cuda()
    elif args.model_name == "gte-qwen2:1.5b":
        tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct", cache_dir="./ckpts/llm")
        model = AutoModel.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct", cache_dir="./ckpts/llm", trust_remote_code=True).cuda()
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    return tokenizer, model

def save_embeddings_in_batch(ent2idx, tokenizer, model, save_path, batch_size=256):
    text_list = list(ent2idx.keys())
    for i in tqdm(range(0, len(text_list), batch_size), desc="Processing batches", ncols=100):
        batch_text = text_list[i:i + batch_size]
        batch_inputs = tokenizer(batch_text, padding=True, return_tensors="pt", truncation=True).to("cuda")
        with torch.no_grad():
            last_hidden = model(**batch_inputs).last_hidden_state
            node_embeddings = get_node_embeddings(args, last_hidden, batch_inputs["attention_mask"])
            node_embeddings = node_embeddings.cpu().numpy()
        
        for j, text in enumerate(batch_text):
            idx = ent2idx[text]
            if not os.path.exists(f"{save_path}/{idx - idx % 50000}"):
                os.makedirs(f"{save_path}/{idx - idx % 50000}")
            np.save(f"{save_path}/{idx - idx % 50000}/{idx}.npy", node_embeddings[j])


def save_hrt_embeddings(args):
    if not os.path.exists(f"./data/preprocessed_data/{args.dataset}/embeddings/{args.model_name}/ori/ent"):
        os.makedirs(f"./data/preprocessed_data/{args.dataset}/embeddings/{args.model_name}/ori/ent")
    if not os.path.exists(f"./data/preprocessed_data/{args.dataset}/embeddings/{args.model_name}/ori/rel"):
        os.makedirs(f"./data/preprocessed_data/{args.dataset}/embeddings/{args.model_name}/ori/rel")

    tokenizer, model = load_model_and_tokenizer(args)

    with open(f"./data/preprocessed_data/{args.dataset}/ent2idx.pkl", "rb") as f:
        ent2idx = pickle.load(f)
    with open(f"./data/preprocessed_data/{args.dataset}/rel2idx.pkl", "rb") as f:
        rel2idx = pickle.load(f)
    
    save_embeddings_in_batch(ent2idx, tokenizer, model, 
                             f"./data/preprocessed_data/{args.dataset}/embeddings/{args.model_name}/ori/ent", batch_size=256)
    save_embeddings_in_batch(rel2idx, tokenizer, model,
                                f"./data/preprocessed_data/{args.dataset}/embeddings/{args.model_name}/ori/rel", batch_size=256)

if __name__ == "__main__":
    save_hrt_embeddings(args)