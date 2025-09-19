import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=str)
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--dataset_name", type=str, default="webqsp")
parser.add_argument("--model_name", type=str, default="relbert")
args = parser.parse_args()


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from tqdm import tqdm
import json
import torch
import pickle
import numpy as np

def get_max_query_token(dataset):

    data = []
    with open(f"./data/preprocessed_data/{dataset}/_domains/total/train.json", "r") as rf:
        data += [json.loads(line)["question"] for line in tqdm(rf.readlines(), ncols=100, desc="load train data")]
    with open(f"./data/preprocessed_data/{dataset}/_domains/total/dev.json", "r") as rf:
        data += [json.loads(line)["question"] for line in tqdm(rf.readlines(), ncols=100, desc="load dev data")]
    with open(f"./data/preprocessed_data/{dataset}/_domains/total/test.json", "r") as rf:
        data += [json.loads(line)["question"] for line in tqdm(rf.readlines(), ncols=100, desc="load test data")]

    if args.model_name == "gemma2:2b":
        tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it', cache_dir="./ckpts/llm")
        tokens = tokenizer(data, padding="longest", return_attention_mask=True, return_tensors="pt")
        print(f"Max query token of {dataset} in gemma2:2b: {tokens.input_ids.size(1)}")
        return tokens.input_ids.size(1)
    
    elif args.model_name == "qwen2:0.5b":
        tokenizer = AutoTokenizer.from_pretrained(
                "PrunaAI/Qwen-Qwen2-0.5B-Instruct-bnb-4bit-smashed", cache_dir="./ckpts/llm")
        tokens = tokenizer(data, padding="longest", return_attention_mask=True, return_tensors="pt")
        print(f"Max query token of {dataset} in qwen2:0.5b: {tokens.input_ids.size(1)}")
        return tokens.input_ids.size(1)

    elif args.model_name == "phi3.5":
        tokenizer = AutoTokenizer.from_pretrained(
                "unsloth/Phi-3.5-mini-instruct-bnb-4bit", cache_dir="./ckpts/llm")
        tokens = tokenizer(data, padding="longest", return_attention_mask=True, return_tensors="pt")
        print(f"Max query token of {dataset} in phi3.5: {tokens.input_ids.size(1)}")
        return tokens.input_ids.size(1)

    elif args.model_name == "llama3.1:8b":
        tokenizer = AutoTokenizer.from_pretrained(
                "unsloth/Meta-Llama-3.1-8B-bnb-4bit", cache_dir="./ckpts/llm")
        tokens = tokenizer(data, padding="longest", return_attention_mask=True, return_tensors="pt")
        print(f"Max query token of {dataset} in llama3.1: {tokens.input_ids.size(1)}")
        return tokens.input_ids.size(1)

    elif args.model_name == "relbert":
        tokenizer = AutoTokenizer.from_pretrained("./ckpts/pretrained_lms/sr-simbert")
        tokens = tokenizer(data, padding="longest", return_attention_mask=True, return_tensors="pt")
        print(f"Max query token of {dataset} in relbert: {tokens.input_ids.size(1)}")
        return tokens.input_ids.size(1)
    
    elif args.model_name == "gte-qwen2:1.5b":
        tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct", cache_dir="./ckpts/llm")
        tokens = tokenizer(data, padding="longest", return_attention_mask=True, return_tensors="pt")
        print(f"Max query token of {dataset} in gte-Qwen2:1.5b: {tokens.input_ids.size(1)}")
        return tokens.input_ids.size(1)
    
    elif args.model_name == "gte-large":
        tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", cache_dir="./ckpts/llm")
        tokens = tokenizer(data, padding="longest", return_attention_mask=True, return_tensors="pt")
        print(f"Max query token of {dataset} in gte-large: {tokens.input_ids.size(1)}")
        return tokens.input_ids.size(1)


def load_tokenizer_model(model_name, dataset):

    if model_name == "gemma2:2b":
        tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it', cache_dir="./ckpts/llm")
        model = AutoModelForCausalLM.from_pretrained(
            'google/gemma-2-2b-it',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16), 
            low_cpu_mem_usage=True,
            cache_dir="./ckpts/llm")
        if dataset == "webqsp":
            max_query_token = 19
        elif dataset == "cwq":
            max_query_token = 44
        elif dataset == "metaqa-1hop":
            max_query_token = 30
        elif dataset == "metaqa-2hop":
            max_query_token = 27
        elif dataset == "metaqa-3hop":
            max_query_token = 32

    elif model_name == "qwen2:0.5b":
        tokenizer = AutoTokenizer.from_pretrained(
            "PrunaAI/Qwen-Qwen2-0.5B-Instruct-bnb-4bit-smashed", cache_dir="./ckpts/llm")
        model = AutoModelForCausalLM.from_pretrained(
            "PrunaAI/Qwen-Qwen2-0.5B-Instruct-bnb-4bit-smashed",
            low_cpu_mem_usage=True,
            cache_dir="./ckpts/llm")
        if dataset == "webqsp":
            max_query_token = 19
        elif dataset == "cwq":
            max_query_token = 42
        elif dataset == "metaqa-1hop":
            max_query_token = 29
        elif dataset == "metaqa-2hop":
            max_query_token = 26
        elif dataset == "metaqa-3hop":
            max_query_token = 32

    elif model_name == "phi3.5":
        tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/Phi-3.5-mini-instruct-bnb-4bit", cache_dir="./ckpts/llm")
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
            low_cpu_mem_usage=True,
            cache_dir="./ckpts/llm")
        if dataset == "webqsp":
            max_query_token = 21
        elif dataset == "cwq":
            max_query_token = 50
        elif dataset == "metaqa-1hop":
            max_query_token = 31
        elif dataset == "metaqa-2hop":
            max_query_token = 32
        elif dataset == "metaqa-3hop":
            max_query_token = 38

    elif model_name == "llama3.1":
        tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/Meta-Llama-3.1-8B-bnb-4bit", cache_dir="./ckpts/llm")
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
            low_cpu_mem_usage=True,
            cache_dir="./ckpts/llm")
        if dataset == "webqsp":
            max_query_token = 19
        elif dataset == "cwq":
            max_query_token = 38
        elif dataset == "metaqa-1hop":
            max_query_token = 29
        elif dataset == "metaqa-2hop":
            max_query_token = 27
        elif dataset == "metaqa-3hop":
            max_query_token = 33

    elif model_name == "sbert":
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2", cache_dir="./ckpts/llm")
        model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            low_cpu_mem_usage=True,
            cache_dir="./ckpts/llm").cuda()
        if dataset == "webqsp":
            max_query_token = 17
        if dataset == "cwq":
            max_query_token = 42

    elif model_name == "relbert":
        tokenizer = AutoTokenizer.from_pretrained("./ckpts/pretrained_lms/sr-simbert")
        model = AutoModel.from_pretrained("./ckpts/pretrained_lms/sr-simbert").cuda()
        if dataset == "webqsp":
            max_query_token = 17
        if dataset == "cwq":
            max_query_token = 42
    
    elif model_name == "gte-large":
        tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", cache_dir="./ckpts/llm")
        model = AutoModel.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", cache_dir="./ckpts/llm", trust_remote_code=True).cuda()
        if dataset == "webqsp":
            max_query_token = 17
        if dataset == "cwq":
            max_query_token = 42

    elif model_name == "gte-qwen2:1.5b":
        tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct", cache_dir="./ckpts/llm")
        model = AutoModelForCausalLM.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct", cache_dir="./ckpts/llm", trust_remote_code=True).cuda()
        if dataset == "webqsp":
            max_query_token = 19
        if dataset == "cwq":
            max_query_token = 42
    
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return tokenizer, model, max_query_token


def save_last_hidden_state():
    batch_size = 10
    model_name = args.model_name
    dataset = args.dataset_name
    domain = args.domain
    split = args.split

    print(f"Processing {dataset} dataset with {model_name} model")
    tokenizer, model, max_query_token = load_tokenizer_model(model_name, dataset)

    rel_dir = f"./data/preprocessed_data/{dataset}/last_hidden_state/{model_name}/ori/rel/"
    if not os.path.exists(rel_dir):
        os.makedirs(rel_dir)
        skip_rel = False
    else:
        skip_rel = True

    rel_inv_dir = f"./data/preprocessed_data/{dataset}/last_hidden_state/{model_name}/ori/rel_inv/"
    if not os.path.exists(rel_inv_dir):
        os.makedirs(rel_inv_dir)
        skip_rel_inv = False
    else:
        skip_rel_inv = True


    if dataset == "webqsp" or dataset == "cwq":
        with open(f"./data/preprocessed_data/{dataset}/rel2idx.pkl", "rb") as rf:
            rel2idx = pickle.load(rf)
        rel_words = []
        for rel in tqdm(rel2idx):
            rel = rel.strip()
            fields = rel.split(".")
            try:
                words = fields[-2].split('_') + fields[-1].split('_')
                rel_words.append(words)
            except:
                words = ['UNK']
                rel_words.append(words)
                pass
    
    if skip_rel is False:
        rel_text = [" ".join(words) for words in rel_words]
        tokens = tokenizer(rel_text, padding=True, return_attention_mask=True, return_tensors="pt")
        torch.save(tokens.attention_mask, f"{rel_dir}/rel_mask.pt")
        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()
        rel_all = []
        for idx in tqdm(range(0, input_ids.size(0), batch_size), desc="save relation last hidden state"):
            batch = input_ids[idx:idx+batch_size]
            batch_mask = attention_mask[idx:idx+batch_size]
            outputs = model(input_ids=batch, attention_mask=batch_mask, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1] # (batch_size, seq_len, hidden_size)
            last_hidden_state = last_hidden_state.cpu().detach()
            rel_all.append(last_hidden_state)
            last_hidden_state = last_hidden_state.numpy()
            for i, rep in enumerate(last_hidden_state):
                num = idx + i
                np.save(f"{rel_dir}/rel_{num}.npy", rep)
            del outputs, last_hidden_state, batch
        rel_all = torch.cat(rel_all, dim=0)
        torch.save(rel_all, f"{rel_dir}/rel_all.pt")
        del rel_all
    
    if skip_rel_inv is False:
        rel_text_inv = [" ".join(words[::-1]) for words in rel_words]
        tokens_inv = tokenizer(rel_text_inv, padding=True, return_attention_mask=True, return_tensors="pt")
        torch.save(tokens_inv.attention_mask, f"{rel_inv_dir}/rel_inv_mask.pt")
        input_ids_inv = tokens_inv.input_ids.cuda()
        attention_mask_inv = tokens_inv.attention_mask.cuda()
        rel_inv_all = []
        for idx in tqdm(range(0, input_ids_inv.size(0), batch_size), desc="save inverse relation last hidden state"):
            batch_inv = input_ids_inv[idx:idx+batch_size]
            batch_mask_inv = attention_mask_inv[idx:idx+batch_size]
            outputs = model(input_ids=batch_inv, attention_mask=batch_mask_inv, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            last_hidden_state = last_hidden_state.cpu().detach()
            rel_inv_all.append(last_hidden_state)
            last_hidden_state = last_hidden_state.numpy()
            for i, rep in enumerate(last_hidden_state):
                num = idx + i
                np.save(f"{rel_inv_dir}/rel_inv_{num}.npy", rep)
            del outputs, last_hidden_state, batch_inv, batch_mask_inv
        rel_inv_all = torch.cat(rel_inv_all, dim=0)
        torch.save(rel_inv_all, f"{rel_inv_dir}/rel_inv_all.pt")
        del rel_inv_all
    
    if skip_rel is False and skip_rel_inv is False:
        tokens_pad = tokenizer("", padding="max_length", max_length=tokens.input_ids.size(1), return_attention_mask=True, return_tensors="pt")
        torch.save(tokens_pad.attention_mask, f"{rel_dir}/rel_pad_mask.pt")
        torch.save(tokens_pad.attention_mask, f"{rel_inv_dir}/rel_inv_pad_mask.pt")
        input_ids_pad = tokens_pad.input_ids.cuda()
        attention_mask_pad = tokens_pad.attention_mask.cuda()
        outputs = model(input_ids=input_ids_pad, attention_mask=attention_mask_pad, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        torch.save(last_hidden_state, f"{rel_dir}/rel_pad.pt")
        torch.save(last_hidden_state, f"{rel_inv_dir}/rel_inv_pad.pt")

    print(f"Processing {domain} {split} split")
    query_dir = f"./data/preprocessed_data/{dataset}/last_hidden_state/{model_name}/ori/queries/{domain}/{split}"
    if not os.path.exists(query_dir):
        os.makedirs(query_dir)

    if not os.path.exists(f"./data/preprocessed_data/{dataset}/_domains/{domain}/{split}.json"):
        print(f"File not found: {dataset}/_domains/{domain}/{split}.json. Skip {dataset} {domain} {split} split")
    else:
        with open(f"./data/preprocessed_data/{dataset}/_domains/{domain}/{split}.json", "r") as rf:
            data = [json.loads(line)["question"].strip() for line in rf.readlines()]

        for idx in tqdm(range(0, len(data), batch_size)):
            batch = data[idx:idx+batch_size]
            inputs = tokenizer(batch, 
                                padding="max_length",
                                max_length=max_query_token,
                                return_attention_mask=True,
                                truncation=True, 
                                return_tensors="pt")
            
            assert inputs.attention_mask.size(1) == max_query_token
            for i in range(batch_size):
                if i < len(inputs.attention_mask):
                    torch.save(inputs.attention_mask[i], f"{query_dir}/{idx+i}_mask.pt")
                else:
                    break

            outputs = model(inputs.input_ids.cuda(), inputs.attention_mask.cuda(), output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            del outputs 
            assert last_hidden_state.size(1) == max_query_token
            for i in range(batch_size):
                if i < len(last_hidden_state):
                    torch.save(last_hidden_state[i], f"{query_dir}/{idx+i}.pt")
                else:
                    break
            
if __name__ == "__main__":
    # get_max_query_token(args.dataset)
    with torch.no_grad():
        save_last_hidden_state()


