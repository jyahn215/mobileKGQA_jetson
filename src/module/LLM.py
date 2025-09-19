import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from torch import nn
import torch
import json
from tqdm import tqdm


class BASE_MODEL:
    def  __init__(self, max_token=512):
        self.max_token = max_token
        self.cumul_token_cnt = 0
        self.cumul_call_cnt = 0

    def convert_dict_into_text(self, message, prompt=False):
        if type(message) == list and type(message[0]) == dict:
            if prompt is True:
                text = ""
                for data in message:
                    speaker, content = data["role"], data["content"]
                    text += f"{speaker}: "
                    text += f"{content}\n\n"
            else:
                text = message[-1]["content"]
        else:
            raise ValueError("Invalid input")

        return text

    def generate_chat_response(self, message):
        if type(message) == list and type(message[0]) == dict:
            input_text = self.convert_dict_into_text(message)
        elif type(message) == str:
            input_text = message
        else:
            raise ValueError("Input message should be a list of dictionaries or a string.")
        input_ids = self.tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **input_ids, 
            max_new_tokens=self.max_token,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            length_penalty=-0.5,
            pad_token_id=self.tokenizer.eos_token_id)
        num_generated_tokens = outputs.shape[1] - input_ids['input_ids'].shape[1]
        self.cumul_token_cnt += num_generated_tokens
        self.cumul_call_cnt += 1
        print(f"cumul_token_cnt: {self.cumul_token_cnt}, cumul_call_cnt: {self.cumul_call_cnt}")
        outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return outputs[len(input_text):] 

    def get_last_hidden_state(self, message):
        if type(message) == list and type(message[0]) == dict:
            input_text = self.convert_dict_into_text(message)
        elif type(message) == str:
            input_text = message
        else:
            raise ValueError("Input message should be a list of dictionaries or a string.")
        input_ids = self.tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = self.model(**input_ids, output_hidden_states=True)
        return outputs.hidden_states[-1]

    def get_last_hidden_state_wo_tokenizer(self, input_ids):
        '''
        input_ids: torch.tensor (batch_size, seq_len)
        return: torch.tensor (batch_size, seq_len, hidden_size)
        '''
        outputs = self.model(input_ids, output_hidden_states=True)
        return outputs["hidden_states"][0]

    def freeze_LLM(self):
        print("Freezing LM params")
        for param in self.model.parameters():
            param.requires_grad = False


class Gemma2_2B(BASE_MODEL):
    def __init__(self, max_token=256, only_tokenizer=False):
        super().__init__(max_token)
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", cache_dir="./ckpts/llm")
        self.model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16), cache_dir="./ckpts/llm"
        )


class Qwen2_0_5B(BASE_MODEL):
    def __init__(self, max_token=256, only_tokenizer=False):
        super().__init__(max_token)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "PrunaAI/Qwen-Qwen2-0.5B-Instruct-bnb-4bit-smashed", cache_dir="./ckpts/llm")
        self.model = AutoModelForCausalLM.from_pretrained(
            "PrunaAI/Qwen-Qwen2-0.5B-Instruct-bnb-4bit-smashed",
            low_cpu_mem_usage=True, cache_dir="./ckpts/llm")


class LLAMA3_1_8B(BASE_MODEL):
    def __init__(self, max_token=256, only_tokenizer=False):
        super().__init__(max_token)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/Meta-Llama-3.1-8B-bnb-4bit")
        self.model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
            low_cpu_mem_usage=True)

class Phi3_5_mini_3_8B(BASE_MODEL):
    def __init__(self, max_token=256, only_tokenizer=False):
        super().__init__(max_token)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/Phi-3.5-mini-instruct-bnb-4bit", cache_dir="./ckpts/llm")
        self.model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Phi-3.5-mini-instruct-bnb-4bit", 
            low_cpu_mem_usage=True, cache_dir="./ckpts/llm")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3.1")
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    with open("./data/preprocessed_data/webqsp/total/train.json", "r") as rf:
        question_data = [json.loads(line)["question"] for line in tqdm(rf.readlines())]

    if args.model == "qwen2:0.5b":
        tokenizer = AutoTokenizer.from_pretrained(
            "PrunaAI/Qwen-Qwen2-0.5B-Instruct-bnb-4bit-smashed", cache_dir="./ckpts/llm")
        model = AutoModelForCausalLM.from_pretrained(
            "PrunaAI/Qwen-Qwen2-0.5B-Instruct-bnb-4bit-smashed",
            cache_dir="./ckpts/llm")
        
        for question in tqdm(question_data):
            input_ids = tokenizer(question, return_tensors="pt").to("cuda")
            outputs = model(**input_ids, output_hidden_states=True)
        del model
    
    elif args.model == "gemma2:2b":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", cache_dir="./ckpts/llm")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16), cache_dir="./ckpts/llm"
        )
        for question in tqdm(question_data):
            input_ids = tokenizer(question, return_tensors="pt").to("cuda")
            outputs = model(**input_ids, output_hidden_states=True)
        del model

    elif args.model == "phi3.5":
        tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/Phi-3.5-mini-instruct-bnb-4bit", cache_dir="./ckpts/llm")
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Phi-3.5-mini-instruct-bnb-4bit", 
            low_cpu_mem_usage=True, cache_dir="./ckpts/llm")
        for question in tqdm(question_data):
            input_ids = tokenizer(question, return_tensors="pt").to("cuda")
            outputs = model(**input_ids, output_hidden_states=True)
        del model
        
    elif args.model == "llama3.1":
        tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/Meta-Llama-3.1-8B-bnb-4bit", cache_dir="./ckpts/llm")
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
            low_cpu_mem_usage=True, cache_dir="./ckpts/llm")
        for question in tqdm(question_data):
            input_ids = tokenizer(question, return_tensors="pt").to("cuda")
            outputs = model(**input_ids, output_hidden_states=True)
        del model
