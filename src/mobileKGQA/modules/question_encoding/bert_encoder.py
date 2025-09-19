from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig #DistilBertModel, BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from torch.nn import LayerNorm
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TRANSFORMERS_CACHE'] = '/export/scratch/costas/home/mavro016/.cache'

from .base_encoder import BaseInstruction
# from src.module.LLM import Qwen2_0_5B, Gemma2_2B, Phi3_5_mini_3_8B, LLAMA3_1_8B

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class BERTInstruction(BaseInstruction):

    def __init__(self, args, model, constraint=False):
        super(BERTInstruction, self).__init__(args, constraint)
        # self.word_embedding = word_embedding
        # self.num_word = num_word
        self.constraint = constraint
        
        entity_dim = self.entity_dim
        self.model = model
        
        if model == "gemma2:2b":
            self.tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it', cache_dir="./ckpts/llm")
            self.pretrained_weights = 'google/gemma-2-2b-it'
            word_dim = 2304 if args["bit"] == None else int(args["bit"])
        elif model == "qwen2:0.5b":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "PrunaAI/Qwen-Qwen2-0.5B-Instruct-bnb-4bit-smashed", cache_dir="./ckpts/llm")
            self.pretrained_weights = "PrunaAI/Qwen-Qwen2-0.5B-Instruct-bnb-4bit-smashed"
            word_dim = 896 if args["bit"] == None else int(args["bit"])
        elif model == "phi3.5":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "unsloth/Phi-3.5-mini-instruct-bnb-4bit", cache_dir="./ckpts/llm")
            self.pretrained_weights = "unsloth/Phi-3.5-mini-instruct-bnb-4bit"
            word_dim = 3072 if args["bit"] == None else int(args["bit"])
        elif model == "llama3.1":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "unsloth/Meta-Llama-3.1-8B-bnb-4bit", cache_dir="./ckpts/llm")
            self.pretrained_weights = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
            word_dim = 4096 if args["bit"] == None else int(args["bit"])
        elif model == "relbert":
            self.tokenizer = AutoTokenizer.from_pretrained('ckpts/pretrained_lms/sr-simbert/')
            self.pretrained_weights = 'ckpts/pretrained_lms/sr-simbert/'
            word_dim = 768 if args["bit"] == None else int(args["bit"])
        elif model == "gte-large":
            self.tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", cache_dir="./ckpts/llm")
            self.pretrained_weights = "Alibaba-NLP/gte-large-en-v1.5"
            word_dim = 1024 if args["bit"] == None else int(args["bit"])
        elif model == "gte-qwen2:1.5b":
            self.tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct", cache_dir="./ckpts/llm")
            self.pretrained_weights = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
            word_dim = 1536 if args["bit"] == None else int(args["bit"])
        elif model == "sbert":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2", cache_dir="./ckpts/llm")
            self.pretrained_weights = "sentence-transformers/all-MiniLM-L6-v2"
            word_dim = 384 if args["bit"] == None else int(args["bit"])
        elif model == "GloVe_300dim":
            self.tokenizer = None
            self.pretrained_weights = None
            word_dim = 300 if args["bit"] == None else int(args["bit"])
 
        self.word_dim = word_dim
        print('word_dim', self.word_dim)
        self.cq_linear = nn.Linear(in_features=4 * entity_dim, out_features=entity_dim)
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_ins):
            self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
        self.question_emb = nn.Linear(in_features=word_dim, out_features=entity_dim)

        # self.pad_val = self.node_encoder.tokenizer.convert_tokens_to_ids(self.node_encoder.tokenizer.pad_token)
        # self.node_encoder.freeze_LLM()


    def encode_question(self, query_hidden, query_mask):
        # # query_text: (batch_size, seq_len)
        # batch_size = query_text.size(0)
        # max_B = 50
        # if batch_size >= max_B:
        #     query_hidden_emb = []
        #     for idx in tqdm(range(0, batch_size, max_B), desc="create hidden states for questions"):
        #         query_text_batch = query_text[idx:idx + max_B]
        #         query_hidden_emb_batch = self.node_encoder.get_last_hidden_state_wo_tokenizer(
        #             query_text_batch
        #         ) # max_B, seq_len, word_dim
        #         query_hidden_emb.append(query_hidden_emb_batch)
        #     query_hidden_emb = torch.cat(query_hidden_emb, dim=0)
        #     query_hidden_emb = query_hidden_emb.to(torch.float32)
        # else:
        #     query_hidden_emb = self.node_encoder.get_last_hidden_state_wo_tokenizer(query_text)
        #     query_hidden_emb = query_hidden_emb.to(torch.float32)
        
        # if store:
        #     self.query_hidden_emb = self.question_emb(query_hidden_emb)
        #     self.query_node_emb = query_hidden_emb.transpose(1,0)[0].unsqueeze(1)
        #     #print(self.query_node_emb.size())
        #     self.query_node_emb = self.question_emb(self.query_node_emb)
            
        #     self.query_mask = (query_text != self.pad_val).float()
        #     return query_hidden_emb, self.query_node_emb
        # else:
        #     return  query_hidden_emb 

        query_hidden_emb = self.question_emb(query_hidden)              # (B, L, D) -> (B, L, E)
        if self.pretrained_weights != 'Alibaba-NLP/gte-Qwen2-1.5B-instruct':
            query_node_emb = self.question_emb(query_hidden[:, 0:1, :])     # (B, 1, D) -> (B, 1, E)
        else:
            query_node_emb = last_token_pool(query_hidden, query_mask) 
            query_node_emb = query_node_emb.unsqueeze(dim=1) # (B, D) -> (B, 1, D)
            query_node_emb = self.question_emb(query_node_emb) # (B, 1, D) -> (B, 1, E)         

        return query_hidden_emb, query_node_emb


