from tqdm import tqdm
import torch
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
from src.module.LLM import Qwen2_0_5B, Gemma2_2B, Phi3_5_mini_3_8B, LLAMA3_1_8B


class BERTInstruction(BaseInstruction):

    def __init__(self, args, model, constraint=False):
        super(BERTInstruction, self).__init__(args, constraint)
        # self.word_embedding = word_embedding
        # self.num_word = num_word
        self.constraint = constraint
        
        entity_dim = self.entity_dim
        self.model = model
        
        if model == "qwen2:0.5b":
            self.node_encoder = Qwen2_0_5B()
            word_dim = 896
        elif model == "gemma2:2b":
            self.node_encoder = Gemma2_2B()
            word_dim = 2304
        elif model == "phi3.5":
            self.node_encoder = Phi3_5_mini_3_8B()
            word_dim = 3072
        elif model == "llama3.1":
            self.node_encoder = LLAMA3_1_8B()
            word_dim = 4096
 
        self.word_dim = word_dim
        print('word_dim', self.word_dim)
        self.cq_linear = nn.Linear(in_features=4 * entity_dim, out_features=entity_dim)
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_ins):
            self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
        self.question_emb = nn.Linear(in_features=word_dim, out_features=entity_dim)

        self.pad_val = self.node_encoder.tokenizer.convert_tokens_to_ids(self.node_encoder.tokenizer.pad_token)
        self.node_encoder.freeze_LLM()


    def encode_question(self, query_text, store=True):
        # query_text: (batch_size, seq_len)
        batch_size = query_text.size(0)
        max_B = 50
        if batch_size >= max_B:
            query_hidden_emb = []
            for idx in tqdm(range(0, batch_size, max_B), desc="create hidden states for questions"):
                query_text_batch = query_text[idx:idx + max_B]
                query_hidden_emb_batch = self.node_encoder.get_last_hidden_state_wo_tokenizer(
                    query_text_batch
                ) # max_B, seq_len, word_dim
                query_hidden_emb.append(query_hidden_emb_batch)
            query_hidden_emb = torch.cat(query_hidden_emb, dim=0)
            query_hidden_emb = query_hidden_emb.to(torch.float32)
        else:
            query_hidden_emb = self.node_encoder.get_last_hidden_state_wo_tokenizer(query_text)
            query_hidden_emb = query_hidden_emb.to(torch.float32)
        
        if store:
            self.query_hidden_emb = self.question_emb(query_hidden_emb)
            self.query_node_emb = query_hidden_emb.transpose(1,0)[0].unsqueeze(1)
            #print(self.query_node_emb.size())
            self.query_node_emb = self.question_emb(self.query_node_emb)
            
            self.query_mask = (query_text != self.pad_val).float()
            return query_hidden_emb, self.query_node_emb
        else:
            return  query_hidden_emb 

