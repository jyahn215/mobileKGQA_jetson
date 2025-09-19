from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000
import torch


from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
 #DistilBertModel, BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from torch.nn import LayerNorm
import warnings
warnings.filterwarnings("ignore")
import os
try:
    os.environ['TRANSFORMERS_CACHE'] = '/export/scratch/costas/home/mavro016/.cache'
except:
    pass

from .base_encoder import BaseInstruction

# from .GENBERT.gen_bert.modeling import BertTransformer


class BERTInstruction(BaseInstruction):

    def __init__(self, args, model, constraint=False):
        super(BERTInstruction, self).__init__(args, constraint)
        self.constraint = constraint
        
        entity_dim = self.entity_dim
        self.model = model
        
        if model == "gemma2:2b":
            self.tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it', cache_dir="./ckpts/llm")
            self.pretrained_weights = 'google/gemma-2-2b-it'
            word_dim = 2304
        elif model == "qwen2:0.5b":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "PrunaAI/Qwen-Qwen2-0.5B-Instruct-bnb-4bit-smashed", cache_dir="./ckpts/llm")
            self.pretrained_weights = "PrunaAI/Qwen-Qwen2-0.5B-Instruct-bnb-4bit-smashed"
            word_dim = 896
        elif model == "phi3.5":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "unsloth/Phi-3.5-mini-instruct-bnb-4bit", cache_dir="./ckpts/llm")
            self.pretrained_weights = "unsloth/Phi-3.5-mini-instruct-bnb-4bit"
            word_dim = 3072
        elif model == "llama3.1":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "unsloth/Meta-Llama-3.1-8B-bnb-4bit", cache_dir="./ckpts/llm")
            self.pretrained_weights = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
            word_dim = 4096

        self.pad_val = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.word_dim = word_dim

        print('word_dim', self.word_dim)
        self.cq_linear = nn.Linear(in_features=4 * entity_dim, out_features=entity_dim)
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_ins):
            self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
        self.question_emb = nn.Linear(in_features=word_dim, out_features=entity_dim)

    def encode_question(self, query_hidden):
        # batch_size = query_text.size(0)
        # max_B = 50
        # if batch_size >= max_B:
        #     query_hidden_emb = []
        #     for idx in tqdm(range(0, batch_size, max_B), desc="create hidden states"):
        #         query_text_batch = query_text[idx:idx + max_B]
        #         query_hidden_emb_batch = self.node_encoder(query_text_batch, output_hidden_states=True).hidden_states[-1]
        #         query_hidden_emb.append(query_hidden_emb_batch)
        #     query_hidden_emb = torch.cat(query_hidden_emb, dim=0)
        #     query_hidden_emb = query_hidden_emb.to(torch.float32)
        # else:
        #     query_hidden_emb = self.node_encoder(query_text, output_hidden_states=True).hidden_states[-1]
        #     query_hidden_emb = query_hidden_emb.to(torch.float32)
        #     print(f"query_hidden_emb: {query_hidden_emb.size()}")

        query_hidden_emb = self.question_emb(query_hidden)              # (B, L, D) -> (B, L, E)
        query_node_emb = self.question_emb(query_hidden[:, 0:1, :])     # (B, 1, D) -> (B, 1, E)                                

        return query_hidden_emb, query_node_emb


