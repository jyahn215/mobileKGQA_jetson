
import torch.nn.functional as F
import torch.nn as nn
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000
from tqdm import tqdm
import torch

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig #DistilBertModel, BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
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

    def __init__(self, args, word_embedding, num_word, model, constraint=False):
        super(BERTInstruction, self).__init__(args, constraint)
        self.word_embedding = word_embedding
        self.num_word = num_word
        self.constraint = constraint
        
        entity_dim = self.entity_dim
        self.model = model
        
        
        if model == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.pretrained_weights = 'bert-base-uncased'
            word_dim = 768#self.word_dim
        elif model == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            self.pretrained_weights = 'roberta-base'
            word_dim = 768#self.word_dim
        elif model == 'sbert':
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.pretrained_weights = 'sentence-transformers/all-MiniLM-L6-v2'
            word_dim = 384#self.word_dim
        elif model == 'sbert2':
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            self.pretrained_weights = 'sentence-transformers/all-mpnet-base-v2'
            word_dim = 768#self.word_dim
        elif model == 't5':
            self.tokenizer = AutoTokenizer.from_pretrained('t5-small')
            self.pretrained_weights = 't5-small'
            word_dim = 512#self.word_dim

        elif model == "gemma2:2b":
            self.tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it', cache_dir="./ckpts/llm")
            self.pretrained_weights = 'google/gemma-2-2b-it'
            word_dim = 2304

        elif model == "qwen2:0.5b":
            self.tokenizer = AutoTokenizer.from_pretrained(
            "PrunaAI/Qwen-Qwen2-0.5B-Instruct-bnb-4bit-smashed", cache_dir="./ckpts/llm")
            self.pretrained_weights  = "PrunaAI/Qwen-Qwen2-0.5B-Instruct-bnb-4bit-smashed",
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
            
        # elif model == 'genbert' :
        #     self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        #     self.pretrained_weights = './modules/question_encoding/GENBERT/gen_bert/out_drop_finetune_syntext_and_numeric/'
        #     word_dim = 768#self.word_dim
        #self.mask = mask
        self.pad_val = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.word_dim = word_dim

        print('word_dim', self.word_dim)
        self.cq_linear = nn.Linear(in_features=4 * entity_dim, out_features=entity_dim)
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_ins):
            self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
        self.question_emb = nn.Linear(in_features=word_dim, out_features=entity_dim)
        
        if not self.constraint:
            self.encoder_def()

    def encoder_def(self):
        # initialize entity embedding
        word_dim = self.word_dim
        entity_dim = self.entity_dim
        if self.model == 'genbert' :
            self.node_encoder = BertTransformer.from_pretrained(self.pretrained_weights)
        elif self.model == "gemma2:2b":
            self.node_encoder = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16), cache_dir="./ckpts/llm"
            )

        elif self.model == "qwen2:0.5b":
            self.node_encoder = AutoModelForCausalLM.from_pretrained(
            "PrunaAI/Qwen-Qwen2-0.5B-Instruct-bnb-4bit-smashed",
            cache_dir="./ckpts/llm")

        elif self.model == "phi3.5":
            self.node_encoder = AutoModelForCausalLM.from_pretrained(
            "unsloth/Phi-3.5-mini-instruct-bnb-4bit", 
            low_cpu_mem_usage=True, cache_dir="./ckpts/llm")
        
        elif self.model == "llama3.1":
            self.node_encoder = AutoModelForCausalLM.from_pretrained(
                "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
                low_cpu_mem_usage=True, cache_dir="./ckpts/llm")
        else :
            self.node_encoder = AutoModel.from_pretrained(self.pretrained_weights)
        print('LM Params', sum(p.numel() for p in self.node_encoder.parameters()))
        if self.lm_frozen == 1:
            print('Freezing LM params')
            for param in self.node_encoder.parameters():
                param.requires_grad = False
        else:
            print('Unfrozen LM params')

    def encode_question(self, query_text, store=True):
        batch_size = query_text.size(0)
        query_text = query_text.cuda()
        max_B = 50
        if batch_size >= max_B:
            query_hidden_emb = []
            for idx in tqdm(range(0, batch_size, max_B), desc="create hidden states"):
                
                query_text_batch = query_text[idx:idx + max_B]
                query_hidden_emb_batch = self.node_encoder(query_text_batch, output_hidden_states=True).hidden_states[-1]
                query_hidden_emb.append(query_hidden_emb_batch)
            query_hidden_emb = torch.cat(query_hidden_emb, dim=0)
            query_hidden_emb = query_hidden_emb.to(torch.float32)
        else:
            query_hidden_emb = self.node_encoder(query_text, output_hidden_states=True).hidden_states[-1]
            query_hidden_emb = query_hidden_emb.to(torch.float32)
        

        if store:
            self.query_hidden_emb = self.question_emb(query_hidden_emb)                # (B, L, D) -> (B, L, E)
            self.query_node_emb = query_hidden_emb.transpose(1,0)[0].unsqueeze(1)      # (B, L, D) -> (B, 1, D)
            #print(self.query_node_emb.size())
            self.query_node_emb = self.question_emb(self.query_node_emb)               # (B, 1, D) -> (B, 1, E)
            
            self.query_mask = (query_text != self.pad_val).float()
            return query_hidden_emb, self.query_node_emb
        else:
            return  query_hidden_emb 

