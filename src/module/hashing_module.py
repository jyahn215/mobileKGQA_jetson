import torch
import random
import numpy as np
from torch.nn import functional
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional
from collections import Counter, defaultdict

from torch import nn
from torch.nn import functional as F
# from pykeen.losses import Loss
# from pykeen.triples import TriplesFactory
# from pykeen.models import Model
from tqdm import tqdm
# from utils.tokenizer import EntTokenizer

class Hashing_Module(nn.Module):
    def __init__(self, args, rep_dim):
        super().__init__()
        args.dim_list = [rep_dim] + args.dim_list
        layers = []
        for i in range(len(args.dim_list) - 1):
            layers.append(nn.Linear(args.dim_list[i], args.dim_list[i + 1]))
            if i < len(args.dim_list) - 2:  
                layers.append(nn.ReLU())
        self.act = nn.Tanh()
        # self.BN_1 = nn.BatchNorm1d(rep_dim)
        self.BN_2 = nn.BatchNorm1d(args.dim_list[-1])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x = self.BN_1(x)
        x = self.model(x)
        x = self.BN_2(x)
        x = self.act(x)
        return x


# class ReconHashEncoder(nn.Module):
#     def __init__(self, tokenizer, args):
#         super().__init__()
#         self.input_dim = 0
#         if "close_anchors" in args.hash_info:
#             self.anchors_per_ent = args.anchors_per_ent
#             self.input_dim += self.anchors_per_ent * args.dim
#         if "connected_rs" in args.hash_info:
#             self.conrel_max = args.con_rel_max
#             self.input_dim += self.conrel_max * args.dim
#         self.dim_list = [self.input_dim] + args.dim_list + [args.dim]
#         assert args.dim % 2 == 0
#         self.margin = args.margin

#         if args.pooler == "mlp":
#             layers = []
#             for i in range(len(self.dim_list) - 1):
#                 layers.append(nn.Linear(self.dim_list[i], self.dim_list[i + 1]))
#                 if i < len(self.dim_list) - 2:  
#                     layers.append(nn.ReLU())
#             self.pooler = nn.Sequential(*layers)
#             print("mlp type pooler is used")
#             print(self.pooler)

#         self.r_embeddings = nn.Embedding(tokenizer.rel_num, args.dim // 2)

#         if "close_anchors" in args.hash_info:
#             anc_padding_idx = tokenizer.token2id["close_anchors"]["anchor"][tokenizer.NO_ANCHORS]
#             self.anchor_embeddings = nn.Embedding(len(tokenizer.token2id["close_anchors"]["anchor"]),
#                                                 embedding_dim=args.dim, 
#                                                 padding_idx=anc_padding_idx)
#             dist_padding_idx = tokenizer.token2id["close_anchors"]["distance"][tokenizer.UNDEFINED_DISTANCE]
#             self.dist_embeddings = nn.Embedding(len(tokenizer.token2id["close_anchors"]["distance"]), 
#                                                 embedding_dim=args.dim,
#                                                 padding_idx=dist_padding_idx)
            
#         if "connected_rs" in args.hash_info:
#             rel_padding_idx = tokenizer.token2id["connected_rs"][tokenizer.NO_rS]
#             self.conrel_embeddings = nn.Embedding(len(tokenizer.token2id["connected_rs"]), 
#                                                     embedding_dim=args.dim, 
#                                                     padding_idx=rel_padding_idx)

#     def get_ent_embeds_train(self, ent_hash):
#         '''
#         ent_hash
#         {"close_anchors": [positive_ancs: (batch_size, :, 2, anc_num_per_ent),
#                         positive_dists: (batch_size, :, 2, anc_num_per_ent)],
#         "connected_rs": [positive_rels: (batch_size, :, 2, conrel_max)]}
#         => 
#         h_embeds: (batch_size, :, dim)
#         t_embeds: (batch_size, :, dim)
#         '''
#         h_embeds, t_embeds = [], []
#         if "close_anchors" in ent_hash.keys():
#             ancs_hash, dists_hash = ent_hash["close_anchors"]
#             anc_embs = self.anchor_embeddings(ancs_hash)
#             dist_embs = self.dist_embeddings(dists_hash)
#             anc_embs += dist_embs
#             h_embeds.append(anc_embs[:, :, 0, :, :])
#             t_embeds.append(anc_embs[:, :, 1, :, :])
        
#         if "connected_rs" in ent_hash.keys():
#             conrel_hash = ent_hash["connected_rs"]
#             conrel_embs = self.conrel_embeddings(conrel_hash)
#             h_embeds.append(conrel_embs[:, :, 0, :, :])
#             t_embeds.append(conrel_embs[:, :, 1, :, :])

#         h_embeds = torch.cat(h_embeds, dim=2)
#         h_embeds = h_embeds.reshape(h_embeds.shape[0], h_embeds.shape[1], -1)
#         t_embeds = torch.cat(t_embeds, dim=2)
#         t_embeds = t_embeds.reshape(t_embeds.shape[0], t_embeds.shape[1], -1)

#         return self.pooler(h_embeds), self.pooler(t_embeds)
    
#     def get_ent_embeds_eval(self, ent_hash):
#         '''
#         ent_hash
#         {"close_anchors": [positive_ancs: (batch_size, anc_num_per_ent),
#                         positive_dists: (batch_size, anc_num_per_ent)],
#         "connected_rs": [positive_rels: (batch_size, conrel_max)]}
#         => 
#         ent_embeds: (batch_size, dim)
#         '''
#         ent_embeds = []
#         if "close_anchors" in ent_hash.keys():
#             ancs_hash, dists_hash = ent_hash["close_anchors"]
#             anc_embs = self.anchor_embeddings(ancs_hash)
#             dist_embs = self.dist_embeddings(dists_hash)
#             anc_embs += dist_embs
#             ent_embeds.append(anc_embs)
        
#         if "connected_rs" in ent_hash.keys():
#             conrel_hash = ent_hash["connected_rs"]
#             conrel_embs = self.conrel_embeddings(conrel_hash)
#             ent_embeds.append(conrel_embs)

#         ent_embeds = torch.cat(ent_embeds, dim=1)
#         ent_embeds = ent_embeds.reshape(ent_embeds.shape[0], -1)

#         return self.pooler(ent_embeds)

#     def forward(self, pos_ent_hash, pos_rel_hash, neg_ent_hash, neg_rel_hash):
#         '''
#         pos_ent_hash
#         {"close_anchors": [positive_ancs: (batch_size, 1, 2, anc_num_per_ent),
#                         positive_dists: (batch_size, 1, 2, anc_num_per_ent)],
#         "connected_rs": [positive_rels: (batch_size, 1, 2, conrel_max)]}
#         pos_rel_hash (batch_size, 1)

#         neg_ent_hash
#         {"close_anchors": [negative_ancs: (batch_size, neg_num, 2, anc_num_per_ent),
#                         negative_dists: (batch_size, neg_num, 2, anc_num_per_ent)],
#         "connected_rs": [negative_rels: (batch_size, neg_num, 2, conrel_max)]}
#         neg_rel_hash (batch_size, neg_num)
#         '''
    
#         pos_h_embeds, pos_t_embeds = self.get_ent_embeds_train(pos_ent_hash)
#         pos_r_embeds = self.r_embeddings(pos_rel_hash)
#         neg_h_embeds, neg_t_embeds = self.get_ent_embeds_train(neg_ent_hash)
#         neg_r_embeds = self.r_embeddings(neg_rel_hash)

#         pos_score = self.interaction_function(pos_h_embeds, pos_r_embeds, pos_t_embeds)
#         neg_score = self.interaction_function(neg_h_embeds, neg_r_embeds, neg_t_embeds)

#         score = - F.logsigmoid(pos_score) - F.logsigmoid(-neg_score)
#         score = score.mean()
#         return score
    
#     def pairwise_interaction_function(self, h_embeds, r_embeds):
#         re_h, im_h = torch.chunk(h_embeds, 2, dim=2)
#         re_r, im_r = torch.cos(r_embeds), torch.sin(r_embeds)
#         re_q = re_h * re_r - im_h * im_r
#         im_q = re_h * im_r + im_h * re_r

#         return torch.cat([re_q, im_q], dim=2)
    
#     def interaction_function(self, h_embeds, r_embeds, t_embeds):
#         '''
#         h_embeds: [batch_size, :, dim]
#         r_embeds: [batch_size, :, dim // 2]
#         t_embeds: [batch_size, :, dim]
#         => 
#         score: [batch_size, :]
#         '''
        
#         rot_h_embeds = self.pairwise_interaction_function(h_embeds, r_embeds)
#         return self.margin - F.pairwise_distance(rot_h_embeds, t_embeds, p=1)
    
#     def get_score(self, triples_factory, rank_filter, all_ent_hash, device):
#         ent_num, rel_num = triples_factory.num_entities, triples_factory.num_relations
#         all_ent_embeds = self.get_ent_embeds_eval(all_ent_hash)
#         score = dict.fromkeys(rank_filter.keys())

#         for query in score.keys():
#             head = torch.tensor([query[0]]).to(device)
#             rel = torch.tensor([query[1]]).to(device)
#             h_embeds = all_ent_embeds[head].repeat(ent_num, 1, 1)
#             r_embeds = self.r_embeddings(rel).repeat(ent_num, 1, 1)
#             t_embeds = all_ent_embeds.unsqueeze(1)
            
#             batch_score = self.interaction_function(h_embeds, r_embeds, t_embeds)
#             batch_score = batch_score.squeeze(1)
#             score[query] = batch_score.detach().cpu().numpy()

#         return score
                    


        

