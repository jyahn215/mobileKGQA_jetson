import argparse
from datetime import datetime

def get_hash_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="hashing")
    parser.add_argument("-model_name", type=str, default="sbert")
    # option gemma2:2b / qwen2:0.5b / llama3.1 / phi3.5 / sbert
    parser.add_argument('-dataset', type=str, default='webqsp', help='dataset name')
    parser.add_argument("-eval_cnt", type=int, default=5)
    # option cwq / webqsp
    parser.add_argument('-domain', type=str, default="total") # train domain selection
    parser.add_argument("-max_epochs", type=int, default=1000)
    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("-gpu", type=str)
    parser.add_argument("-num_workers", type=int, default=5)
    parser.add_argument("-config", type=str)
    parser.add_argument("-batch_size", type=int, default=3)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-patience", type=int, default=4)
    parser.add_argument("-reg_level", type=float, default=0.01)
    parser.add_argument("-dim_list", type=int, nargs="+", default=[128])
    parser.add_argument("-beta", type=float, default=1.0)
    parser.add_argument("-p", type=float, default=2.0)
    parser.add_argument("-loss_fn", type=str, default="LogRatioLoss")
    parser.add_argument("-period", type=int, default=1)
    # for LogRatioLoss
    parser.add_argument("-eps", type=float, default=1e-6)
    # for triplet loss and NSSALoss
    parser.add_argument("-margin", type=float, default=5.0)
    parser.add_argument("-thr_cossim", type=float, default=None)
    parser.add_argument("-thr_num", type=int, default=10)
    parser.add_argument("-pos_num", type=int, default=1)
    parser.add_argument("-neg_num", type=int, default=1)
    # for infoNCEloss
    parser.add_argument("-noise_level", type=float, default=0.1)
    parser.add_argument("-temp", type=float, default=0.1)

    args, _ = parser.parse_known_args()
    if args.config is None:
        args.config = "SH-" + datetime.now().strftime("%m-%d-%H-%M-%S-%f")
    args.data_path = f"./data/preprocessed_data/{args.dataset}"

    return args

# def get_retrieval_params():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-llm", type=str, default="gemma2:2b")
#     parser.add_argument("-dataset", type=str, default="webqsp")
#     parser.add_argument("-seed", type=int, default=0)
#     parser.add_argument("-gpu", type=str)
#     parser.add_argument("-num_workers", type=int, default=10)
#     parser.add_argument("-config", type=str)
#     parser.add_argument("-max_q_tokens", type=int, default=22)
#     parser.add_argument("-max_graph_tokens", type=int, default=3999)
#     parser.add_argument("-d_model", type=int, default=2304)
#     parser.add_argument("-d_proj", type=int, default=64)
#     parser.add_argument("-num_heads", type=int, default=4)
#     parser.add_argument("-num_layers", type=int, default=1)
#     parser.add_argument("-dropout", type=float, default=0.1)

    
#     # for knowledge graph learning
#     parser.add_argument("-loss_fn", type=str, default="NSSALoss")
#     parser.add_argument("-margin", type=float, default=9.)
#     parser.add_argument("-neg_num", type=int, default=30)
#     parser.add_argument("-max_epochs", type=int, default=500)
#     parser.add_argument("-lr", type=float, default=0.000001)
#     parser.add_argument("-batch_size", type=int, default=1)
#     parser.add_argument("-dim", type=int, default=400)
#     parser.add_argument("-pooler", type=str, default="mlp")
#     parser.add_argument("-hidden_dim_list", type=int, nargs="+", default=[400])
#     parser.add_argument("-layer_num", type=int, default=1)
#     parser.add_argument("-create_inverse_triples", type=bool, default=True)
#     parser.add_argument("-part_num", type=int, default=1)

#     args, _ = parser.parse_known_args()
#     if args.config is None:
#         args.config = "retriever-" + datetime.now().strftime("%m-%d-%H-%M-%S")
#     args.data_path = f"./../data/preprocessed_data/{args.dataset}"


#     return args

# def get_infer_params():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-dataset", type=str, default="expla_graphs")
#     parser.add_argument("-seed", type=int, default=0)
#     parser.add_argument("-gpu", type=str)
#     parser.add_argument("-num_workers", type=int, default=5)
#     parser.add_argument("-config", type=str)
#     parser.add_argument("-kg_module", type=str, default="RotatE")
#     parser.add_argument("-batch_size", type=int, default=40)
#     parser.add_argument("-anchor_num", type=float, default=40)
#     parser.add_argument("-anchor_strategy", type=dict, default={"degree": 0.4, "pagerank": 0.4, "random": 0.2})
#     parser.add_argument("-anchors_per_ent", type=dict, default=100)
#     parser.add_argument("-gamma", type=float, default=5)
#     parser.add_argument("-dim", type=int, default=300)
#     parser.add_argument("-neg_num", type=int, default=30)
#     parser.add_argument("-max_epochs", type=int, default=600)
#     parser.add_argument("-lr", type=float, default=0.001)

#     args, _ = parser.parse_known_args()
#     if args.config is None:
#         args.config = datetime.now().strftime("%m-%d-%H-%M-%S")
#         args.config += f"-{args.dataset}-s{args.seed}-{args.kg_module}"
#         args.config += f"-b{args.batch_size}-a{args.anchor_num}-g{args.gamma}"
#         args.config += f"-d{args.dim}-n{args.neg_num}-e{args.max_epochs}-l{args.lr}"

#     return args

# def get_finetune_params():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-dataset", type=str, default="expla_graphs")
#     parser.add_argument("-seed", type=int, default=0)
#     parser.add_argument("-gpu", type=str)
#     parser.add_argument("-num_workers", type=int, default=5)
#     parser.add_argument("-config", type=str)
#     parser.add_argument("-load_SH", type=str,
#         default=f"06-18-18-23-21-SH-mlp-expla_graphs-s0-b40-l0.004-p10-"
#         f"tcNone-tn10-pos1-neg1-reg0.001-d[1024, 256, 128]-LogRatioLoss-n0.1-t0.1-m5.0")
#     parser.add_argument("-load_RH", type=str, 
#         default=f"06-17-18-45-26-RH-nodepiece-expla_graphs-s0-lNSSALoss-t1-c5-b50-a700-"
#         f"g9.0-d400-n30-l0.001-lyn1")
#     parser.add_argument("-batch_size", type=int, default=5)
#     parser.add_argument("-max_epochs", type=int, default=50)
#     parser.add_argument("-warmup_epochs", type=int, default=1)
#     parser.add_argument("-lr", type=float, default=0.001)
#     parser.add_argument("-wd", type=float, default=0.05)
#     parser.add_argument("-grad_steps", type=int, default=2)
#     parser.add_argument("-create_inverse_triples", type=bool, default=True)
#     parser.add_argument("-ent_ret_num", type=int, default=7)
#     parser.add_argument("-rel_ret_num", type=int, default=7)
#     parser.add_argument("-triple_ret_num", type=int, default=10)
#     parser.add_argument("-thr", type=str, default=None)
#     parser.add_argument("-patience", type=float, default=2)

#     # for model
#     parser.add_argument("-llm_model_name", type=str, default="7b")
#     parser.add_argument("-max_txt_len", type=int, default=512)
#     parser.add_argument("-max_new_tokens", type=int, default=32)
#     parser.add_argument("-quant", type=int, default=8)
#     parser.add_argument("-gnn_model_name", type=str, default="gt")
#     parser.add_argument("-gnn_num_layers", type=int, default=4)
#     parser.add_argument("-gnn_in_dim", type=int, default=128)
#     parser.add_argument("-gnn_hidden_dim", type=int, default=128)
#     parser.add_argument("-gnn_num_heads", type=int, default=4)
#     parser.add_argument("-gnn_dropout", type=float, default=0.0)

#     args, _ = parser.parse_known_args()
#     if args.config is None:
#         args.config = datetime.now().strftime("%m-%d-%H-%M-%S")
#     args.data_path = f"./../data/preprocessed_data/{args.dataset}"

#     return args

# def get_LORA_params():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-dataset", type=str, default="expla_graphs")
#     parser.add_argument("-seed", type=int, default=0)
#     parser.add_argument("-gpu", type=str)
#     parser.add_argument("-num_workers", type=int, default=5)
#     parser.add_argument("-config", type=str)
#     parser.add_argument("-kg_module", type=str, default="RotatE")
#     parser.add_argument("-batch_size", type=int, default=40)
#     parser.add_argument("-anchor_num", type=float, default=40)
#     parser.add_argument("-anchor_strategy", type=dict, default={"degree": 0.4, 
#                                                                 "pagerank": 0.4, 
#                                                                 "random": 0.2})
#     parser.add_argument("-anchors_per_ent", type=dict, default=100)
#     parser.add_argument("-vocab_info", type=str, nargs="+", default=["close_anchors"])
#     # for close anchor search
#     parser.add_argument("-max_hop", type=int, default=50)
#     parser.add_argument("-gamma", type=float, default=5)
#     parser.add_argument("-dim", type=int, default=300)
#     parser.add_argument("-neg_num", type=int, default=30)
#     parser.add_argument("-max_epochs", type=int, default=600)
#     parser.add_argument("-lr", type=float, default=0.001)

#     args, _ = parser.parse_known_args()
#     if args.config is None:
#         args.config = datetime.now().strftime("%m-%d-%H-%M-%S")
#         args.config += f"-{args.dataset}-s{args.seed}-{args.kg_module}"
#         args.config += f"-b{args.batch_size}-a{args.anchor_num}-g{args.gamma}"
#         args.config += f"-d{args.dim}-n{args.neg_num}-e{args.max_epochs}-l{args.lr}"

#     return args
