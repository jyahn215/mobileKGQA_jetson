import os
from utils.get_params import get_hash_params
args = get_hash_params()

# gpu setting
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import json
import wandb
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn.utils import clip_grad_norm_ as clip
# from deepspeed.profiling.flops_profiler import FlopsProfiler
from module.hashing_module import Hashing_Module
from module.loss_fn import load_loss_fn, QuantError, QuantError_
from utils.evaluate import get_cossim_matrix, get_mAP
from utils.dataset import load_dataset, load_all_rep, get_idx, sample_rep



def train_SearchHashEncoder(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device, args.gpu)
    args.ckpts_path = f"./ckpts/hashing/{args.config}"
    if not os.path.exists(args.ckpts_path):
        os.makedirs(args.ckpts_path)

    wandb.init(
        project="mkg",
        name=args.config,
        config=args
    )

    # seed setting
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # define model and optimizer
    rep_dim_dict = {
        "qwen2:0.5b": 896,
        "gemma2:2b": 2304,
        "phi3.5": 3072,
        "llama3.1": 4096,
        "sbert": 384,
        "relbert": 768,
        "gte-large": 1024,
        "gte-qwen2:1.5b": 1536
    }
    model = Hashing_Module(args, rep_dim_dict[args.model_name]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params/1e6}M")
    print(f"Trainable Parameters: {trainable_params/1e6}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load dataset
    if args.loss_fn == "LogRatioLoss":
        if args.domain != "total":
            with open(f"./data/preprocessed_data/{args.dataset}/_domains/{args.domain}/rel_in_domain.json", "r") as f:
                rel_list = json.load(f)
        else:
            if args.dataset == "webqsp":
                rel_list = [i for i in range(6094)]
            elif args.dataset == "cwq":
                rel_list = [i for i in range(6649)]
            elif args.dataset == "metaqa-3hop" or args.dataset == "metaqa-2hop" or args.dataset == "metaqa-1hop":
                rel_list = [i for i in range(9)]
            else:
                raise ValueError("Invalid dataset")

        query_num = len(os.listdir(f"./data/preprocessed_data/{args.dataset}/last_hidden_state/{args.model_name}/ori/queries/{args.domain}/train"))
        assert query_num % 2 == 0
        query_list = [i for i in range(query_num // 2)]
        train_dataset = load_dataset[args.loss_fn](args, query_list, rel_list, split="train")
        print("train dataset size: ", len(train_dataset))

        query_num = len(os.listdir(f"./data/preprocessed_data/{args.dataset}/last_hidden_state/{args.model_name}/ori/queries/{args.domain}/dev"))
        assert query_num % 2 == 0
        query_list = [i for i in range(query_num // 2)]
        valid_dataset = load_dataset[args.loss_fn](args, query_list, rel_list, split="dev")
        print("valid dataset size: ", len(valid_dataset))

        query_num = len(os.listdir(f"./data/preprocessed_data/{args.dataset}/last_hidden_state/{args.model_name}/ori/queries/{args.domain}/test"))
        assert query_num % 2 == 0
        query_list = [i for i in range(query_num // 2)]
        test_dataset = load_dataset[args.loss_fn](args, query_list, rel_list, split="test")
        print("test dataset size: ", len(test_dataset))

    # elif args.loss_fn == "InfoNCELoss":
    #     train_dataset = load_dataset[args.loss_fn](args, "train")
    #     valid_dataset = load_dataset[args.loss_fn](args, "validation")
        
    # elif args.loss_fn == "TripletLoss":
    #     pos_idx, neg_idx = get_idx(cossim_matrix, args)
    #     train_dataset = load_dataset[args.loss_fn]\
    #         (args, pos_idx, neg_idx, all_rep)
    # elif args.loss_fn == "NSSALoss":
    #     train_dataset = load_dataset[args.loss_fn](args)
    # else:
    #     raise NotImplementedError

    # define dataloader
    # if args.loss_fn != "LogRatioLoss":

    print("batch_size: ", args.batch_size)
    train_loader = data.DataLoader(train_dataset, 
                            batch_size = 1, 
                            shuffle=True,
                            num_workers=args.num_workers, 
                            collate_fn=train_dataset.collate_fn)

    valid_loader = data.DataLoader(valid_dataset,
                            batch_size = 1,
                            shuffle=True,
                            num_workers=args.num_workers,
                            collate_fn=valid_dataset.collate_fn)
    
    test_loader = data.DataLoader(test_dataset,
                            batch_size = 1,
                            shuffle=True,
                            num_workers=args.num_workers,
                            collate_fn=test_dataset.collate_fn)
    
    loss_fn = load_loss_fn[args.loss_fn](args)

    # generate cossim matrix
    best_score, patience = 0, 0
    escape = False
    cnt = 0
    for epoch in range(1, args.max_epochs + 1):
        # if cnt == 0:
        #     FLOPs_per_iter = 0
        #     # profiler = FlopsProfiler(model)
        #     # profiler.start_profile()

        if escape == True:
            break
        model.train()
        tqdm_obj = tqdm(train_loader, desc=f"train SH(epoch:{epoch})", ncols=80)
        for batch in tqdm_obj:
            if args.loss_fn == "InfoNCELoss":
                rep1, rep2 = batch
                rep1, rep2 = rep1.to(device), rep2.to(device)
                rep1, rep2 = rep1.to(torch.float32), rep2.to(torch.float32)
                rep1, rep2 = model(rep1), model(rep2)
                loss = loss_fn(rep1, rep2)
                tqdm_obj.set_postfix({"loss": loss.item()})
                # continue
                # if args.reg_level > 0:
                #     reg = QuantError(rep1) + QuantError(rep2)
                #     loss = loss + args.reg_level * reg

            elif args.loss_fn == "TripletLoss":
                rep, rep1, rep2 = batch
                rep, rep1, rep2 = rep.to(device), rep1.to(device), rep2.to(device)
                rep, rep1, rep2 = model(rep), model(rep1), model(rep2)
                loss = loss_fn(rep, rep1, rep2)
                if args.reg_level > 0:
                    reg = QuantError(rep) + QuantError(rep1) + QuantError(rep2)
                    loss = loss + args.reg_level * reg

            elif args.loss_fn == "LogRatioLoss":
                rep = batch

                rep = rep.to(device) if rep.device != device else rep
                true_cossim = get_cossim_matrix(rep)
                true_cossim = torch.from_numpy(true_cossim).to(device)

                for _, param in model.named_parameters():
                    if param.data.isnan().any().item():
                        print("Nan in model parameters")

                rep = model(rep)
                loss = loss_fn(rep, true_cossim)

                if args.reg_level > 0:
                    rep = torch.sign(rep)
                    true_cossim = get_cossim_matrix(rep)
                    true_cossim = torch.from_numpy(true_cossim).to(device)
                    quant_loss = loss_fn(rep, true_cossim)
                    loss = loss + args.reg_level * quant_loss

            # elif args.loss_fn == "NSSALoss":
            #     rep, rep1, rep2 = batch
            #     rep, rep1, rep2 = rep.to(device), rep1.to(device), rep2.to(device)
            #     rep, rep1, rep2 = model(rep), model(rep1), model(rep2)
            #     loss = loss_fn(rep, rep1, rep2)
            #     if args.reg_level > 0:
            #         reg = QuantError(rep) + QuantError(rep1) + QuantError(rep2)
            #         loss = loss + args.reg_level * reg

            optimizer.zero_grad()
            loss.backward()
            for _, param in model.named_parameters():
                param.grad = torch.nan_to_num(param.grad, nan=0.0)
            optimizer.step()
            cnt += 1

            # if cnt == 1:
            #     GFLOPs_per_iter = profiler.get_total_flops() / 1e9
            #     print(f"GFLOPs per iter: {GFLOPs_per_iter}")
            #     profiler.end_profile()
        
            if cnt % 2 == 0:
                model.eval()
                mAP_list = []
                with torch.no_grad():
                    cnt2 = 0
                    for sampled_rep in valid_loader:
                        sampled_rep = sampled_rep[:100] # we sample 100 hidden representations for each token.
                        sampled_rep = sampled_rep.to(device)
                        cossim_matrix = get_cossim_matrix(sampled_rep)
                        rep = model(sampled_rep)
                        quant_error = QuantError_(rep).item()
                        quantized_rep = torch.sign(rep)
                        mAP = get_mAP(cossim_matrix, quantized_rep)
                        mAP_list.append(mAP)
                        cnt2 += 1
                        if cnt2 > 30:
                            break
                mAP = np.mean(mAP_list)
                    
                if mAP > best_score:
                    patience = 0
                    best_score = mAP
                    ckpts = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_score": best_score,
                        "args": args
                    }

                    torch.save(ckpts, 
                            os.path.join(args.ckpts_path, "SearchEncoder.pt"))
                    print(f"save best model")
                    print(f"path: {args.ckpts_path}")
                else:
                    patience += 1

                wandb.log({"epoch": epoch, "valid_mAP": mAP, "cnt": cnt, "best_score": best_score}, step=cnt)
                
                print(f"\nepoch: {epoch}, valid_mAP: {np.round(mAP, 4)}, "
                    f"patience: {patience}, best_score: {np.round(best_score, 4)}, "
                    f"quant error: {np.round(quant_error, 4)}\n")
                
            if patience >= args.patience:
                print("early stopping")
                print(f"Total GFLOPs: {GFLOPs_per_iter * cnt}")
                wandb.log({"Total GFLOPs": GFLOPs_per_iter * cnt})
                escape = True
                break
    
    print("load best model")
    best_ckpts = torch.load(os.path.join(args.ckpts_path, f"SearchEncoder.pt"))
    model.load_state_dict(ckpts["model_state_dict"])
    model.eval()
    mAP_list = []
    ret_wise_precision_list = []
    with torch.no_grad():
        cnt2 = 0
        for sampled_rep in test_loader:
            sampled_rep = sampled_rep[:100]
            if cnt2 == 0:
                print("test batch_size: ", sampled_rep.size(0))
            sampled_rep = sampled_rep.to(device)
            cossim_matrix = get_cossim_matrix(sampled_rep)
            rep = model(sampled_rep)
            quant_error = QuantError_(rep).item()
            quantized_rep = torch.sign(rep)
            mAP = get_mAP(cossim_matrix, quantized_rep)
            mAP_list.append(mAP)
            cnt2 += 1
            if cnt2 > 30:
                break
    mAP = np.mean(mAP_list)
    wandb.log({"test_mAP_new": mAP})

    print(f"test mAP: {np.round(mAP, 4)}")


    # ckpts = torch.load(os.path.join(args.ckpts_path, f"SearchEncoder.pt"))
    # model.load_state_dict(ckpts["model_state_dict"])

    # query_path = os.path.join(args.data_path, "hash", args.config, "queries")
    # if not os.path.exists(query_path):
    #     os.makedirs(query_path)
    # ent_rel_path = os.path.join(args.data_path, "hash", args.config, "ent_rel")
    # if not os.path.exists(ent_rel_path):
    #     os.makedirs(ent_rel_path)

    # if all_rep is None:
    #     ent_rel_dataset = load_dataset["ent_rel"](args)
    #     ent_rel_loader = data.DataLoader(ent_rel_dataset, batch_size=1,
    #                                     shuffle=False, drop_last=False)
    #     for idx, ent_rel_rep in enumerate(tqdm(ent_rel_loader, desc="ent_rel hash", ncols=80)):
    #         ent_rel_rep = ent_rel_rep.to(device)
    #         ent_rel_rep = model(ent_rel_rep)
    #         ent_rel_rep = torch.sign(ent_rel_rep)
    #         torch.save(ent_rel_rep, os.path.join(ent_rel_path, f"ent_rel_hash_{idx}.pt"))
        
    # else:
    #     rep = model(torch.from_numpy(all_rep).to(device))
    #     rep = torch.sign(rep)
    #     torch.save(rep, os.path.join(ent_rel_path, f"ent_rel_hash.pt"))

    # qurey_rep = torch.load(os.path.join(args.data_path, "queries", "query.pt"))
    # qurey_rep = model(qurey_rep.to(device))
    # qurey_rep = torch.sign(qurey_rep)
    # torch.save(qurey_rep, os.path.join(query_path, f"query_hash.pt"))
    # print(args)

if __name__ == "__main__":
    train_SearchHashEncoder(args)
