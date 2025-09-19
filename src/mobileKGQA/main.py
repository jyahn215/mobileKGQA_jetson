import argparse
import os
from parsing import add_parse_args
import time
import wandb

parser = argparse.ArgumentParser()
add_parse_args(parser)  
args = parser.parse_args()
if args.is_eval:
    args.experiment_name = f"eval_{args.load_experiment}" if args.load_experiment is not None else time.strftime("%Y-%m-%d-%H-%M-%S")
else:
    args.experiment_name = time.strftime("%Y-%m-%d-%H-%M-%S")
args.model = 'mobileKGQA'

if args.slurm is False:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

wandb.init(
    project="mkg",
    config=vars(args),
    name=time.strftime("%Y-%m-%d-%H-%M-%S")
)

from utils import create_logger
import torch
args.use_cuda = torch.cuda.is_available()
import numpy as np
import sys
workspace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./../../")
sys.path.append(workspace_dir)
from train_model import Trainer_KBQA
import random

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def main():
    args.checkpoint_dir = "./ckpts/mobileKGQA"
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    # logger = create_logger(args)

    trainer = Trainer_KBQA(args=vars(args), model_name=args.model_name)
    
    if not args.is_eval:
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            print("Loading pre trained model from {}".format(ckpt_path))
            trainer.load_ckpt(ckpt_path)
        print("Start training")
        trainer.train(1, args.num_epoch)
    else:
        assert args.load_experiment is not None
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            print("Loading pre trained model from {}".format(ckpt_path))
        else:
            print("Randomly initialized model is used for evaluation")
            ckpt_path = None
        print("Start evaluating")
        trainer.evaluate_single(ckpt_path)


if __name__ == '__main__':
    main()
