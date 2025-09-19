import argparse
import os
from parsing import add_parse_args

parser = argparse.ArgumentParser()
add_parse_args(parser)
args = parser.parse_args()
args.model = 'ReaRev'

if args.slurm is False:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import time
import wandb
args.experiment_name = time.strftime("%Y-%m-%d-%H-%M-%S")
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
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    logger = create_logger(args)

    trainer = Trainer_KBQA(args=vars(args), model_name=args.model_name, logger=logger)
    
    if not args.is_eval:
        trainer.train(1, args.num_epoch)
    else:
        assert args.load_experiment is not None
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            print("Loading pre trained model from {}".format(ckpt_path))
        else:
            ckpt_path = None
        trainer.evaluate_single(ckpt_path)


if __name__ == '__main__':
    main()
