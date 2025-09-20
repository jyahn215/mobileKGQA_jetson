from dataset_load import load_data
import argparse
import os
from parsing import add_parse_args
import time
from tqdm import tqdm
import h5py
import pickle
import numpy as np
import math

parser = argparse.ArgumentParser()
add_parse_args(parser)
args = parser.parse_args()
if args.is_eval:
    args.experiment_name = (
        f"eval_{args.load_experiment}"
        if args.load_experiment is not None
        else time.strftime("%Y-%m-%d-%H-%M-%S")
    )
else:
    args.experiment_name = time.strftime("%Y-%m-%d-%H-%M-%S")
args.model = "mobileKGQA"

if args.slurm is False:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch

args.use_cuda = torch.cuda.is_available()


def main():
    args.checkpoint_dir = "./ckpts/mobileKGQA"
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    config = vars(args)
    dataset = load_data(config, config["lm"])
    train_data = dataset["train"]
    test_data = dataset["test"]

    out_path = os.path.join(
        args.data_folder,
        args.domain,
        f"train_batches_{args.lm}_{args.bit}_{args.batch_size}.h5",
    )
    with h5py.File(out_path, "w") as f:
        num_batches = math.ceil(train_data.num_data / args.batch_size)
        for idx in tqdm(range(num_batches)):
            batch = train_data.get_batch(idx, 1, 0.0)
            data_bytes = pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
            f.create_dataset(f"batch_{idx}", data=np.void(data_bytes))
        f.create_dataset("max_local_entity", data=train_data.max_local_entity)
        f.create_dataset("num_data", data=train_data.num_data)

    out_path = os.path.join(
        args.data_folder,
        args.domain,
        f"test_batches_{args.lm}_{args.bit}_{args.batch_size}.h5",
    )
    with h5py.File(out_path, "w") as f:
        num_batches = math.ceil(test_data.num_data / args.batch_size)
        for idx in tqdm(range(num_batches)):
            batch = test_data.get_batch(idx, 1, 0.0, test=True)
            data_bytes = pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
            f.create_dataset(f"batch_{idx}", data=np.void(data_bytes))
        f.create_dataset("max_local_entity", data=test_data.max_local_entity)
        f.create_dataset("num_data", data=test_data.num_data)


if __name__ == "__main__":
    main()
