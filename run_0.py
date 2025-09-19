import os
gpu = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

for dim in [128]:
    for lr in [0.00004, 0.00008, 0.0001, 0.0002, 0.0004]:
        for reg_level in [0.002]:
            os.system(f"python3 ./src/train_hashing_module.py -lr {lr} -reg_level {reg_level} -gpu {gpu} -dim_list {dim} -model_name gte-large")