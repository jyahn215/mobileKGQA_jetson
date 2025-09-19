#!/bin/bash
#SBATCH -J NuTrea
#SBATCH -p gpu-farm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --qos=low_gpu_users
#SBATCH --output=./save/outputs/%x_%j_%N_%n.out
#SBATCH --error=./save/outputs/%x_%j_%N_%n.err

module load cuda/12.1

lr=$1
num_iter=$2
num_expansion_ins=$3
num_backup_ins=$4
backup_depth=$5
num_gnn=$6
context_coef=$7
lm=$8
gpu=$9
domain=${10}
batch_size=${11}

echo "lr: $lr"
echo "num_iter: $num_iter"
echo "num_expansion_ins: $num_expansion_ins"
echo "num_backup_ins: $num_backup_ins"
echo "backup_depth: $backup_depth"
echo "num_gnn: $num_gnn"
echo "context_coef: $context_coef"
echo "lm: $lm"
echo "gpu: $gpu"
echo "domain: $domain"
echo "batch_size: $batch_size"


python ./src/NuTrea_minimal/src/main.py NuTrea \
    --dataset_name cwq \
    --domain $domain \
    --lm $lm \
    --relation_word_emb True \
    --num_epoch 200 \
    --eval_every 2 \
    --batch_size $batch_size \
    --decay_rate 0.99 \
    --linear_dropout 0.3 \
    --num_iter $num_iter \
    --num_expansion_ins $num_expansion_ins \
    --num_backup_ins $num_backup_ins \
    --backup_depth $backup_depth \
    --num_layers $num_gnn \
    --context_coef $context_coef \
    --rf_ief \
    --gpu $gpu \
    --checkpoint_dir ./ckpts/nutrea \
    --lr $lr \
    --gradient_clip 1.0 \
    --slurm