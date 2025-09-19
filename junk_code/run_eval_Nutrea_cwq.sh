#!/bin/bash

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
test_domain=${11}
load_experiment=${12}  # load_experiment 인자 추가

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
echo "test_domain: $test_domain"
echo "load_experiment: $load_experiment"


python ./src/NuTrea_minimal/src/main.py NuTrea \
    --dataset_name cwq \
    --data_folder ./data/preprocessed_data/cwq/_domains/ \
    --domain $domain \
    --lm $lm \
    --relation_word_emb True \
    --num_epoch 100 \
    --eval_every 2 \
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
    --test_domain $test_domain \
    --load_experiment $load_experiment \
    --batch_size 8 \
    --is_eval \
    --rf_ief