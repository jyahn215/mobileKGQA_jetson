#!/bin/bash

lr=$1
num_iter=$2
num_ins=$3
num_gnn=$4
entity_dim=$5
lm=$6
gpu=$7
domain=$8

echo "dataset: cwq"
echo "lr: $lr"
echo "num_iter: $num_iter"
echo "num_ins: $num_ins"
echo "num_gnn: $num_gnn"
echo "entity_dim: $entity_dim"
echo "lm: $lm"
echo "gpu: $gpu"
echo "domain: $domain"

python ./src/ReaRev/main.py ReaRev \
--entity_dim $entity_dim \
--lr $lr \
--gradient_clip 1 \
--fact_drop 0 \
--batch_size 8 \
--eval_every 2 \
--lm $lm \
--num_iter $num_iter \
--num_ins $num_ins \
--num_gnn $num_gnn \
--name cwq \
--data_folder ./data/preprocessed_data/cwq/_domains/$domain/ \
--warmup_epoch 80 \
--checkpoint_dir ./ckpts/ReaRev/ \
--num_epoch 200 \
--gpu $gpu \
--dataset_name cwq