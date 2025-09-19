#!/bin/bash


gpu=$1
domain=$2
split=$3
model_name=$4
dataset_name=$5
bit=$6

export CUDA_VISIBLE_DEVICES=$gpu

echo "gpu: $gpu"
echo "domain: $domain"
echo "split: $split"
echo "model_name: $model_name"
echo "dataset_name: $dataset_name"
echo "bit: ${bit:-'float embedding'}"

cmd="python ./src/save_hash_code.py"

if [ -n "$domain" ]; then
    cmd="$cmd -domain $domain"
fi
if [ -n "$split" ]; then
    cmd="$cmd -split $split"
fi
if [ -n "$model_name" ]; then
    cmd="$cmd -model_name $model_name"
fi
if [ -n "$dataset_name" ]; then
    cmd="$cmd -dataset_name $dataset_name"
fi
if [ -n "$bit" ]; then
    cmd="$cmd -bit $bit"
fi

cmd="python ./src/mobileKGQA/main.py ReaRev \
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
--data_folder ./data/preprocessed_data/webqsp/_domains/ \
--warmup_epoch 80 \
--num_epoch 200 \
--gpu $gpu \
--dataset_name webqsp \
--domain $domain \
--test_domain $test_domain"

# bit 인자가 지정된 경우에만 --bit 추가
if [ -n "$bit" ]; then
    cmd="$cmd --bit $bit"
fi

# 최종 명령 실행
echo "Executing command:"
echo $cmd
eval $cmd