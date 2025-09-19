#!/bin/bash

lr=$1
num_iter=$2
num_ins=$3
num_gnn=$4
entity_dim=$5
lm=$6
gpu=$7
domain=$8
test_domain=$9
bit=${10}
load_experiment=${11}  # load_experiment 인자 추가
# is_eval=${12}  # is_eval 인자 추가

echo "dataset: cwq"
echo "lr: $lr"
echo "num_iter: $num_iter"
echo "num_ins: $num_ins"
echo "num_gnn: $num_gnn"
echo "entity_dim: $entity_dim"
echo "lm: $lm"
echo "gpu: $gpu"
echo "domain: $domain"
echo "test_domain: $test_domain"
echo "bit: ${bit:-'float embedding'}"
echo "load_experiment: $load_experiment"

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
--data_folder ./data/preprocessed_data/cwq/_domains/ \
--warmup_epoch 80 \
--num_epoch 400 \
--gpu $gpu \
--dataset_name cwq \
--domain $domain \
--test_domain $test_domain"

# bit 인자가 지정된 경우에만 --bit 추가
if [ -n "$bit" ]; then
    cmd="$cmd --bit $bit"
fi

if [ -n "$load_experiment" ]; then
    cmd="$cmd --load_experiment $load_experiment"
fi

# 최종 명령 실행
echo "Executing command:"
echo $cmd
eval $cmd