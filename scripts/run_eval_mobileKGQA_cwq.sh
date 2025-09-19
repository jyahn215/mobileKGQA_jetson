#!/bin/bash

num_iter=$1
num_ins=$2
num_gnn=$3
entity_dim=$4
lm=$5
gpu=$6
test_domain=$7
load_experiment=${8:-""}
bit=${9:-""}

echo "dataset: cwq"
echo "num_iter: $num_iter"
echo "num_ins: $num_ins"
echo "num_gnn: $num_gnn"
echo "entity_dim: $entity_dim"
echo "lm: $lm"
echo "gpu: $gpu"
echo "test_domain: $test_domain"
echo "load_experiment: $load_experiment" 
echo "bit: $bit"


# 기본 실행 명령어
CMD="python ./src/mobileKGQA/main.py ReaRev \
--entity_dim $entity_dim \
--lm $lm \
--num_iter $num_iter \
--num_ins $num_ins \
--num_gnn $num_gnn \
--data_folder ./data/preprocessed_data/cwq/_domains/ \
--warmup_epoch 80 \
--num_epoch 400 \
--gpu $gpu \
--dataset_name cwq \
--test_domain $test_domain"

# load_experiment가 설정된 경우만 추가
if [ -n "$load_experiment" ]; then
    CMD="$CMD --load_experiment $load_experiment"
fi

# bit 인자가 지정된 경우에만 --bit 추가
if [ -n "$bit" ]; then
    CMD="$CMD --bit $bit"
fi

# is_eval 플래그 추가
CMD="$CMD --is_eval"

# 최종 실행
echo "Executing: $CMD"
eval $CMD