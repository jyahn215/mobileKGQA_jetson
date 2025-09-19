#!/bin/bash
#SBATCH -J ReaRev
#SBATCH -p gpu-farm
#SBATCH --gres=gpu:1
#SBATCH --output=./save/outputs/%x_%j_%N_%n.out
#SBATCH --error=./save/outputs/%x_%j_%N_%n.err

module load cuda/12.1

# 테스트할 여러 값들을 배열로 정의
dataset_list=(cwq)
lr_list=(0.0001 0.0003 0.0005 0.001)
num_iter=$1
num_ins=$2
num_gnn=$3
entity_dim_list=(50)
gradient_clip_list=(1.0)
fact_drop_list=(0)
warmup_epoch_list=(80)
lm_list=("gemma2:2b")


# 중첩 for문으로 모든 조합 실행
for dataset in "${dataset_list[@]}"; do
  for lr in "${lr_list[@]}"; do
    for entity_dim in "${entity_dim_list[@]}"; do
      for gradient_clip in "${gradient_clip_list[@]}"; do
        for fact_drop in "${fact_drop_list[@]}"; do
          for warmup_epoch in "${warmup_epoch_list[@]}"; do
            for lm in "${lm_list[@]}"; do

              echo "-----------------------------------------------------------------"
              echo "Running with:"
              echo "dataset       = $dataset"
              echo "lr            = $lr"
              echo "num_iter      = $num_iter"
              echo "num_ins       = $num_ins"
              echo "num_gnn       = $num_gnn"
              echo "entity_dim    = $entity_dim"
              echo "gradient_clip = $gradient_clip"
              echo "fact_drop     = $fact_drop"
              echo "warmup_epoch  = $warmup_epoch"
              echo "lm            = $lm"
              echo "-----------------------------------------------------------------"

              # 실제 실행할 python 명령어
              python ./src/ReaRev/main.py ReaRev \
                --entity_dim "$entity_dim" \
                --lr "$lr" \
                --gradient_clip "$gradient_clip" \
                --fact_drop "$fact_drop" \
                --batch_size 8 \
                --eval_every 2 \
                --lm "$lm" \
                --num_iter "$num_iter" \
                --num_ins "$num_ins" \
                --num_gnn "$num_gnn" \
                --name "$dataset" \
                --data_folder "./data/preprocessed_data/$dataset/total/" \
                --warmup_epoch "$warmup_epoch" \
                --checkpoint_dir ./ckpts/ReaRev/ \
                --num_epoch 200 \

              # 만약 에러가 발생한 경우 다음 반복으로 넘어가기
              if [ $? -ne 0 ]; then
                echo "##################################################################"
                echo "############################ERROR#################################"
                echo "##################################################################"
                echo "Running with:"
                echo "dataset       = $dataset"
                echo "lr            = $lr"
                echo "num_iter      = $num_iter"
                echo "num_ins       = $num_ins"
                echo "num_gnn       = $num_gnn"
                echo "entity_dim    = $entity_dim"
                echo "gradient_clip = $gradient_clip"
                echo "fact_drop     = $fact_drop"
                echo "warmup_epoch  = $warmup_epoch"
                echo "lm            = $lm"
                echo "##################################################################"
                echo "############################ERROR#################################"
                echo "##################################################################"
                continue
              fi
            done
          done
        done
      done
    done
  done
done
