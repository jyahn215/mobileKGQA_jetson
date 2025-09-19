#!/bin/bash
#SBATCH -J ReaRev
#SBATCH -p gpu-farm
#SBATCH --gres=gpu:1
#SBATCH --output=./save/outputs/%x_%j_%N_%n.out
#SBATCH --error=./save/outputs/%x_%j_%N_%n.err

module load cuda/12.1

# 테스트할 여러 값들을 배열로 정의
dataset_list=("cwq" "webqsp")
model_list=("gemma2:2b" "qwen2:0.5b" "phi3.5")


# 중첩 for문으로 모든 조합 실행
for dataset in "${dataset_list[@]}"; do
  for model in "${model_list[@]}"; do
  
    echo "-----------------------------------------------------------------"
    echo "Running with:"
    echo "dataset       = $dataset"
    echo "model         = $model"
    echo "-----------------------------------------------------------------"

    # 실제 실행할 python 명령어
    python ./preprocess/save_last_hidden_state.py --dataset $dataset --model_name $model --slurm
  done
done
