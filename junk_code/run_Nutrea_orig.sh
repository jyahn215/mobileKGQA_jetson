#!/bin/bash
#SBATCH -J ReaRev
#SBATCH -p gpu-farm
#SBATCH --gres=gpu:1
#SBATCH --output=./save/outputs/%x_%j_%N_%n.out
#SBATCH --error=./save/outputs/%x_%j_%N_%n.err

module load cuda/12.1

python ./preprocess/save_last_hidden_state.py