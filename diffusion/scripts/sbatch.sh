#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=20G
#SBATCH -t 10:00:00
module load cudnn cuda
source ../../pytorch.venv/bin/activate
bash scripts/run_train.sh