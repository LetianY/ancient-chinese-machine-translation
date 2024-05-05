#!/bin/bash
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
module load cudnn cuda
source pytorch.venv/bin/activate
bash run_train.sh