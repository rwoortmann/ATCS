#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -p gpu_titanrtx_shared_course
#SBATCH --ntasks=1
#SBATCH --gpus=3
module load 2020
module load Python
python train_multitask.py multitask --finetuned_layers 1 --num_workers 12
