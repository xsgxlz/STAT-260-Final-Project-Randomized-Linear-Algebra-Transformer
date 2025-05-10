#!/bin/bash -l
#SBATCH --time=4:05:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=RLALLaMA_Array
#SBATCH --array=0-3
#SBATCH --output=RLALLaMA_%A_%a.log

cd "/accounts/grad/zhangyunzhe2023/stat 260/STAT-260-Final-Project-Randomized-Linear-Algebra-Transformer/RLALLaMA3"

# Activate Conda environment
conda activate neuralode

echo "Starting SLURM array task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID on host $HOSTNAME"

python -u train_0.75.py

echo "Finished SLURM array task $SLURM_ARRAY_TASK_ID"

exit