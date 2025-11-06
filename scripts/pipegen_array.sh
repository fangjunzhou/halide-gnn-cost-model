#!/bin/bash
#SBATCH --job-name=pipegen
#SBATCH --array=0-31
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --output=logs/out_%A_%a.txt

srun echo "${SLURM_ARRAY_TASK_ID}"