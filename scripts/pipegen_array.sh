#!/bin/bash
#SBATCH --job-name=pipegen
#SBATCH --array=0-31
#SBATCH --time=00:10:00
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --output=logs/job_%A_%a.log


rm -rf /var/tmp/piplines-job${SLURM_ARRAY_TASK_ID}
rm -rf /var/tmp/build-job${SLURM_ARRAY_TASK_ID}

srun apptainer run -B.:/workspace /home/groups/kayvonf/fzhou48/simg/pipebench.simg \
    --pipeline-id ${SLURM_ARRAY_TASK_ID} \
    --num-schedules 32 \
    --pipelines-dir /var/tmp/piplines-job${SLURM_ARRAY_TASK_ID} \
    --build-dir /var/tmp/build-job${SLURM_ARRAY_TASK_ID} -v

mkdir -p /home/groups/kayvonf/fzhou48/pipelines
cp -r /var/tmp/piplines-job${SLURM_ARRAY_TASK_ID}/* /home/groups/kayvonf/fzhou48/pipelines