#!/bin/bash
#SBATCH --job-name=pipegen
#SBATCH --array=0-256
#SBATCH --time=00:45:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --output=logs/job_%A_%a.log

export RESOURCE_PATH=/home/groups/kayvonf/fzhou48/halide-gnn-cost-model/resources

rm -rf /var/tmp/piplines-job${SLURM_ARRAY_TASK_ID}
rm -rf /var/tmp/build-job${SLURM_ARRAY_TASK_ID}
rm $RESOURCE_PATH/pipeline-job-artifacts/piplines-job${SLURM_ARRAY_TASK_ID}.tar

srun apptainer run -B.:/workspace $RESOURCE_PATH/simgs/pipeline-dataset.simg \
    --pipeline-id ${SLURM_ARRAY_TASK_ID} \
    --num-schedules 128 \
    --pipelines-dir /var/tmp/piplines-job${SLURM_ARRAY_TASK_ID} \
    --build-dir /var/tmp/build-job${SLURM_ARRAY_TASK_ID} -v

mkdir -p $RESOURCE_PATH/pipeline-job-artifacts
cd /var/tmp/piplines-job${SLURM_ARRAY_TASK_ID}
tar -cvf piplines-job${SLURM_ARRAY_TASK_ID}.tar *
cp /var/tmp/piplines-job${SLURM_ARRAY_TASK_ID}/piplines-job${SLURM_ARRAY_TASK_ID}.tar $RESOURCE_PATH/pipeline-job-artifacts