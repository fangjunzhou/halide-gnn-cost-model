#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=train_gcn
#SBATCH --time=00:10:00
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH --output=logs/job_%A_%a.log

export TRAINING_SCRIPT=/home/groups/kayvonf/fzhou48/halide-gnn-cost-model/scripts/training/train_gcn.py
export DATASET_PATH=/home/groups/kayvonf/fzhou48/halide-gnn-cost-model/resources/datasets
export LOG_DIR=/home/groups/kayvonf/fzhou48/halide-gnn-cost-model/resources/runs
export DATASET_NAME=pipelines-32k-data
export SCRATCH_PATH=/lscratch/fzhou48
export TRAINING_IMG=/home/groups/kayvonf/fzhou48/halide-gnn-cost-model/resources/simgs/training.simg
export MODELS_DIR=/home/groups/kayvonf/fzhou48/halide-gnn-cost-model/resources/models

export NUM_EPOCHS=250
export SAVE_EVERY=25

export HIDDEN_CHANNELS=64
export OUT_CHANNELS=64
export NUM_LAYERS=4
export MODEL_HIDDEN=32

nvidia-smi

# Copy and untar dataset
cp $DATASET_PATH/$DATASET_NAME.tar $SCRATCH_PATH/
tar -xvf $SCRATCH_PATH/$DATASET_NAME.tar -C $SCRATCH_PATH/

srun apptainer run --nv -B.:/workspace -B/lscratch:/lscratch $TRAINING_IMG \
    $TRAINING_SCRIPT \
    --pipelines-dir $SCRATCH_PATH/$DATASET_NAME \
    --models-dir $MODELS_DIR/gcn-h$HIDDEN_CHANNELS-o$OUT_CHANNELS-l$NUM_LAYERS-mh$MODEL_HIDDEN \
    --log-dir $LOG_DIR \
    --num-epochs $NUM_EPOCHS \
    --num-workers 8 \
    --hidden-channels $HIDDEN_CHANNELS \
    --out-channels $OUT_CHANNELS \
    --num-layers $NUM_LAYERS \
    --model-hidden $MODEL_HIDDEN \
    --save-every $SAVE_EVERY