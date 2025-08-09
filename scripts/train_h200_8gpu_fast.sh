#!/bin/bash

# Auto-generated training script for H200 8-GPU Fast Training
# Generated from configuration: h200_8gpu_fast

set -e

# Configuration
DATASET="hf"
BATCH_SIZE_PER_GPU=64
EPOCHS=3
LEARNING_RATE=5e-05
STREAMING="true"
NUM_GPUS=8
MAX_SAMPLES="500k"  # Dataset size configuration

echo "=========================================="
echo "H200 8-GPU Fast Training"
echo "=========================================="
echo "Description: Fast training on H200 8-GPU setup with smaller model"
echo "Dataset: $DATASET"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Total batch size: $((BATCH_SIZE_PER_GPU * NUM_GPUS))"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Number of GPUs: $NUM_GPUS"
echo "Model: 4L/4H/8D"
echo "Dataset samples: $MAX_SAMPLES"
echo "Estimated time per epoch: 1.6 minutes"
echo "Estimated total time: 0.1 hours"
echo "=========================================="

# Check available GPUs
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS are available."
    echo "Adjusting to use $AVAILABLE_GPUS GPUs."
    NUM_GPUS=$AVAILABLE_GPUS
fi

# Create logs directory
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/h200_8gpu_fast_training_$TIMESTAMP.log"

echo "Starting training..."
echo "Log file: $LOG_FILE"

# Launch distributed training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    mlm_pretrain_v3.py \
    --dataset $DATASET \
    --batch-size $BATCH_SIZE_PER_GPU \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --streaming $STREAMING \
    --max-samples $MAX_SAMPLES \
    2>&1 | tee $LOG_FILE

echo "Fast training completed!"
echo "Results saved to: .mlm_pretrained_v3/"
echo "Log file: $LOG_FILE"
echo "=========================================="
