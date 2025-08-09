#!/bin/bash

# Optimized training script for MicroBERT v4 (24GB Memory Optimized)
# This script is designed to maximize GPU memory utilization while ensuring compatibility

set -e

# Configuration optimized for 24GB GPU memory
DATASET="hf"
BATCH_SIZE_PER_GPU=96  # Will be automatically adjusted based on GPU memory
EPOCHS=5
LEARNING_RATE=3e-05  # Standard learning rate for medium model
STREAMING="true"
NUM_GPUS=8
MAX_SAMPLES="10M"  # Optimized for 24GB memory

echo "=========================================="
echo "MicroBERT v4 Training (24GB Memory Optimized)"
echo "=========================================="
echo "Description: Optimized training for 24GB GPU memory (H200/A10 compatible)"
echo "Dataset: $DATASET"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU (will be auto-adjusted)"
echo "Total batch size: $((BATCH_SIZE_PER_GPU * NUM_GPUS)) (estimated)"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Number of GPUs: $NUM_GPUS"
echo "Model: Dynamic (4-6L/8H/128D for large, 6L/8H/128D for medium, 4L/8H/128D for small)"
echo "Dataset samples: $MAX_SAMPLES"
echo "Sequence length: 128 (reduced for memory)"
echo "Vocabulary size: 50,000 (limited)"
echo "Estimated time per epoch: 15 minutes"
echo "Estimated total time: 1.25 hours"
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
LOG_FILE="logs/v4_training_$TIMESTAMP.log"

echo "Starting MicroBERT v4 training..."
echo "Log file: $LOG_FILE"

# Launch distributed training with optimized settings
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    mlm_pretrain_v4.py \
    --dataset $DATASET \
    --batch-size $BATCH_SIZE_PER_GPU \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --streaming $STREAMING \
    --max-samples $MAX_SAMPLES \
    2>&1 | tee $LOG_FILE

echo "Training completed!"
echo "Results saved to: .mlm_pretrained_v4/"
echo "Log file: $LOG_FILE"
echo "=========================================="
