#!/bin/bash

# Optimized training script for MicroBERT v4 (140GB H200 Memory Optimized)
# This script is designed to maximize GPU memory utilization for 140GB H200

set -e

# Configuration optimized for 140GB GPU memory
DATASET="hf"
BATCH_SIZE_PER_GPU=256  # Optimized for 140GB memory
EPOCHS=5
LEARNING_RATE=3e-05  # Standard learning rate for large model
STREAMING="true"
NUM_GPUS=8
MAX_SAMPLES="50M"  # Optimized for 140GB memory

echo "=========================================="
echo "MicroBERT v4 Training (140GB H200 Memory Optimized)"
echo "=========================================="
echo "Description: Optimized training for 140GB H200 GPU memory"
echo "Dataset: $DATASET"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Total batch size: $((BATCH_SIZE_PER_GPU * NUM_GPUS))"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Number of GPUs: $NUM_GPUS"
echo "Model: 8L/8H/256D (Large model for 140GB memory)"
echo "Dataset samples: $MAX_SAMPLES"
echo "Sequence length: 256"
echo "Vocabulary size: 25,000"
echo "Estimated time per epoch: 45 minutes"
echo "Estimated total time: 3.8 hours"
echo "Expected memory usage: ~120GB/140GB (85% utilization)"
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
LOG_FILE="logs/h200_140gb_v4_training_$TIMESTAMP.log"

echo "Starting MicroBERT v4 training (140GB optimized)..."
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

echo "MicroBERT v4 training (140GB optimized) completed!"
echo "Results saved to: .mlm_pretrained_v4/"
echo "Log file: $LOG_FILE"
echo "=========================================="
