#!/bin/bash

# Optimized training script for H200 8-GPU with 140GB Memory
# This script is designed to maximize GPU memory utilization

set -e

# Configuration optimized for 24GB GPU memory
DATASET="hf"
BATCH_SIZE_PER_GPU=96  # Optimized for 24GB memory
EPOCHS=5
LEARNING_RATE=3e-05  # Standard learning rate for medium model
STREAMING="true"
NUM_GPUS=8
MAX_SAMPLES="10M"  # Optimized for 24GB memory

echo "=========================================="
echo "H200 8-GPU Optimized Training (24GB Memory)"
echo "=========================================="
echo "Description: Optimized training on H200 8-GPU setup with medium model for 24GB memory"
echo "Dataset: $DATASET"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Total batch size: $((BATCH_SIZE_PER_GPU * NUM_GPUS))"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Number of GPUs: $NUM_GPUS"
echo "Model: 8L/8H/256D (Medium model for 24GB memory)"
echo "Dataset samples: $MAX_SAMPLES"
echo "Sequence length: 256"
echo "Vocabulary size: 25,000"
echo "Estimated time per epoch: 20 minutes"
echo "Estimated total time: 1.7 hours"
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
LOG_FILE="logs/h200_8gpu_optimized_training_$TIMESTAMP.log"

echo "Starting optimized training..."
echo "Log file: $LOG_FILE"

# Launch distributed training with optimized settings
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

echo "Optimized training completed!"
echo "Results saved to: .mlm_pretrained_v3/"
echo "Log file: $LOG_FILE"

# Show GPU utilization summary
echo "=========================================="
echo "GPU Utilization Summary"
echo "=========================================="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
