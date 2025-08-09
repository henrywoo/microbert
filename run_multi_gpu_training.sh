#!/bin/bash

# Multi-GPU Training Script for MicroBERT MLM
# Optimized for H200 8-card environment

set -e

# Default parameters
DATASET=${1:-"hf"}
BATCH_SIZE_PER_GPU=${2:-32}
EPOCHS=${3:-5}
LEARNING_RATE=${4:-3e-5}
STREAMING=${5:-"true"}
MAX_SAMPLES=${6:-"500k"}  # Default to 500k samples

# Number of GPUs (adjust based on your setup)
NUM_GPUS=8

echo "=========================================="
echo "Multi-GPU MLM Training Setup"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Total batch size: $((BATCH_SIZE_PER_GPU * NUM_GPUS))"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Streaming: $STREAMING"
echo "Number of GPUs: $NUM_GPUS"
echo "Dataset samples: $MAX_SAMPLES"
echo "=========================================="

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please ensure CUDA is installed."
    exit 1
fi

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
LOG_FILE="logs/multi_gpu_training_${DATASET}_${TIMESTAMP}.log"

echo "Starting multi-GPU training..."
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

echo "Training completed!"
echo "Results saved to: .mlm_pretrained_v3/"
echo "Log file: $LOG_FILE"

# Optional: Show GPU utilization summary
echo "=========================================="
echo "GPU Utilization Summary"
echo "=========================================="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
