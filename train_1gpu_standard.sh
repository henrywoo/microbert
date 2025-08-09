#!/bin/bash

# Auto-generated training script for 1-GPU Standard Training
# Generated from configuration: 1gpu_standard

set -e

# Configuration
DATASET="hf"
BATCH_SIZE_PER_GPU=16
EPOCHS=5
LEARNING_RATE=3e-05
STREAMING="true"
NUM_GPUS=1
MAX_SAMPLES="250k"  # Dataset size configuration

echo "=========================================="
echo "1-GPU Standard Training"
echo "=========================================="
echo "Description: Standard training on single GPU"
echo "Dataset: $DATASET"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Total batch size: $((BATCH_SIZE_PER_GPU * NUM_GPUS))"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Number of GPUs: $NUM_GPUS"
echo "Model: 4L/4H/8D"
echo "Dataset samples: $MAX_SAMPLES"
echo "Estimated time per epoch: 130.2 minutes"
echo "Estimated total time: 10.9 hours"
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
LOG_FILE="logs/1gpu_standard_training_$TIMESTAMP.log"

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

echo "Training completed!"
echo "Results saved to: .mlm_pretrained_v3/"
echo "Log file: $LOG_FILE"

# Show GPU utilization summary
echo "=========================================="
echo "GPU Utilization Summary"
echo "=========================================="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
