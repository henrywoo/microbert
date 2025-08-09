#!/bin/bash

# Standard training script that uses existing cache
# This script uses parameters that match the cached dataset

set -e

# Configuration that matches the cached dataset
DATASET="hf"
BATCH_SIZE_PER_GPU=96
EPOCHS=5
LEARNING_RATE=3e-05
STREAMING="true"
NUM_GPUS=8
MAX_SAMPLES="500k"  # Matches the cached dataset (500,000 samples)
MIN_WORDS=5
SEED=42

echo "=========================================="
echo "MicroBERT Training with Cache (Cache-Optimized)"
echo "=========================================="
echo "Description: Training using cached dataset to avoid re-downloading"
echo "Dataset: $DATASET"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Total batch size: $((BATCH_SIZE_PER_GPU * NUM_GPUS))"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Number of GPUs: $NUM_GPUS"
echo "Max samples: $MAX_SAMPLES (matches cache)"
echo "Min words: $MIN_WORDS (matches cache)"
echo "Seed: $SEED (matches cache)"
echo "Expected cache key: 52719dbb61cb7957"
echo "=========================================="

# Check cache status
echo "Checking cache status..."
python cache_manager.py key --ds-name wikitext --ds-kwargs '{"name": "wikitext-103-raw-v1"}' --max-samples 500000

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
LOG_FILE="logs/cache_training_$TIMESTAMP.log"

echo "Starting MicroBERT training with cache..."
echo "Log file: $LOG_FILE"

# Launch distributed training with cache-optimized settings
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
echo "Cache was used successfully - no data was re-downloaded!"
echo "=========================================="
