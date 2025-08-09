#!/usr/bin/env python3
"""
Multi-GPU training configurations for MicroBERT MLM
Optimized for different hardware setups
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TrainingConfig:
    """Training configuration for multi-GPU setup"""
    name: str
    description: str
    num_gpus: int
    batch_size_per_gpu: int
    epochs: int
    learning_rate: float
    model_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    optimization_config: Dict[str, Any]

# Predefined configurations
CONFIGS = {
    # H200 8-card configurations
    "h200_8gpu_fast": TrainingConfig(
        name="H200 8-GPU Fast Training",
        description="Fast training on H200 8-GPU setup with smaller model",
        num_gpus=8,
        batch_size_per_gpu=64,
        epochs=3,
        learning_rate=5e-5,
        model_config={
            "n_layers": 4,
            "n_heads": 4,
            "n_embed": 8,
            "max_seq_len": 128
        },
        dataset_config={
            "max_samples": 500_000,
            "streaming": True,
            "min_words": 5
        },
        optimization_config={
            "gradient_clip": 1.0,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "mixed_precision": True
        }
    ),
    
    "h200_8gpu_standard": TrainingConfig(
        name="H200 8-GPU Standard Training",
        description="Standard training on H200 8-GPU setup with medium model",
        num_gpus=8,
        batch_size_per_gpu=32,
        epochs=5,
        learning_rate=3e-5,
        model_config={
            "n_layers": 6,
            "n_heads": 8,
            "n_embed": 16,
            "max_seq_len": 128
        },
        dataset_config={
            "max_samples": 1_000_000,
            "streaming": True,
            "min_words": 5
        },
        optimization_config={
            "gradient_clip": 1.0,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "mixed_precision": True
        }
    ),
    
    "h200_8gpu_quality": TrainingConfig(
        name="H200 8-GPU Quality Training",
        description="High-quality training on H200 8-GPU setup with large model",
        num_gpus=8,
        batch_size_per_gpu=16,
        epochs=10,
        learning_rate=2e-5,
        model_config={
            "n_layers": 8,
            "n_heads": 12,
            "n_embed": 24,
            "max_seq_len": 128
        },
        dataset_config={
            "max_samples": 2_000_000,
            "streaming": True,
            "min_words": 5
        },
        optimization_config={
            "gradient_clip": 1.0,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "mixed_precision": True
        }
    ),
    
    # 4-GPU configurations
    "4gpu_standard": TrainingConfig(
        name="4-GPU Standard Training",
        description="Standard training on 4-GPU setup",
        num_gpus=4,
        batch_size_per_gpu=32,
        epochs=5,
        learning_rate=3e-5,
        model_config={
            "n_layers": 4,
            "n_heads": 4,
            "n_embed": 8,
            "max_seq_len": 128
        },
        dataset_config={
            "max_samples": 500_000,
            "streaming": True,
            "min_words": 5
        },
        optimization_config={
            "gradient_clip": 1.0,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "mixed_precision": True
        }
    ),
    
    # 2-GPU configurations
    "2gpu_standard": TrainingConfig(
        name="2-GPU Standard Training",
        description="Standard training on 2-GPU setup",
        num_gpus=2,
        batch_size_per_gpu=32,
        epochs=5,
        learning_rate=3e-5,
        model_config={
            "n_layers": 4,
            "n_heads": 4,
            "n_embed": 8,
            "max_seq_len": 128
        },
        dataset_config={
            "max_samples": 500_000,
            "streaming": True,
            "min_words": 5
        },
        optimization_config={
            "gradient_clip": 1.0,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "mixed_precision": True
        }
    ),
    
    # Single GPU configurations
    "1gpu_standard": TrainingConfig(
        name="1-GPU Standard Training",
        description="Standard training on single GPU",
        num_gpus=1,
        batch_size_per_gpu=16,
        epochs=5,
        learning_rate=3e-5,
        model_config={
            "n_layers": 4,
            "n_heads": 4,
            "n_embed": 8,
            "max_seq_len": 128
        },
        dataset_config={
            "max_samples": 250_000,
            "streaming": True,
            "min_words": 5
        },
        optimization_config={
            "gradient_clip": 1.0,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "mixed_precision": True
        }
    )
}

def get_config(config_name: str) -> TrainingConfig:
    """Get training configuration by name"""
    if config_name not in CONFIGS:
        raise ValueError(f"Configuration '{config_name}' not found. Available: {list(CONFIGS.keys())}")
    return CONFIGS[config_name]

def list_configs():
    """List all available configurations"""
    print("Available Multi-GPU Training Configurations:")
    print("=" * 80)
    
    for name, config in CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Description: {config.description}")
        print(f"  GPUs: {config.num_gpus}")
        print(f"  Batch size per GPU: {config.batch_size_per_gpu}")
        print(f"  Total batch size: {config.num_gpus * config.batch_size_per_gpu}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Model: {config.model_config['n_layers']}L/{config.model_config['n_heads']}H/{config.model_config['n_embed']}D")
        print(f"  Dataset samples: {config.dataset_config['max_samples']:,}")

def estimate_training_time(config: TrainingConfig) -> Dict[str, float]:
    """Estimate training time for a configuration"""
    # Rough estimates based on model size and dataset
    total_params = (
        config.model_config['n_layers'] * 
        config.model_config['n_heads'] * 
        config.model_config['n_embed'] * 
        10000  # vocab size
    )
    
    total_batch_size = config.num_gpus * config.batch_size_per_gpu
    samples_per_epoch = config.dataset_config['max_samples']
    batches_per_epoch = samples_per_epoch // total_batch_size
    
    # Rough time per batch (seconds) - varies by GPU
    if config.num_gpus >= 8:
        time_per_batch = 0.1  # H200 is fast
    elif config.num_gpus >= 4:
        time_per_batch = 0.2
    elif config.num_gpus >= 2:
        time_per_batch = 0.3
    else:
        time_per_batch = 0.5
    
    time_per_epoch = batches_per_epoch * time_per_batch
    total_time = time_per_epoch * config.epochs
    
    return {
        "time_per_epoch_minutes": time_per_epoch / 60,
        "total_time_minutes": total_time / 60,
        "total_time_hours": total_time / 3600
    }

def generate_training_script(config_name: str, output_file: str = None):
    """Generate training script for a configuration"""
    config = get_config(config_name)
    time_estimate = estimate_training_time(config)
    
    script_content = f"""#!/bin/bash

# Auto-generated training script for {config.name}
# Generated from configuration: {config_name}

set -e

# Configuration
DATASET="hf"
BATCH_SIZE_PER_GPU={config.batch_size_per_gpu}
EPOCHS={config.epochs}
LEARNING_RATE={config.learning_rate}
STREAMING="true"
NUM_GPUS={config.num_gpus}

echo "=========================================="
echo "{config.name}"
echo "=========================================="
echo "Description: {config.description}"
echo "Dataset: $DATASET"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Total batch size: $((BATCH_SIZE_PER_GPU * NUM_GPUS))"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Number of GPUs: $NUM_GPUS"
echo "Model: {config.model_config['n_layers']}L/{config.model_config['n_heads']}H/{config.model_config['n_embed']}D"
echo "Dataset samples: {config.dataset_config['max_samples']:,}"
echo "Estimated time per epoch: {time_estimate['time_per_epoch_minutes']:.1f} minutes"
echo "Estimated total time: {time_estimate['total_time_hours']:.1f} hours"
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
LOG_FILE="logs/{config_name}_training_$TIMESTAMP.log"

echo "Starting training..."
echo "Log file: $LOG_FILE"

# Launch distributed training
torchrun \\
    --nproc_per_node=$NUM_GPUS \\
    --nnodes=1 \\
    --node_rank=0 \\
    --master_addr=localhost \\
    --master_port=12355 \\
    mlm_pretrain_v3.py \\
    --dataset $DATASET \\
    --batch-size $BATCH_SIZE_PER_GPU \\
    --epochs $EPOCHS \\
    --lr $LEARNING_RATE \\
    --streaming $STREAMING \\
    2>&1 | tee $LOG_FILE

echo "Training completed!"
echo "Results saved to: .mlm_pretrained_v3/"
echo "Log file: $LOG_FILE"

# Show GPU utilization summary
echo "=========================================="
echo "GPU Utilization Summary"
echo "=========================================="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
"""
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(script_content)
        os.chmod(output_file, 0o755)
        print(f"Training script generated: {output_file}")
    else:
        print(script_content)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python multi_gpu_configs.py list                    # List all configurations")
        print("  python multi_gpu_configs.py info <config_name>      # Show config details")
        print("  python multi_gpu_configs.py generate <config_name>  # Generate training script")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'list':
        list_configs()
    
    elif command == 'info':
        if len(sys.argv) < 3:
            print("Please specify configuration name")
            sys.exit(1)
        
        config_name = sys.argv[2]
        try:
            config = get_config(config_name)
            time_estimate = estimate_training_time(config)
            
            print(f"Configuration: {config.name}")
            print(f"Description: {config.description}")
            print(f"GPUs: {config.num_gpus}")
            print(f"Batch size per GPU: {config.batch_size_per_gpu}")
            print(f"Total batch size: {config.num_gpus * config.batch_size_per_gpu}")
            print(f"Epochs: {config.epochs}")
            print(f"Learning rate: {config.learning_rate}")
            print(f"Model: {config.model_config}")
            print(f"Dataset: {config.dataset_config}")
            print(f"Optimization: {config.optimization_config}")
            print(f"Estimated time per epoch: {time_estimate['time_per_epoch_minutes']:.1f} minutes")
            print(f"Estimated total time: {time_estimate['total_time_hours']:.1f} hours")
        
        except ValueError as e:
            print(f"Error: {e}")
            list_configs()
    
    elif command == 'generate':
        if len(sys.argv) < 3:
            print("Please specify configuration name")
            sys.exit(1)
        
        config_name = sys.argv[2]
        output_file = f"train_{config_name}.sh"
        
        try:
            generate_training_script(config_name, output_file)
        except ValueError as e:
            print(f"Error: {e}")
            list_configs()
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: list, info, generate")
