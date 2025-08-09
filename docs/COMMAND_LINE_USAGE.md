# MLM Pre-training Command Line Usage Guide

## Basic Usage

### Run with Default Parameters
```bash
python mlm_pretrain_v1.py
```

### View All Available Parameters
```bash
python mlm_pretrain_v1.py --help
```

## Model Architecture Parameters

### Adjust Model Size
```bash
# Ultra-small model (1 layer, 1 head, 8 dimensions)
python mlm_pretrain_v1.py --n_layers 1 --n_heads 1 --n_embed 8

# Small model (2 layers, 2 heads, 16 dimensions) - Default configuration
python mlm_pretrain_v1.py --n_layers 2 --n_heads 2 --n_embed 16

# Medium model (4 layers, 4 heads, 32 dimensions)
python mlm_pretrain_v1.py --n_layers 4 --n_heads 4 --n_embed 32

# Large model (6 layers, 8 heads, 64 dimensions)
python mlm_pretrain_v1.py --n_layers 6 --n_heads 8 --n_embed 64
```

### Adjust Sequence Length
```bash
# Short sequence (64 tokens)
python mlm_pretrain_v1.py --max_seq_len 64

# Long sequence (256 tokens)
python mlm_pretrain_v1.py --max_seq_len 256
```

## Training Parameters

### Adjust Learning Rate
```bash
# High learning rate (fast convergence)
python mlm_pretrain_v1.py --learning_rate 2e-4

# Low learning rate (stable training)
python mlm_pretrain_v1.py --learning_rate 5e-5
```

### Adjust Batch Size
```bash
# Small batch (suitable for small GPUs)
python mlm_pretrain_v1.py --batch_size 16

# Large batch (suitable for large GPUs)
python mlm_pretrain_v1.py --batch_size 64
```

### Adjust Training Epochs
```bash
# Quick test
python mlm_pretrain_v1.py --epochs 3

# Full training
python mlm_pretrain_v1.py --epochs 20
```

### Adjust Early Stopping Patience
```bash
# Quick early stopping
python mlm_pretrain_v1.py --patience 2

# Patient waiting
python mlm_pretrain_v1.py --patience 5
```

## Data Loading Parameters

### Adjust Number of Worker Processes
```bash
# Single process (for debugging)
python mlm_pretrain_v1.py --num_workers 0

# Multi-process (for production)
python mlm_pretrain_v1.py --num_workers 4
```

## Other Options

### Force Fresh Training Start
```bash
python mlm_pretrain_v1.py --force-fresh
```

### Custom Save Directory
```bash
python mlm_pretrain_v1.py --save-dir ./my_mlm_model
```

## Common Configuration Combinations

### Quick Test Configuration
```bash
python mlm_pretrain_v1.py \
    --n_layers 1 \
    --n_heads 1 \
    --n_embed 8 \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 1e-4
```

### Production Training Configuration
```bash
python mlm_pretrain_v1.py \
    --n_layers 4 \
    --n_heads 8 \
    --n_embed 64 \
    --epochs 15 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --patience 5 \
    --num_workers 4
```

### Memory-Constrained Configuration
```bash
python mlm_pretrain_v1.py \
    --n_layers 2 \
    --n_heads 2 \
    --n_embed 16 \
    --batch_size 16 \
    --max_seq_len 64
```

## Parameter Description

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n_layers` | int | 2 | Number of Transformer layers |
| `--n_heads` | int | 2 | Number of attention heads |
| `--n_embed` | int | 16 | Embedding dimensions |
| `--max_seq_len` | int | 128 | Maximum sequence length |
| `--batch_size` | int | 32 | Training batch size |
| `--learning_rate` | float | 1e-4 | Learning rate |
| `--epochs` | int | 10 | Number of training epochs |
| `--patience` | int | 3 | Early stopping patience |
| `--num_workers` | int | 2 | Number of data loading worker processes |
| `--save-dir` | str | .mlm_v1 | Model save directory |
| `--force-fresh` | flag | False | Force fresh training start |

## Notes

1. **n_embed must be divisible by n_heads**
2. **batch_size should be adjusted based on GPU memory**
3. **num_workers is recommended to be set to 1/2 to 1/4 of CPU cores**
4. **Using --force-fresh will delete existing checkpoints**

## Example: Progressive Training

```bash
# Phase 1: Small model quick validation
python mlm_pretrain_v1.py --n_layers 1 --n_heads 1 --n_embed 8 --epochs 3

# Phase 2: Medium model training
python mlm_pretrain_v1.py --n_layers 2 --n_heads 2 --n_embed 16 --epochs 10

# Phase 3: Large model fine-tuning
python mlm_pretrain_v1.py --n_layers 4 --n_heads 4 --n_embed 32 --epochs 20
```

Now you can flexibly adjust all model configurations through command line parameters without needing to modify the code!
