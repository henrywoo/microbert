# Multi-GPU Training Guide

This guide explains how to use `mlm_pretrain_v3.py` for multi-GPU training on your H200 8-card setup.

## Quick Start

### Method 1: Using Pre-configured Scripts

1. **List available configurations:**
   ```bash
   python multi_gpu_configs.py list
   ```

2. **Generate training script for H200 8-GPU:**
   ```bash
   python multi_gpu_configs.py generate h200_8gpu_standard
   ```

3. **Run the generated script:**
   ```bash
   ./train_h200_8gpu_standard.sh
   ```

### Method 2: Using torchrun Directly

```bash
# For H200 8-GPU setup
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    mlm_pretrain_v3.py \
    --dataset hf \
    --batch-size 32 \
    --epochs 5 \
    --lr 3e-5 \
    --streaming true \
    --max-samples 500k
```

### Method 3: Using the Generic Script

```bash
# Run with default H200 8-GPU settings
./run_multi_gpu_training.sh

# Or with custom parameters
./run_multi_gpu_training.sh hf 32 5 3e-5 true
```

## Available Configurations

### H200 8-GPU Configurations

| Configuration | Model | Batch Size | Epochs | Time | Use Case |
|---------------|-------|------------|--------|------|----------|
| `h200_8gpu_fast` | 4L/4H/8D | 512 | 3 | ~20min | Quick testing |
| `h200_8gpu_standard` | 6L/8H/16D | 256 | 5 | ~30min | Standard training |
| `h200_8gpu_quality` | 8L/12H/24D | 128 | 10 | ~2hrs | High quality |

### Other Configurations

| Configuration | GPUs | Model | Batch Size | Use Case |
|---------------|------|-------|------------|----------|
| `4gpu_standard` | 4 | 4L/4H/8D | 128 | 4-GPU setup |
| `2gpu_standard` | 2 | 4L/4H/8D | 64 | 2-GPU setup |
| `1gpu_standard` | 1 | 4L/4H/8D | 16 | Single GPU |

## Command Line Arguments

```bash
python mlm_pretrain_v3.py [OPTIONS]

Options:
  --dataset {imdb,hf}     Dataset choice (default: hf)
  --streaming {true,false} Streaming mode (default: true)
  --batch-size INT        Batch size per GPU (default: 32)
  --epochs INT            Number of training epochs (default: 5)
  --lr FLOAT              Learning rate (default: 3e-5)
  --max-samples STR       Maximum number of samples to load (e.g., 500k, 5M, 50M, 500M)
  --local_rank INT        Local rank for distributed training (auto-detected)
```

## Environment Requirements

### Software Requirements

1. **PyTorch with CUDA support:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Additional dependencies:**
   ```bash
   pip install transformers datasets tqdm matplotlib
   ```

3. **NCCL for multi-GPU communication:**
   ```bash
   # Usually included with PyTorch, but you can install separately
   conda install pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

### Hardware Requirements

- **GPU Memory:** At least 8GB per GPU (H200 has 80GB)
- **System Memory:** At least 32GB RAM
- **Storage:** At least 50GB free space for datasets and models

## Monitoring Training

### Log Files

Training logs are automatically saved to:
```
logs/
├── h200_8gpu_standard_training_20241201_143022.log
├── h200_8gpu_fast_training_20241201_143156.log
└── ...
```

### GPU Monitoring

During training, you can monitor GPU usage:
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use the built-in summary at the end of training
```

### Training Progress

The script shows:
- Progress bars for each epoch
- Training and validation loss
- Estimated time remaining
- GPU utilization

## Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   - Reduce batch size per GPU
   - Use smaller model configuration
   - Enable gradient checkpointing

2. **NCCL errors:**
   - Check if all GPUs are visible: `nvidia-smi`
   - Ensure NCCL is installed: `python -c "import torch; print(torch.cuda.nccl.version())"`

3. **Port conflicts:**
   - Change master port: `--master_port=12356`

4. **Permission denied:**
   - Make scripts executable: `chmod +x *.sh`

### Performance Optimization

1. **For H200 8-GPU:**
   - Use batch size 32-64 per GPU
   - Enable mixed precision (automatic)
   - Use streaming datasets for large datasets

2. **Memory optimization:**
   - Reduce sequence length if needed
   - Use gradient accumulation for larger effective batch sizes

## Results

### Model Output

Trained models are saved to:
```
.mlm_pretrained_v3/
├── mlm_model.pth           # Model weights
├── tokenizer.json          # Tokenizer vocabulary
├── mlm_training_history.json  # Training history
└── training_history.png    # Training plots
```

### Model Performance

Expected performance improvements with multi-GPU:
- **Speedup:** 6-8x faster than single GPU
- **Memory efficiency:** Better utilization of GPU memory
- **Scalability:** Linear scaling with number of GPUs

## Advanced Usage

### Custom Configurations

You can create custom configurations by modifying `multi_gpu_configs.py`:

```python
# Add custom configuration
CONFIGS["my_custom_config"] = TrainingConfig(
    name="My Custom Training",
    description="Custom training setup",
    num_gpus=4,
    batch_size_per_gpu=16,
    epochs=10,
    learning_rate=1e-5,
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
)
```

### Distributed Training Across Nodes

For multi-node training, modify the torchrun command:

```bash
# Node 0
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=12355 \
    mlm_pretrain_v3.py

# Node 1
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=12355 \
    mlm_pretrain_v3.py
```

## Support

For issues and questions:
1. Check the logs in `logs/` directory
2. Verify GPU availability with `nvidia-smi`
3. Test single GPU first: `python mlm_pretrain_v3.py --dataset imdb`
4. Check PyTorch installation: `python -c "import torch; print(torch.cuda.device_count())"`
