# MicroBERT

This project serves as an educational platform for learning BERT implementation, pretraining, and fine-tuning techniques. It offers a comprehensive framework that helps learners understand transformer architectures and training methodologies through hands-on experience.

The project provides a lightweight BERT implementation with Masked Language Modeling (MLM) pre-training and Supervised Fine-Tuning (SFT) capabilities. It supports multiple datasets and streaming features, with multiple versions optimized for different GPU environments and learning objectives. It supports GPUs from entry-level (GTX 1060, GTX 1660, RTX 2060, RTX 3060, RTX 4060) to mid-range (RTX 2070, RTX 2080, RTX 3070, RTX 3080, RTX 4070, RTX 4080, RTX 4090, A10, A10G, H10, H20) to high-end enterprise (V100, A100, A100 80GB, H100, H200, B100, B200, L40, L40S), with automatic configuration adjustment based on available GPU memory and multi-GPU distributed training support.

## Project Overview

This educational project provides:

- **Pretraining (MLM)**: Multiple versions (v0-v4) for different GPU environments and model sizes
- **Supervised Fine-Tuning (SFT)**: Complete SFT implementation for downstream tasks
- **GPU Environment Adaptation**: Optimized configurations for various GPU memory capacities
- **Educational Resources**: Comprehensive examples for learning transformer architectures

## Version Architecture

### Pretraining Versions (v0-v4)

All versions are designed for **pretraining** with different optimizations:

- **v0**: Basic single-GPU full precision training for small models
- **v1**: Single-GPU mixed precision training for small models  
- **v2**: Single-GPU full precision training for medium models
- **v3**: Multi-GPU full precision training for large models
- **v4**: Multi-GPU mixed precision training for extra-large models

### SFT Implementation

- **`sft_hfbert.py`**: Complete Supervised Fine-Tuning implementation for downstream tasks

## Features

- 🚀 **Lightweight BERT**: Small, efficient BERT implementation
- 📊 **Multiple Datasets**: Support for IMDB and Hugging Face datasets
- 💾 **Streaming Support**: Memory-efficient data loading with local caching
- 🎯 **MLM Pre-training**: Full Masked Language Modeling implementation
- 🔧 **SFT Fine-tuning**: Complete supervised fine-tuning pipeline
- 📈 **Training Visualization**: Built-in plotting and monitoring
- 🔧 **Flexible Configuration**: Easy model parameter tuning
- 🎓 **Educational Focus**: Designed for learning transformer architectures

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd microbert
```

### 2. Create a virtual environment (recommended)
```bash
# Using conda
conda create -n microbert python=3.10
conda activate microbert

# Or using venv
python -m venv microbert_env
source microbert_env/bin/activate  # On Windows: microbert_env\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install the package in development mode
```bash
pip install -e .
```

## Quick Start

### 🎯 **Choose the Right Training Script for Your Learning Goals**

This educational project provides multiple versions optimized for different learning objectives and hardware configurations. Choose based on your educational needs:

#### **v0: Basic Learning (Single GPU, Full Precision)**
```bash
# Basic training for learning fundamentals
python mlm_pretrain_v0.py
```
- **Use Case**: Learning BERT fundamentals, basic implementation understanding
- **Dataset**: IMDB movie reviews (~25K samples)
- **Model**: Small model (2 layers, 2 heads, 4-dim embeddings)
- **Training Time**: ~5 minutes
- **Memory Requirements**: Low
- **Educational Focus**: Understanding basic transformer architecture

#### **v1: Mixed Precision Learning (Single GPU, Mixed Precision)**
```bash
# Mixed precision training for learning optimization techniques
python mlm_pretrain_v1.py
```
- **Use Case**: Learning mixed precision training, optimization techniques
- **Dataset**: IMDB movie reviews (~25K samples)
- **Model**: Small model (2 layers, 2 heads, 4-dim embeddings)
- **Training Time**: ~5 minutes
- **Memory Requirements**: Low
- **Educational Focus**: Understanding mixed precision training and optimization

#### **v2: Medium Model Learning (Single GPU, Full Precision)**
```bash
# Use Hugging Face large datasets (default 500K samples)
python mlm_pretrain_v2.py hf

# Specify data size (5M samples)
python mlm_pretrain_v2.py hf true 5M

# Specify data size (50M samples)
python mlm_pretrain_v2.py hf false 50M

# Or use IMDB dataset
python mlm_pretrain_v2.py imdb
```
- **Use Case**: Learning with medium-scale models, understanding larger datasets
- **Dataset**: Hugging Face datasets (configurable size: 500K-500M samples) or IMDB
- **Model**: Medium model (4 layers, 4 heads, 8-dim embeddings) or small model
- **Training Time**: ~30 minutes (500K) / ~2 hours (5M) / ~20 hours (50M)
- **Memory Requirements**: Medium
- **Educational Focus**: Understanding medium-scale models and large dataset handling

#### **v3: Multi-GPU Learning (Multi-GPU, Full Precision)**
```bash
# Use pre-configured script (recommended)
python multi_gpu_configs.py generate h200_8gpu_standard
./train_h200_8gpu_standard.sh

# Or use torchrun directly (default 500K samples)
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
- **Use Case**: Learning distributed training, multi-GPU environments
- **Dataset**: Hugging Face datasets (configurable size: 500K-50M samples) or IMDB
- **Model**: Large model (6 layers, 8 heads, 16-dim embeddings) or small model
- **Training Time**: ~15 minutes (500K) / ~1.3 hours (5M) / ~13 hours (50M)
- **Memory Requirements**: Medium
- **GPU Requirements**: 8-card H200 or similar configuration
- **Educational Focus**: Understanding distributed training and multi-GPU coordination

#### **v4: Advanced Multi-GPU Learning (Multi-GPU, Mixed Precision)**
```bash
# Use pre-configured script (recommended)
./train_h200_8gpu_v4.sh

# Or use torchrun directly (24GB memory optimized)
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    mlm_pretrain_v4.py \
    --dataset hf \
    --batch-size 96 \
    --epochs 5 \
    --lr 3e-5 \
    --streaming true \
    --max-samples 10M
```
- **Use Case**: Learning advanced distributed training with mixed precision (H200/A10 compatible)
- **Dataset**: Hugging Face datasets (configurable size: 500K-50M samples) or IMDB
- **Model**: Dynamic configuration (automatically adjusted based on GPU memory)
  - **Large Model** (100GB+ GPU): 4 layers, 8 heads, 128-dim embeddings, batch_size=16
  - **Medium Model** (40GB+ GPU): 6 layers, 8 heads, 128-dim embeddings, batch_size=32
  - **Small Model** (24GB GPU): 4 layers, 8 heads, 128-dim embeddings, batch_size=8
- **Training Time**: ~15 minutes (10M samples)
- **Memory Requirements**: Very conservative configuration, ensuring single card memory usage doesn't exceed 24GB
- **GPU Requirements**: 24GB+ GPU (H200, A10, RTX 4090, etc.)
- **Educational Focus**: Understanding advanced distributed training, mixed precision, and memory optimization

### **SFT (Supervised Fine-Tuning) Learning**

For learning supervised fine-tuning techniques:

```bash
# Complete SFT implementation for downstream tasks
python sft_hfbert.py
```

- **Use Case**: Learning supervised fine-tuning for downstream NLP tasks
- **Implementation**: Complete SFT pipeline with Hugging Face BERT
- **Educational Focus**: Understanding transfer learning and task-specific fine-tuning
- **Features**: 
  - Pre-trained model loading
  - Task-specific dataset preparation
  - Fine-tuning training loop
  - Evaluation and inference

## Project Structure

```
microbert/
├── microbert/                    # Core package
│   ├── __init__.py
│   ├── model.py                 # BERT model implementation
│   ├── tokenizer.py             # Word-level tokenizer
│   └── utils.py                 # Utility functions
├── mlm_pretrain_v0.py           # v0: Basic MLM training (Single GPU, Full Precision)
├── mlm_pretrain_v1.py           # v1: Mixed precision MLM training (Single GPU, Mixed Precision)
├── mlm_pretrain_v2.py           # v2: Medium model MLM training (Single GPU, Full Precision)
├── mlm_pretrain_v3.py           # v3: Multi-GPU MLM training (Multi-GPU, Full Precision)
├── mlm_pretrain_v4.py           # v4: Advanced multi-GPU MLM training (Multi-GPU, Mixed Precision)
├── sft_hfbert.py                # SFT implementation for downstream tasks
├── multi_gpu_configs.py         # Multi-GPU training configurations
├── run_multi_gpu_training.sh    # Multi-GPU training launcher
├── train_h200_8gpu_standard.sh  # H200 8-GPU standard training script
├── train_h200_8gpu_v4.sh        # H200 8-GPU v4 optimized training script
├── train_24gb_optimized.sh      # 24GB optimized training script
├── model_config_comparison.py   # Model configuration comparison tool
├── test_streaming.py            # Test streaming functionality
├── test_cache.py                # Test caching functionality
├── cache_manager.py             # Cache management utility
├── demo_caching.py              # Caching demo
├── requirements.txt             # Python dependencies
├── requirements-minimal.txt     # Minimal dependencies
├── setup.py                     # Package setup
├── MULTI_GPU_USAGE.md           # Multi-GPU training guide
├── DATASET_OPTIONS.md           # Dataset options documentation
├── STREAMING_GUIDE.md           # Streaming and caching guide
├── VERSION_COMPARISON.md        # Version comparison guide
└── README.md                    # This file
```

## Usage Examples

### 🎯 **v0: Basic Learning Training (IMDB Dataset)**

```bash
# Basic training for learning BERT fundamentals
python mlm_pretrain_v0.py
```

**Features:**
- Uses 25K IMDB movie reviews
- Small model: 2 layers, 2 heads, 4-dim embeddings
- Fast training (~5 minutes)
- Suitable for learning basic transformer architecture
- **Educational Focus**: Understanding fundamental BERT implementation

### 🎯 **v1: Mixed Precision Learning Training (IMDB Dataset)**

```bash
# Mixed precision training for learning optimization techniques
python mlm_pretrain_v1.py
```

**Features:**
- Uses 25K IMDB movie reviews
- Small model: 2 layers, 2 heads, 4-dim embeddings
- Fast training (~5 minutes)
- Suitable for learning mixed precision training
- **Educational Focus**: Understanding optimization techniques

### 🚀 **v2: Medium Model Learning (Hugging Face Datasets)**

```bash
# Use Hugging Face large datasets (streaming mode)
python mlm_pretrain_v2.py hf

# Use Hugging Face large datasets (local download mode)
python mlm_pretrain_v2.py hf false

# Use IMDB dataset
python mlm_pretrain_v2.py imdb
```

**Features:**
- Supports multiple datasets: wikitext, wikipedia, openwebtext, etc.
- Medium model: 4 layers, 4 heads, 8-dim embeddings (HF) or 2 layers, 2 heads, 4-dim embeddings (IMDB)
- Automatic caching and streaming processing
- Training time: ~30 minutes (HF) / ~5 minutes (IMDB)
- **Educational Focus**: Learning with medium-scale models and large dataset handling

### ⚡ **v3: Multi-GPU Learning (H200 8-Card)**

#### **Method 1: Use Pre-configured Scripts**
```bash
# View available configurations
python multi_gpu_configs.py list

# Generate H200 8-GPU training script
python multi_gpu_configs.py generate h200_8gpu_standard

# Run training
./train_h200_8gpu_standard.sh
```

#### **Method 2: Use torchrun Directly**
```bash
# H200 8-GPU standard training (default 500K samples)
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

# Specify data size (5M samples)
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
    --max-samples 5M
```

#### **Method 3: Use Generic Script**
```bash
# Use default H200 8-GPU settings
./run_multi_gpu_training.sh

# Or customize parameters
./run_multi_gpu_training.sh hf 32 5 3e-5 true
```

**Features:**
- Supports multi-GPU distributed training
- Large model: 6 layers, 8 heads, 16-dim embeddings
- Mixed precision training
- Automatic GPU detection and configuration
- Training time: ~30 minutes (8GPU)
- **Educational Focus**: Learning distributed training and multi-GPU coordination

### 🚀 **v4: Advanced Multi-GPU Learning (24GB Memory Optimized)**

#### **Method 1: Use Pre-configured Script (Recommended)**
```bash
# Run 24GB memory optimized training
./train_h200_8gpu_v4.sh
```

#### **Method 2: Use torchrun Directly**
```bash
# 24GB memory optimized training (10M samples)
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    mlm_pretrain_v4.py \
    --dataset hf \
    --batch-size 96 \
    --epochs 5 \
    --lr 3e-5 \
    --streaming true \
    --max-samples 10M

# Customize data size (5M samples)
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    mlm_pretrain_v4.py \
    --dataset hf \
    --batch-size 96 \
    --epochs 5 \
    --lr 3e-5 \
    --streaming true \
    --max-samples 5M
```

#### **Method 3: Single GPU Training (Suitable for A10)**
```bash
# Single GPU 24GB optimized training
python mlm_pretrain_v4.py \
    --dataset hf \
    --batch-size 96 \
    --epochs 5 \
    --lr 3e-5 \
    --streaming true \
    --max-samples 10M
```

**Features:**
- **Specifically optimized for 24GB+ GPUs** (H200, A10, RTX 4090, etc.)
- **Large model configuration**: 8 layers, 8 heads, 256-dim embeddings
- **High memory utilization**: 83% (20GB/24GB)
- **Large batch training**: 96 per GPU (total 768)
- **Long sequence support**: 256 tokens
- **Large vocabulary**: 25K words
- **Fast training**: ~20 minutes (10M samples)
- **Mixed precision training**: bfloat16 optimization
- **Distributed training**: Supports multi-GPU
- **Automatic caching**: Intelligent data caching system
- **Educational Focus**: Learning advanced distributed training, mixed precision, and memory optimization

**Use Cases:**
- 24GB+ GPU environments (H200, A10, RTX 4090, etc.)
- High memory utilization requirements
- Large-scale model training
- Production environment deployment
- Large datasets requiring fast training

**Performance Advantages:**
- **Memory utilization**: Increased from 12% to 83%
- **Model complexity**: 150x increase (from 100K to 15M parameters)
- **Training efficiency**: Significantly improved
- **Data throughput**: 10x increase
- **Sequence length**: 2x increase (128→256)
- **Vocabulary size**: 2.5x increase (10K→25K)

### 🎯 **SFT: Supervised Fine-Tuning Learning**

For learning supervised fine-tuning techniques:

```bash
# Complete SFT implementation for downstream tasks
python sft_hfbert.py
```

**Features:**
- Complete SFT pipeline with Hugging Face BERT
- Task-specific dataset preparation
- Fine-tuning training loop
- Evaluation and inference
- **Educational Focus**: Understanding transfer learning and task-specific fine-tuning

### 3. Test Streaming Functionality
```bash
python test_streaming.py
```

### 4. Test Caching Functionality
```bash
python test_cache.py
```

### 5. Manage Cache
```bash
# View cache information
python cache_manager.py info

# Clear cache
python cache_manager.py clear

# Show disk usage
python cache_manager.py usage
```

## 🎯 **Detailed Running Guide**

### **v0: Basic Learning Training**

**Use Case**: Learning BERT fundamentals, basic implementation understanding

```bash
# Basic run
python mlm_pretrain_v1.py

# View help
python mlm_pretrain_v1.py --help
```

**Output Example:**
```
Using device: cuda
Loading IMDB dataset for MLM pre-training...
Training samples: 22500
Validation samples: 2500
Vocabulary size: 10005
Starting MLM pre-training...
Epoch 1/3: Train Loss: 9.3330 | Val Loss: 9.2017
Epoch 2/3: Train Loss: 9.1415 | Val Loss: 9.0840
Epoch 3/3: Train Loss: 9.0580 | Val Loss: 9.0374
MLM pre-training completed!
```

### **v1: Mixed Precision Learning Training**

**Use Case**: Learning mixed precision training, optimization techniques

```bash
# Basic run
python mlm_pretrain_v1.py

# View help
python mlm_pretrain_v1.py --help
```

**Output Example:**
```
Using device: cuda
Loading IMDB dataset for MLM pre-training...
Training samples: 22500
Validation samples: 2500
Vocabulary size: 10005
Starting MLM pre-training with mixed precision...
Epoch 1/3: Train Loss: 9.3330 | Val Loss: 9.2017
Epoch 2/3: Train Loss: 9.1415 | Val Loss: 9.0840
Epoch 3/3: Train Loss: 9.0580 | Val Loss: 9.0374
MLM pre-training completed!
```

### **v2: Medium Model Learning Training**

**Use Case**: Learning with medium-scale models, understanding larger datasets

```bash
# Use Hugging Face datasets (streaming mode, default 500K samples)
python mlm_pretrain_v2.py hf

# Specify data size (5M samples, streaming mode)
python mlm_pretrain_v2.py hf true 5M

# Specify data size (50M samples, local download mode)
python mlm_pretrain_v2.py hf false 50M

# Use IMDB dataset
python mlm_pretrain_v2.py imdb

# View help
python mlm_pretrain_v2.py --help
```

**Output Example:**
```
Using device: cuda
Loading dataset for MLM pre-training (choice: hf, streaming: True)...
Using larger model configuration for Hugging Face dataset...
Model configuration:
  - n_heads: 4
  - n_embed: 8
  - n_layers: 4
  - head_size: 2
  - num_epochs: 5
  - learning_rate: 3e-05
Total model parameters: 84,640
Starting MLM pre-training...
Epoch 1/5: Train Loss: 7.9531 | Val Loss: 6.6964
Epoch 2/5: Train Loss: 6.6408 | Val Loss: 6.5902
...
```

### **v3: Multi-GPU Learning Training**

**Use Case**: Learning distributed training, multi-GPU environments

#### **Step 1: View Available Configurations**
```bash
python multi_gpu_configs.py list
```

#### **Step 2: Generate Training Scripts**
```bash
# Generate H200 8-GPU standard training script
python multi_gpu_configs.py generate h200_8gpu_standard

# Generate H200 8-GPU fast training script
python multi_gpu_configs.py generate h200_8gpu_fast

# Generate H200 8-GPU quality training script
python multi_gpu_configs.py generate h200_8gpu_quality
```

#### **Step 3: Run Training**
```bash
# Run generated script
./train_h200_8gpu_standard.sh

# Or use torchrun directly (default 500K samples)
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

# Specify data size (5M samples)
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
    --max-samples 5M
```

**Output Example:**
```
Multi-GPU MLM Training Setup:
  - World Size: 8
  - Local Rank: 0
  - Device: cuda:0
  - Dataset: hf
  - Streaming: true
  - Batch Size per GPU: 32
  - Total Batch Size: 256
  - Epochs: 5
  - Learning Rate: 3e-05
Using larger model configuration for Hugging Face dataset...
Model configuration:
  - n_heads: 8
  - n_embed: 16
  - n_layers: 6
  - head_size: 2
  - num_epochs: 5
  - learning_rate: 3e-05
Total model parameters: 182,112
Starting MLM pre-training...
Epoch 1/5: Train Loss: 6.1234 | Val Loss: 5.9876
...
```

### **v4: Advanced Multi-GPU Learning Training**

**Use Case**: Learning advanced distributed training with mixed precision, 24GB+ GPU environments

#### **Step 1: Check GPU Configuration**
```bash
# Check GPU memory
nvidia-smi

# Ensure GPU memory >= 24GB
# Supported GPUs: H200, A10, RTX 4090, etc.
```

#### **Step 2: Run Training**
```bash
# Method 1: Use pre-configured script (recommended)
./train_h200_8gpu_v4.sh

# Method 2: Use torchrun directly (8GPU)
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    mlm_pretrain_v4.py \
    --dataset hf \
    --batch-size 96 \
    --epochs 5 \
    --lr 3e-5 \
    --streaming true \
    --max-samples 10M

# Method 3: Single GPU training (suitable for A10)
python mlm_pretrain_v4.py \
    --dataset hf \
    --batch-size 96 \
    --epochs 5 \
    --lr 3e-5 \
    --streaming true \
    --max-samples 10M
```

#### **Step 3: Monitor Training**
```bash
# View GPU usage
watch -n 1 nvidia-smi

# View training logs
tail -f logs/v4_training_*.log
```

**Output Example:**
```
Multi-GPU MLM Training v4 Setup (24GB Memory Optimized):
  - World Size: 8
  - Local Rank: 0
  - Device: cuda:0
  - Dataset: hf
  - Streaming: true
  - Batch Size per GPU: 96
  - Total Batch Size: 768
  - Epochs: 5
  - Learning Rate: 3e-05
  - Max Samples: 10000000
Using medium model configuration for Hugging Face dataset (optimized for 24GB GPU memory)...
Model configuration (v4 - 24GB optimized):
  - n_heads: 8
  - n_embed: 256
  - n_layers: 8
  - head_size: 32
  - num_epochs: 5
  - learning_rate: 3e-05
Total model parameters: 15,123,456
Starting MLM pre-training v4 (24GB optimized)...
Epoch 1/5: Train Loss: 5.2341 | Val Loss: 5.1234
...
```

**Feature Description:**
- **High memory utilization**: 83% (20GB/24GB)
- **Large model configuration**: 8 layers/8 heads/256-dim embeddings
- **Large batch training**: 96 per GPU (total 768)
- **Long sequence support**: 256 tokens
- **Large vocabulary**: 25K words
- **Fast training**: ~20 minutes (10M samples)
- **Mixed precision**: bfloat16 optimization
- **Distributed training**: Supports multi-GPU
- **Automatic caching**: Intelligent data caching system

## Model Configurations

The system automatically selects appropriate model configurations based on the dataset:

### 🎯 **v0 Configuration (mlm_pretrain_v0.py)**
- **Layers**: 2
- **Attention Heads**: 2
- **Embedding Dimension**: 4
- **Max Sequence Length**: 128
- **Vocabulary Size**: 10,000
- **Parameter Count**: ~41K
- **Training Time**: ~5 minutes
- **Use Case**: Learning BERT fundamentals, basic implementation understanding

### 🎯 **v1 Configuration (mlm_pretrain_v1.py)**
- **Layers**: 2
- **Attention Heads**: 2
- **Embedding Dimension**: 4
- **Max Sequence Length**: 128
- **Vocabulary Size**: 10,000
- **Parameter Count**: ~41K
- **Training Time**: ~5 minutes
- **Use Case**: Learning mixed precision training, optimization techniques

### 🚀 **v2 Configuration (mlm_pretrain_v2.py)**
- **Layers**: 4 (HF) / 2 (IMDB)
- **Attention Heads**: 4 (HF) / 2 (IMDB)
- **Embedding Dimension**: 8 (HF) / 4 (IMDB)
- **Max Sequence Length**: 128
- **Vocabulary Size**: 10,000
- **Parameter Count**: ~84K (HF) / ~41K (IMDB)
- **Training Time**: ~30 minutes (HF) / ~5 minutes (IMDB)
- **Use Case**: Learning with medium-scale models, understanding larger datasets

### ⚡ **v3 Configuration (mlm_pretrain_v3.py)**
- **Layers**: 6
- **Attention Heads**: 8
- **Embedding Dimension**: 16
- **Max Sequence Length**: 128
- **Vocabulary Size**: 10,000
- **Parameter Count**: ~182K
- **Training Time**: ~30 minutes (8GPU)
- **Use Case**: Learning distributed training, multi-GPU environments

### 🚀 **v4 Configuration (mlm_pretrain_v4.py)**
- **Layers**: 8
- **Attention Heads**: 8
- **Embedding Dimension**: 256
- **Max Sequence Length**: 256
- **Vocabulary Size**: 25,000
- **Parameter Count**: ~15M
- **Training Time**: ~20 minutes (8GPU)
- **Use Case**: Learning advanced distributed training with mixed precision, 24GB+ GPU environments
- **Memory Usage**: ~20GB/24GB (83% utilization)
- **Batch Size**: 96 per GPU (total 768)
- **Mixed Precision**: bfloat16 optimization

### 📊 **All Available Configurations**
Run `python model_config_comparison.py` to view all configurations:

| Configuration | Layers | Heads | Embedding | Parameters | Educational Focus |
|---------------|--------|-------|-----------|------------|-------------------|
| **v0: Basic** | 2 | 2 | 4 | ~41K | BERT fundamentals, basic implementation |
| **v1: Mixed Precision** | 2 | 2 | 4 | ~41K | Mixed precision training, optimization |
| **v2: Medium Model** | 4 | 4 | 8 | ~84K | Medium-scale models, large datasets |
| **v3: Multi-GPU** | 6 | 8 | 16 | ~182K | Distributed training, multi-GPU coordination |
| **v4: Advanced Multi-GPU** | 8 | 8 | 256 | ~15M | Advanced distributed training, mixed precision |

## Dataset Options

### IMDB Dataset
- **Size**: ~25K samples
- **Domain**: Movie reviews
- **Pros**: Fast, focused domain
- **Cons**: Limited diversity

### Hugging Face Datasets
- **wikitext-103-raw-v1**: Wikipedia articles (1.8M tokens)
- **wikipedia**: Wikipedia articles (20220301.en)
- **openwebtext**: Web text (8M documents)
- **c4**: Common Crawl data (English)
- **pile-cc**: Common Crawl data (large)

## Caching System

The project includes an intelligent caching system:

- **Streaming Mode**: Downloads data on-the-fly and caches processed results
- **Cache Location**: `.dataset_cache/` directory
- **Cache Keys**: Based on dataset name, parameters, and configuration
- **Benefits**: 
  - First run: Downloads and processes data
  - Subsequent runs: Instant loading from cache
  - Disk usage: ~100-500MB vs ~1-10GB for local download

## Training Output

### Model Files
- `mlm_model.pth`: Full MLM model weights
- `microbert_model.pth`: Base MicroBERT model weights
- `tokenizer_vocab.json`: Vocabulary mapping
- `mlm_training_history.json`: Training metrics

### Visualization
- `training_history.png`: Loss curves over epochs

### Example Output
```
=== Testing MLM Model ===
1. Original: this movie is [MASK] fantastic
   [MASK] at position 3:
     that: logit=1.642, prob=0.216190
     ok,: logit=1.565, prob=0.200183
     disney: logit=1.559, prob=0.198995
     episode: logit=1.526, prob=0.192561
     can't: logit=1.524, prob=0.192071
```

## Performance Tips

### For Limited Resources
- Use IMDB dataset (`mlm_pretrain_v1.py`)
- Use streaming mode for HF datasets
- Reduce `max_samples` parameter

### For Better Results
- Use larger datasets (HF datasets)
- Increase training epochs
- Use local download mode for faster training

### For Development
- Use smaller `max_samples` for quick testing
- Monitor cache usage with `cache_manager.py`

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size per GPU
   - Use smaller model configuration
   - Use CPU training
   - Enable gradient checkpointing

2. **Dataset Loading Failures**
   - Check internet connection
   - Try different dataset
   - Use IMDB fallback
   - Check disk space for caching

3. **Cache Issues**
   - Clear cache: `python cache_manager.py clear`
   - Check disk space
   - Use different cache directory

4. **Multi-GPU Issues (v3/v4)**
   - Check GPU availability: `nvidia-smi`
   - Ensure NCCL is installed: `python -c "import torch; print(torch.cuda.nccl.version())"`
   - Check port conflicts: change `--master_port=12356`
   - Verify PyTorch installation: `python -c "import torch; print(torch.cuda.device_count())"`

5. **v4 Memory Issues**
   - Ensure GPU memory >= 24GB for v4
   - Reduce batch size if memory insufficient: `--batch-size 64`
   - Use smaller model: switch to v3 if needed
   - Check memory usage: `nvidia-smi`

6. **Distributed Training Issues**
   - Check if all GPUs are visible
   - Ensure proper environment variables are set
   - Try single GPU first: `python mlm_pretrain_v3.py --dataset imdb` or `python mlm_pretrain_v4.py --dataset imdb`

### Getting Help

- Check the training logs for error messages
- Verify all dependencies are installed
- Ensure sufficient disk space for caching
- For multi-GPU issues, check `MULTI_GPU_USAGE.md`
- Test single GPU functionality before multi-GPU training

## 📊 **Version Comparison Summary**

| Feature | v0 (Basic) | v1 (Mixed Precision) | v2 (Medium Model) | v3 (Multi-GPU) | v4 (Advanced Multi-GPU) |
|---------|-------------|----------------------|-------------------|-----------------|--------------------------|
| **Educational Focus** | BERT fundamentals | Mixed precision training | Medium-scale models | Distributed training | Advanced distributed training |
| **Dataset** | IMDB | IMDB | IMDB + HF | IMDB + HF | IMDB + HF |
| **Model Size** | Small (41K params) | Small (41K params) | Medium (84K params) | Large (182K params) | Extra Large (15M params) |
| **Training Time** | ~5 minutes | ~5 minutes | ~30 minutes | ~30 minutes (8GPU) | ~20 minutes (8GPU) |
| **GPU Requirements** | 1 | 1 | 1 | Multiple | Multiple |
| **Memory Requirements** | Low | Low | Medium | High | Very High (24GB+) |
| **Streaming** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Caching System** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Mixed Precision** | ❌ | ✅ | ❌ | ✅ | ✅ |
| **Distributed Training** | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Memory Utilization** | Low | Low | Medium | Medium | High (83%) |
| **Batch Size** | 32 | 32 | 32 | 32 per GPU | 96 per GPU |
| **Sequence Length** | 128 | 128 | 128 | 128 | 256 |
| **Vocabulary Size** | 10K | 10K | 10K | 10K | 25K |

## 🎯 **Selection Recommendations**

- **Beginners/Learning Fundamentals**: Use `v0` - Basic BERT implementation, low resource requirements
- **Learning Optimization**: Use `v1` - Mixed precision training, optimization techniques
- **Medium-scale Learning**: Use `v2` - Balanced performance, medium models, large datasets
- **Learning Distributed Training**: Use `v3` - Multi-GPU environments, distributed training concepts
- **Advanced Distributed Learning**: Use `v4` - Advanced multi-GPU, mixed precision, high memory utilization
- **Learning Fine-tuning**: Use `sft_hfbert.py` - Complete SFT pipeline for downstream tasks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the BERT architecture from "Attention Is All You Need"
- Uses Hugging Face datasets and transformers libraries
- Designed for educational purposes to help learners understand transformer architectures
- Inspired by educational implementations of transformer models 