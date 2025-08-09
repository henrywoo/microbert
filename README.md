# MicroBERT MLM Pre-training

A lightweight BERT implementation for Masked Language Modeling (MLM) pre-training with support for multiple datasets and streaming capabilities.

## Features

- 🚀 **Lightweight BERT**: Small, efficient BERT implementation
- 📊 **Multiple Datasets**: Support for IMDB and Hugging Face datasets
- 💾 **Streaming Support**: Memory-efficient data loading with local caching
- 🎯 **MLM Pre-training**: Full Masked Language Modeling implementation
- 📈 **Training Visualization**: Built-in plotting and monitoring
- 🔧 **Flexible Configuration**: Easy model parameter tuning

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

### 🎯 **选择适合你的训练脚本**

根据你的需求和硬件配置，选择以下三个版本之一：

#### **v1: 快速入门 (IMDB数据集)**
```bash
# 使用IMDB数据集进行快速训练
python mlm_pretrain_v1.py
```
- **适用场景**: 学习、测试、快速验证
- **数据集**: IMDB电影评论 (~25K样本)
- **模型**: 小模型 (2层, 2头, 4维嵌入)
- **训练时间**: ~5分钟
- **内存需求**: 低

#### **v2: 标准训练 (Hugging Face数据集)**
```bash
# 使用Hugging Face大数据集
python mlm_pretrain_v2.py hf

# 或使用IMDB数据集
python mlm_pretrain_v2.py imdb
```
- **适用场景**: 标准训练、中等规模数据集
- **数据集**: Hugging Face数据集 (~500K-1M样本) 或 IMDB
- **模型**: 中等模型 (4层, 4头, 8维嵌入) 或小模型
- **训练时间**: ~30分钟 (HF) / ~5分钟 (IMDB)
- **内存需求**: 中等

#### **v3: 多GPU训练 (H200 8卡)**
```bash
# 使用预配置脚本 (推荐)
python multi_gpu_configs.py generate h200_8gpu_standard
./train_h200_8gpu_standard.sh

# 或直接使用torchrun
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
    --streaming true
```
- **适用场景**: 大规模训练、多GPU环境
- **数据集**: Hugging Face大数据集
- **模型**: 大模型 (6层, 8头, 16维嵌入)
- **训练时间**: ~30分钟 (8GPU)
- **内存需求**: 高 (需要多GPU)

## Project Structure

```
microbert/
├── microbert/                    # Core package
│   ├── __init__.py
│   ├── model.py                 # BERT model implementation
│   ├── tokenizer.py             # Word-level tokenizer
│   └── utils.py                 # Utility functions
├── mlm_pretrain_v1.py           # v1: IMDB-only MLM training (快速入门)
├── mlm_pretrain_v2.py           # v2: Standard MLM training with dataset options (标准训练)
├── mlm_pretrain_v3.py           # v3: Multi-GPU MLM training (多GPU训练)
├── multi_gpu_configs.py         # Multi-GPU training configurations
├── run_multi_gpu_training.sh    # Multi-GPU training launcher
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
└── README.md                    # This file
```

## Usage Examples

### 🎯 **v1: 快速入门训练 (IMDB数据集)**

```bash
# 使用IMDB数据集进行快速训练
python mlm_pretrain_v1.py
```

**特点:**
- 使用25K IMDB电影评论
- 小模型: 2层, 2头, 4维嵌入
- 快速训练 (~5分钟)
- 适合学习和测试

### 🚀 **v2: 标准训练 (Hugging Face数据集)**

```bash
# 使用Hugging Face大数据集 (流式模式)
python mlm_pretrain_v2.py hf

# 使用Hugging Face大数据集 (本地下载模式)
python mlm_pretrain_v2.py hf false

# 使用IMDB数据集
python mlm_pretrain_v2.py imdb
```

**特点:**
- 支持多种数据集: wikitext, wikipedia, openwebtext等
- 中等模型: 4层, 4头, 8维嵌入 (HF) 或 2层, 2头, 4维嵌入 (IMDB)
- 自动缓存和流式处理
- 训练时间: ~30分钟 (HF) / ~5分钟 (IMDB)

### ⚡ **v3: 多GPU训练 (H200 8卡)**

#### **方法1: 使用预配置脚本**
```bash
# 查看可用配置
python multi_gpu_configs.py list

# 生成H200 8-GPU训练脚本
python multi_gpu_configs.py generate h200_8gpu_standard

# 运行训练
./train_h200_8gpu_standard.sh
```

#### **方法2: 直接使用torchrun**
```bash
# H200 8-GPU标准训练
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
    --streaming true
```

#### **方法3: 使用通用脚本**
```bash
# 使用默认H200 8-GPU设置
./run_multi_gpu_training.sh

# 或自定义参数
./run_multi_gpu_training.sh hf 32 5 3e-5 true
```

**特点:**
- 支持多GPU分布式训练
- 大模型: 6层, 8头, 16维嵌入
- 混合精度训练
- 自动GPU检测和配置
- 训练时间: ~30分钟 (8GPU)

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

## 🎯 **详细运行指南**

### **v1: 快速入门训练**

**适用场景**: 学习、测试、快速验证

```bash
# 基本运行
python mlm_pretrain_v1.py

# 查看帮助
python mlm_pretrain_v1.py --help
```

**输出示例:**
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

### **v2: 标准训练**

**适用场景**: 标准训练、中等规模数据集

```bash
# 使用Hugging Face数据集 (流式模式)
python mlm_pretrain_v2.py hf

# 使用Hugging Face数据集 (本地下载模式)
python mlm_pretrain_v2.py hf false

# 使用IMDB数据集
python mlm_pretrain_v2.py imdb

# 查看帮助
python mlm_pretrain_v2.py --help
```

**输出示例:**
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

### **v3: 多GPU训练**

**适用场景**: 大规模训练、多GPU环境

#### **步骤1: 查看可用配置**
```bash
python multi_gpu_configs.py list
```

#### **步骤2: 生成训练脚本**
```bash
# 生成H200 8-GPU标准训练脚本
python multi_gpu_configs.py generate h200_8gpu_standard

# 生成H200 8-GPU快速训练脚本
python multi_gpu_configs.py generate h200_8gpu_fast

# 生成H200 8-GPU高质量训练脚本
python multi_gpu_configs.py generate h200_8gpu_quality
```

#### **步骤3: 运行训练**
```bash
# 运行生成的脚本
./train_h200_8gpu_standard.sh

# 或直接使用torchrun
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
    --streaming true
```

**输出示例:**
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

## Model Configurations

系统根据数据集自动选择合适的模型配置：

### 🎯 **v1配置 (mlm_pretrain_v1.py)**
- **层数**: 2
- **注意力头**: 2
- **嵌入维度**: 4
- **最大序列长度**: 128
- **词汇表大小**: 10,000
- **参数数量**: ~41K
- **训练时间**: ~5分钟
- **适用场景**: 学习、测试、小数据集

### 🚀 **v2配置 (mlm_pretrain_v2.py)**
- **层数**: 4 (HF) / 2 (IMDB)
- **注意力头**: 4 (HF) / 2 (IMDB)
- **嵌入维度**: 8 (HF) / 4 (IMDB)
- **最大序列长度**: 128
- **词汇表大小**: 10,000
- **参数数量**: ~84K (HF) / ~41K (IMDB)
- **训练时间**: ~30分钟 (HF) / ~5分钟 (IMDB)
- **适用场景**: 大数据集、更好性能

### ⚡ **v3配置 (mlm_pretrain_v3.py)**
- **层数**: 6
- **注意力头**: 8
- **嵌入维度**: 16
- **最大序列长度**: 128
- **词汇表大小**: 10,000
- **参数数量**: ~182K
- **训练时间**: ~30分钟 (8GPU)
- **适用场景**: 大规模训练、多GPU环境

### 📊 **所有可用配置**
运行 `python model_config_comparison.py` 查看所有配置：

| 配置 | 层数 | 头数 | 嵌入 | 参数 | 适用场景 |
|------|------|------|------|------|----------|
| **IMDB Small** | 2 | 2 | 4 | ~41K | IMDB数据集 |
| **HF Medium** | 4 | 4 | 8 | ~84K | HF数据集 |
| **HF Large** | 6 | 8 | 16 | ~182K | 大数据集 |
| **HF Extra Large** | 8 | 12 | 24 | ~301K | 超大数据集 |

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

4. **Multi-GPU Issues (v3)**
   - Check GPU availability: `nvidia-smi`
   - Ensure NCCL is installed: `python -c "import torch; print(torch.cuda.nccl.version())"`
   - Check port conflicts: change `--master_port=12356`
   - Verify PyTorch installation: `python -c "import torch; print(torch.cuda.device_count())"`

5. **Distributed Training Issues**
   - Check if all GPUs are visible
   - Ensure proper environment variables are set
   - Try single GPU first: `python mlm_pretrain_v3.py --dataset imdb`

### Getting Help

- Check the training logs for error messages
- Verify all dependencies are installed
- Ensure sufficient disk space for caching
- For multi-GPU issues, check `MULTI_GPU_USAGE.md`
- Test single GPU functionality before multi-GPU training

## 📊 **版本对比总结**

| 特性 | v1 (快速入门) | v2 (标准训练) | v3 (多GPU训练) |
|------|---------------|---------------|----------------|
| **适用场景** | 学习、测试 | 标准训练 | 大规模训练 |
| **数据集** | IMDB | IMDB + HF | IMDB + HF |
| **模型大小** | 小 (41K参数) | 中 (84K参数) | 大 (182K参数) |
| **训练时间** | ~5分钟 | ~30分钟 | ~30分钟 (8GPU) |
| **GPU需求** | 1个 | 1个 | 多个 |
| **内存需求** | 低 | 中等 | 高 |
| **流式处理** | ❌ | ✅ | ✅ |
| **缓存系统** | ❌ | ✅ | ✅ |
| **混合精度** | ❌ | ❌ | ✅ |
| **分布式训练** | ❌ | ❌ | ✅ |

## 🎯 **选择建议**

- **初学者/测试**: 使用 `v1` - 快速、简单、资源需求低
- **标准训练**: 使用 `v2` - 平衡性能和资源需求
- **大规模训练**: 使用 `v3` - 充分利用多GPU资源

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
- Inspired by educational implementations of transformer models 