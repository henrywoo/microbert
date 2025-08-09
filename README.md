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

### Basic MLM Training (IMDB Dataset)
```bash
python mlm_pretrain_v1.py
```

### Advanced MLM Training (Hugging Face Datasets)
```bash
# Use streaming mode (recommended for limited disk space)
python mlm_pretrain.py hf

# Use local download mode (faster, uses more disk space)
python mlm_pretrain.py hf false
```

## Project Structure

```
microbert/
├── microbert/                 # Core package
│   ├── __init__.py
│   ├── model.py              # BERT model implementation
│   ├── tokenizer.py          # Word-level tokenizer
│   └── utils.py              # Utility functions
├── mlm_pretrain_v1.py        # IMDB-only MLM training
├── mlm_pretrain.py           # Full MLM training with dataset options
├── test_streaming.py         # Test streaming functionality
├── test_cache.py             # Test caching functionality
├── cache_manager.py          # Cache management utility
├── demo_caching.py           # Caching demo
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

## Usage Examples

### 1. Train with IMDB Dataset
```bash
python mlm_pretrain_v1.py
```
- Uses 25K IMDB movie reviews
- Small model: 2 layers, 2 heads, 4-dim embeddings
- Fast training (~5 minutes on GPU)

### 2. Train with Hugging Face Datasets
```bash
# Streaming mode (saves disk space)
python mlm_pretrain.py hf

# Local download mode (faster)
python mlm_pretrain.py hf false
```
- Supports multiple datasets: wikitext, wikipedia, openwebtext, etc.
- Larger model: 2 layers, 2 heads, 4-dim embeddings
- Automatic caching for reuse

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

## Model Configurations

### Default Configuration (mlm_pretrain_v1.py)
- **Layers**: 2
- **Heads**: 2
- **Embedding Dimension**: 4
- **Max Sequence Length**: 128
- **Vocabulary Size**: 10,000
- **Parameters**: ~41K

### Advanced Configuration (mlm_pretrain.py)
- **Layers**: 2
- **Heads**: 2
- **Embedding Dimension**: 4
- **Max Sequence Length**: 128
- **Vocabulary Size**: 10,000
- **Parameters**: ~41K

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
   - Reduce batch size
   - Use smaller model configuration
   - Use CPU training

2. **Dataset Loading Failures**
   - Check internet connection
   - Try different dataset
   - Use IMDB fallback

3. **Cache Issues**
   - Clear cache: `python cache_manager.py clear`
   - Check disk space
   - Use different cache directory

### Getting Help

- Check the training logs for error messages
- Verify all dependencies are installed
- Ensure sufficient disk space for caching

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