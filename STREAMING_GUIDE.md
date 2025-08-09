# Streaming Guide for MicroBERT MLM Training

## Overview

The MicroBERT MLM training now supports **streaming mode** for Hugging Face datasets, which significantly reduces disk usage from gigabytes to megabytes.

## What is Streaming?

- **Streaming Mode**: Data is downloaded and processed on-the-fly, then cached locally for reuse
- **Local Download Mode**: Data is downloaded to local disk for faster subsequent access
- **Caching**: Processed datasets are automatically saved to `.dataset_cache/` for instant reuse

## Usage

### Command Line Options

```bash
# Use IMDB dataset (always fast, small)
python mlm_pretrain.py imdb

# Use Hugging Face dataset with streaming (default, saves disk space)
python mlm_pretrain.py hf

# Use Hugging Face dataset with local download (faster, uses more disk)
python mlm_pretrain.py hf false

# Explicit streaming mode
python mlm_pretrain.py hf true
```

### Programmatic Usage

```python
from mlm_pretrain import main

# Streaming mode (default)
main(dataset_choice='hf', streaming=True)

# Local download mode
main(dataset_choice='hf', streaming=False)
```

## Storage Comparison

| Mode | Disk Usage | Speed | Use Case |
|------|------------|-------|----------|
| **Streaming + Cache** | ~100-500MB | Fast after first run | Limited disk space |
| **Local Download** | ~1-10GB | Fastest | Sufficient disk space |

**Note**: Streaming mode now automatically caches processed data, so subsequent runs are much faster.

## Supported Datasets

The system tries these datasets in order:

1. **wikitext-103-raw-v1** - Wikipedia articles (1.8M tokens)
2. **wikipedia** - Wikipedia articles (20220301.en)
3. **openwebtext** - Web text (8M documents)
4. **c4** - Common Crawl data (English)
5. **pile-cc** - Common Crawl data (large)

## Technical Details

### Streaming Implementation

```python
# Streaming mode
dataset = load_dataset(dataset_name, split='train', streaming=True)

# Local download mode  
dataset = load_dataset(dataset_name, split='train', streaming=False)
```

### Memory Management

- **Streaming**: Only keeps current batch in memory during processing
- **Local**: Keeps entire dataset in memory after download
- **Caching**: Processed data is saved to disk for instant reuse

### Error Handling

- Falls back to IMDB if all Hugging Face datasets fail
- Graceful handling of unsupported shuffle operations
- Clear error messages for debugging

## Testing

Run the test scripts to verify functionality:

```bash
# Test streaming functionality
python test_streaming.py

# Test caching functionality
python test_cache.py

# Show cache information
python test_cache.py
```

## Troubleshooting

### Common Issues

1. **Network Timeout**: Increase timeout or try different dataset
2. **Memory Issues**: Use streaming mode or reduce `max_samples`
3. **Dataset Unavailable**: System will automatically fall back to IMDB

### Performance Tips

- Use streaming mode for large datasets if disk space is limited
- Use local download mode for faster training if disk space is available
- Adjust `max_samples` parameter to control dataset size

## Configuration

Key parameters in `load_hf_dataset()`:

- `max_samples`: Maximum number of samples to load (default: 500,000)
- `min_words`: Minimum words per sample (default: 5)
- `seed`: Random seed for reproducibility (default: 42)
- `streaming`: Enable/disable streaming mode (default: True)
- `cache_dir`: Directory to cache processed datasets (default: ".dataset_cache")

## Caching

### How it works:
1. **First run**: Downloads and processes data, saves to cache
2. **Subsequent runs**: Loads directly from cache (instant)
3. **Cache key**: Based on dataset name, parameters, and configuration
4. **Cache location**: `.dataset_cache/` directory

### Cache management:
```bash
# View cache information
python test_cache.py

# Clear cache manually
rm -rf .dataset_cache/

# Use custom cache directory
load_hf_dataset(cache_dir="my_cache")
```
