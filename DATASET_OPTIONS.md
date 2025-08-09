# MLM Training Dataset Options

## Quick Start

### Option 1: Fast Training (IMDB Dataset)
```bash
python mlm_pretrain.py imdb
# or simply
python mlm_pretrain.py
```

**Pros:**
- ✅ Fast training (~5-10 minutes)
- ✅ Small dataset (~50K samples)
- ✅ Good for testing and development
- ✅ No internet required (uses local files)

**Cons:**
- ❌ Limited vocabulary (movie reviews only)
- ❌ Smaller model capacity
- ❌ Less diverse language patterns

### Option 2: Large Dataset Training (Hugging Face)
```bash
python mlm_pretrain.py hf
# or use the dedicated script
python train_large_mlm.py
```

**Pros:**
- ✅ Much larger dataset (~100K+ samples)
- ✅ Diverse vocabulary and language patterns
- ✅ Better model performance
- ✅ More realistic language understanding

**Cons:**
- ❌ Longer training time (30+ minutes)
- ❌ Requires internet connection
- ❌ More memory usage

## Dataset Details

### IMDB Dataset
- **Size**: ~50K movie reviews
- **Vocabulary**: ~10K words (limited to movie domain)
- **Training Time**: 5-10 minutes
- **Best for**: Quick testing, development, learning

### Hugging Face Datasets (in order of preference)
1. **wikitext-103-raw-v1**: Wikipedia articles (1.8M tokens)
2. **bookcorpus**: Books (74M sentences)
3. **wikipedia**: General Wikipedia articles
4. **openwebtext**: Web text (8M documents)
5. **pile-cc**: Common Crawl data

## Usage Examples

```bash
# Quick test with IMDB
python mlm_pretrain.py imdb

# Full training with large dataset
python mlm_pretrain.py hf

# See help
python mlm_pretrain.py

# Use dedicated large dataset script
python train_large_mlm.py
```

## Recommendations

1. **For Learning/Testing**: Use `imdb` dataset
2. **For Production**: Use `hf` dataset
3. **For Development**: Start with `imdb`, then switch to `hf` for final training

## Troubleshooting

- If Hugging Face datasets fail to load, the script will automatically fall back to IMDB
- Make sure you have internet connection for Hugging Face datasets
- Large datasets require more RAM and GPU memory
