# MicroBERT Cache Usage Guide

## Problem Description

The "repeated data processing" issue you encountered has the following main reasons:

### 1. Cache Key Mismatch
Each time you run with different parameters, different cache keys will be generated:
- `max_samples` different
- `min_words` different  
- `seed` different
- Dataset parameters different

### 2. Multi-Dataset Strategy
Even after loading sufficient data from cache, the code will still try other datasets, resulting in repeated "processed xxx samples" messages.

## Solutions

### 1. Use Consistent Parameters
Ensure to use the same parameters each time you run:

```bash
# Recommended: Use fixed parameter combinations
python mlm_pretrain_v4.py \
    --dataset hf \
    --max-samples 10M \
    --min-words 5 \
    --seed 42 \
    --streaming true
```

### 2. Check Cache Status
Use the cache manager to view current cache:

```bash
# View cache information
python cache_manager.py info

# View cache key for specific configuration
python cache_manager.py key --ds-name c4 --ds-kwargs '{"name": "en"}' --max-samples 10000000
```

### 3. Cache Key Generation Rules
Cache keys are generated based on the following parameters:
```python
config_str = f"{ds_name}_{str(ds_kwargs)}_{max_samples}_{min_words}_{seed}"
cache_key = hashlib.md5(config_str.encode()).hexdigest()[:16]
```

## Best Practices

### 1. Parameter Standardization
Create configuration files or scripts to ensure parameter consistency:

```bash
# Create standard training script
cat > train_standard.sh << 'EOF'
#!/bin/bash
python mlm_pretrain_v4.py \
    --dataset hf \
    --max-samples 10M \
    --min-words 5 \
    --seed 42 \
    --streaming true \
    --batch-size 96 \
    --epochs 5 \
    --lr 3e-05
EOF

chmod +x train_standard.sh
```

### 2. Cache Warming
On first run, let the data download and cache completely:

```bash
# First run (will download and cache)
./train_standard.sh

# Subsequent runs (use cache)
./train_standard.sh
```

### 3. Monitor Cache Hits
Observe cache information in the output:
```
✓ Loaded 500,000 samples from cache for c4
✓ Used 500,000 samples from cache, total so far: 500,000
✓ Reached target sample count (500,000) from cache, stopping dataset loading
```

## Troubleshooting

### 1. Cache Miss
If you see "Cache MISS", check:
- Whether parameters are consistent with previous runs
- Whether cache files exist
- Whether cache files are corrupted

### 2. Repeated Downloads
If downloads still repeat:
- Check the `max_samples` parameter
- Confirm `seed` value is consistent
- Verify dataset name and parameters

### 3. Clear Cache
If you need to start over:
```bash
# Clear all cache
python cache_manager.py clear --no-confirm

# Clear specific cache file
python cache_manager.py clear --file 52719dbb61cb7957.json
```

## Performance Optimization

### 1. Cache Size Control
- Cache up to 1 million samples per dataset
- Avoid overly large cache files
- Regularly clean up unnecessary cache

### 2. Streaming Mode
- Use `--streaming true` to avoid complete downloads
- Load data on-demand, saving disk space
- Support large-scale datasets

### 3. Memory Management
- Adjust batch size based on GPU memory
- Use appropriate sequence lengths
- Monitor GPU memory usage

## 示例工作流

```bash
# 1. 检查当前缓存
python cache_manager.py info

# 2. 运行标准训练（首次会下载）
./train_standard.sh

# 3. 再次运行（应该使用缓存）
./train_standard.sh

# 4. 验证缓存命中
python cache_manager.py key --ds-name c4 --ds-kwargs '{"name": "en"}' --max-samples 10000000
```

通过遵循这些指南，你应该能够避免重复下载数据，充分利用缓存功能。
