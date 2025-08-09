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

### 2. 缓存预热
首次运行时，让数据完全下载并缓存：

```bash
# 第一次运行（会下载并缓存）
./train_standard.sh

# 后续运行（使用缓存）
./train_standard.sh
```

### 3. 监控缓存命中
观察输出中的缓存信息：
```
✓ Loaded 500,000 samples from cache for c4
✓ Used 500,000 samples from cache, total so far: 500,000
✓ Reached target sample count (500,000) from cache, stopping dataset loading
```

## 故障排除

### 1. 缓存未命中
如果看到"Cache MISS"，检查：
- 参数是否与之前运行一致
- 缓存文件是否存在
- 缓存文件是否损坏

### 2. 重复下载
如果仍然重复下载：
- 检查 `max_samples` 参数
- 确认 `seed` 值一致
- 验证数据集名称和参数

### 3. 清理缓存
如果需要重新开始：
```bash
# 清理所有缓存
python cache_manager.py clear --no-confirm

# 清理特定缓存文件
python cache_manager.py clear --file 52719dbb61cb7957.json
```

## 性能优化

### 1. 缓存大小控制
- 每个数据集最多缓存100万样本
- 避免缓存文件过大
- 定期清理不需要的缓存

### 2. 流式模式
- 使用 `--streaming true` 避免完整下载
- 数据按需加载，节省磁盘空间
- 支持大规模数据集

### 3. 内存管理
- 根据GPU内存调整批次大小
- 使用适当的序列长度
- 监控GPU内存使用

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
