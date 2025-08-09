# MicroBERT 缓存使用指南

## 问题描述

你遇到的"重复处理数据"问题主要有以下几个原因：

### 1. 缓存键不匹配
每次运行时如果使用不同的参数，会生成不同的缓存键：
- `max_samples` 不同
- `min_words` 不同  
- `seed` 不同
- 数据集参数不同

### 2. 多数据集策略
即使从缓存加载了足够的数据，代码仍会尝试其他数据集，导致重复的"processed xxx samples"消息。

## 解决方案

### 1. 使用一致的参数
确保每次运行时使用相同的参数：

```bash
# 推荐：使用固定的参数组合
python mlm_pretrain_v4.py \
    --dataset hf \
    --max-samples 10M \
    --min-words 5 \
    --seed 42 \
    --streaming true
```

### 2. 检查缓存状态
使用缓存管理器查看当前缓存：

```bash
# 查看缓存信息
python cache_manager.py info

# 查看特定配置的缓存键
python cache_manager.py key --ds-name c4 --ds-kwargs '{"name": "en"}' --max-samples 10000000
```

### 3. 缓存键生成规则
缓存键基于以下参数生成：
```python
config_str = f"{ds_name}_{str(ds_kwargs)}_{max_samples}_{min_words}_{seed}"
cache_key = hashlib.md5(config_str.encode()).hexdigest()[:16]
```

## 最佳实践

### 1. 参数标准化
创建配置文件或脚本，确保参数一致：

```bash
# 创建标准训练脚本
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
