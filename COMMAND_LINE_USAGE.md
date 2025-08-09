# MLM Pre-training 命令行使用指南

## 基本用法

### 使用默认参数运行
```bash
python mlm_pretrain_v1.py
```

### 查看所有可用参数
```bash
python mlm_pretrain_v1.py --help
```

## 模型架构参数

### 调整模型大小
```bash
# 超小模型 (1层, 1头, 8维)
python mlm_pretrain_v1.py --n_layers 1 --n_heads 1 --n_embed 8

# 小模型 (2层, 2头, 16维) - 默认配置
python mlm_pretrain_v1.py --n_layers 2 --n_heads 2 --n_embed 16

# 中等模型 (4层, 4头, 32维)
python mlm_pretrain_v1.py --n_layers 4 --n_heads 4 --n_embed 32

# 大模型 (6层, 8头, 64维)
python mlm_pretrain_v1.py --n_layers 6 --n_heads 8 --n_embed 64
```

### 调整序列长度
```bash
# 短序列 (64 tokens)
python mlm_pretrain_v1.py --max_seq_len 64

# 长序列 (256 tokens)
python mlm_pretrain_v1.py --max_seq_len 256
```

## 训练参数

### 调整学习率
```bash
# 高学习率 (快速收敛)
python mlm_pretrain_v1.py --learning_rate 2e-4

# 低学习率 (稳定训练)
python mlm_pretrain_v1.py --learning_rate 5e-5
```

### 调整批次大小
```bash
# 小批次 (适合小GPU)
python mlm_pretrain_v1.py --batch_size 16

# 大批次 (适合大GPU)
python mlm_pretrain_v1.py --batch_size 64
```

### 调整训练轮数
```bash
# 快速测试
python mlm_pretrain_v1.py --epochs 3

# 充分训练
python mlm_pretrain_v1.py --epochs 20
```

### 调整早停耐心值
```bash
# 快速早停
python mlm_pretrain_v1.py --patience 2

# 耐心等待
python mlm_pretrain_v1.py --patience 5
```

## 数据加载参数

### 调整工作进程数
```bash
# 单进程 (调试时)
python mlm_pretrain_v1.py --num_workers 0

# 多进程 (生产环境)
python mlm_pretrain_v1.py --num_workers 4
```

## 其他选项

### 强制重新开始训练
```bash
python mlm_pretrain_v1.py --force-fresh
```

### 自定义保存目录
```bash
python mlm_pretrain_v1.py --save-dir ./my_mlm_model
```

## 常用配置组合

### 快速测试配置
```bash
python mlm_pretrain_v1.py \
    --n_layers 1 \
    --n_heads 1 \
    --n_embed 8 \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 1e-4
```

### 生产训练配置
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

### 内存受限配置
```bash
python mlm_pretrain_v1.py \
    --n_layers 2 \
    --n_heads 2 \
    --n_embed 16 \
    --batch_size 16 \
    --max_seq_len 64
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--n_layers` | int | 2 | Transformer层数 |
| `--n_heads` | int | 2 | 注意力头数 |
| `--n_embed` | int | 16 | Embedding维度 |
| `--max_seq_len` | int | 128 | 最大序列长度 |
| `--batch_size` | int | 32 | 训练批次大小 |
| `--learning_rate` | float | 1e-4 | 学习率 |
| `--epochs` | int | 10 | 训练轮数 |
| `--patience` | int | 3 | 早停耐心值 |
| `--num_workers` | int | 2 | 数据加载工作进程数 |
| `--save-dir` | str | .mlm_v1 | 模型保存目录 |
| `--force-fresh` | flag | False | 强制重新开始训练 |

## 注意事项

1. **n_embed必须能被n_heads整除**
2. **batch_size根据GPU内存调整**
3. **num_workers建议设置为CPU核心数的1/2到1/4**
4. **使用--force-fresh会删除现有检查点**

## 示例：渐进式训练

```bash
# 第一阶段：小模型快速验证
python mlm_pretrain_v1.py --n_layers 1 --n_heads 1 --n_embed 8 --epochs 3

# 第二阶段：中等模型训练
python mlm_pretrain_v1.py --n_layers 2 --n_heads 2 --n_embed 16 --epochs 10

# 第三阶段：大模型精调
python mlm_pretrain_v1.py --n_layers 4 --n_heads 4 --n_embed 32 --epochs 20
```

现在你可以灵活地通过命令行参数调整所有模型配置，不再需要修改代码！
