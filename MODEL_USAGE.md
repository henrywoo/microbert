# MicroBERT 模型使用指南

## 模型保存

训练完成后，模型会自动保存到 `~/.microbert_model/` 目录下，包含以下文件：

- `microbert_classification.pth` - 模型权重文件
- `tokenizer_vocab.json` - 分词器词汇表
- `training_history.json` - 训练历史记录
- `model_config.json` - 模型配置信息

## 模型加载和使用

### 1. 使用预测脚本

运行预制的预测脚本：

```bash
python predict.py
```

这将加载训练好的模型并对示例文本进行情感分析。

### 2. 在代码中使用

```python
from microbert.utils import load_model, predict_sentiment

# 加载模型
model, tokenizer, config = load_model('~/.microbert_model')

# 进行预测
text = "This movie is amazing!"
prediction, confidence = predict_sentiment(model, tokenizer, text)

sentiment = "正面" if prediction == 1 else "负面"
print(f"预测结果: {sentiment}, 置信度: {confidence:.3f}")
```

### 3. 手动保存模型

如果需要手动保存模型，可以使用：

```python
from microbert.utils import save_model

config = {
    'vocab_size': len(tokenizer.vocab),
    'n_layers': 1,
    'n_heads': 1,
    'max_seq_len': 128,
    'n_classes': 2,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'num_epochs': 200
}

save_model(model, tokenizer, history, config, '~/my_model')
```

## 文件说明

### 模型权重文件 (.pth)
包含训练好的神经网络权重参数。

### 词汇表文件 (.json)
包含分词器使用的词汇表，用于文本预处理。

### 训练历史文件 (.json)
包含训练过程中的损失、准确率、F1分数等指标。

### 配置文件 (.json)
包含模型架构和训练参数，用于重建模型结构。

## 注意事项

1. 确保在加载模型时使用相同的模型架构参数
2. 模型文件较大，请确保有足够的存储空间
3. 建议定期备份训练好的模型文件
4. 在不同设备间迁移时，注意设备兼容性（CPU/GPU） 