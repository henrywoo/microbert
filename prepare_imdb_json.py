from datasets import load_dataset
import json
from microbert.tokenizer import WordTokenizer

# 加载原始 IMDb 数据
dataset = load_dataset("imdb")

# 收集词表（注意这里用了一个简单的 word splitter）
def build_vocab(data):
    vocab = set()
    for example in data:
        words = example["text"].lower().split()
        vocab.update(words)
    return list(vocab)

# 从训练集构建词表
vocab = build_vocab(dataset["train"])
tokenizer = WordTokenizer(vocab=vocab, sep=' ', max_seq_len=128)

# 将每条样本转换为 tokenized 格式（但我们这里只存原始 tokens 和标签）
def to_json_format(example):
    return {
        "text": example["text"].lower().split(),  # 不再用 nltk，而是和 tokenizer 的逻辑一致
        "label": "pos" if example["label"] == 1 else "neg"
    }

# 构建数据集
train_data = [to_json_format(example) for example in dataset["train"]]
test_data = [to_json_format(example) for example in dataset["test"]]

# 保存 JSONL 文件
with open("imdb_train.json", "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open("imdb_test.json", "w") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")
