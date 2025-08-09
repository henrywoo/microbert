from datasets import load_dataset
import json
from microbert.tokenizer import WordTokenizer

# Load original IMDb data
dataset = load_dataset("imdb")

# Collect vocabulary (note: using a simple word splitter here)
def build_vocab(data):
    vocab = set()
    for example in data:
        words = example["text"].lower().split()
        vocab.update(words)
    return list(vocab)

# Build vocabulary from training set
vocab = build_vocab(dataset["train"])
tokenizer = WordTokenizer(vocab=vocab, sep=' ', max_seq_len=128)

# Convert each sample to tokenized format (but here we only store original tokens and labels)
def to_json_format(example):
    return {
        "text": example["text"].lower().split(),  # No longer using nltk, but consistent with tokenizer logic
        "label": "pos" if example["label"] == 1 else "neg"
    }

# Build dataset
train_data = [to_json_format(example) for example in dataset["train"]]
test_data = [to_json_format(example) for example in dataset["test"]]

# Save JSONL files
with open("imdb_train.json", "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open("imdb_test.json", "w") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")
