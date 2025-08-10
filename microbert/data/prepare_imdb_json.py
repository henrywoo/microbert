from datasets import load_dataset
import json
from microbert.tokenizer import WordTokenizer

print("Starting IMDb dataset preparation...")

# Load original IMDb data
print("Loading IMDb dataset from Hugging Face...")
dataset = load_dataset("imdb")
print(f"Dataset loaded successfully. Train samples: {len(dataset['train'])}, Test samples: {len(dataset['test'])}")

# Collect vocabulary (note: using a simple word splitter here)
def build_vocab(data):
    print("Building vocabulary from training data...")
    vocab = set()
    for i, example in enumerate(data):
        words = example["text"].lower().split()
        vocab.update(words)
    
    print(f"Vocabulary built successfully. Total unique words: {len(vocab)}")
    return list(vocab)

# Build vocabulary from training set
vocab = build_vocab(dataset["train"])
print(f"Creating WordTokenizer with vocabulary size: {len(vocab)}, max_seq_len: 128")
tokenizer = WordTokenizer(vocab=vocab, sep=' ', max_seq_len=128)

# Convert each sample to tokenized format (but here we only store original tokens and labels)
def to_json_format(example):
    return {
        "text": example["text"].lower().split(),  # No longer using nltk, but consistent with tokenizer logic
        "label": "pos" if example["label"] == 1 else "neg"
    }

# Build dataset
print("Converting training data to JSON format...")
train_data = []
for i, example in enumerate(dataset["train"]):
    train_data.append(to_json_format(example))

print("Converting test data to JSON format...")
test_data = []
for i, example in enumerate(dataset["test"]):
    test_data.append(to_json_format(example))

# Count label distribution
train_pos = sum(1 for item in train_data if item["label"] == "pos")
train_neg = sum(1 for item in train_data if item["label"] == "neg")
test_pos = sum(1 for item in test_data if item["label"] == "pos")
test_neg = sum(1 for item in test_data if item["label"] == "neg")

print(f"Training data - Positive: {train_pos}, Negative: {train_neg}")
print(f"Test data - Positive: {test_pos}, Negative: {test_neg}")

# Save JSONL files
print("Saving training data to imdb_train.json...")
with open("imdb_train.json", "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

print("Saving test data to imdb_test.json...")
with open("imdb_test.json", "w") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")

print("Dataset preparation completed successfully!")
print(f"Files saved: imdb_train.json ({len(train_data)} samples), imdb_test.json ({len(test_data)} samples)")
