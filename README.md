# MicroBERT

A lightweight BERT implementation for text classification tasks, designed to be simple, efficient, and easy to use.

## Features

- **Lightweight**: Minimal dependencies and optimized for small to medium datasets
- **Easy to use**: Simple API for training and inference
- **Customizable**: Easy to modify for different text classification tasks
- **Fast training**: Optimized training loop with progress tracking
- **IMDB Dataset Support**: Built-in support for IMDB movie review classification

## Installation

```bash
pip install microbert
```

Or install from source:

```bash
git clone https://github.com/henrywoo/microbert.git
cd microbert
pip install -e .
```

## Quick Start

### Training on IMDB Dataset

```python
import train

# Train the model on IMDB dataset
train.train_imdb(
    model_name="bert-base-uncased",
    batch_size=16,
    learning_rate=2e-5,
    epochs=3,
    max_length=512
)
```

### Using the Trained Model

```python
from microbert.model import MicroBERTClassifier
from microbert.tokenizer import MicroBERTTokenizer

# Load the trained model
model = MicroBERTClassifier.from_pretrained("path/to/saved/model")
tokenizer = MicroBERTTokenizer.from_pretrained("path/to/saved/tokenizer")

# Make predictions
text = "This movie was absolutely fantastic!"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
prediction = model(**inputs)
```

## Project Structure

```
microbert/
├── microbert/
│   ├── __init__.py
│   ├── model.py          # BERT classifier implementation
│   ├── tokenizer.py      # Tokenizer utilities
│   └── utils.py          # Utility functions
├── train.py              # Training script
├── prepare_imdb_json.py  # IMDB dataset preparation
├── imdb_train.json       # Training data
├── imdb_test.json        # Test data
└── setup.py             # Package configuration
```

## API Reference

### MicroBERTClassifier

The main model class for text classification.

```python
from microbert.model import MicroBERTClassifier

model = MicroBERTClassifier(
    model_name="bert-base-uncased",
    num_labels=2,
    dropout=0.1
)
```

**Parameters:**
- `model_name` (str): Pre-trained BERT model name
- `num_labels` (int): Number of classification labels
- `dropout` (float): Dropout rate for classification head

### MicroBERTTokenizer

Utility class for text tokenization.

```python
from microbert.tokenizer import MicroBERTTokenizer

tokenizer = MicroBERTTokenizer("bert-base-uncased")
```

### Training Functions

#### `train_imdb()`

Train the model on the IMDB dataset.

```python
import train

train.train_imdb(
    model_name="bert-base-uncased",
    batch_size=16,
    learning_rate=2e-5,
    epochs=3,
    max_length=512,
    save_dir="./models"
)
```

**Parameters:**
- `model_name` (str): Pre-trained model name
- `batch_size` (int): Training batch size
- `learning_rate` (float): Learning rate for optimization
- `epochs` (int): Number of training epochs
- `max_length` (int): Maximum sequence length
- `save_dir` (str): Directory to save the trained model

## Data Preparation

### IMDB Dataset

The project includes utilities to prepare the IMDB dataset:

```python
import prepare_imdb_json

# Prepare IMDB dataset
prepare_imdb_json.prepare_imdb_data(
    data_dir="./aclImdb",
    output_dir="./",
    max_samples=1000  # Limit samples for faster training
)
```

## Examples

### Custom Dataset Training

```python
from microbert.model import MicroBERTClassifier
from microbert.tokenizer import MicroBERTTokenizer
import torch
from torch.utils.data import DataLoader

# Initialize model and tokenizer
model = MicroBERTClassifier("bert-base-uncased", num_labels=3)
tokenizer = MicroBERTTokenizer("bert-base-uncased")

# Prepare your custom dataset
# ... your data preparation code ...

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch['labels'])
        loss.backward()
        optimizer.step()
```

## Requirements

- Python >= 3.7
- PyTorch
- Transformers
- Datasets
- NumPy
- tqdm

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Fuheng Wu** - [wufuheng@gmail.com](mailto:wufuheng@gmail.com)

## Acknowledgments

- Hugging Face for the transformers library
- The BERT paper authors for the original architecture
- The IMDB dataset creators

## Citation

If you use this project in your research, please cite:

```bibtex
@software{microbert2024,
  title={MicroBERT: A Lightweight BERT Implementation for Text Classification},
  author={Wu, Fuheng},
  year={2024},
  url={https://github.com/yourusername/microbert}
}
``` 