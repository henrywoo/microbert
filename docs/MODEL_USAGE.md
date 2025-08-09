# MicroBERT Model Usage Guide

## Model Saving

After training is completed, the model will be automatically saved to the `~/.microbert_model/` directory, containing the following files:

- `microbert_classification.pth` - Model weight file
- `tokenizer_vocab.json` - Tokenizer vocabulary
- `training_history.json` - Training history record
- `model_config.json` - Model configuration information

## Model Loading and Usage

### 1. Using Prediction Script

Run the pre-made prediction script:

```bash
python predict.py
```

This will load the trained model and perform sentiment analysis on example text.

### 2. Using in Code

```python
from microbert.utils import load_model, predict_sentiment

# Load model
model, tokenizer, config = load_model('~/.microbert_model')

# Make prediction
text = "This movie is amazing!"
prediction, confidence = predict_sentiment(model, tokenizer, text)

sentiment = "Positive" if prediction == 1 else "Negative"
print(f"Prediction result: {sentiment}, Confidence: {confidence:.3f}")
```

### 3. Manually Saving Model

If you need to manually save the model, you can use:

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

## File Description

### Model Weight File (.pth)
Contains the trained neural network weight parameters.

### Vocabulary File (.json)
Contains the vocabulary used by the tokenizer for text preprocessing.

### Training History File (.json)
Contains metrics such as loss, accuracy, F1 score during training.

### Configuration File (.json)
Contains model architecture and training parameters for rebuilding the model structure.

## Notes

1. Ensure to use the same model architecture parameters when loading the model
2. Model files are large, please ensure sufficient storage space
3. It is recommended to regularly backup trained model files
4. When migrating between different devices, pay attention to device compatibility (CPU/GPU) 