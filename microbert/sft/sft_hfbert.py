import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np


class SentimentClassifier(torch.nn.Module):
    """
    Sentiment classifier using Hugging Face BERT model
    """
    def __init__(self, model_name="bert-base-cased", num_classes=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # Use [CLS] token representation for classification
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class IMDBDataset(Dataset):
    """
    Dataset class for IMDB sentiment analysis
    """
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = " ".join(item['text'])  # Join tokens back to text

        # Encode text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Convert label
        label = 1 if item['label'] == 'pos' else 0

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_imdb_data():
    """
    Load IMDB dataset from JSON files
    """
    # Load training data
    train_data = []
    with open('imdb_train.json', 'r') as f:
        for line in f:
            train_data.append(json.loads(line.strip()))

    # Load test data
    test_data = []
    with open('imdb_test.json', 'r') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))

    return train_data, test_data


def train_model(model, train_loader, val_loader, device, tokenizer, num_epochs=3, learning_rate=2e-5):
    """
    Train the sentiment classifier
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }

    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in tqdm(train_loader, desc='Training'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            # Get predictions
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids, attention_mask, token_type_ids)
                loss = F.cross_entropy(logits, labels)

                val_loss += loss.item()

                # Get predictions
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        train_f1 = f1_score(train_labels, train_preds)
        val_f1 = f1_score(val_labels, val_preds)

        # Store history
        history['train_losses'].append(train_loss / len(train_loader))
        history['val_losses'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        print(f'Train Loss: {history["train_losses"][-1]:.4f} | Val Loss: {history["val_losses"][-1]:.4f}')
        print(f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
        print(f'Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}')
        print()

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_model(model, tokenizer, history, '.sft_hf_bert')

    return history


def save_model(model, tokenizer, history, save_dir):
    """
    Save the trained model and tokenizer
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)

    # Save tokenizer
    tokenizer_path = os.path.join(save_dir, 'tokenizer')
    tokenizer.save_pretrained(tokenizer_path)

    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f'Model saved to {save_dir}')


def load_model(model, tokenizer, save_dir):
    """
    Load the trained model and tokenizer
    """
    # Load model
    model_path = os.path.join(save_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))

    # Load tokenizer
    tokenizer_path = os.path.join(save_dir, 'tokenizer')
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    return model, tokenizer


def predict_sentiment(model, tokenizer, text, device):
    """
    Predict sentiment for a given text
    """
    model.eval()

    # Tokenize text
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1)[0]

    sentiment = "Positive" if pred.item() == 1 else "Negative"
    return sentiment, confidence.item()


def main():
    """
    Main training function
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data
    print('Loading IMDB dataset...')
    train_data, test_data = load_imdb_data()

    # Split training data into train and validation
    train_size = int(0.9 * len(train_data))
    val_data = train_data[train_size:]
    train_data = train_data[:train_size]

    print(f'Training samples: {len(train_data)}')
    print(f'Validation samples: {len(val_data)}')
    print(f'Test samples: {len(test_data)}')

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # Create datasets
    train_dataset = IMDBDataset(train_data, tokenizer)
    val_dataset = IMDBDataset(val_data, tokenizer)
    test_dataset = IMDBDataset(test_data, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model
    model = SentimentClassifier().to(device)

    # Check if model already exists
    if os.path.exists('.sft_hf_bert/model.pth'):
        print('Loading existing model...')
        model, tokenizer = load_model(model, tokenizer, '.sft_hf_bert')
        print('Model loaded successfully!')
    else:
        print('Training new model...')
        # Train model
        history = train_model(model, train_loader, val_loader, device, tokenizer, num_epochs=3)
        print('Training completed!')

    # Test the model
    print('\n=== Testing Model ===')
    test_texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Terrible film, waste of time and money. Don't watch it.",
        "The acting was okay but the plot was confusing.",
        "Amazing performance by all actors, highly recommended!",
        "Boring and predictable, I fell asleep halfway through."
    ]

    for i, text in enumerate(test_texts, 1):
        sentiment, confidence = predict_sentiment(model, tokenizer, text, device)
        print(f'{i}. Text: {text}')
        print(f'   Prediction: {sentiment} (Confidence: {confidence:.3f})\n')


if __name__ == '__main__':
    main()
