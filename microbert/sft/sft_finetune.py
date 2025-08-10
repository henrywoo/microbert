import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
from microbert.model import MicroBERTForClassification


class SentimentClassifier(torch.nn.Module):
    """
    Sentiment classifier using MLM pre-trained MicroBERT model
    """
    def __init__(self, pretrained_microbert_path, vocab_size, num_classes=2, dropout=0.1):
        super().__init__()
        # Load MLM pre-trained MicroBERT
        from microbert.model import MicroBERT
        self.micro_bert = MicroBERT(
            vocab_size=vocab_size,
            n_layers=2,
            n_heads=1,
            n_embed=3,
            max_seq_len=128
        )
        self.micro_bert.load_state_dict(torch.load(pretrained_microbert_path, map_location='cpu', weights_only=True))
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(3, num_classes)  # n_embed=3
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Get MicroBERT outputs
        embeddings = self.micro_bert.embedding(input_ids)
        encoded = self.micro_bert.encoder(embeddings, None)  # No mask for classification
        
        # Use first token representation for classification
        pooled_output = encoded[:, 0, :]  # Take first token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class IMDBDataset(Dataset):
    """
    Dataset class for IMDB sentiment analysis using our own tokenizer
    """
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Use our own tokenizer (already tokenized)
        tokens = item['text'][:self.max_length]
        
        # Convert tokens to IDs
        input_ids = []
        for token in tokens:
            if token in self.tokenizer.vocab:
                input_ids.append(self.tokenizer.vocab[token])
            else:
                input_ids.append(self.tokenizer.vocab['[UNK]'])
        
        # Pad to max_length
        while len(input_ids) < self.max_length:
            input_ids.append(self.tokenizer.vocab['[PAD]'])
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Convert label
        label = 1 if item['label'] == 'pos' else 0
        
        return {
            'input_ids': input_ids,
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


def train_sft_model_selective(model, train_loader, val_loader, device, tokenizer, 
                              trainable_layers='classifier', num_epochs=3, learning_rate=1e-3):
    """
    Selective fine-tuning: Choose which layers to train
    Options for trainable_layers:
    - 'classifier': Only train classifier layer (fastest)
    - 'last_2': Train classifier + dropout + last transformer layer
    - 'last_3': Train classifier + dropout + last 2 transformer layers
    - 'all': Train everything (slowest, original behavior)
    """
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
    
    trainable_params = []
    
    if trainable_layers == 'classifier':
        # Only train classifier and dropout
        for param in model.classifier.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        for param in model.dropout.parameters():
            param.requires_grad = True
            trainable_params.append(param)
            
    elif trainable_layers == 'last_2':
        # Train classifier, dropout, and last transformer layer
        for param in model.classifier.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        for param in model.dropout.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        # Unfreeze last transformer layer
        for param in model.micro_bert.encoder.layers[-1].parameters():
            param.requires_grad = True
            trainable_params.append(param)
            
    elif trainable_layers == 'last_3':
        # Train classifier, dropout, and last 2 transformer layers
        for param in model.classifier.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        for param in model.dropout.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        # Unfreeze last 2 transformer layers
        for param in model.micro_bert.encoder.layers[-2:]:
            for p in param.parameters():
                p.requires_grad = True
                trainable_params.append(p)
                
    elif trainable_layers == 'all':
        # Train everything (original behavior)
        for param in model.parameters():
            param.requires_grad = True
            trainable_params.append(param)
    else:
        raise ValueError(f"Unknown trainable_layers option: {trainable_layers}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
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
    
    print(f"Selective fine-tuning: Training {trainable_layers} layers")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc='Training'):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids)
            loss = F.cross_entropy(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
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
                labels = batch['label'].to(device)
                
                logits = model(input_ids)
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
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f'New best validation F1: {best_val_f1:.4f}')
    
    return history


def train_sft_model_fast(model, train_loader, val_loader, device, tokenizer, num_epochs=3, learning_rate=1e-3):
    """
    Fast fine-tuning: Only train the last layer (classifier) while freezing the pre-trained MicroBERT layers
    This approach is much faster and uses less memory
    """
    # Freeze all layers except the classifier
    for param in model.micro_bert.parameters():
        param.requires_grad = False
    
    # Only train the classifier and dropout layers
    trainable_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
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
    
    print("Fast fine-tuning: Only training classifier layer (MicroBERT layers frozen)")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc='Training'):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids)
            loss = F.cross_entropy(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
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
                labels = batch['label'].to(device)
                
                logits = model(input_ids)
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
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f'New best validation F1: {best_val_f1:.4f}')
    
    return history


def train_sft_model(model, train_loader, val_loader, device, tokenizer, num_epochs=3, learning_rate=2e-5):
    """
    Train the SFT model
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
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids)
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
                labels = batch['label'].to(device)
                
                logits = model(input_ids)
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
            save_sft_model(model, tokenizer, history, '.sft_finetuned')
    
    return history


def save_sft_model(model, tokenizer, history, save_dir):
    """
    Save the SFT fine-tuned model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'sft_model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer
    tokenizer_path = os.path.join(save_dir, 'tokenizer')
    tokenizer.save_pretrained(tokenizer_path)
    
    # Save training history
    history_path = os.path.join(save_dir, 'sft_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'SFT model saved to {save_dir}')


def load_sft_model(model, tokenizer, save_dir):
    """
    Load the SFT fine-tuned model
    """
    # Load model
    model_path = os.path.join(save_dir, 'sft_model.pth')
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
    
    # Tokenize text using our tokenizer
    tokens = text.lower().split()[:128]  # Simple word-level tokenization
    
    # Convert tokens to IDs
    input_ids = []
    for token in tokens:
        if token in tokenizer.vocab:
            input_ids.append(tokenizer.vocab[token])
        else:
            input_ids.append(tokenizer.vocab['[UNK]'])
    
    # Pad to max_length
    while len(input_ids) < 128:
        input_ids.append(tokenizer.vocab['[PAD]'])
    
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(input_ids)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1)[0]
    
    sentiment = "Positive" if pred.item() == 1 else "Negative"
    return sentiment, confidence.item()


def main():
    """
    Main SFT fine-tuning function
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Check if MLM pre-trained model exists
    mlm_model_path = '.mlm_pretrained/microbert_model.pth'
    if not os.path.exists(mlm_model_path):
        print('Error: MLM pre-trained model not found!')
        print('Please run mlm_pretrain.py first to pre-train the MicroBERT model.')
        return
    
    # Load data
    print('Loading IMDB dataset for SFT fine-tuning...')
    train_data, test_data = load_imdb_data()
    
    # Split training data into train and validation
    train_size = int(0.9 * len(train_data))
    val_data = train_data[train_size:]
    train_data = train_data[:train_size]
    
    print(f'Training samples: {len(train_data)}')
    print(f'Validation samples: {len(val_data)}')
    print(f'Test samples: {len(test_data)}')
    
    # Load tokenizer from MLM pre-trained model
    with open('.mlm_pretrained/tokenizer_vocab.json', 'r') as f:
        vocab = json.load(f)
    
    from microbert.tokenizer import WordTokenizer
    tokenizer = WordTokenizer(vocab=list(vocab.keys()), max_seq_len=128)
    
    # Create datasets
    train_dataset = IMDBDataset(train_data, tokenizer)
    val_dataset = IMDBDataset(val_data, tokenizer)
    test_dataset = IMDBDataset(test_data, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model with MLM pre-trained MicroBERT
    model = SentimentClassifier(mlm_model_path, len(tokenizer.vocab)).to(device)
    
    # Check if SFT model already exists
    if os.path.exists('.sft_finetuned/sft_model.pth'):
        print('Loading existing SFT model...')
        model, tokenizer = load_sft_model(model, tokenizer, '.sft_finetuned')
        print('SFT model loaded successfully!')
    else:
        print('Starting SFT fine-tuning...')
        # Train model
        history = train_sft_model(model, train_loader, val_loader, device, tokenizer, num_epochs=3)
        print('SFT fine-tuning completed!')
    
    # Test the model
    print('\n=== Testing SFT Model ===')
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