import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from microbert.model import MicroBERT
from microbert.utils import plot_results, plot_mlm_results


class MicroBertMLM(torch.nn.Module):
    """
    MicroBERT model for Masked Language Modeling (MLM) pre-training
    """

    def __init__(self, vocab_size, n_layers=2, n_heads=1, n_embed=3, max_seq_len=128):
        super().__init__()
        self.micro_bert = MicroBERT(
            vocab_size=vocab_size,
            n_layers=n_layers,
            n_heads=n_heads,
            n_embed=n_embed,
            max_seq_len=max_seq_len
        )
        self.mlm_head = torch.nn.Linear(n_embed, vocab_size)

    def forward(self, input_ids, labels=None):
        # Get embeddings from MicroBERT
        embeddings = self.micro_bert.embedding(input_ids)
        # Create attention mask for the encoder (same as in MicroBERT)
        attention_mask = (input_ids > 0).unsqueeze(1).repeat(1, input_ids.size(1), 1)
        encoded = self.micro_bert.encoder(embeddings, attention_mask)
        # Apply MLM head to predict masked tokens
        logits = self.mlm_head(encoded)  # (B, seq_len, vocab_size)
        if labels is not None:
            # Only compute loss on masked positions
            input_, target_ = logits.view(-1, logits.size(-1)), labels.view(-1)
            loss = F.cross_entropy(input_, target_, ignore_index=-100)
            return loss
        return logits


class MLMDataset(Dataset):
    """
    Dataset for MLM pre-training with masking using our own tokenizer
    """

    def __init__(self, data, tokenizer, max_length=128, mlm_probability=0.15):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

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
        # Ensure all IDs are within valid range
        vocab_size = len(self.tokenizer.vocab)
        input_ids = [min(id, vocab_size - 1) for id in input_ids]
        # Pad to max_length
        while len(input_ids) < self.max_length:
            input_ids.append(self.tokenizer.vocab['[PAD]'])
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # Debug: Check for invalid indices
        if input_ids.max() >= len(self.tokenizer.vocab):
            print(f"Warning: Found invalid token ID {input_ids.max()} >= vocab size {len(self.tokenizer.vocab)}")
            input_ids = torch.clamp(input_ids, 0, len(self.tokenizer.vocab) - 1)
        # Create masked input and labels
        masked_input_ids, labels = self.mask_tokens(input_ids)
        return {
            'input_ids': masked_input_ids,
            'labels': labels
        }

    def mask_tokens(self, input_ids):
        """
        Prepare masked tokens inputs/labels for masked language modeling
        """
        labels = input_ids.clone()
        # Create probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # Don't mask special tokens ([PAD], [CLS], [SEP], [UNK])
        special_token_ids = [
            self.tokenizer.vocab['[PAD]'],
            self.tokenizer.vocab['[CLS]'],
            self.tokenizer.vocab['[SEP]'],
            self.tokenizer.vocab['[UNK]']
        ]
        for special_id in special_token_ids:
            probability_matrix.masked_fill_(labels == special_id, value=0.0)
        # Mask tokens
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.vocab['[MASK]']
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # Ensure random indices are within valid vocabulary range
        vocab_size = len(self.tokenizer.vocab)
        random_words = torch.randint(0, vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels


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


def train_mlm(model, train_loader, val_loader, device, tokenizer, num_epochs=10, learning_rate=5e-5):
    """
    Train the MLM model
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    # Training history
    history = {
        'train_losses': [],
        'val_losses': []
    }
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc='Training'):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            loss = model(input_ids, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                loss = model(input_ids, labels)
                val_loss += loss.item()
        # Store history
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        print()
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_mlm_model(model, tokenizer, history, '.mlm_v0')
    return history


def save_mlm_model(model, tokenizer, history, save_dir):
    """
    Save the MLM pre-trained model
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, 'mlm_model.pth')
    torch.save(model.state_dict(), model_path)

    # Save MicroBERT model for later use
    microbert_path = os.path.join(save_dir, 'microbert_model.pth')
    torch.save(model.micro_bert.state_dict(), microbert_path)

    # Save tokenizer
    tokenizer_path = os.path.join(save_dir, 'tokenizer_vocab.json')
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer.vocab, f, indent=2)

    # Save training history
    history_path = os.path.join(save_dir, 'mlm_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f'MLM model saved to {save_dir}')


def main():
    """
    Main MLM pre-training function
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data
    print('Loading IMDB dataset for MLM pre-training...')
    train_data, test_data = load_imdb_data()

    # Use all data for MLM pre-training (unsupervised)
    all_data = train_data + test_data

    # Split into train and validation
    train_size = int(0.9 * len(all_data))
    val_data = all_data[train_size:]
    train_data = all_data[:train_size]

    print(f'Training samples: {len(train_data)}')
    print(f'Validation samples: {len(val_data)}')

    # Build vocabulary from data with frequency filtering
    from collections import Counter
    word_counts = Counter()
    for item in all_data:
        word_counts.update(item['text'])

    # Keep only the most frequent words (top 10000) to limit vocabulary size
    max_vocab_size = 10000
    most_common_words = [word for word, count in word_counts.most_common(max_vocab_size)]

    # Add special tokens
    special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]']
    vocab = set(special_tokens + most_common_words)

    print(f'Vocabulary size: {len(vocab)} (limited from {len(word_counts)} total unique words)')

    # Create our own tokenizer
    from microbert.tokenizer import WordTokenizer
    tokenizer = WordTokenizer(vocab=list(vocab), max_seq_len=128)

    # Create datasets
    train_dataset = MLMDataset(train_data, tokenizer)
    val_dataset = MLMDataset(val_data, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize model
    model = MicroBertMLM(
        vocab_size=len(tokenizer.vocab),
        n_layers=2,
        n_heads=2,
        n_embed=4,
        max_seq_len=128
    ).to(device)

    # Check if model already exists
    if os.path.exists('.mlm_v0/mlm_model.pth'):
        print('Loading existing MLM model...')
        model.load_state_dict(torch.load('.mlm_v0/mlm_model.pth', map_location=device, weights_only=True))
        print('MLM model loaded successfully!')

        # Load training history for plotting
        history_path = os.path.join('.mlm_v0', 'mlm_training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            print('Training history loaded for plotting.')
        else:
            history = None
            print('No training history found.')
    else:
        print('Starting MLM pre-training...')
        # Train model
        history = train_mlm(model, train_loader, val_loader, device, tokenizer, num_epochs=10)
        print('MLM pre-training completed!')

    # Test the model with some examples
    print('\n=== Testing MLM Model ===')
    test_texts = [
        ["this", "movie", "is", "[MASK]", "fantastic"],
        ["the", "acting", "was", "[MASK]", "but", "the", "plot", "was", "confusing"],
        ["amazing", "[MASK]", "by", "all", "actors", "highly", "recommended"]
    ]

    model.eval()
    for i, text_tokens in enumerate(test_texts, 1):
        print(f'{i}. Original: {" ".join(text_tokens)}')

        # Convert tokens to IDs
        input_ids = []
        for token in text_tokens:
            if token in tokenizer.vocab:
                input_ids.append(tokenizer.vocab[token])
            else:
                input_ids.append(tokenizer.vocab['[UNK]'])

        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

        # Get predictions
        with torch.no_grad():
            logits = model(input_ids)

            # Debug: Check logits values
            print(f'   Logits shape: {logits.shape}')
            print(f'   Logits min/max: {logits.min().item():.3f}/{logits.max().item():.3f}')
            print(f'   Logits mean: {logits.mean().item():.3f}')

            probs = F.softmax(logits, dim=-1)

            # Debug: Check probabilities
            print(f'   Probs min/max: {probs.min().item():.6f}/{probs.max().item():.6f}')
            print(f'   Probs mean: {probs.mean().item():.6f}')

            # Find masked positions
            mask_token_id = tokenizer.vocab['[MASK]']
            masked_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

            for pos in masked_positions:
                top_k = 5
                # Use logits directly for better numerical stability
                top_logits, top_indices = torch.topk(logits[0, pos], top_k)

                print(f'   [MASK] at position {pos}:')
                for j in range(top_k):
                    token_id = top_indices[j].item()
                    # Find token by ID
                    token = None
                    for t, tid in tokenizer.vocab.items():
                        if tid == token_id:
                            token = t
                            break
                    logit = top_logits[j].item()
                    prob = F.softmax(top_logits, dim=0)[j].item()
                    print(f'     {token}: logit={logit:.3f}, prob={prob:.6f}')
        print()

    # Plot training history if available
    if history is not None:
        print('\n=== Plotting Training History ===')
        # Use MLM-specific plotting function from utils
        plot_mlm_results(history, save_path='.mlm_v0/training_history.png')
    else:
        print('\nNo training history available for plotting.')


if __name__ == '__main__':
    main() 