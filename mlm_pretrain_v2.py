import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import GradScaler
# autocast is used directly from torch.amp.autocast
from microbert.model import MicroBERT
from microbert.utils import plot_results, plot_mlm_results
from hiq.vis import print_model

# Add current directory to path for imports
import sys
sys.path.append('.')


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
    print("Loading IMDB dataset...")
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


def load_hf_dataset(max_samples: int = 500_000, min_words: int = 5, seed: int = 42, streaming: bool = True, cache_dir: str = ".dataset_cache"):
    """
    Load large text dataset from Hugging Face with streaming support and local caching
    
    Args:
        max_samples: Maximum number of samples to load
        min_words: Minimum number of words per sample
        seed: Random seed for reproducibility
        streaming: If True, use streaming mode to avoid downloading full dataset to disk
                  If False, download dataset to local disk (uses more space but faster)
        cache_dir: Directory to cache processed dataset
    
    Returns: (data, []) where data is a list[{'text': List[str]}]
    """
    from datasets import load_dataset
    import itertools
    import os
    import json
    import hashlib

    print("Loading large text dataset from Hugging Face...")
    if streaming:
        print("Using streaming mode - data will be cached locally for reuse")
    else:
        print("Using local download mode - data will be saved to disk (faster but uses more space)")

    # 数据集与其可选配置（尽量选择公开可用、可流式，按大小排序）
    # 为了开发/测试，优先选择较小的数据集以避免磁盘空间问题
    # 但如果max_samples很大，优先选择大数据集
    if max_samples and max_samples > 1_000_000:  # 如果请求超过1M样本
        candidates = [
            # (dataset_name, dict(kwargs_for_load_dataset))
            ("c4",         {"name": "en"}),  # Common Crawl data, very large (大数据集优先)
            ("dbpedia_14", {}),  # Wikipedia articles
            ("ag_news",    {}),  # News articles
            ("ag_news",    {"name": "default"}),  # AG News default
            ("yelp_polarity", {}),  # Yelp reviews
            ("yelp_review_full", {}),  # Full Yelp reviews (larger)
            ("amazon_polarity", {}),  # Amazon reviews
            ("squad",      {}),  # Question answering dataset
            ("squad",      {"name": "plain_text"}),  # SQuAD plain text
            ("squad_v2",   {}),  # SQuAD v2 (larger)
            ("imdb",       {}),  # Movie reviews
            ("imdb",       {"name": "plain_text"}),  # IMDB plain text
            ("wikitext",   {"name": "wikitext-103-raw-v1"}),  # ~1.8M tokens
            ("wikitext",   {"name": "wikitext-2-raw-v1"}),  # Alternative wikitext
            ("wikitext",   {"name": "wikitext-103-v1"}),  # Another wikitext variant
            ("wikitext",   {"name": "wikitext-2-v1"}),  # Another wikitext variant
        ]
    else:
        candidates = [
            # (dataset_name, dict(kwargs_for_load_dataset))
            ("wikitext",   {"name": "wikitext-103-raw-v1"}),  # ~1.8M tokens (smaller)
            ("squad",      {}),  # Question answering dataset
            ("imdb",       {}),  # Movie reviews
            ("ag_news",    {}),  # News articles
            ("yelp_polarity", {}),  # Yelp reviews
            ("dbpedia_14", {}),  # Wikipedia articles
            ("c4",         {"name": "en"}),  # Common Crawl data, very large (last resort)
        ]

    def extract_text(item: dict) -> str | None:
        # 按常见字段顺序取文本
        text_fields = [
            'text', 'content', 'sentence', 'passage', 'article', 'question', 'context', 'title', 'summary',
            'review', 'review_text', 'body', 'document', 'raw_content', 'source', 'comment', 'description',
            'abstract', 'caption', 'transcript', 'utterance', 'dialogue', 'conversation', 'story', 'narrative'
        ]
        for field in text_fields:
            if field in item and item[field]:
                text = item[field]
                if isinstance(text, str) and len(text.strip()) > 10:
                    return text
                elif isinstance(text, list) and len(text) > 0:
                    # Handle list of strings
                    if all(isinstance(t, str) for t in text):
                        combined_text = ' '.join(text)
                        if len(combined_text.strip()) > 10:
                            return combined_text
        return None

    def generate_cache_key(ds_name, ds_kwargs, max_samples, min_words, seed):
        """Generate cache key for dataset configuration - fixed to be consistent"""
        import hashlib
        # Remove total_loaded from cache key to make it consistent
        config_str = f"{ds_name}_{ds_kwargs}_{max_samples}_{min_words}_{seed}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def load_from_cache(cache_key):
        """Load dataset from cache if available"""
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                print(f"Loading cached dataset from {cache_file}...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"Successfully loaded {len(data)} samples from cache")
                return data
            except Exception as e:
                print(f"Failed to load cache: {e}")
        return None

    def save_to_cache(data, cache_key):
        """Save dataset to cache"""
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")
        try:
            print(f"Saving dataset to cache: {cache_file}")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Successfully cached {len(data)} samples")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    # Try each dataset option sequentially until we reach max_samples
    all_data = []
    total_samples = 0
    
    for ds_name, ds_kwargs in candidates:
        if total_samples >= max_samples:
            print(f"Reached target sample count ({max_samples:,}), stopping dataset loading")
            break
            
        remaining_samples = max_samples - total_samples
        print(f"Trying to load {ds_name} {ds_kwargs} (need {remaining_samples:,} more samples)...")
        
        try:
            # Generate cache key for this specific dataset - FIXED: use max_samples for consistency
            cache_key = generate_cache_key(ds_name, ds_kwargs, max_samples, min_words, seed)
            
            # Try to load from cache first
            cached_data = load_from_cache(cache_key)
            if cached_data is not None:
                print(f"Loaded {len(cached_data)} samples from cache for {ds_name}")
                # Only take what we need from cache
                needed_samples = min(len(cached_data), remaining_samples)
                all_data.extend(cached_data[:needed_samples])
                total_samples += needed_samples
                print(f"Used {needed_samples} samples from cache, total so far: {total_samples:,}")
                continue
            
            # Load dataset from Hugging Face
            try:
                ds = load_dataset(
                    ds_name,
                    **ds_kwargs,
                    split="train",
                    streaming=streaming
                )
            except Exception as e:
                print(f"Failed to load {ds_name} with streaming={streaming}: {e}")
                # Try with opposite streaming setting
                try:
                    streaming_fallback = not streaming
                    print(f"Trying with streaming={streaming_fallback}...")
                    ds = load_dataset(
                        ds_name,
                        **ds_kwargs,
                        split="train",
                        streaming=streaming_fallback
                    )
                except Exception as e2:
                    print(f"Failed to load {ds_name} with streaming={streaming_fallback}: {e2}")
                    continue

            # 可选：shuffle。注意：streaming模式下需要设置一个足够大的缓冲区
            if streaming:
                try:
                    ds = ds.shuffle(seed=seed, buffer_size=10_000)
                    print("Applied streaming shuffle with buffer_size=10_000")
                    print(f"Dataset {ds_name} loaded in streaming mode")
                except Exception as _:
                    # 某些数据集/版本可能不支持流式 shuffle，忽略即可
                    print("Streaming shuffle not supported, continuing without shuffle")
                    print(f"Dataset {ds_name} loaded in streaming mode")
            else:
                # 非streaming模式可以直接shuffle
                try:
                    ds = ds.shuffle(seed=seed)
                    print("Applied local shuffle")
                    print(f"Dataset {ds_name} loaded with {len(ds)} total samples")
                except Exception as _:
                    print("Local shuffle failed, continuing without shuffle")
                    print(f"Dataset {ds_name} loaded with {len(ds)} total samples")

            data = []
            kept = 0

            for item in ds:
                text = extract_text(item)
                if not text:
                    continue
                tokens = text.strip().lower().split()
                if len(tokens) >= min_words:
                    data.append({"text": tokens})
                    kept += 1
                    if kept >= remaining_samples:
                        break

            if data:
                print(f"Successfully loaded {kept} samples from {ds_name}")
                
                # Save to cache for future use
                save_to_cache(data, cache_key)
                
                # Add to total data
                all_data.extend(data)
                total_samples += len(data)
                
                print(f"Total samples so far: {total_samples:,}/{max_samples:,}")
                
                if total_samples >= max_samples:
                    print(f"Reached target sample count ({max_samples:,}), stopping dataset loading")
                    break
            else:
                print(f"No valid samples found in {ds_name}")

        except Exception as e:
            print(f"Failed to load {ds_name}: {e}")
            continue

    if all_data:
        print(f"Successfully loaded {len(all_data):,} total samples from multiple datasets")
        return all_data, []
    else:
        print("Failed to load any datasets, falling back to IMDB")
        return load_imdb_data()


def load_text_data(dataset_choice='imdb', streaming=True, max_samples=None):
    """
    Load text dataset based on user choice
    
    Args:
        dataset_choice: 'imdb' for IMDB dataset, 'hf' for Hugging Face dataset
        streaming: If True, use streaming mode for Hugging Face datasets
        max_samples: Maximum number of samples to load (for HF datasets)
    """
    if dataset_choice.lower() == 'imdb':
        return load_imdb_data()
    elif dataset_choice.lower() in ['hf', 'huggingface', 'large']:
        return load_hf_dataset(max_samples=max_samples, streaming=streaming)
    else:
        print(f"Unknown dataset choice: {dataset_choice}. Using IMDB dataset.")
        return load_imdb_data()


def train_mlm(model, train_loader, val_loader, device, tokenizer, num_epochs=3, learning_rate=5e-5):
    """
    Train the MLM model with bfloat16 mixed precision
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler('cuda')
    
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
            
            # Forward pass with bfloat16 mixed precision
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = model(input_ids, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
            save_mlm_model(model, tokenizer, history, '.mlm_pretrained')
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


def main(dataset_choice='imdb', streaming=True, max_samples=None):
    """
    Main MLM pre-training function
    
    Args:
        dataset_choice: 'imdb' for IMDB dataset (fast), 'hf' for Hugging Face dataset (large)
        streaming: If True, use streaming mode for Hugging Face datasets
        max_samples: Maximum number of samples to load (for HF datasets)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print(f'Loading dataset for MLM pre-training (choice: {dataset_choice}, streaming: {streaming}, max_samples: {max_samples})...')
    train_data, test_data = load_text_data(dataset_choice, streaming=streaming, max_samples=max_samples)
    
    # Use all data for MLM pre-training (unsupervised)
    if test_data:
        all_data = train_data + test_data
    else:
        all_data = train_data
    
    # Split into train and validation
    train_size = int(0.9 * len(all_data))
    val_data = all_data[train_size:]
    train_data = all_data[:train_size]
    
    print(f'Training samples: {len(train_data)}')
    print(f'Validation samples: {len(val_data)}')
    
    # Build vocabulary from data
    from collections import Counter
    word_counts = Counter()
    for item in all_data:
        word_counts.update(item['text'])
    
    # Use all unique words from the dataset (no artificial limits)
    all_unique_words = list(word_counts.keys())
    
    # Add special tokens (ensure they're at the beginning and no duplicates)
    special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]']
    # Create vocabulary list with special tokens first, then all dataset words
    vocab_list = special_tokens + [word for word in all_unique_words if word not in special_tokens]
    
    print(f'Vocabulary size: {len(vocab_list)} (all unique words from dataset)')
    print(f'Total unique words in dataset: {len(word_counts)}')
    
    # Create our own tokenizer
    from microbert.tokenizer import WordTokenizer
    tokenizer = WordTokenizer(vocab=list(vocab_list), max_seq_len=128)
    
    # Create datasets
    train_dataset = MLMDataset(train_data, tokenizer)
    val_dataset = MLMDataset(val_data, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model based on dataset choice
    if dataset_choice.lower() == 'hf':
        # Use larger model for Hugging Face datasets
        print("Using larger model configuration for Hugging Face dataset...")
        n_heads = 4
        n_embed = 8
        n_layers = 4
        head_size = n_embed // n_heads  # 8 // 4 = 2
        num_epochs = 5
        learning_rate = 3e-5
        batch_size = 16
    else:
        # Use smaller model for IMDB dataset
        print("Using smaller model configuration for IMDB dataset...")
        n_heads = 2
        n_embed = 4
        n_layers = 2
        head_size = n_embed // n_heads  # 4 // 2 = 2
        num_epochs = 3
        learning_rate = 5e-5
        batch_size = 16
    
    print(f"Model configuration:")
    print(f"  - n_heads: {n_heads}")
    print(f"  - n_embed: {n_embed}")
    print(f"  - n_layers: {n_layers}")
    print(f"  - head_size: {head_size}")
    print(f"  - num_epochs: {num_epochs}")
    print(f"  - learning_rate: {learning_rate}")
    
    model = MicroBertMLM(
        vocab_size=len(tokenizer.vocab),
        n_layers=n_layers,
        n_heads=n_heads,
        n_embed=n_embed,
        max_seq_len=128
    ).to(device)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Print model structure using hiq
    print("\n=== Model Structure ===")
    print_model(model)
    print("=== End Model Structure ===\n")
    
    # Check if model already exists
    if os.path.exists('.mlm_pretrained/mlm_model.pth'):
        print('Loading existing MLM model...')
        model.load_state_dict(torch.load('.mlm_pretrained/mlm_model.pth', map_location=device, weights_only=True))
        print('MLM model loaded successfully!')
        
        # Load training history for plotting
        history_path = os.path.join('.mlm_pretrained', 'mlm_training_history.json')
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
        history = train_mlm(model, train_loader, val_loader, device, tokenizer, num_epochs=num_epochs, learning_rate=learning_rate)
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
        plot_mlm_results(history, save_path='.mlm_pretrained/training_history.png')
    else:
        print('\nNo training history available for plotting.')


if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        dataset_choice = sys.argv[1]
    else:
        print("Usage: python mlm_pretrain_v2.py [imdb|hf] [streaming] [max_samples]")
        print("  imdb: Use IMDB dataset (fast, ~50K samples)")
        print("  hf:   Use Hugging Face dataset (large, configurable size)")
        print("  streaming: 'true' for streaming mode (default, saves disk space)")
        print("             'false' for local download mode (faster, uses more disk)")
        print("  max_samples: Maximum number of samples to load (e.g., 500k, 5M, 50M, 500M)")
        print("Examples:")
        print("  python mlm_pretrain_v2.py imdb                    # IMDB with streaming")
        print("  python mlm_pretrain_v2.py hf                      # HF dataset with streaming (default 500k)")
        print("  python mlm_pretrain_v2.py hf true 5M              # HF dataset with 5M samples")
        print("  python mlm_pretrain_v2.py hf false 50M            # HF dataset with 50M samples (local download)")
        print("Defaulting to IMDB dataset with streaming...")
        dataset_choice = 'imdb'
    
    # Parse streaming argument
    streaming = True  # Default to streaming
    if len(sys.argv) > 2:
        streaming_arg = sys.argv[2].lower()
        if streaming_arg in ['false', '0', 'no', 'local']:
            streaming = False
        elif streaming_arg in ['true', '1', 'yes', 'stream']:
            streaming = True
        else:
            print(f"Unknown streaming argument: {sys.argv[2]}. Using streaming=True")
    
    # Parse max_samples argument
    max_samples = None  # Default to 500k for HF datasets
    if len(sys.argv) > 3:
        max_samples_str = sys.argv[3].upper()
        if max_samples_str.endswith('K'):
            max_samples = int(max_samples_str[:-1]) * 1000
        elif max_samples_str.endswith('M'):
            max_samples = int(max_samples_str[:-1]) * 1000000
        elif max_samples_str.isdigit():
            max_samples = int(max_samples_str)
        else:
            print(f"Unknown max_samples format: {sys.argv[3]}. Using default (500k)")
    
    print(f"Dataset choice: {dataset_choice}")
    print(f"Streaming mode: {streaming}")
    print(f"Max samples: {max_samples}")
    print()
    
    main(dataset_choice, streaming=streaming, max_samples=max_samples) 