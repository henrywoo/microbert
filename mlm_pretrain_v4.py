#!/usr/bin/env python3
"""
Multi-GPU MLM pre-training script for MicroBERT v4
Optimized for 24GB GPU memory (H200/A10 compatible)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch.distributed as dist
from torch.amp import GradScaler, autocast
import time
from pathlib import Path
import datetime

# Add current directory to path for imports
import sys
sys.path.append('.')

from microbert.model import MicroBERT, BertEmbeddings, BertEncoder
from microbert.utils import plot_mlm_results
from hiq.vis import print_model


class MicroBertMLM(nn.Module):
    """MicroBERT for Masked Language Modeling - v4 optimized for 24GB memory"""
    
    def __init__(self, vocab_size, n_layers=8, n_heads=8, n_embed=256, max_seq_len=256):
        super().__init__()
        # Only include the components needed for MLM
        self.embedding = BertEmbeddings(vocab_size, n_embed, max_seq_len)
        self.encoder = BertEncoder(n_layers, n_heads, dropout=0.1, n_embed=n_embed)
        self.mlm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, input_ids, labels=None):
        # Get embeddings
        embeddings = self.embedding(input_ids)
        
        # Create attention mask for padding - ensure proper dimensions
        # input_ids shape: (batch_size, seq_len)
        # attention_mask shape: (batch_size, seq_len, seq_len)
        batch_size, seq_len = input_ids.shape
        # Create mask where 1 indicates valid tokens (non-padding) and 0 indicates padding
        # Use boolean mask for better compatibility
        attention_mask = (input_ids > 0).unsqueeze(1).expand(batch_size, seq_len, seq_len).bool()
        
        # Pass through encoder
        encoded = self.encoder(embeddings, attention_mask)
        
        # MLM head
        logits = self.mlm_head(encoded)
        
        if labels is not None:
            # Calculate loss only for masked positions
            loss_fct = nn.CrossEntropyLoss()
            # Flatten for loss calculation
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, logits.size(-1))
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits


class MLMDataset(Dataset):
    """Dataset for MLM pre-training - v4 optimized for 24GB memory"""
    
    def __init__(self, data, tokenizer, max_length=128, mlm_probability=0.15):  # Reduced from 256 to 128
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # Tokenize - tokenizer.encode returns a torch.Tensor with proper padding
        input_ids = self.tokenizer.encode(text)
        
        # Ensure the tensor has the correct length and is on the right device
        if isinstance(input_ids, torch.Tensor):
            # If the tokenizer returned a tensor, ensure it has the correct length
            if len(input_ids) != self.max_length:
                if len(input_ids) > self.max_length:
                    input_ids = input_ids[:self.max_length]
                else:
                    # Pad with [PAD] tokens
                    pad_length = self.max_length - len(input_ids)
                    pad_tokens = torch.full((pad_length,), self.tokenizer.vocab['[PAD]'], dtype=torch.long)
                    input_ids = torch.cat([input_ids, pad_tokens])
        else:
            # If tokenizer returned something else, convert to tensor
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            if len(input_ids) != self.max_length:
                if len(input_ids) > self.max_length:
                    input_ids = input_ids[:self.max_length]
                else:
                    # Pad with [PAD] tokens
                    pad_length = self.max_length - len(input_ids)
                    pad_tokens = torch.full((pad_length,), self.tokenizer.vocab['[PAD]'], dtype=torch.long)
                    input_ids = torch.cat([input_ids, pad_tokens])
        
        # Ensure all token IDs are within valid vocabulary range
        vocab_size = len(self.tokenizer.vocab)
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        
        # Apply MLM masking
        input_ids, labels = self.mask_tokens(input_ids)
        
        # Final validation - ensure all token IDs are still within bounds after masking
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        labels = torch.clamp(labels, -100, vocab_size - 1)  # -100 is the ignore_index for loss calculation
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }
    
    def mask_tokens(self, input_ids):
        """Apply MLM masking to input tokens"""
        labels = input_ids.clone()
        
        # Find positions to mask (exclude special tokens)
        special_tokens = [
            self.tokenizer.vocab['[PAD]'],
            self.tokenizer.vocab['[CLS]'],
            self.tokenizer.vocab['[SEP]']
        ]
        
        maskable_positions = []
        for i, token_id in enumerate(input_ids):
            if token_id not in special_tokens:
                maskable_positions.append(i)
        
        # Randomly mask 15% of tokens
        num_to_mask = max(1, int(len(maskable_positions) * self.mlm_probability))
        masked_positions = random.sample(maskable_positions, min(num_to_mask, len(maskable_positions)))
        
        vocab_size = len(self.tokenizer.vocab)
        
        for pos in masked_positions:
            # 80% chance to replace with [MASK]
            if random.random() < 0.8:
                mask_token_id = self.tokenizer.vocab.get('[MASK]', 0)
                input_ids[pos] = mask_token_id
            # 10% chance to replace with random word
            elif random.random() < 0.5:
                # Ensure the random token ID is within valid range
                random_token_id = random.randint(0, vocab_size - 1)
                input_ids[pos] = random_token_id
            # 10% chance to keep original
            # (labels already contain original)
        
        # Final validation - ensure all token IDs are within bounds
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        labels = torch.clamp(labels, -100, vocab_size - 1)
        
        return input_ids, labels


def load_imdb_data():
    """Load IMDB dataset"""
    print("Loading IMDB dataset for MLM pre-training...")
    
    # Load training data
    with open('imdb_train.json', 'r') as f:
        train_data = json.load(f)
    
    # Load test data
    with open('imdb_test.json', 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples from IMDB")
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
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not found. Please install with: pip install datasets")
        return [], []
    
    print("Loading large text dataset from Hugging Face...")
    
    if streaming:
        print("Using streaming mode - data will be cached locally for reuse")
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define dataset options in order of preference
    # For large sample requests (>1M), prioritize larger datasets like C4
    # For smaller requests, use smaller datasets to avoid disk space issues
    if max_samples and max_samples > 1_000_000:  # If requesting more than 1M samples
        dataset_options = [
            {'name': 'c4', 'kwargs': {'name': 'en'}},  # Common Crawl data, very large (first choice for large samples)
            {'name': 'c4', 'kwargs': {'name': 'en', 'split': 'train'}},  # Alternative C4 configuration
            {'name': 'dbpedia_14', 'kwargs': {}},  # Wikipedia articles
            {'name': 'ag_news', 'kwargs': {}},  # News articles
            {'name': 'yelp_polarity', 'kwargs': {}},  # Yelp reviews
            {'name': 'yelp_review_full', 'kwargs': {}},  # Full Yelp reviews (larger)
            {'name': 'amazon_polarity', 'kwargs': {}},  # Amazon reviews
            {'name': 'amazon_reviews_multi', 'kwargs': {'language': 'en'}},  # Multi-language Amazon reviews
            {'name': 'squad', 'kwargs': {}},  # Question answering dataset
            {'name': 'squad_v2', 'kwargs': {}},  # SQuAD v2 (larger)
            {'name': 'imdb', 'kwargs': {}},  # Movie reviews
            {'name': 'wikitext', 'kwargs': {'name': 'wikitext-103-raw-v1'}},  # ~1.8M tokens
            {'name': 'wikitext', 'kwargs': {'name': 'wikitext-2-raw-v1'}},  # Alternative wikitext
            {'name': 'bookcorpus', 'kwargs': {}},  # Book corpus
            {'name': 'openwebtext', 'kwargs': {}},  # OpenWebText (if available)
            {'name': 'wikipedia', 'kwargs': {'language': 'en', 'date': '20220301'}},  # Wikipedia articles
            {'name': 'wikipedia', 'kwargs': {'language': 'en', 'date': '20221201'}},  # Alternative Wikipedia date
            {'name': 'oscar', 'kwargs': {'language': 'en', 'split': 'train'}},  # OSCAR corpus
            {'name': 'oscar', 'kwargs': {'language': 'en', 'split': 'train', 'cache_dir': cache_dir}},  # OSCAR with cache
        ]
    else:
        dataset_options = [
            {'name': 'wikitext', 'kwargs': {'name': 'wikitext-103-raw-v1'}},  # ~1.8M tokens (smaller)
            {'name': 'squad', 'kwargs': {}},  # Question answering dataset
            {'name': 'imdb', 'kwargs': {}},  # Movie reviews
            {'name': 'ag_news', 'kwargs': {}},  # News articles
            {'name': 'yelp_polarity', 'kwargs': {}},  # Yelp reviews
            {'name': 'dbpedia_14', 'kwargs': {}},  # Wikipedia articles
            {'name': 'c4', 'kwargs': {'name': 'en'}},  # Common Crawl data, very large (last resort)
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
    
    def generate_cache_key(ds_name, ds_kwargs, max_samples, min_words, seed, total_loaded=0):
        """Generate cache key for dataset configuration"""
        import hashlib
        config_str = f"{ds_name}_{str(ds_kwargs)}_{max_samples}_{min_words}_{seed}_{total_loaded}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def load_from_cache(cache_key):
        """Load dataset from cache"""
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                print(f"Loaded {len(data)} samples from cache: {cache_file}")
                return data
            except Exception as e:
                print(f"Failed to load cache: {e}")
        return None
    
    def save_to_cache(data, cache_key):
        """Save dataset to cache"""
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            print(f"Saving dataset to cache: {cache_file}")
            return True
        except Exception as e:
            print(f"Failed to save cache: {e}")
            return False
    
    # Try each dataset option sequentially until we reach max_samples
    all_data = []
    total_samples = 0
    
    for ds_option in dataset_options:
        if total_samples >= max_samples:
            print(f"Reached target sample count ({max_samples:,}), stopping dataset loading")
            break
            
        ds_name = ds_option['name']
        ds_kwargs = ds_option['kwargs']
        
        remaining_samples = max_samples - total_samples
        print(f"Trying to load {ds_name} {ds_kwargs} (need {remaining_samples:,} more samples)...")
        
        try:
            # Generate cache key for this specific dataset and remaining samples
            cache_key = generate_cache_key(ds_name, ds_kwargs, remaining_samples, min_words, seed, total_loaded=total_samples)
            
            # Try to load from cache first
            cached_data = load_from_cache(cache_key)
            if cached_data:
                print(f"Loaded {len(cached_data)} samples from cache for {ds_name}")
                all_data.extend(cached_data)
                total_samples += len(cached_data)
                continue
            
            # Load dataset
            if streaming:
                try:
                    dataset = load_dataset(ds_name, **ds_kwargs, streaming=True, split='train')
                    print(f"Dataset {ds_name} loaded in streaming mode")
                except Exception as e:
                    print(f"Streaming mode failed for {ds_name}: {e}")
                    print(f"Trying non-streaming mode for {ds_name}...")
                    dataset = load_dataset(ds_name, **ds_kwargs, split='train')
                    print(f"Dataset {ds_name} loaded with {len(dataset)} total samples")
            else:
                dataset = load_dataset(ds_name, **ds_kwargs, split='train')
                print(f"Dataset {ds_name} loaded with {len(dataset)} total samples")
            
            # Process dataset
            data = []
            sample_count = 0
            
            if streaming:
                # For streaming datasets, we need to iterate
                for item in dataset:
                    if sample_count >= remaining_samples:
                        break
                    
                    text = extract_text(item)
                    if text and len(text.split()) >= min_words:
                        data.append({'text': text.split()})
                        sample_count += 1
                        
                        if sample_count % 10000 == 0:
                            print(f"Processed {sample_count} samples from {ds_name}...")
            else:
                # For non-streaming datasets, we can process all at once
                for item in dataset:
                    if sample_count >= remaining_samples:
                        break
                    
                    text = extract_text(item)
                    if text and len(text.split()) >= min_words:
                        data.append({'text': text.split()})
                        sample_count += 1
            
            if data:
                print(f"Successfully loaded {len(data)} samples from {ds_name}")
                
                # Save to cache
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
    """Load text data based on choice"""
    if dataset_choice.lower() == 'hf':
        return load_hf_dataset(max_samples=max_samples, streaming=streaming)
    else:
        return load_imdb_data()


def train_mlm_multi_gpu(model, train_loader, val_loader, device, tokenizer, num_epochs=3, learning_rate=5e-5, local_rank=0):
    """Train MLM model with multi-GPU support - v4 optimized for 24GB memory"""
    
    # Move model to device
    model = model.to(device)
    
    # Enable gradient checkpointing for memory efficiency (especially for large models)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print(f"Enabled gradient checkpointing for memory efficiency")
    
    # Wrap model with DDP if using multiple GPUs (check world_size instead of device_count)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler('cuda')
    
    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps // 10, 
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        # Progress bar only on main process
        if local_rank == 0:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        else:
            pbar = train_loader
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass with bfloat16 mixed precision (better for H200)
            with autocast('cuda', dtype=torch.bfloat16):
                loss = model(input_ids, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_train_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            if local_rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / num_batches
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                with autocast('cuda', dtype=torch.bfloat16):
                    loss = model(input_ids, labels)
                
                total_val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = total_val_loss / val_batches
        
        # Save history
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        
        # Print results (only on main process)
        if local_rank == 0:
            print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
            
            # Save model checkpoint
            save_dir = '.mlm_pretrained_v4'
            os.makedirs(save_dir, exist_ok=True)
            
            # Save model state dict
            if isinstance(model, DDP):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            
            torch.save(model_state_dict, os.path.join(save_dir, 'mlm_model.pth'))
            
            # Save tokenizer
            tokenizer_path = os.path.join(save_dir, 'tokenizer.json')
            with open(tokenizer_path, 'w') as f:
                json.dump(tokenizer.vocab, f, indent=2)
            
            # Save training history
            history_path = os.path.join(save_dir, 'mlm_training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            print(f'MLM model saved to {save_dir}')
    
    return history


def save_mlm_model(model, tokenizer, history, save_dir):
    """Save MLM model and related files"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state dict
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    torch.save(model_state_dict, os.path.join(save_dir, 'mlm_model.pth'))
    
    # Save tokenizer
    tokenizer_path = os.path.join(save_dir, 'tokenizer.json')
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer.vocab, f, indent=2)
    
    # Save training history
    history_path = os.path.join(save_dir, 'mlm_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'MLM model saved to {save_dir}')


def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Set environment variables to avoid warnings
        os.environ['OMP_NUM_THREADS'] = '1'
        
        # Check if CUDA is available and local_rank is valid
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if local_rank >= num_gpus:
                print(f"Warning: local_rank {local_rank} >= num_gpus {num_gpus}. Using local_rank 0 instead.")
                local_rank = 0
            
            # Set CUDA device BEFORE initializing process group
            torch.cuda.set_device(local_rank)
        else:
            print("Warning: CUDA not available. Using CPU.")
            local_rank = 0
        
        # Initialize process group with proper configuration to avoid warnings
        dist.init_process_group(
            backend='nccl', 
            rank=rank, 
            world_size=world_size,
            init_method='env://',
            timeout=datetime.timedelta(seconds=3600)  # 1 hour timeout
        )
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    """Main function for multi-GPU MLM training - v4 optimized for 24GB memory"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Multi-GPU MLM Pre-training v4 (24GB Memory Optimized)')
    parser.add_argument('--dataset', choices=['imdb', 'hf'], default='hf',
                       help='Dataset choice: imdb or hf')
    parser.add_argument('--streaming', type=str, default='true',
                       help='Streaming mode: true or false')
    parser.add_argument('--batch-size', type=int, default=96,
                       help='Batch size per GPU (optimized for 24GB memory)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-5,
                       help='Learning rate')
    parser.add_argument('--max-samples', type=str, default='10M',
                       help='Maximum number of samples to load (e.g., 500k, 5M, 10M, 50M)')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Set device - ensure it's valid
    if torch.cuda.is_available():
        # Use the local_rank that was already validated in setup_distributed
        device = torch.device(f'cuda:{local_rank}')
        
        # Get GPU memory info and adjust configuration accordingly
        gpu_memory = torch.cuda.get_device_properties(local_rank).total_memory / (1024**3)  # GB
        if rank == 0:
            print(f"GPU {local_rank} memory: {gpu_memory:.1f} GB")
        
        # Adjust batch size and model configuration based on available memory
        # Target: Stay under 24GB per GPU for all configurations
        if gpu_memory >= 100:  # 100GB+ GPU (like H200)
            # Use very conservative configuration to stay under 24GB per GPU
            adjusted_batch_size = min(args.batch_size, 16)  # Very conservative batch size, max 16
            model_scale = "large"
        elif gpu_memory >= 40:  # 40GB+ GPU (like A100)
            # Use medium configuration
            adjusted_batch_size = min(args.batch_size, 32)  # Conservative batch size, max 32
            model_scale = "medium"
        else:  # 24GB or less (like A10)
            # Use smaller configuration
            adjusted_batch_size = max(args.batch_size // 4, 8)  # Quarter batch size, min 8
            model_scale = "small"
        
        if rank == 0:
            print(f"Detected {gpu_memory:.1f}GB GPU, using {model_scale} configuration")
            print(f"Adjusted batch size: {adjusted_batch_size} (original: {args.batch_size})")
    else:
        device = torch.device('cpu')
        local_rank = 0
        adjusted_batch_size = args.batch_size
        model_scale = "cpu"
    
    # Parse max_samples argument
    max_samples = None
    if args.max_samples:
        max_samples_str = args.max_samples.upper()
        if max_samples_str.endswith('K'):
            max_samples = int(max_samples_str[:-1]) * 1000
        elif max_samples_str.endswith('M'):
            max_samples = int(max_samples_str[:-1]) * 1000000
        elif max_samples_str.isdigit():
            max_samples = int(max_samples_str)
        else:
            print(f"Unknown max_samples format: {args.max_samples}. Using default (10M)")
            max_samples = 10000000
    else:
        max_samples = 10000000  # Default 10M
    
    # Print setup info (only on main process)
    if rank == 0:
        print(f'Multi-GPU MLM Training v4 Setup (Dynamic Memory Optimization):')
        print(f'  - World Size: {world_size}')
        print(f'  - Local Rank: {local_rank}')
        print(f'  - Device: {device}')
        print(f'  - Dataset: {args.dataset}')
        print(f'  - Streaming: {args.streaming}')
        print(f'  - Original Batch Size per GPU: {args.batch_size}')
        print(f'  - Adjusted Batch Size per GPU: {adjusted_batch_size}')
        print(f'  - Total Batch Size: {adjusted_batch_size * world_size}')
        print(f'  - Epochs: {args.epochs}')
        print(f'  - Learning Rate: {args.lr}')
        print(f'  - Max Samples: {max_samples:,}')
        print(f'  - Model Scale: {model_scale}')
    
    try:
        # Load data (only on main process)
        if rank == 0:
            print(f'Loading dataset for MLM pre-training (choice: {args.dataset}, streaming: {args.streaming}, max_samples: {max_samples:,})...')
            train_data, test_data = load_text_data(args.dataset, streaming=args.streaming.lower() == 'true', max_samples=max_samples)
            
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
            
            # Build vocabulary from data with frequency filtering
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
            tokenizer = WordTokenizer(vocab=vocab_list, max_seq_len=128)  # Match MLMDataset max_length
            
            # Save tokenizer for other processes
            torch.save(tokenizer, '.temp_tokenizer.pth')
        
        # Wait for main process to finish data loading
        if world_size > 1:
            dist.barrier()
        
        # Load tokenizer on all processes
        if world_size > 1:
            tokenizer = torch.load('.temp_tokenizer.pth', weights_only=False)
        
        # Create datasets
        if rank == 0:
            train_dataset = MLMDataset(train_data, tokenizer)
            val_dataset = MLMDataset(val_data, tokenizer)
            
            # Save datasets for other processes
            torch.save(train_dataset, '.temp_train_dataset.pth')
            torch.save(val_dataset, '.temp_val_dataset.pth')
        
        # Wait for main process to finish dataset creation
        if world_size > 1:
            dist.barrier()
        
        # Load datasets on all processes
        if world_size > 1:
            train_dataset = torch.load('.temp_train_dataset.pth', weights_only=False)
            val_dataset = torch.load('.temp_val_dataset.pth', weights_only=False)
        
        # Create data loaders with distributed sampling
        if world_size > 1:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=adjusted_batch_size, 
                sampler=train_sampler,
                num_workers=6,  # Optimized for 24GB memory
                pin_memory=True,
                prefetch_factor=2  # Added prefetch_factor for better performance
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=adjusted_batch_size, 
                sampler=val_sampler,
                num_workers=6,  # Optimized for 24GB memory
                pin_memory=True,
                prefetch_factor=2  # Added prefetch_factor for better performance
            )
        else:
            train_loader = DataLoader(train_dataset, batch_size=adjusted_batch_size, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=2)
            val_loader = DataLoader(val_dataset, batch_size=adjusted_batch_size, shuffle=False, num_workers=6, pin_memory=True, prefetch_factor=2)
        
        # Initialize model based on dataset choice and GPU memory
        if args.dataset.lower() == 'hf':
            # Use configuration based on GPU memory
            if model_scale == "large":  # 100GB+ GPU
                if rank == 0:
                    print("Using large model configuration for high-memory GPU (100GB+) - very conservative for 24GB per GPU...")
                n_heads = 8
                n_embed = 128  # Reduced from 256
                n_layers = 4   # Reduced from 6
            elif model_scale == "medium":  # 40GB+ GPU
                if rank == 0:
                    print("Using medium model configuration for medium-memory GPU (40GB+)...")
                n_heads = 8
                n_embed = 128  # Reduced from 256
                n_layers = 6   # Reduced from 8
            else:  # 24GB or less
                if rank == 0:
                    print("Using small model configuration for low-memory GPU (24GB or less)...")
                n_heads = 8
                n_embed = 128  # Reduced from 256
                n_layers = 4   # Reduced from 8
            head_size = n_embed // n_heads
            num_epochs = args.epochs
            learning_rate = args.lr
        else:
            # Use configuration based on GPU memory for IMDB dataset too
            if model_scale == "large":  # 100GB+ GPU
                if rank == 0:
                    print("Using large model configuration for IMDB dataset (high-memory GPU) - very conservative for 24GB per GPU...")
                n_heads = 4
                n_embed = 64   # Reduced from 128
                n_layers = 4   # Reduced from 6
            elif model_scale == "medium":  # 40GB+ GPU
                if rank == 0:
                    print("Using medium model configuration for IMDB dataset (medium-memory GPU)...")
                n_heads = 4
                n_embed = 64   # Reduced from 128
                n_layers = 4   # Reduced from 6
            else:  # 24GB or less
                if rank == 0:
                    print("Using small model configuration for IMDB dataset (low-memory GPU)...")
                n_heads = 4
                n_embed = 64   # Reduced from 128
                n_layers = 4   # Reduced from 6
            head_size = n_embed // n_heads
            num_epochs = args.epochs
            learning_rate = args.lr
        
        # Ensure n_embed is divisible by n_heads
        if n_embed % n_heads != 0:
            n_embed = ((n_embed + n_heads - 1) // n_heads) * n_heads
            if rank == 0:
                print(f"Warning: n_embed adjusted to {n_embed} to be divisible by n_heads {n_heads}")
        
        # Limit vocabulary size to reduce memory usage
        max_vocab_size = 50000  # Limit vocabulary size to save memory
        if len(tokenizer.vocab) > max_vocab_size:
            if rank == 0:
                print(f"Warning: Vocabulary size {len(tokenizer.vocab)} exceeds limit {max_vocab_size}")
                print("This may cause memory issues. Consider reducing dataset size or using smaller vocabulary.")
        
        if rank == 0:
            print(f"Model configuration (v4 - Dynamic Memory Optimization):")
            print(f"  - Model Scale: {model_scale}")
            print(f"  - n_heads: {n_heads}")
            print(f"  - n_embed: {n_embed}")
            print(f"  - n_layers: {n_layers}")
            print(f"  - head_size: {head_size}")
            print(f"  - num_epochs: {num_epochs}")
            print(f"  - learning_rate: {learning_rate}")
            print(f"  - batch_size_per_gpu: {adjusted_batch_size}")
            print(f"  - total_batch_size: {adjusted_batch_size * world_size}")
            print(f"  - max_seq_len: 128 (reduced for memory)")
            print(f"  - max_vocab_size: {max_vocab_size}")
        
        model = MicroBertMLM(
            vocab_size=min(len(tokenizer.vocab), max_vocab_size),  # Limit vocabulary size
            n_layers=n_layers,
            n_heads=n_heads,
            n_embed=n_embed,
            max_seq_len=128  # Reduced from 256 to save memory
        )
        
        # Calculate model parameters (only on main process)
        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total model parameters: {total_params:,}")
            
            # Print model structure using hiq if available
            print("\n=== Model Structure ===")
            print_model(model)
            print("=== End Model Structure ===\n")
        
        # Check if model already exists
        save_dir = '.mlm_pretrained_v4'
        model_path = os.path.join(save_dir, 'mlm_model.pth')
        
        if os.path.exists(model_path) and rank == 0:
            print('Loading existing MLM model...')
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print('MLM model loaded successfully!')
            
            # Load training history for plotting
            history_path = os.path.join(save_dir, 'mlm_training_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
                print('Training history loaded for plotting.')
            else:
                history = None
                print('No training history found.')
        else:
            if rank == 0:
                print('Starting MLM pre-training v4 (24GB optimized)...')
            
            # Train model
            history = train_mlm_multi_gpu(
                model, train_loader, val_loader, device, tokenizer, 
                num_epochs=num_epochs, learning_rate=learning_rate, local_rank=local_rank
            )
            
            if rank == 0:
                print('MLM pre-training v4 completed!')
        
        # Cleanup temporary files
        if rank == 0:
            for temp_file in ['.temp_tokenizer.pth', '.temp_train_dataset.pth', '.temp_val_dataset.pth']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        # Plot results (only on main process)
        if rank == 0 and history:
            print('=== Plotting Training History ===')
            plot_mlm_results(history, save_path=os.path.join(save_dir, 'training_history.png'))
        
        # Test model (only on main process)
        if rank == 0:
            print('=== Testing MLM Model v4 ===')
            
            # Load model for testing
            if isinstance(model, DDP):
                test_model = model.module
            else:
                test_model = model
            
            test_model.eval()
            
            # Test sentences
            test_sentences = [
                "this movie is [MASK] fantastic",
                "the acting was [MASK] but the plot was confusing",
                "amazing [MASK] by all actors highly recommended"
            ]
            
            for i, sentence in enumerate(test_sentences, 1):
                print(f"{i}. Original: {sentence}")
                
                # Tokenize
                tokens = sentence.split()
                input_ids = []
                
                for token in tokens:
                    if token == '[MASK]':
                        input_ids.append(tokenizer.vocab['[MASK]'])
                    else:
                        input_ids.append(tokenizer.vocab.get(token, tokenizer.vocab['[UNK]']))
                
                # Pad or truncate
                if len(input_ids) > 256:
                    input_ids = input_ids[:256]
                else:
                    input_ids = input_ids + [tokenizer.vocab['[PAD]']] * (256 - len(input_ids))
                
                input_ids = torch.tensor([input_ids]).to(device)
                
                # Get predictions
                with torch.no_grad():
                    logits = test_model(input_ids)
                
                # Find [MASK] positions
                mask_positions = []
                for j, token_id in enumerate(input_ids[0]):
                    if token_id == tokenizer.vocab['[MASK]']:
                        mask_positions.append(j)
                
                # Get top predictions for each [MASK]
                for pos in mask_positions:
                    print(f"   [MASK] at position {pos}:")
                    
                    # Get logits for this position
                    pos_logits = logits[0, pos, :]
                    
                    # Get top 5 predictions
                    top_k = 5
                    top_probs, top_indices = torch.topk(F.softmax(pos_logits, dim=-1), top_k)
                    
                    for prob, idx in zip(top_probs, top_indices):
                        word = list(tokenizer.vocab.keys())[list(tokenizer.vocab.values()).index(idx.item())]
                        print(f"      {word}: prob={prob.item():.6f}")
        
    finally:
        # Cleanup distributed training
        cleanup_distributed()


if __name__ == '__main__':
    main()
