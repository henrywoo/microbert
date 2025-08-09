#!/usr/bin/env python3
"""
Quick test script for optimized MLM pre-training parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.amp import GradScaler
from microbert.model import MicroBERT
from microbert.tokenizer import WordTokenizer
import json
import os

class MicroBertMLM(torch.nn.Module):
    """
    MicroBERT model for Masked Language Modeling (MLM) pre-training
    """
    def __init__(self, vocab_size, n_layers=4, n_heads=4, n_embed=64, max_seq_len=128):
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
        # Create attention mask for the encoder
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

def quick_test():
    """Quick test with optimized parameters"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create a small vocabulary for testing
    vocab = ['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]'] + [f'word_{i}' for i in range(100)]
    tokenizer = WordTokenizer(vocab=vocab, max_seq_len=32)
    
    # Create model with optimized parameters
    model = MicroBertMLM(
        vocab_size=len(vocab),
        n_layers=4,
        n_heads=4,
        n_embed=64,
        max_seq_len=32
    ).to(device)
    
    print(f'Model created with {sum(p.numel() for p in model.parameters()):,} parameters')
    
    # Create dummy data
    batch_size = 8
    seq_len = 32
    input_ids = torch.randint(0, len(vocab), (batch_size, seq_len)).to(device)
    labels = torch.randint(0, len(vocab), (batch_size, seq_len)).to(device)
    
    # Test forward pass
    print('Testing forward pass...')
    with torch.no_grad():
        logits = model(input_ids)
        print(f'Output shape: {logits.shape}')
    
    # Test training step
    print('Testing training step...')
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = GradScaler('cuda')
    
    for step in range(5):
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_ids, labels)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        
        print(f'Step {step + 1}: Loss = {loss.item():.4f}')
    
    print('Quick test completed successfully!')
    print('Optimized parameters are working correctly.')

if __name__ == '__main__':
    quick_test()
