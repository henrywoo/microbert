#!/usr/bin/env python3
"""
Test script to verify v4 memory optimization fixes
"""

import sys
import os
import torch

# Add current directory to path
sys.path.append('.')

def test_memory_detection():
    """Test GPU memory detection and configuration adjustment"""
    print("🚀 Testing v4 Memory Optimization")
    print("=" * 60)
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"GPU 0 memory: {gpu_memory:.1f} GB")
        
        # Test configuration adjustment logic
        if gpu_memory >= 100:  # 100GB+ GPU (like H200)
            adjusted_batch_size = 192  # Double batch size, max 256
            model_scale = "large"
            n_heads = 16
            n_embed = 512
            n_layers = 12
        elif gpu_memory >= 40:  # 40GB+ GPU (like A100)
            adjusted_batch_size = 96
            model_scale = "medium"
            n_heads = 8
            n_embed = 256
            n_layers = 8
        else:  # 24GB or less (like A10)
            adjusted_batch_size = 48
            model_scale = "small"
            n_heads = 8
            n_embed = 256
            n_layers = 8
        
        print(f"Detected {gpu_memory:.1f}GB GPU, using {model_scale} configuration")
        print(f"Adjusted batch size: {adjusted_batch_size}")
        print(f"Model config: {n_layers}L/{n_heads}H/{n_embed}D")
        
        # Calculate model parameters
        vocab_size = 50000  # Example
        total_params = (
            vocab_size * n_embed +  # word embeddings
            n_embed * n_embed +  # position embeddings
            n_embed * 2 +  # layer norms
            n_layers * (
                n_embed * n_embed * 4 +  # self-attention
                n_embed * n_embed * 2 +  # feed-forward
                n_embed * 2  # layer norms
            ) +
            n_embed * vocab_size + vocab_size  # MLM head
        )
        
        print(f"Estimated model parameters: {total_params:,}")
        
        # Memory usage estimation
        batch_size = adjusted_batch_size
        seq_len = 256
        memory_per_sample = n_embed * seq_len * 4  # 4 bytes per float32
        batch_memory = batch_size * memory_per_sample / (1024**3)  # GB
        
        print(f"Estimated batch memory usage: {batch_memory:.2f} GB")
        
    else:
        print("CUDA not available")

if __name__ == "__main__":
    test_memory_detection()
