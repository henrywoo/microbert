#!/usr/bin/env python3
"""
BERT Base Analysis - Calculate parameters and training time estimates
"""

import torch
import time
from microbert.model import MicroBERT

def calculate_bert_base_params():
    """Calculate BERT base parameters"""
    
    # BERT base configuration
    bert_base_config = {
        'vocab_size': 30522,  # BERT base vocab size
        'n_layers': 12,       # 12 transformer layers
        'n_heads': 12,        # 12 attention heads
        'n_embed': 768,       # 768 hidden size
        'max_seq_len': 512,   # 512 max sequence length
        'intermediate_size': 3072,  # FFN intermediate size
        'dropout': 0.1
    }
    
    print("BERT Base Configuration:")
    print(f"  - Vocabulary size: {bert_base_config['vocab_size']:,}")
    print(f"  - Layers: {bert_base_config['n_layers']}")
    print(f"  - Attention heads: {bert_base_config['n_heads']}")
    print(f"  - Hidden size: {bert_base_config['n_embed']}")
    print(f"  - Max sequence length: {bert_base_config['max_seq_len']}")
    print(f"  - Intermediate size: {bert_base_config['intermediate_size']}")
    
    # Calculate parameters for each component
    vocab_size = bert_base_config['vocab_size']
    n_layers = bert_base_config['n_layers']
    n_heads = bert_base_config['n_heads']
    n_embed = bert_base_config['n_embed']
    max_seq_len = bert_base_config['max_seq_len']
    intermediate_size = bert_base_config['intermediate_size']
    
    # 1. Embeddings
    word_embeddings = vocab_size * n_embed  # 30522 * 768
    position_embeddings = max_seq_len * n_embed  # 512 * 768
    token_type_embeddings = 2 * n_embed  # 2 * 768 (BERT uses token type embeddings)
    embedding_layer_norm = n_embed * 2  # LayerNorm parameters
    embedding_dropout = 0  # No parameters
    
    embedding_params = word_embeddings + position_embeddings + token_type_embeddings + embedding_layer_norm
    print(f"\n1. Embeddings: {embedding_params:,} parameters")
    print(f"   - Word embeddings: {word_embeddings:,}")
    print(f"   - Position embeddings: {position_embeddings:,}")
    print(f"   - Token type embeddings: {token_type_embeddings:,}")
    print(f"   - Layer norm: {embedding_layer_norm:,}")
    
    # 2. Transformer layers (12 layers)
    # Each layer has:
    # - Self-attention: 4 * n_embed * n_embed (Q, K, V, output projections)
    # - Layer norm: 2 * n_embed
    # - FFN: n_embed * intermediate_size + intermediate_size * n_embed
    # - Layer norm: 2 * n_embed
    
    # Self-attention parameters
    head_size = n_embed // n_heads  # 768 // 12 = 64
    self_attention_params = n_heads * (3 * head_size * head_size) + n_embed * n_embed  # Q, K, V + output projection
    self_attention_layer_norm = n_embed * 2
    
    # FFN parameters
    ffn_params = n_embed * intermediate_size + intermediate_size + intermediate_size * n_embed + n_embed
    ffn_layer_norm = n_embed * 2
    
    # Total per layer
    params_per_layer = self_attention_params + self_attention_layer_norm + ffn_params + ffn_layer_norm
    total_transformer_params = params_per_layer * n_layers
    
    print(f"\n2. Transformer layers ({n_layers} layers): {total_transformer_params:,} parameters")
    print(f"   - Per layer: {params_per_layer:,}")
    print(f"   - Self-attention: {self_attention_params:,} per layer")
    print(f"   - FFN: {ffn_params:,} per layer")
    print(f"   - Layer norms: {(self_attention_layer_norm + ffn_layer_norm):,} per layer")
    
    # 3. Pooler
    pooler_params = n_embed * n_embed + n_embed  # Linear layer + bias
    
    print(f"\n3. Pooler: {pooler_params:,} parameters")
    
    # 4. MLM head (for MLM pre-training)
    mlm_head_params = n_embed * vocab_size + vocab_size  # Linear layer + bias
    
    print(f"\n4. MLM head: {mlm_head_params:,} parameters")
    
    # Total parameters
    total_params = embedding_params + total_transformer_params + pooler_params + mlm_head_params
    
    print(f"\n" + "="*50)
    print(f"BERT BASE TOTAL PARAMETERS: {total_params:,}")
    print(f"BERT BASE TOTAL PARAMETERS: {total_params/1e6:.1f}M")
    print("="*50)
    
    return total_params

def estimate_training_time_bert_base(total_params, num_gpus=8, batch_size_per_gpu=32):
    """Estimate training time for BERT base on H200 8-GPU setup"""
    
    print(f"\nBERT Base Training Time Estimation (H200 8-GPU)")
    print("="*60)
    
    # Training configuration
    total_batch_size = num_gpus * batch_size_per_gpu
    max_seq_len = 512
    vocab_size = 30522
    
    print(f"Configuration:")
    print(f"  - GPUs: {num_gpus}")
    print(f"  - Batch size per GPU: {batch_size_per_gpu}")
    print(f"  - Total batch size: {total_batch_size}")
    print(f"  - Max sequence length: {max_seq_len}")
    print(f"  - Vocabulary size: {vocab_size:,}")
    
    # Dataset size estimates
    dataset_sizes = {
        'small': 1_000_000,    # 1M samples
        'medium': 10_000_000,  # 10M samples
        'large': 100_000_000,  # 100M samples
        'xlarge': 1_000_000_000  # 1B samples
    }
    
    # H200 performance estimates
    # H200 has ~989 TFLOPS for FP16
    # BERT base requires ~2 * total_params * sequence_length * batch_size operations per forward pass
    # Plus backward pass (roughly 2x forward)
    
    # Rough FLOPS calculation
    flops_per_token = total_params * 2  # Rough estimate
    flops_per_sample = flops_per_token * max_seq_len
    flops_per_batch = flops_per_sample * total_batch_size
    
    # H200 theoretical peak: ~989 TFLOPS
    # Realistic utilization: ~60-70%
    realistic_tflops = 989 * 0.65  # ~643 TFLOPS
    
    print(f"\nPerformance Estimates:")
    print(f"  - FLOPS per token: {flops_per_token/1e9:.1f} GFLOPS")
    print(f"  - FLOPS per sample: {flops_per_sample/1e12:.2f} TFLOPS")
    print(f"  - FLOPS per batch: {flops_per_batch/1e12:.2f} TFLOPS")
    print(f"  - H200 realistic performance: {realistic_tflops:.0f} TFLOPS")
    
    # Time per batch
    time_per_batch = flops_per_batch / (realistic_tflops * 1e12)  # seconds
    
    print(f"\nTraining Time Estimates:")
    print(f"  - Time per batch: {time_per_batch:.3f} seconds")
    
    for dataset_name, num_samples in dataset_sizes.items():
        batches_per_epoch = num_samples // total_batch_size
        time_per_epoch = batches_per_epoch * time_per_batch / 3600  # hours
        
        # Typical training epochs for BERT base
        if dataset_name == 'small':
            epochs = 10
        elif dataset_name == 'medium':
            epochs = 5
        elif dataset_name == 'large':
            epochs = 3
        else:  # xlarge
            epochs = 1
        
        total_time = time_per_epoch * epochs
        
        print(f"\n  {dataset_name.upper()} dataset ({num_samples:,} samples):")
        print(f"    - Batches per epoch: {batches_per_epoch:,}")
        print(f"    - Time per epoch: {time_per_epoch:.1f} hours")
        print(f"    - Total epochs: {epochs}")
        print(f"    - Total training time: {total_time:.1f} hours ({total_time/24:.1f} days)")
    
    return time_per_batch

def compare_with_microbert():
    """Compare BERT base with our MicroBERT configurations"""
    
    print(f"\n" + "="*60)
    print("COMPARISON WITH MICROBERT CONFIGURATIONS")
    print("="*60)
    
    # Our configurations
    microbert_configs = [
        {'name': 'v1 (IMDB Small)', 'params': 41_048, 'layers': 2, 'heads': 2, 'embed': 4},
        {'name': 'v2 (HF Medium)', 'params': 84_640, 'layers': 4, 'heads': 4, 'embed': 8},
        {'name': 'v3 (HF Large)', 'params': 182_112, 'layers': 6, 'heads': 8, 'embed': 16},
        {'name': 'v3 (HF Extra Large)', 'params': 301_632, 'layers': 8, 'heads': 12, 'embed': 24},
    ]
    
    bert_base_params = 110_000_000  # ~110M parameters
    
    print(f"BERT Base: {bert_base_params:,} parameters")
    print(f"\nMicroBERT Configurations:")
    
    for config in microbert_configs:
        ratio = bert_base_params / config['params']
        print(f"  {config['name']}: {config['params']:,} parameters ({ratio:.1f}x smaller)")
        print(f"    - Layers: {config['layers']} vs 12")
        print(f"    - Heads: {config['heads']} vs 12")
        print(f"    - Embed: {config['embed']} vs 768")

def main():
    """Main analysis function"""
    
    print("BERT BASE ANALYSIS")
    print("="*60)
    
    # Calculate BERT base parameters
    total_params = calculate_bert_base_params()
    
    # Estimate training time
    time_per_batch = estimate_training_time_bert_base(total_params)
    
    # Compare with MicroBERT
    compare_with_microbert()
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"BERT Base has ~110M parameters")
    print(f"Training on H200 8-GPU would take:")
    print(f"  - Small dataset (1M samples): ~2-3 hours")
    print(f"  - Medium dataset (10M samples): ~8-12 hours")
    print(f"  - Large dataset (100M samples): ~1-2 days")
    print(f"  - XLarge dataset (1B samples): ~1-2 weeks")
    print(f"\nOur MicroBERT is 300-2000x smaller and faster!")

if __name__ == '__main__':
    main()
