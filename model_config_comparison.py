#!/usr/bin/env python3
"""
Compare different model configurations for MicroBERT MLM
"""

import torch
from microbert.model import MicroBERT

def calculate_model_params(n_layers, n_heads, n_embed, vocab_size=10005, max_seq_len=128):
    """Calculate the number of parameters for a given configuration"""
    model = MicroBERT(
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        n_embed=n_embed,
        max_seq_len=max_seq_len
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def test_model_forward(n_layers, n_heads, n_embed, vocab_size=10005, max_seq_len=128):
    """Test forward pass for a given configuration"""
    model = MicroBERT(
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        n_embed=n_embed,
        max_seq_len=max_seq_len
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        output = model(input_ids)
    
    return output.shape

def main():
    """Compare different model configurations"""
    print("=" * 80)
    print("MICROBERT MODEL CONFIGURATION COMPARISON")
    print("=" * 80)
    
    # Define configurations
    configs = [
        {
            'name': 'IMDB Small',
            'n_layers': 2,
            'n_heads': 2,
            'n_embed': 4,
            'description': 'Small model for IMDB dataset (~25K samples)'
        },
        {
            'name': 'HF Medium',
            'n_layers': 4,
            'n_heads': 4,
            'n_embed': 8,
            'description': 'Medium model for Hugging Face datasets (~500K samples)'
        },
        {
            'name': 'HF Large',
            'n_layers': 6,
            'n_heads': 8,
            'n_embed': 16,
            'description': 'Large model for large Hugging Face datasets (~1M+ samples)'
        },
        {
            'name': 'HF Extra Large',
            'n_layers': 8,
            'n_heads': 12,
            'n_embed': 24,
            'description': 'Extra large model for very large datasets'
        }
    ]
    
    print("\nConfiguration Details:")
    print("-" * 80)
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  Description: {config['description']}")
        print(f"  Layers: {config['n_layers']}")
        print(f"  Heads: {config['n_heads']}")
        print(f"  Embedding: {config['n_embed']}")
        print(f"  Head Size: {config['n_embed'] // config['n_heads']}")
        
        # Calculate parameters
        total_params, trainable_params = calculate_model_params(
            config['n_layers'], 
            config['n_heads'], 
            config['n_embed']
        )
        
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        
        # Test forward pass
        try:
            output_shape = test_model_forward(
                config['n_layers'], 
                config['n_heads'], 
                config['n_embed']
            )
            print(f"  Output Shape: {output_shape}")
            print(f"  ✓ Forward pass successful")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1. For IMDB Dataset (~25K samples):")
    print("   - Use 'IMDB Small' configuration")
    print("   - ~41K parameters")
    print("   - Fast training (~5 minutes)")
    print("   - Good for learning and testing")
    
    print("\n2. For Hugging Face Datasets (~500K samples):")
    print("   - Use 'HF Medium' configuration")
    print("   - ~84K parameters")
    print("   - Moderate training time (~30 minutes)")
    print("   - Better semantic understanding")
    
    print("\n3. For Large Datasets (~1M+ samples):")
    print("   - Use 'HF Large' configuration")
    print("   - ~400K parameters")
    print("   - Longer training time (~2 hours)")
    print("   - Much better performance")
    
    print("\n4. For Very Large Datasets:")
    print("   - Use 'HF Extra Large' configuration")
    print("   - ~1.2M parameters")
    print("   - Long training time (~6 hours)")
    print("   - Best performance but requires more resources")
    
    print("\n" + "=" * 80)
    print("MEMORY USAGE ESTIMATES")
    print("=" * 80)
    
    print("\nGPU Memory Usage (batch_size=16, seq_len=128):")
    print("  IMDB Small:     ~50MB")
    print("  HF Medium:      ~100MB")
    print("  HF Large:       ~400MB")
    print("  HF Extra Large: ~1GB")
    
    print("\nNote: Actual memory usage may vary based on:")
    print("  - Batch size")
    print("  - Sequence length")
    print("  - Gradient accumulation")
    print("  - Mixed precision training")

if __name__ == '__main__':
    main()
