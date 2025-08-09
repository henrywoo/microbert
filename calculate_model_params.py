#!/usr/bin/env python3
"""
Calculate model parameters for different MicroBERT configurations
"""

def calculate_params(vocab_size, n_layers, n_heads, n_embed, max_seq_len=128):
    """Calculate total parameters for a given configuration"""
    
    # Word embeddings
    word_emb_params = vocab_size * n_embed
    
    # Position embeddings  
    pos_emb_params = max_seq_len * n_embed
    
    # Layer normalization parameters (2 per layer)
    ln_params = n_layers * 2 * n_embed * 2  # weight + bias
    
    # Self-attention parameters per layer
    # Query, Key, Value projections: 3 * n_embed * n_embed
    # Output projection: n_embed * n_embed
    # Total per layer: 4 * n_embed * n_embed
    attn_params_per_layer = 4 * n_embed * n_embed
    
    # Feed-forward parameters per layer
    # First layer: n_embed * (4 * n_embed)
    # Second layer: (4 * n_embed) * n_embed
    ff_params_per_layer = n_embed * (4 * n_embed) + (4 * n_embed) * n_embed
    
    # Total attention + FF per layer
    layer_params = attn_params_per_layer + ff_params_per_layer
    
    # Pooler
    pooler_params = n_embed * n_embed + n_embed
    
    # MLM head
    mlm_head_params = n_embed * vocab_size + vocab_size
    
    # Total parameters
    total_params = (word_emb_params + pos_emb_params + ln_params + 
                   n_layers * layer_params + pooler_params + mlm_head_params)
    
    return {
        'word_embeddings': word_emb_params,
        'position_embeddings': pos_emb_params,
        'layer_norms': ln_params,
        'attention_layers': n_layers * attn_params_per_layer,
        'feedforward_layers': n_layers * ff_params_per_layer,
        'pooler': pooler_params,
        'mlm_head': mlm_head_params,
        'total': total_params
    }

def main():
    vocab_size = 251642  # IMDB dataset vocabulary size
    
    print("=== MicroBERT Parameter Calculation ===\n")
    
    # Original configuration (too large)
    print("1. Original Large Config (4L/4H/64D):")
    orig_params = calculate_params(vocab_size, 4, 4, 64)
    print(f"   Total: {orig_params['total']:,} parameters")
    print(f"   Word embeddings: {orig_params['word_embeddings']:,}")
    print(f"   MLM head: {orig_params['mlm_head']:,}")
    print()
    
    # New micro configuration
    print("2. New Micro Config (2L/2H/16D):")
    micro_params = calculate_params(vocab_size, 2, 2, 16)
    print(f"   Total: {micro_params['total']:,} parameters")
    print(f"   Word embeddings: {micro_params['word_embeddings']:,}")
    print(f"   MLM head: {micro_params['mlm_head']:,}")
    print()
    
    # Even smaller configuration
    print("3. Ultra Micro Config (1L/1H/8D):")
    ultra_params = calculate_params(vocab_size, 1, 1, 8)
    print(f"   Total: {ultra_params['total']:,} parameters")
    print(f"   Word embeddings: {ultra_params['word_embeddings']:,}")
    print(f"   MLM head: {ultra_params['mlm_head']:,}")
    print()
    
    # Calculate reduction
    reduction = (orig_params['total'] - micro_params['total']) / orig_params['total'] * 100
    print(f"=== Summary ===")
    print(f"Parameter reduction: {reduction:.1f}%")
    print(f"Memory savings: ~{(orig_params['total'] - micro_params['total']) * 4 / 1024 / 1024:.1f} MB")
    print()
    
    print("Recommendation: Use the new micro config (2L/2H/16D)")
    print("It provides a good balance between performance and model size.")

if __name__ == '__main__':
    main()
