#!/usr/bin/env python3
"""
Test script to compare IMDB vs Hugging Face dataset performance
"""

import sys
import os
import time

# Add current directory to path
sys.path.append('.')

from mlm_pretrain import main

def test_dataset_comparison():
    """Compare different datasets"""
    print("Testing dataset comparison...")
    print("=" * 60)
    
    datasets_to_test = [
        ('imdb', True, "IMDB with streaming"),
        ('hf', True, "Hugging Face with streaming"),
        ('hf', False, "Hugging Face with local download")
    ]
    
    for dataset_choice, streaming, description in datasets_to_test:
        print(f"\n{'='*20} Testing {description} {'='*20}")
        
        start_time = time.time()
        try:
            # Run with limited samples for quick testing
            from mlm_pretrain import load_hf_dataset, load_imdb_data
            
            if dataset_choice == 'imdb':
                data, _ = load_imdb_data()
            else:
                data, _ = load_hf_dataset(max_samples=1000, streaming=streaming)
            
            end_time = time.time()
            print(f"✓ Successfully loaded {len(data)} samples in {end_time - start_time:.2f}s")
            
            if data:
                # Show sample data
                sample_text = ' '.join(data[0]['text'][:10])
                print(f"   Sample: {sample_text}...")
                
                # Show vocabulary stats
                from collections import Counter
                word_counts = Counter()
                for item in data[:100]:  # Check first 100 samples
                    word_counts.update(item['text'])
                
                print(f"   Unique words in first 100 samples: {len(word_counts)}")
                print(f"   Most common words: {[w for w, _ in word_counts.most_common(5)]}")
                
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("Dataset comparison completed!")

if __name__ == '__main__':
    test_dataset_comparison()
