#!/usr/bin/env python3
"""
Test script to verify large dataset loading functionality
"""

import sys
import os
import time
import json
import hashlib

# Add current directory to path
sys.path.append('.')

def test_large_datasets():
    """Test loading large datasets"""
    print("🚀 Testing Large Dataset Loading")
    print("=" * 60)
    
    # Test parameters
    test_configs = [
        {"max_samples": 1000, "name": "Small test"},
        {"max_samples": 10000, "name": "Medium test"},
        {"max_samples": 100000, "name": "Large test"},
    ]
    
    for config in test_configs:
        max_samples = config["max_samples"]
        test_name = config["name"]
        
        print(f"\n📊 {test_name} - {max_samples:,} samples")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Try to import datasets library
            try:
                from datasets import load_dataset
            except ImportError:
                print("❌ datasets library not found. Please install with: pip install datasets")
                continue
            
            # Define dataset options (same as in v3)
            dataset_options = [
                {'name': 'openwebtext', 'kwargs': {}},  # ~8M documents, very large
                {'name': 'wikipedia', 'kwargs': {'name': '20220301.en'}},  # ~6M articles
                {'name': 'pile-cc', 'kwargs': {'name': 'pile-cc'}},  # Common Crawl data, very large
                {'name': 'bookcorpus', 'kwargs': {}},  # ~11K books
                {'name': 'wikitext', 'kwargs': {'name': 'wikitext-103-raw-v1'}},  # ~1.8M tokens (smaller)
            ]
            
            def extract_text(item):
                text_fields = ['text', 'content', 'sentence', 'passage', 'article']
                for field in text_fields:
                    if field in item and item[field]:
                        return item[field]
                return None
            
            def generate_cache_key(ds_name, ds_kwargs, max_samples, min_words, seed):
                config_str = f"{ds_name}_{str(ds_kwargs)}_{max_samples}_{min_words}_{seed}"
                return hashlib.md5(config_str.encode()).hexdigest()[:16]
            
            # Try each dataset option
            data = []
            for ds_option in dataset_options:
                ds_name = ds_option['name']
                ds_kwargs = ds_option['kwargs']
                
                print(f"  Trying {ds_name}...")
                
                try:
                    # Load dataset
                    dataset = load_dataset(ds_name, **ds_kwargs, streaming=True, split='train')
                    print(f"    ✓ {ds_name} loaded in streaming mode")
                    
                    # Process dataset
                    sample_count = 0
                    for item in dataset:
                        if sample_count >= max_samples:
                            break
                        
                        text = extract_text(item)
                        if text and len(text.split()) >= 5:  # min_words=5
                            data.append({'text': text.split()})
                            sample_count += 1
                            
                            if sample_count % 1000 == 0:
                                print(f"    Processed {sample_count} samples...")
                    
                    if data:
                        print(f"    ✓ Successfully loaded {len(data)} samples from {ds_name}")
                        break
                    else:
                        print(f"    ⚠ No valid samples found in {ds_name}")
                        
                except Exception as e:
                    print(f"    ❌ Failed to load {ds_name}: {e}")
                    continue
            
            end_time = time.time()
            duration = end_time - start_time
            
            if data:
                print(f"✅ Successfully loaded {len(data):,} samples")
                print(f"   Duration: {duration:.2f}s")
                print(f"   Rate: {len(data)/duration:.0f} samples/second")
                
                # Show sample info
                sample_text = ' '.join(data[0]['text'][:10])
                print(f"   Sample: {sample_text}...")
                
                # Calculate average sample length
                avg_length = sum(len(sample['text']) for sample in data[:100]) / min(100, len(data))
                print(f"   Average sample length: {avg_length:.1f} words")
            else:
                print("❌ Failed to load any dataset")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎯 Test completed!")

if __name__ == "__main__":
    test_large_datasets()
