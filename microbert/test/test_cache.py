#!/usr/bin/env python3
"""
Test script to verify dataset caching functionality
"""

import sys
import os
import time

# Add current directory to path
sys.path.append('.')

from mlm_pretrain import load_hf_dataset

def test_caching():
    """Test caching functionality"""
    print("Testing dataset caching functionality...")
    print("=" * 60)
    
    # Test parameters
    max_samples = 100  # Small number for quick testing
    min_words = 5
    seed = 42
    cache_dir = ".test_cache"
    
    print(f"Test parameters:")
    print(f"  - max_samples: {max_samples}")
    print(f"  - min_words: {min_words}")
    print(f"  - seed: {seed}")
    print(f"  - cache_dir: {cache_dir}")
    print()
    
    # First run - should download and cache
    print("1. First run - downloading and caching...")
    start_time = time.time()
    try:
        data1, _ = load_hf_dataset(
            max_samples=max_samples,
            min_words=min_words,
            seed=seed,
            streaming=True,
            cache_dir=cache_dir
        )
        end_time = time.time()
        print(f"✓ First run completed in {end_time - start_time:.2f}s")
        print(f"   Loaded {len(data1)} samples")
        
        if data1:
            print(f"   Sample: {' '.join(data1[0]['text'][:10])}...")
            
    except Exception as e:
        print(f"✗ First run failed: {e}")
        return
    
    print()
    
    # Second run - should load from cache
    print("2. Second run - loading from cache...")
    start_time = time.time()
    try:
        data2, _ = load_hf_dataset(
            max_samples=max_samples,
            min_words=min_words,
            seed=seed,
            streaming=True,
            cache_dir=cache_dir
        )
        end_time = time.time()
        print(f"✓ Second run completed in {end_time - start_time:.2f}s")
        print(f"   Loaded {len(data2)} samples")
        
        if data2:
            print(f"   Sample: {' '.join(data2[0]['text'][:10])}...")
            
    except Exception as e:
        print(f"✗ Second run failed: {e}")
        return
    
    print()
    
    # Verify data consistency
    print("3. Verifying data consistency...")
    if len(data1) == len(data2):
        print(f"✓ Sample count matches: {len(data1)}")
        
        # Check if first few samples are identical
        identical = True
        for i in range(min(3, len(data1))):
            if data1[i]['text'] != data2[i]['text']:
                identical = False
                break
        
        if identical:
            print("✓ Data content is identical")
        else:
            print("✗ Data content differs")
    else:
        print(f"✗ Sample count differs: {len(data1)} vs {len(data2)}")
    
    print()
    
    # Check cache files
    print("4. Checking cache files...")
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        print(f"✓ Found {len(cache_files)} cache files:")
        for cache_file in cache_files:
            file_path = os.path.join(cache_dir, cache_file)
            file_size = os.path.getsize(file_path)
            print(f"   - {cache_file}: {file_size:,} bytes")
    else:
        print("✗ Cache directory not found")
    
    print()
    
    # Clean up test cache
    print("5. Cleaning up test cache...")
    try:
        import shutil
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("✓ Test cache cleaned up")
        else:
            print("No test cache to clean")
    except Exception as e:
        print(f"✗ Failed to clean cache: {e}")
    
    print("\n" + "=" * 60)
    print("Caching test completed!")

def show_cache_info():
    """Show information about existing cache"""
    print("\n" + "=" * 60)
    print("CACHE INFORMATION")
    print("=" * 60)
    
    cache_dir = ".dataset_cache"
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        print(f"Found {len(cache_files)} cached datasets:")
        
        total_size = 0
        for cache_file in cache_files:
            file_path = os.path.join(cache_dir, cache_file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            
            # Try to get sample count
            try:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                sample_count = len(data)
                print(f"  - {cache_file}: {sample_count:,} samples, {file_size:,} bytes")
            except:
                print(f"  - {cache_file}: {file_size:,} bytes (could not read)")
        
        print(f"\nTotal cache size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    else:
        print("No cache directory found")

if __name__ == '__main__':
    test_caching()
    show_cache_info()
