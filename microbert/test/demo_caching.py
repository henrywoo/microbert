#!/usr/bin/env python3
"""
Demo script to show caching functionality
"""

import sys
import os
import time

# Add current directory to path
sys.path.append('.')

from mlm_pretrain import load_hf_dataset

def demo_caching():
    """Demonstrate caching functionality"""
    print("🚀 MicroBERT Dataset Caching Demo")
    print("=" * 60)
    
    # Demo parameters
    max_samples = 50  # Small number for quick demo
    min_words = 5
    seed = 42
    
    print(f"Demo parameters:")
    print(f"  - max_samples: {max_samples}")
    print(f"  - min_words: {min_words}")
    print(f"  - seed: {seed}")
    print()
    
    # First run - should be slow (downloading and processing)
    print("📥 First run - downloading and processing...")
    start_time = time.time()
    
    try:
        data1, _ = load_hf_dataset(
            max_samples=max_samples,
            min_words=min_words,
            seed=seed,
            streaming=True
        )
        
        first_run_time = time.time() - start_time
        print(f"✅ First run completed in {first_run_time:.2f}s")
        print(f"   Loaded {len(data1)} samples")
        
        if data1:
            sample_text = ' '.join(data1[0]['text'][:8])
            print(f"   Sample: {sample_text}...")
        
    except Exception as e:
        print(f"❌ First run failed: {e}")
        return
    
    print()
    
    # Second run - should be fast (loading from cache)
    print("⚡ Second run - loading from cache...")
    start_time = time.time()
    
    try:
        data2, _ = load_hf_dataset(
            max_samples=max_samples,
            min_words=min_words,
            seed=seed,
            streaming=True
        )
        
        second_run_time = time.time() - start_time
        print(f"✅ Second run completed in {second_run_time:.2f}s")
        print(f"   Loaded {len(data2)} samples")
        
        if data2:
            sample_text = ' '.join(data2[0]['text'][:8])
            print(f"   Sample: {sample_text}...")
        
    except Exception as e:
        print(f"❌ Second run failed: {e}")
        return
    
    print()
    
    # Show speed improvement
    if first_run_time > 0:
        speedup = first_run_time / second_run_time
        print("📊 Performance Comparison:")
        print(f"   First run:  {first_run_time:.2f}s")
        print(f"   Second run: {second_run_time:.2f}s")
        print(f"   Speedup:    {speedup:.1f}x faster!")
    
    print()
    
    # Verify data consistency
    print("🔍 Data Consistency Check:")
    if len(data1) == len(data2):
        print(f"   ✅ Sample count matches: {len(data1)}")
        
        # Check if first few samples are identical
        identical = True
        for i in range(min(3, len(data1))):
            if data1[i]['text'] != data2[i]['text']:
                identical = False
                break
        
        if identical:
            print("   ✅ Data content is identical")
        else:
            print("   ❌ Data content differs")
    else:
        print(f"   ❌ Sample count differs: {len(data1)} vs {len(data2)}")
    
    print()
    
    # Show cache information
    print("📁 Cache Information:")
    cache_dir = ".dataset_cache"
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        print(f"   Found {len(cache_files)} cache files:")
        
        total_size = 0
        for cache_file in cache_files:
            file_path = os.path.join(cache_dir, cache_file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            print(f"   - {cache_file}: {file_size:,} bytes")
        
        print(f"   Total cache size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    else:
        print("   No cache directory found")
    
    print()
    print("=" * 60)
    print("🎉 Demo completed!")
    print()
    print("💡 Tips:")
    print("   - Use 'python cache_manager.py info' to view cache details")
    print("   - Use 'python cache_manager.py clear' to clear cache")
    print("   - Cache is automatically managed - no manual intervention needed")

if __name__ == '__main__':
    demo_caching()
