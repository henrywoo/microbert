#!/usr/bin/env python3
"""
Quick test script for v4 with small sample sizes to avoid downloading large datasets
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

def quick_test_v4():
    """Quick test of v4 with small sample sizes"""
    print("🚀 Quick Test v4 (Small Sample Sizes)")
    print("=" * 60)
    
    # Test parameters - small sample sizes to avoid disk space issues
    test_configs = [
        {"max_samples": 1000, "name": "Tiny test"},
        {"max_samples": 5000, "name": "Small test"},
        {"max_samples": 10000, "name": "Medium test"},
    ]
    
    for config in test_configs:
        max_samples = config["max_samples"]
        test_name = config["name"]
        
        print(f"\n📊 {test_name} - {max_samples:,} samples")
        print("-" * 40)
        
        try:
            # Import the function from v4
            from mlm_pretrain_v4 import load_hf_dataset
            
            print(f"Loading dataset with max_samples={max_samples:,}...")
            data, _ = load_hf_dataset(
                max_samples=max_samples,
                min_words=5,
                seed=42,
                streaming=True,
                cache_dir=".quick_test_cache"
            )
            
            print(f"✅ Successfully loaded {len(data):,} samples")
            
            if data:
                # Show sample info
                sample_text = ' '.join(data[0]['text'][:10])
                print(f"   Sample: {sample_text}...")
                
                # Calculate average sample length
                avg_length = sum(len(sample['text']) for sample in data[:100]) / min(100, len(data))
                print(f"   Average sample length: {avg_length:.1f} words")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎯 Quick test completed!")

if __name__ == "__main__":
    quick_test_v4()
