#!/usr/bin/env python3
"""
Script to check actual sample count and file size
"""

import sys
import os
import json
import glob

def check_sample_count():
    """Check actual sample count and file size"""
    print("🔍 Checking Sample Count and File Size")
    print("=" * 60)
    
    # Check cache directory
    cache_dir = ".dataset_cache"
    if not os.path.exists(cache_dir):
        print(f"❌ Cache directory {cache_dir} not found")
        return
    
    # Find all cache files
    cache_files = glob.glob(os.path.join(cache_dir, "*.json"))
    
    if not cache_files:
        print(f"❌ No cache files found in {cache_dir}")
        return
    
    print(f"📁 Found {len(cache_files)} cache files:")
    print("-" * 40)
    
    total_samples = 0
    total_size = 0
    
    for cache_file in cache_files:
        try:
            # Get file size
            file_size = os.path.getsize(cache_file)
            file_size_mb = file_size / (1024 * 1024)
            
            # Load and count samples
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            sample_count = len(data)
            total_samples += sample_count
            total_size += file_size
            
            print(f"📄 {os.path.basename(cache_file)}:")
            print(f"   - Samples: {sample_count:,}")
            print(f"   - Size: {file_size_mb:.1f} MB")
            
            # Show sample info
            if data:
                sample_text = ' '.join(data[0]['text'][:10])
                print(f"   - Sample: {sample_text}...")
                
                # Calculate average sample length
                avg_length = sum(len(sample['text']) for sample in data[:100]) / min(100, len(data))
                print(f"   - Avg length: {avg_length:.1f} words")
            
            print()
            
        except Exception as e:
            print(f"❌ Error reading {cache_file}: {e}")
    
    print("📊 Summary:")
    print("-" * 40)
    print(f"Total samples: {total_samples:,}")
    print(f"Total size: {total_size / (1024 * 1024):.1f} MB")
    
    if total_samples > 0:
        avg_size_per_sample = total_size / total_samples
        print(f"Average size per sample: {avg_size_per_sample:.1f} bytes")
        
        # Estimate for 10M samples
        estimated_size_10m = total_samples * avg_size_per_sample * (10_000_000 / total_samples)
        print(f"Estimated size for 10M samples: {estimated_size_10m / (1024 * 1024 * 1024):.1f} GB")

if __name__ == "__main__":
    check_sample_count()
