#!/usr/bin/env python3
"""
Cache management utility for MicroBERT dataset caching
"""

import os
import json
import shutil
import argparse
import hashlib
from pathlib import Path

def generate_cache_key(ds_name, ds_kwargs, max_samples, min_words, seed):
    """Generate cache key for dataset configuration - same logic as in training script"""
    config_str = f"{ds_name}_{str(ds_kwargs)}_{max_samples}_{min_words}_{seed}"
    return hashlib.md5(config_str.encode()).hexdigest()[:16]

def show_cache_info(cache_dir=".dataset_cache"):
    """Show information about cached datasets"""
    print("=" * 60)
    print("DATASET CACHE INFORMATION")
    print("=" * 60)
    
    if not os.path.exists(cache_dir):
        print("No cache directory found.")
        return
    
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
    
    if not cache_files:
        print("No cached datasets found.")
        return
    
    print(f"Found {len(cache_files)} cached datasets:")
    print()
    
    total_size = 0
    total_samples = 0
    
    for cache_file in sorted(cache_files):
        file_path = os.path.join(cache_dir, cache_file)
        file_size = os.path.getsize(file_path)
        total_size += file_size
        
        # Try to get sample count and dataset info
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sample_count = len(data)
            total_samples += sample_count
            
            # Try to extract dataset info from first sample
            if data and 'text' in data[0]:
                sample_text = ' '.join(data[0]['text'][:5])
                print(f"  📄 {cache_file}")
                print(f"     Samples: {sample_count:,}")
                print(f"     Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
                print(f"     Sample: {sample_text}...")
            else:
                print(f"  📄 {cache_file}")
                print(f"     Samples: {sample_count:,}")
                print(f"     Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        except Exception as e:
            print(f"  📄 {cache_file}")
            print(f"     Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            print(f"     Error reading: {e}")
        
        print()
    
    print("=" * 60)
    print(f"SUMMARY:")
    print(f"  Total datasets: {len(cache_files)}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    print("=" * 60)

def show_cache_key_info(ds_name, ds_kwargs, max_samples, min_words, seed, cache_dir=".dataset_cache"):
    """Show information about a specific cache key and whether it exists"""
    cache_key = generate_cache_key(ds_name, ds_kwargs, max_samples, min_words, seed)
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    print("=" * 60)
    print("CACHE KEY INFORMATION")
    print("=" * 60)
    print(f"Dataset: {ds_name}")
    print(f"Arguments: {ds_kwargs}")
    print(f"Max samples: {max_samples:,}")
    print(f"Min words: {min_words}")
    print(f"Seed: {seed}")
    print(f"Generated cache key: {cache_key}")
    print(f"Cache file: {cache_file}")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ Cache HIT: Found {len(data):,} samples")
            print(f"File size: {os.path.getsize(cache_file):,} bytes ({os.path.getsize(cache_file)/1024/1024:.1f} MB)")
        except Exception as e:
            print(f"✗ Cache file exists but cannot be read: {e}")
    else:
        print("✗ Cache MISS: File not found")
    
    print("=" * 60)

def clear_cache(cache_dir=".dataset_cache", confirm=True):
    """Clear all cached datasets"""
    if not os.path.exists(cache_dir):
        print("No cache directory found.")
        return
    
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
    
    if not cache_files:
        print("No cached datasets to clear.")
        return
    
    if confirm:
        print(f"Found {len(cache_files)} cached datasets:")
        for cache_file in cache_files:
            print(f"  - {cache_file}")
        
        response = input(f"\nAre you sure you want to delete all cached datasets? (y/N): ")
        if response.lower() != 'y':
            print("Cache clearing cancelled.")
            return
    
    try:
        shutil.rmtree(cache_dir)
        print(f"✓ Successfully cleared cache directory: {cache_dir}")
    except Exception as e:
        print(f"✗ Failed to clear cache: {e}")

def clear_specific_cache(cache_file, cache_dir=".dataset_cache"):
    """Clear a specific cached dataset"""
    file_path = os.path.join(cache_dir, cache_file)
    
    if not os.path.exists(file_path):
        print(f"Cache file not found: {cache_file}")
        return
    
    try:
        os.remove(file_path)
        print(f"✓ Successfully removed: {cache_file}")
    except Exception as e:
        print(f"✗ Failed to remove {cache_file}: {e}")

def show_cache_usage():
    """Show disk usage information"""
    cache_dir = ".dataset_cache"
    
    if not os.path.exists(cache_dir):
        print("No cache directory found.")
        return
    
    # Get disk usage
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_count += 1
    
    print("=" * 60)
    print("CACHE DISK USAGE")
    print("=" * 60)
    print(f"Files: {file_count}")
    print(f"Size: {total_size:,} bytes")
    print(f"Size: {total_size/1024/1024:.1f} MB")
    print(f"Size: {total_size/1024/1024/1024:.2f} GB")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Manage MicroBERT dataset cache")
    parser.add_argument("action", choices=["info", "clear", "usage", "key"], 
                       help="Action to perform")
    parser.add_argument("--cache-dir", default=".dataset_cache",
                       help="Cache directory (default: .dataset_cache)")
    parser.add_argument("--file", help="Specific cache file to clear (for clear action)")
    parser.add_argument("--no-confirm", action="store_true",
                       help="Skip confirmation prompt (for clear action)")
    
    # Cache key generation arguments
    parser.add_argument("--ds-name", help="Dataset name for key generation")
    parser.add_argument("--ds-kwargs", help="Dataset arguments as JSON string")
    parser.add_argument("--max-samples", type=int, help="Max samples for key generation")
    parser.add_argument("--min-words", type=int, default=5, help="Min words for key generation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for key generation")
    
    args = parser.parse_args()
    
    if args.action == "info":
        show_cache_info(args.cache_dir)
    elif args.action == "clear":
        if args.file:
            clear_specific_cache(args.file, args.cache_dir)
        else:
            clear_cache(args.cache_dir, not args.no_confirm)
    elif args.action == "usage":
        show_cache_usage()
    elif args.action == "key":
        if not all([args.ds_name, args.max_samples]):
            print("Error: --ds-name and --max-samples are required for key generation")
            return
        
        try:
            ds_kwargs = json.loads(args.ds_kwargs) if args.ds_kwargs else {}
        except json.JSONDecodeError:
            print("Error: --ds-kwargs must be valid JSON")
            return
        
        show_cache_key_info(args.ds_name, ds_kwargs, args.max_samples, args.min_words, args.seed, args.cache_dir)

if __name__ == '__main__':
    main()
