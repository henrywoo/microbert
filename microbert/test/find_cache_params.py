#!/usr/bin/env python3
"""
Script to find the parameters that generated a specific cache file
"""

import os
import json
import hashlib
from pathlib import Path

def find_cache_params(cache_file, cache_dir=".dataset_cache"):
    """Find the parameters that generated a specific cache file"""
    
    file_path = os.path.join(cache_dir, cache_file)
    if not os.path.exists(file_path):
        print(f"Cache file not found: {cache_file}")
        return
    
    # Read the cache file to get sample count
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        sample_count = len(data)
        print(f"Cache file contains {sample_count:,} samples")
    except Exception as e:
        print(f"Error reading cache file: {e}")
        return
    
    # Common dataset configurations to test
    test_configs = [
        # C4 dataset
        {'name': 'c4', 'kwargs': {'name': 'en'}},
        {'name': 'c4', 'kwargs': {}},
        
        # WikiText datasets
        {'name': 'wikitext', 'kwargs': {'name': 'wikitext-103-raw-v1'}},
        {'name': 'wikitext', 'kwargs': {'name': 'wikitext-2-raw-v1'}},
        {'name': 'wikitext', 'kwargs': {'name': 'wikitext-103-v1'}},
        {'name': 'wikitext', 'kwargs': {'name': 'wikitext-2-v1'}},
        
        # Other common datasets
        {'name': 'dbpedia_14', 'kwargs': {}},
        {'name': 'ag_news', 'kwargs': {}},
        {'name': 'ag_news', 'kwargs': {'name': 'default'}},
        {'name': 'yelp_polarity', 'kwargs': {}},
        {'name': 'yelp_review_full', 'kwargs': {}},
        {'name': 'amazon_polarity', 'kwargs': {}},
        {'name': 'squad', 'kwargs': {}},
        {'name': 'squad', 'kwargs': {'name': 'plain_text'}},
        {'name': 'squad_v2', 'kwargs': {}},
        {'name': 'imdb', 'kwargs': {}},
        {'name': 'imdb', 'kwargs': {'name': 'plain_text'}},
    ]
    
    # Test different parameter combinations
    max_samples_options = [500000, 1000000, 5000000, 10000000, 50000000]
    min_words_options = [3, 5, 10]
    seed_options = [42, 123, 456]
    
    print(f"\nSearching for parameters that generate cache key: {cache_file[:-5]}")
    print("=" * 80)
    
    found = False
    for ds_config in test_configs:
        for max_samples in max_samples_options:
            for min_words in min_words_options:
                for seed in seed_options:
                    # Generate cache key
                    config_str = f"{ds_config['name']}_{str(ds_config['kwargs'])}_{max_samples}_{min_words}_{seed}"
                    cache_key = hashlib.md5(config_str.encode()).hexdigest()[:16]
                    
                    if cache_key == cache_file[:-5]:  # Remove .json extension
                        print(f"✓ FOUND MATCHING PARAMETERS!")
                        print(f"  Dataset: {ds_config['name']}")
                        print(f"  Arguments: {ds_config['kwargs']}")
                        print(f"  Max samples: {max_samples:,}")
                        print(f"  Min words: {min_words}")
                        print(f"  Seed: {seed}")
                        print(f"  Generated key: {cache_key}")
                        print(f"  Expected file: {cache_key}.json")
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            break
    
    if not found:
        print("✗ No matching parameters found for this cache file")
        print("\nPossible reasons:")
        print("1. The cache file was generated with different parameters")
        print("2. The cache key generation logic has changed")
        print("3. The file was manually created or renamed")
        
        # Show some close matches
        print(f"\nClosest matches for key '{cache_file[:-5]}':")
        for ds_config in test_configs[:3]:  # Test first few datasets
            for max_samples in [500000, 1000000]:
                for min_words in [5]:
                    for seed in [42]:
                        config_str = f"{ds_config['name']}_{str(ds_config['kwargs'])}_{max_samples}_{min_words}_{seed}"
                        cache_key = hashlib.md5(config_str.encode()).hexdigest()[:16]
                        print(f"  {ds_config['name']} {max_samples:,} {min_words} {seed} -> {cache_key}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Find parameters for a cache file")
    parser.add_argument("cache_file", help="Cache file name (e.g., 52719dbb61cb7957.json)")
    parser.add_argument("--cache-dir", default=".dataset_cache", help="Cache directory")
    
    args = parser.parse_args()
    
    find_cache_params(args.cache_file, args.cache_dir)

if __name__ == '__main__':
    main()
