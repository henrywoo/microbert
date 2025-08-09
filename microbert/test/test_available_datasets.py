#!/usr/bin/env python3
"""
Test script to check which datasets are currently available and working
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

def test_available_datasets():
    """Test which datasets are currently available and working"""
    print("🔍 Testing Available Datasets")
    print("=" * 60)
    
    # Try to import datasets library
    try:
        from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
    except ImportError:
        print("❌ datasets library not found. Please install with: pip install datasets")
        return
    
    # Test datasets with their configurations
    test_datasets = [
        {'name': 'c4', 'kwargs': {'name': 'en'}},
        {'name': 'c4', 'kwargs': {'name': 'en', 'split': 'train'}},
        {'name': 'wikitext', 'kwargs': {'name': 'wikitext-103-raw-v1'}},
        {'name': 'wikitext', 'kwargs': {'name': 'wikitext-2-raw-v1'}},
        {'name': 'wikitext', 'kwargs': {'name': 'wikitext-103-v1'}},
        {'name': 'wikitext', 'kwargs': {'name': 'wikitext-2-v1'}},
        {'name': 'squad', 'kwargs': {}},
        {'name': 'squad', 'kwargs': {'name': 'plain_text'}},
        {'name': 'squad_v2', 'kwargs': {}},
        {'name': 'squad_v2', 'kwargs': {'name': 'plain_text'}},
        {'name': 'imdb', 'kwargs': {}},
        {'name': 'imdb', 'kwargs': {'name': 'plain_text'}},
        {'name': 'ag_news', 'kwargs': {}},
        {'name': 'ag_news', 'kwargs': {'name': 'default'}},
        {'name': 'yelp_polarity', 'kwargs': {}},
        {'name': 'yelp_polarity', 'kwargs': {'name': 'default'}},
        {'name': 'yelp_review_full', 'kwargs': {}},
        {'name': 'yelp_review_full', 'kwargs': {'name': 'default'}},
        {'name': 'amazon_polarity', 'kwargs': {}},
        {'name': 'amazon_polarity', 'kwargs': {'name': 'default'}},
        {'name': 'amazon_reviews_multi', 'kwargs': {'language': 'en'}},
        {'name': 'amazon_reviews_multi', 'kwargs': {'name': 'en'}},
        {'name': 'dbpedia_14', 'kwargs': {}},
    ]
    
    available_datasets = []
    
    for ds_config in test_datasets:
        ds_name = ds_config['name']
        ds_kwargs = ds_config['kwargs']
        
        print(f"\n📊 Testing {ds_name} {ds_kwargs}...")
        
        try:
            # Try to load a small sample to test if it works
            print(f"  🔍 Testing dataset loading...")
            
            # Try streaming first
            try:
                dataset = load_dataset(ds_name, **ds_kwargs, streaming=True, split='train')
                print(f"     ✅ Streaming mode works")
                
                # Try to get a few samples
                sample_count = 0
                for item in dataset:
                    if sample_count >= 5:  # Just test 5 samples
                        break
                    sample_count += 1
                
                print(f"     ✅ Successfully loaded {sample_count} samples in streaming mode")
                available_datasets.append({**ds_config, 'mode': 'streaming'})
                
            except Exception as e:
                print(f"     ❌ Streaming failed: {e}")
                
                # Try non-streaming
                try:
                    dataset = load_dataset(ds_name, **ds_kwargs, split='train')
                    print(f"     ✅ Non-streaming mode works")
                    print(f"     ✅ Dataset has {len(dataset)} total samples")
                    available_datasets.append({**ds_config, 'mode': 'non-streaming'})
                    
                except Exception as e2:
                    print(f"     ❌ Non-streaming also failed: {e2}")
            
        except Exception as e:
            print(f"  ❌ {ds_name} failed completely: {e}")
    
    print(f"\n🎯 Summary:")
    print(f"Available datasets: {len(available_datasets)}/{len(test_datasets)}")
    for ds in available_datasets:
        print(f"  - {ds['name']} {ds['kwargs']} ({ds['mode']})")
    
    print(f"\n💡 Recommendation:")
    if available_datasets:
        print(f"Use these datasets in order of preference:")
        for i, ds in enumerate(available_datasets, 1):
            print(f"  {i}. {ds['name']} {ds['kwargs']} ({ds['mode']})")
    else:
        print("No datasets available. Consider using local IMDB data or smaller datasets.")
    
    return available_datasets

if __name__ == "__main__":
    test_available_datasets()
