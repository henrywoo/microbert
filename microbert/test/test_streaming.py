#!/usr/bin/env python3
"""
Test script to verify streaming functionality
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

from mlm_pretrain import load_hf_dataset

def test_streaming():
    """Test streaming functionality"""
    print("Testing streaming functionality...")
    print("=" * 50)
    
    # Test streaming mode
    print("\n1. Testing streaming mode (should use minimal disk space):")
    try:
        data, _ = load_hf_dataset(max_samples=100, streaming=True)
        print(f"✓ Streaming mode successful: loaded {len(data)} samples")
        if data:
            print(f"   Sample text: {' '.join(data[0]['text'][:10])}...")
    except Exception as e:
        print(f"✗ Streaming mode failed: {e}")
    
    # Test local download mode
    print("\n2. Testing local download mode (will use more disk space):")
    try:
        data, _ = load_hf_dataset(max_samples=100, streaming=False)
        print(f"✓ Local download mode successful: loaded {len(data)} samples")
        if data:
            print(f"   Sample text: {' '.join(data[0]['text'][:10])}...")
    except Exception as e:
        print(f"✗ Local download mode failed: {e}")
    
    print("\n" + "=" * 50)
    print("Streaming test completed!")

if __name__ == '__main__':
    test_streaming()
