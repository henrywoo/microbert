#!/usr/bin/env python3
"""
Test script for max_samples functionality
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

def test_max_samples_parsing():
    """Test max_samples parsing functionality"""
    
    def parse_max_samples(max_samples_str):
        """Parse max_samples string to integer"""
        if not max_samples_str:
            return None
        
        max_samples_str = max_samples_str.upper()
        if max_samples_str.endswith('K'):
            return int(max_samples_str[:-1]) * 1000
        elif max_samples_str.endswith('M'):
            return int(max_samples_str[:-1]) * 1000000
        elif max_samples_str.isdigit():
            return int(max_samples_str)
        else:
            raise ValueError(f"Unknown max_samples format: {max_samples_str}")
    
    # Test cases
    test_cases = [
        ('500k', 500000),
        ('5M', 5000000),
        ('50M', 50000000),
        ('500M', 500000000),
        ('1000', 1000),
        ('1000000', 1000000),
        (None, None),
    ]
    
    print("Testing max_samples parsing:")
    print("=" * 40)
    
    for input_str, expected in test_cases:
        try:
            result = parse_max_samples(input_str)
            status = "✓" if result == expected else "✗"
            print(f"{status} {input_str} -> {result} (expected: {expected})")
        except Exception as e:
            print(f"✗ {input_str} -> Error: {e}")
    
    print()

def test_v2_usage():
    """Test v2 usage examples"""
    print("v2 Usage Examples:")
    print("=" * 40)
    print("python mlm_pretrain_v2.py hf                    # 500K samples (default)")
    print("python mlm_pretrain_v2.py hf true 5M            # 5M samples")
    print("python mlm_pretrain_v2.py hf false 50M          # 50M samples")
    print("python mlm_pretrain_v2.py hf true 500M          # 500M samples")
    print("python mlm_pretrain_v2.py imdb                  # IMDB dataset")
    print()

def test_v3_usage():
    """Test v3 usage examples"""
    print("v3 Usage Examples:")
    print("=" * 40)
    print("# Default 500K samples")
    print("torchrun --nproc_per_node=8 mlm_pretrain_v3.py --dataset hf")
    print()
    print("# 5M samples")
    print("torchrun --nproc_per_node=8 mlm_pretrain_v3.py --dataset hf --max-samples 5M")
    print()
    print("# 50M samples")
    print("torchrun --nproc_per_node=8 mlm_pretrain_v3.py --dataset hf --max-samples 50M")
    print()
    print("# 500M samples")
    print("torchrun --nproc_per_node=8 mlm_pretrain_v3.py --dataset hf --max-samples 500M")
    print()

def main():
    """Main test function"""
    print("MAX_SAMPLES FUNCTIONALITY TEST")
    print("=" * 50)
    print()
    
    test_max_samples_parsing()
    test_v2_usage()
    test_v3_usage()
    
    print("Test completed!")

if __name__ == '__main__':
    main()
