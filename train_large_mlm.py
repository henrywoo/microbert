#!/usr/bin/env python3
"""
Script to train MLM with large Hugging Face dataset
Usage: python train_large_mlm.py
"""

import os
import sys

# Add current directory to path
sys.path.append('.')

from mlm_pretrain import main

if __name__ == '__main__':
    print("Training MLM with large Hugging Face dataset...")
    print("This will take longer but should give better results.")
    print("You can interrupt with Ctrl+C if it takes too long.")
    print()
    
    # Train with large dataset
    main(dataset_choice='hf')
