#!/usr/bin/env python3
"""
Test script to check vocabulary coverage for test sentences
"""

import json
import os

def test_vocabulary_coverage():
    """Test if vocabulary contains test words"""
    
    # Test sentences
    test_sentences = [
        "this movie is [MASK] fantastic",
        "the acting was [MASK] but the plot was confusing", 
        "amazing [MASK] by all actors highly recommended"
    ]
    
    # Load existing tokenizer vocabulary if available
    vocab = {}
    try:
        if os.path.exists('.mlm_pretrained_v3/tokenizer.json'):
            with open('.mlm_pretrained_v3/tokenizer.json', 'r') as f:
                vocab = json.load(f)
            print("Loaded existing tokenizer vocabulary")
        elif os.path.exists('.mlm_pretrained_v4/tokenizer.json'):
            with open('.mlm_pretrained_v4/tokenizer.json', 'r') as f:
                vocab = json.load(f)
            print("Loaded existing v4 tokenizer vocabulary")
        else:
            print("No existing tokenizer found")
            return
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Special tokens: {[k for k in vocab.keys() if k.startswith('[')]}")
    
    # Test each sentence
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. Testing sentence: {sentence}")
        
        # Split into tokens
        tokens = sentence.split()
        missing_words = []
        found_words = []
        
        for token in tokens:
            if token == '[MASK]':
                found_words.append(token)
                continue
                
            token_lower = token.lower()
            if token_lower in vocab:
                found_words.append(token)
                print(f"   ✓ '{token}' -> {vocab[token_lower]}")
            else:
                missing_words.append(token)
                print(f"   ✗ '{token}' -> [UNK] (not in vocabulary)")
        
        print(f"   Found: {len(found_words)}/{len(tokens)} words")
        if missing_words:
            print(f"   Missing: {missing_words}")
        else:
            print(f"   ✓ All words found in vocabulary!")

if __name__ == '__main__':
    test_vocabulary_coverage()
