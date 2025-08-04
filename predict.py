#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use the trained MicroBERT model for sentiment analysis prediction
"""

import os
from microbert.utils import load_model, predict_sentiment

def main():
    # Model file path
    home_dir = os.path.expanduser('~')
    model_dir = os.path.join(home_dir, '.microbert_model')
    
    # Check if model files exist
    if not os.path.exists(model_dir):
        print(f"Error: Model directory does not exist: {model_dir}")
        print("Please run train.py first to train the model")
        return
    
    # Load model
    print("Loading model...")
    model, tokenizer, config = load_model(model_dir)
    print(f"Model configuration: {config}")
    
    # Test texts
    test_texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Terrible film, waste of time and money. Don't watch it.",
        "The acting was okay but the plot was confusing.",
        "Amazing performance by all actors, highly recommended!",
        "Boring and predictable, I fell asleep halfway through."
    ]
    
    print("\n=== Sentiment Analysis Results ===")
    for i, text in enumerate(test_texts, 1):
        prediction, confidence = predict_sentiment(model, tokenizer, text)
        sentiment = "Positive" if prediction == 1 else "Negative"
        print(f"{i}. Text: {text}")
        print(f"   Prediction: {sentiment} (Confidence: {confidence:.3f})")
        print()

if __name__ == "__main__":
    main() 