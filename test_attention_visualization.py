import matplotlib.pyplot as plt
from microbert.utils import plot_parallel
import numpy as np

print("Testing attention visualization with different modes...")

# Create sample attention matrix and tokens
sample_tokens = ["[CLS]", "This", "movie", "was", "amazing", "!"]
sample_matrix = np.array([
    [0.4, 0.1, 0.2, 0.2, 0.05, 0.05],  # CLS attention to others
    [0.1, 0.3, 0.2, 0.2, 0.1, 0.1],   # "This" attention to others
    [0.05, 0.2, 0.4, 0.2, 0.1, 0.05], # "movie" attention to others
    [0.05, 0.1, 0.2, 0.4, 0.2, 0.05], # "was" attention to others
    [0.05, 0.05, 0.1, 0.2, 0.5, 0.1], # "amazing" attention to others
    [0.05, 0.05, 0.05, 0.1, 0.1, 0.65] # "!" attention to others
])

print("1. Testing CLS token attention only (default mode)...")
plot_parallel(sample_matrix, sample_tokens, show_all_connections=False)

print("2. Testing all token connections...")
plot_parallel(sample_matrix, sample_tokens, show_all_connections=True)

print("Visualization test completed!") 