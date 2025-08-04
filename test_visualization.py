import matplotlib.pyplot as plt
from microbert.utils import plot_results, plot_parallel
import numpy as np

print("Testing improved visualization functions...")

# Create sample training history
history = {
    'train_losses': [450, 430, 420, 410, 400, 390, 380, 370, 360, 350, 340, 330, 320, 310, 300, 290, 280, 270, 260, 250],
    'val_losses': [480, 460, 440, 420, 400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180, 160, 140, 120, 100],
    'train_acc': [0.45, 0.47, 0.49, 0.51, 0.53, 0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67, 0.69, 0.71, 0.73, 0.75, 0.77, 0.79, 0.81, 0.83],
    'val_acc': [0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82],
    'train_f1': [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88],
    'val_f1': [0.49, 0.51, 0.53, 0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67, 0.69, 0.71, 0.73, 0.75, 0.77, 0.79, 0.81, 0.83, 0.85, 0.87]
}

# Test plot_results function
print("1. Testing plot_results function...")
plot_results(history, do_val=True)

# Test plot_parallel function with sample data
print("2. Testing plot_parallel function...")
# Create sample attention matrix and tokens
sample_tokens = ["This", "movie", "was", "amazing", "!"]
sample_matrix = np.array([
    [0.3, 0.1, 0.2, 0.3, 0.1],
    [0.1, 0.4, 0.2, 0.2, 0.1],
    [0.2, 0.2, 0.3, 0.2, 0.1],
    [0.3, 0.2, 0.2, 0.2, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.6]
])

plot_parallel(sample_matrix, sample_tokens)

print("Visualization test completed!") 