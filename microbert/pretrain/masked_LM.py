import matplotlib.pyplot as plt
import numpy as np

# Original tensor - a 2x3 matrix with values 1-6
x = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=float)

# Mask tensor - boolean array where True indicates positions to be masked/filled
# False values will keep the original values, True values will be replaced
mask = np.array([[False, True, False],
                 [True, False, True]])

# Fill value - the value that will replace masked positions
value = -1

# Apply masked_fill_ operation
# Create a copy to avoid modifying the original tensor
x_filled = x.copy()
# Replace values where mask is True with the fill value
x_filled[mask] = value

# Create visualization with three subplots side by side
fig, axes = plt.subplots(1, 3, figsize=(9, 3))

# Plot 1: Original tensor
# Use Blues colormap to show gradient effect based on values
axes[0].imshow(x, cmap='Blues', vmin=-1, vmax=6)
# Add text annotations showing the actual values
for i in range(2):
    for j in range(3):
        axes[0].text(j, i, int(x[i, j]), ha='center', va='center', fontsize=12)
axes[0].set_title("Original Tensor")
axes[0].axis('off')

# Plot 2: Mask tensor
# Use binary colormap for clear black and white distinction
axes[1].imshow(mask, cmap='binary', vmin=0, vmax=1)
# Add text annotations: 'T' for True (masked), 'F' for False (not masked)
for i in range(2):
    for j in range(3):
        axes[1].text(j, i, 'T' if mask[i, j] else 'F', ha='center', va='center', fontsize=12)
axes[1].set_title("Mask Tensor")
axes[1].axis('off')

# Plot 3: Result after applying masked_fill_
# Use Blues colormap to show gradient effect, with -1 as white
axes[2].imshow(x_filled, cmap='Blues', vmin=-1, vmax=6)
# Add text annotations showing the final values
for i in range(2):
    for j in range(3):
        axes[2].text(j, i, int(x_filled[i, j]), ha='center', va='center', fontsize=12)
axes[2].set_title("After masked_fill_")
axes[2].axis('off')

# Adjust layout to prevent overlap between subplots
plt.tight_layout()
# Display the visualization
plt.show()
