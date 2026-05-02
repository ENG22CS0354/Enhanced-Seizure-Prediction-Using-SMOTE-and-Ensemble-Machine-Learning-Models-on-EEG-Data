from scipy.io import loadmat
import numpy as np

data = loadmat("LABELS.mat")
labels = data["LABELS"]

print("Top-level shape:", labels.shape)

print("\nInspecting each cell:\n")

for i in range(labels.shape[1]):
    cell = labels[0, i]
    print(f"Cell {i} type:", type(cell))
    print(f"Cell {i} shape:", np.shape(cell))
    print("-" * 40)
