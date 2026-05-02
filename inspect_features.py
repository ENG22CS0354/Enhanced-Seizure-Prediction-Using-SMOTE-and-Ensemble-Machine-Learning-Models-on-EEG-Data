from scipy.io import loadmat
import numpy as np

data = loadmat("FEATURES.mat")
features = data["FEATURES"]

print("Top-level shape:", features.shape)

print("\nInspecting each cell:\n")

for i in range(features.shape[1]):
    cell = features[0, i]
    print(f"Cell {i} type:", type(cell))
    print(f"Cell {i} shape:", np.shape(cell))
    print("-" * 40)
