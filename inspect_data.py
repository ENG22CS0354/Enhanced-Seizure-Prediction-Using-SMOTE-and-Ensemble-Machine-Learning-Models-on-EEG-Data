import os
from scipy.io import loadmat

# Get all .mat files in current directory
files = [f for f in os.listdir() if f.endswith(".mat")]

print("MAT files found:")
print(files)

if len(files) == 0:
    print("No .mat files found in this folder.")
else:
    file_path = files[0]  # take first file automatically
    print("\nOpening:", file_path)

    data = loadmat(file_path)

    print("\nKeys inside file:")
    for key in data.keys():
        print(key)

    # Print shape of first real variable
    for key in data.keys():
        if not key.startswith("__"):
            print("\nExample variable:", key)
            print("Shape:", data[key].shape)
            break
