#!/usr/bin/env python3

"""
Stream seizure predictions using Epileptic Seizure Recognition dataset
Simulates live EEG streaming row-by-row
"""

import pandas as pd
import numpy as np
import joblib
from time import sleep
from datetime import datetime

# ----------------------------
# Load trained model + scaler
# ----------------------------
model = joblib.load("seizure_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("Epileptic Seizure Recognition.csv")

# Drop first ID column
df = df.iloc[:, 1:]

# Convert to binary
df["y"] = df["y"].apply(lambda x: 1 if x == 1 else 0)

X = df.drop("y", axis=1)
y = df["y"]

print("Feature shape:", X.shape)
print("Scaler expects:", scaler.n_features_in_)

if X.shape[1] != scaler.n_features_in_:
    raise ValueError("Feature mismatch! Training and streaming features are different.")

print("Streaming started...")
print("Total samples:", len(X))
print("----------------------------------")

# ----------------------------
# Stream row by row
# ----------------------------
THRESHOLD = 0.3   # you tuned this earlier

for i in range(len(X)):
    
    # Simulate live window
    sample = X.iloc[i].values.astype(float).reshape(1, -1)
    
    # Scale using trained scaler
    sample_scaled = scaler.transform(sample)
    
    # Predict probability
    prob = model.predict_proba(sample_scaled)[0][1]
    
    # Apply threshold
    prediction = 1 if prob > THRESHOLD else 0
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"[{timestamp}] Sample {i+1}")
    print(f"   True Label : {y.iloc[i]}")
    print(f"   Prediction : {prediction}")
    print(f"   Confidence : {prob:.4f}")
    print("----------------------------------")
    
    sleep(0.1)   # simulate 0.5 sec streaming delay