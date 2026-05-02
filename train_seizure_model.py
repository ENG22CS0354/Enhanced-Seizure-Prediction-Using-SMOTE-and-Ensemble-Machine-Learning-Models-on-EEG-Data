#!/usr/bin/env python3

"""
Train Seizure Detection Model using
Epileptic Seizure Recognition Dataset (178 features)
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ----------------------------
# Load Dataset
# ----------------------------
print("Loading dataset...")
df = pd.read_csv("Epileptic Seizure Recognition.csv")

# Drop first ID column safely
df = df.iloc[:, 1:]

# Convert to binary classification
# 1 = seizure, others = non-seizure
df["y"] = df["y"].apply(lambda x: 1 if x == 1 else 0)

# Separate features and labels
X = df.drop("y", axis=1)
y = df["y"]

# Ensure numeric
X = X.apply(pd.to_numeric)

print("Dataset shape:", X.shape)
print("Seizure distribution:\n", y.value_counts())

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# Handle Imbalance using SMOTE
# ----------------------------
print("\nApplying SMOTE...")
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

print("After SMOTE class distribution:",
      np.bincount(y_train_resampled))

# ----------------------------
# Feature Scaling
# ----------------------------
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

print("Scaler expects:", scaler.n_features_in_, "features")

# ----------------------------
# Train XGBoost Model
# ----------------------------
print("\nTraining model...")

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)

model.fit(X_train_resampled, y_train_resampled)

# ----------------------------
# Evaluate Model
# ----------------------------
y_prob = model.predict_proba(X_test)[:, 1]

print("\nThreshold Tuning Results:")
for threshold in [0.2, 0.3, 0.4, 0.5]:
    print(f"\nThreshold: {threshold}")
    y_pred = (y_prob > threshold).astype(int)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# ----------------------------
# Save Model and Scaler
# ----------------------------
joblib.dump(model, "seizure_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved as seizure_model.pkl")
print("Scaler saved as scaler.pkl")
print("Training complete.")