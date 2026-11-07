
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Create data directory if it does not exist
os.makedirs("data", exist_ok=True)

print("=" * 55)
print("DRY BEAN CLASSIFICATION - DATASET DOWNLOAD")
print("=" * 55)

# Fetch Dry Bean dataset from UCI
print("\nConnecting to UCI repository...")

dry_bean = fetch_ucirepo(id=602)

# Extract features and target
X = dry_bean.data.features
y = dry_bean.data.targets

# Combine features and target
df = pd.concat([X, y], axis=1)

# Save dataset as CSV
output_path = "data/dry_bean_dataset.csv"
df.to_csv(output_path, index=False)

print("\nDataset downloaded successfully!")

print("\nDataset Information")
print("-" * 55)

print("Dataset shape:", df.shape)
print("Number of instances:", df.shape[0])
print("Number of features:", X.shape[1])
print("Target column:", y.columns[0])

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nBean class distribution:")
print(df[y.columns[0]].value_counts())

print("\nFirst five records:")
print(df.head())

print("\nDataset saved at:", output_path)

print("\nSTEP 1 COMPLETED SUCCESSFULLY!")

