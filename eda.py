
# ==========================================================
# DRY BEAN CLASSIFICATION
# STEP 2: EXPLORATORY DATA ANALYSIS AND PREPROCESSING
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# Create output directories
os.makedirs("results", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

# Set visualization style
sns.set_theme(style="whitegrid")

print("=" * 65)
print("DRY BEAN CLASSIFICATION - EXPLORATORY DATA ANALYSIS")
print("=" * 65)

# ----------------------------------------------------------
# 1. LOAD DATASET
# ----------------------------------------------------------

df = pd.read_csv("data/dry_bean_dataset.csv")

print("\n1. ORIGINAL DATASET INFORMATION")
print("-" * 65)

print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
df.info()

# ----------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS
# ----------------------------------------------------------

print("\n2. DESCRIPTIVE STATISTICS")
print("-" * 65)

print(df.describe().round(2))

df.describe().to_csv("results/descriptive_statistics.csv")

# ----------------------------------------------------------
# 3. MISSING VALUE ANALYSIS
# ----------------------------------------------------------

print("\n3. MISSING VALUE ANALYSIS")
print("-" * 65)

print(df.isnull().sum())

total_missing = df.isnull().sum().sum()
print("\nTotal missing values:", total_missing)

# ----------------------------------------------------------
# 4. DUPLICATE VALUE ANALYSIS
# ----------------------------------------------------------

print("\n4. DUPLICATE VALUE ANALYSIS")
print("-" * 65)

duplicate_count = df.duplicated().sum()

print("Duplicate records:", duplicate_count)

# Remove exact duplicate records
df = df.drop_duplicates().reset_index(drop=True)

print("Dataset shape after removing duplicates:", df.shape)

df.to_csv("data/dry_bean_cleaned.csv", index=False)

# ----------------------------------------------------------
# 5. TARGET CLASS DISTRIBUTION
# ----------------------------------------------------------

print("\n5. TARGET CLASS DISTRIBUTION")
print("-" * 65)

print(df["Class"].value_counts())

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Class",
    order=df["Class"].value_counts().index
)

plt.title("Distribution of Dry Bean Classes")
plt.xlabel("Bean Variety")
plt.ylabel("Number of Records")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("results/class_distribution.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 6. FEATURE DISTRIBUTION HISTOGRAMS
# ----------------------------------------------------------

print("\n6. FEATURE DISTRIBUTION ANALYSIS")

features = df.drop(columns=["Class"])

features.hist(
    figsize=(18, 14),
    bins=30,
    edgecolor="black"
)

plt.suptitle(
    "Distribution of Dry Bean Numerical Features",
    fontsize=16
)

plt.tight_layout()

plt.savefig("results/feature_distributions.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 7. CORRELATION HEATMAP
# ----------------------------------------------------------

print("\n7. FEATURE CORRELATION ANALYSIS")

plt.figure(figsize=(15, 12))

correlation = features.corr()

sns.heatmap(
    correlation,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    square=True,
    annot_kws={"size": 7}
)

plt.title("Correlation Heatmap of Dry Bean Features")
plt.tight_layout()

plt.savefig("results/correlation_heatmap.png", dpi=300)
plt.close()

correlation.to_csv("results/correlation_matrix.csv")

# ----------------------------------------------------------
# 8. TRAIN TEST SPLIT
# ----------------------------------------------------------

print("\n8. TRAIN TEST SPLIT")
print("-" * 65)

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save training data
train_data = X_train.copy()
train_data["Class"] = y_train

train_data.to_csv("data/train_data.csv", index=False)

# Save held-out test data in project root
test_data = X_test.copy()
test_data["Class"] = y_test

test_data.to_csv("test_data.csv", index=False)

print("Total cleaned records:", len(df))
print("Training records:", len(train_data))
print("Testing records:", len(test_data))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# ----------------------------------------------------------
# 9. FINAL SUMMARY
# ----------------------------------------------------------

print("\n9. FINAL DATASET SUMMARY")
print("-" * 65)

print("Input features:", X.shape[1])
print("Target variable: Class")
print("Number of classes:", y.nunique())
print("Duplicate records removed:", duplicate_count)
print("Missing values:", df.isnull().sum().sum())

print("\nGenerated files:")
print("data/dry_bean_cleaned.csv")
print("data/train_data.csv")
print("test_data.csv")
print("results/descriptive_statistics.csv")
print("results/class_distribution.png")
print("results/feature_distributions.png")
print("results/correlation_heatmap.png")
print("results/correlation_matrix.csv")

print("\nSTEP 2 COMPLETED SUCCESSFULLY!")
print("=" * 65)

