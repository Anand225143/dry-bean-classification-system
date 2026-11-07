
# ==========================================================
# DRY BEAN CLASSIFICATION
# STEP 4: DETAILED MODEL EVALUATION
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# ----------------------------------------------------------
# 1. CREATE DIRECTORIES
# ----------------------------------------------------------

os.makedirs("results", exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 65)
print("DRY BEAN CLASSIFICATION - DETAILED MODEL EVALUATION")
print("=" * 65)

# ----------------------------------------------------------
# 2. LOAD TEST DATA
# ----------------------------------------------------------

test_data = pd.read_csv("test_data.csv")

X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]

print("\nTest dataset shape:", X_test.shape)

# ----------------------------------------------------------
# 3. LOAD FIVE TRAINED MODELS
# ----------------------------------------------------------

model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib"
}

models = {}

for name, path in model_files.items():
    models[name] = joblib.load(path)
    print("Loaded:", name)

# ----------------------------------------------------------
# 4. CLASS-WISE PERFORMANCE ANALYSIS
# ----------------------------------------------------------

print("\n1. CLASS-WISE PERFORMANCE ANALYSIS")

class_results = []

for name, model in models.items():

    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    for class_name in model.classes_:

        class_results.append({
            "Model": name,
            "Class": class_name,
            "Precision": report[class_name]["precision"],
            "Recall": report[class_name]["recall"],
            "F1 Score": report[class_name]["f1-score"],
            "Support": report[class_name]["support"]
        })

        if name == "SVM":
            print(
                class_name,
                "F1 Score:",
                round(report[class_name]["f1-score"], 4)
            )

class_df = pd.DataFrame(class_results)

class_df.to_csv(
    "results/class_wise_performance.csv",
    index=False
)

print("\nSaved class-wise performance CSV.")

# ----------------------------------------------------------
# 5. MULTICLASS ROC CURVE COMPARISON
# ----------------------------------------------------------

print("\n2. GENERATING ROC CURVE COMPARISON")

classes = models["SVM"].classes_

y_binary = label_binarize(
    y_test,
    classes=classes
)

plt.figure(figsize=(11, 8))

for name, model in models.items():

    probabilities = model.predict_proba(X_test)

    # Micro-average multiclass ROC
    fpr, tpr, thresholds = roc_curve(
        y_binary.ravel(),
        probabilities.ravel()
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Multiclass ROC Curve - Model Comparison")

plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    "results/roc_curve_comparison.png",
    dpi=300
)

plt.close()

print("ROC curve saved successfully.")

# ----------------------------------------------------------
# 6. SVM MISCLASSIFICATION ANALYSIS
# ----------------------------------------------------------

print("\n3. SVM MISCLASSIFICATION ANALYSIS")

svm_model = models["SVM"]

svm_predictions = svm_model.predict(X_test)

svm_probabilities = svm_model.predict_proba(X_test)

misclassified_mask = (
    svm_predictions != y_test.to_numpy()
)

misclassified = X_test.loc[
    misclassified_mask
].copy()

misclassified["Actual Class"] = (
    y_test.to_numpy()[misclassified_mask]
)

misclassified["Predicted Class"] = (
    svm_predictions[misclassified_mask]
)

misclassified["Prediction Confidence"] = (
    svm_probabilities.max(axis=1)[misclassified_mask]
)

misclassified.to_csv(
    "results/svm_misclassified_records.csv",
    index=False
)

print("Total test records:", len(X_test))
print("Correct predictions:", np.sum(~misclassified_mask))
print("Incorrect predictions:", np.sum(misclassified_mask))
print(
    "Misclassification rate:",
    round(np.mean(misclassified_mask) * 100, 2),
    "%"
)

print("\nSample misclassified records:")
print(
    misclassified[
        ["Actual Class", "Predicted Class", "Prediction Confidence"]
    ].head(10)
)

# ----------------------------------------------------------
# 7. CLASS-WISE SVM PERFORMANCE GRAPH
# ----------------------------------------------------------

print("\n4. GENERATING SVM CLASS-WISE F1 GRAPH")

svm_class_results = class_df[
    class_df["Model"] == "SVM"
]

plt.figure(figsize=(10, 6))

sns.barplot(
    data=svm_class_results,
    x="Class",
    y="F1 Score"
)

plt.title("SVM - F1 Score by Dry Bean Class")
plt.xlabel("Bean Variety")
plt.ylabel("F1 Score")
plt.ylim(0, 1.05)

plt.tight_layout()

plt.savefig(
    "results/svm_classwise_f1.png",
    dpi=300
)

plt.close()

# ----------------------------------------------------------
# 8. FINAL SUMMARY
# ----------------------------------------------------------

print("\n5. EVALUATION SUMMARY")

print("Number of models evaluated:", len(models))
print("Number of test records:", len(X_test))
print("Number of bean classes:", len(classes))

print("\nGenerated files:")
print("results/class_wise_performance.csv")
print("results/roc_curve_comparison.png")
print("results/svm_misclassified_records.csv")
print("results/svm_classwise_f1.png")

print("\nSTEP 4 COMPLETED SUCCESSFULLY!")
print("=" * 65)
