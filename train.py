
# ==========================================================
# DRY BEAN CLASSIFICATION
# STEP 3: TRAIN FIVE MACHINE LEARNING CLASSIFIERS
# ==========================================================

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ----------------------------------------------------------
# 1. CREATE DIRECTORIES
# ----------------------------------------------------------

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/confusion_matrices", exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 70)
print("DRY BEAN CLASSIFICATION - MODEL TRAINING")
print("=" * 70)

# ----------------------------------------------------------
# 2. LOAD TRAINING AND TEST DATA
# ----------------------------------------------------------

print("\n1. LOADING DATA")

train_data = pd.read_csv("data/train_data.csv")
test_data = pd.read_csv("test_data.csv")

X_train = train_data.drop(columns=["Class"])
y_train = train_data["Class"]

X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ----------------------------------------------------------
# 3. DEFINE FIVE CLASSIFIERS
# ----------------------------------------------------------

print("\n2. DEFINING CLASSIFIERS")

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Naive Bayes": GaussianNB(),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])
}

# ----------------------------------------------------------
# 4. TRAIN MODELS AND EVALUATE
# ----------------------------------------------------------

results = []

for name, model in models.items():

    print("\n" + "=" * 70)
    print("TRAINING:", name)
    print("=" * 70)

    start_time = time.time()

    # Train model
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Predict classes
    y_pred = model.predict(X_test)

    # Predict class probabilities for AUC
    y_prob = model.predict_proba(X_test)

    # Calculate six evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        y_test,
        y_prob,
        labels=model.classes_,
        multi_class="ovr",
        average="macro"
    )

    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    # Display evaluation results
    print("Accuracy :", round(accuracy, 4))
    print("AUC      :", round(auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("MCC      :", round(mcc, 4))
    print("Training time:", round(training_time, 2), "seconds")

    # Save metrics
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc,
        "Training Time (seconds)": training_time
    })

    # ------------------------------------------------------
    # 5. SAVE TRAINED MODEL
    # ------------------------------------------------------

    filename = name.lower().replace(" ", "_") + ".joblib"

    joblib.dump(model, os.path.join("model", filename))

    print("Saved model:", filename)

    # ------------------------------------------------------
    # 6. GENERATE CONFUSION MATRIX
    # ------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_
    )

    plt.title(name + " - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()

    matrix_filename = (
        name.lower().replace(" ", "_") + "_confusion_matrix.png"
    )

    plt.savefig(
        os.path.join("results/confusion_matrices", matrix_filename),
        dpi=300
    )

    plt.close()

    # ------------------------------------------------------
    # 7. SAVE CLASSIFICATION REPORT
    # ------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    report_filename = (
        name.lower().replace(" ", "_") + "_report.txt"
    )

    with open(
        os.path.join("results", report_filename),
        "w"
    ) as file:
        file.write(name + "\n\n")
        file.write(report)

    print("\nClassification report:")
    print(report)

# ----------------------------------------------------------
# 8. CREATE MODEL COMPARISON TABLE
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

results_df.to_csv("results/model_comparison.csv", index=False)

print(results_df.round(4).to_string(index=False))

# ----------------------------------------------------------
# 9. VISUALIZE MODEL PERFORMANCE
# ----------------------------------------------------------

metrics = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]

for metric in metrics:

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results_df,
        x="Model",
        y=metric
    )

    plt.title("Model Comparison - " + metric)
    plt.xlabel("Classification Model")
    plt.ylabel(metric)
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        "results/" + metric.lower().replace(" ", "_")
        + "_comparison.png",
        dpi=300
    )

    plt.close()

# ----------------------------------------------------------
# 10. IDENTIFY BEST MODEL
# ----------------------------------------------------------

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\nBEST MODEL BASED ON MACRO F1 SCORE")
print("Model:", best_model["Model"])
print("F1 Score:", round(best_model["F1 Score"], 4))

print("\nAll trained models saved successfully!")
print("\nSTEP 3 COMPLETED SUCCESSFULLY!")
print("=" * 70)
