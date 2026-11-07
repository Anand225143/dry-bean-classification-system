
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Dry Bean Classification System",
    page_icon="🌱",
    layout="wide"
)

# Project title
st.title("🌱 Dry Bean Classification System")
st.write(
    "A Machine Learning application to classify dry bean varieties "
    "using 16 numerical features."
)

st.divider()

# Available models
MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbors": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

# Feature names
FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRatio",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "Roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4"
]

# Load selected model
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# Sidebar
st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a classification model:",
    list(MODEL_PATHS.keys())
)

st.sidebar.info(
    "Select one of the five trained machine learning models."
)

# Main input section
st.header("Enter Dry Bean Measurements")

st.write("Enter the 16 numerical measurements of the dry bean.")

# Input fields
input_values = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(FEATURES):
    with col1 if i < 8 else col2:
        input_values[feature] = st.number_input(
            feature,
            min_value=0.0,
            value=1.0,
            format="%.6f",
            key=feature
        )

# Prediction button
if st.button("Predict Bean Class", type="primary"):

    model_path = MODEL_PATHS[selected_model]

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")

    else:
        try:
            model = load_model(model_path)

            input_df = pd.DataFrame(
                [input_values],
                columns=FEATURES
            )

            prediction = model.predict(input_df)[0]

            st.divider()
            st.subheader("Prediction Result")

            st.success(f"Predicted Bean Class: {prediction}")

            # Display prediction confidence when available
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                confidence = max(probabilities) * 100

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                probability_df = pd.DataFrame({
                    "Bean Class": model.classes_,
                    "Probability": probabilities
                })

                probability_df = probability_df.sort_values(
                    "Probability",
                    ascending=False
                )

                st.subheader("Class Probability Distribution")

                st.bar_chart(
                    probability_df.set_index("Bean Class")
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# About section
st.divider()

st.header("About the Project")

st.write("""
This project uses the UCI Dry Bean dataset to classify seven varieties
of dry beans using five machine learning algorithms:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Naive Bayes
5. Support Vector Machine

The application allows users to select a trained model, enter bean
measurements, and obtain a predicted bean variety.
""")

