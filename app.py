import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Kidney Disease Risk Prediction",
    layout="wide"
)

st.title("🩺 Kidney Disease Risk Prediction System")
st.write("Predict kidney disease risk using a trained Random Forest model.")

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# LOAD SAVED ARTIFACTS (MATCHING YOUR FILE NAMES)
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "rf_kidney_disease_model.pkl")
    target_encoder = joblib.load(BASE_DIR / "target_label_encoder.pkl")
    scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
    return model, target_encoder, scaler, feature_columns


model, target_encoder, scaler, feature_columns = load_artifacts()

# --------------------------------------------------
# INPUT FORM (NUMERIC INPUTS – MATCHES TRAINING)
# --------------------------------------------------
st.subheader("🔹 Enter Patient Details")

input_data = {}

with st.form("prediction_form"):
    cols = st.columns(3)

    for i, col_name in enumerate(feature_columns):
        with cols[i % 3]:
            input_data[col_name] = st.number_input(
                col_name,
                value=0.0,
                format="%.4f"
            )

    submit = st.form_submit_button("Predict")

# --------------------------------------------------
# PREDICTION LOGIC (FIXED)
# --------------------------------------------------
if submit:
    input_df = pd.DataFrame([input_data])

    # --------------------------------------------------
    # SCALE ONLY NUMERICAL FEATURES
    # (exact columns used during training)
    # --------------------------------------------------
    scaler_features = list(scaler.feature_names_in_)

    input_df[scaler_features] = scaler.transform(
        input_df[scaler_features]
    )

    # --------------------------------------------------
    # ENSURE CORRECT FEATURE ORDER
    # --------------------------------------------------
    input_df = input_df[feature_columns]

    # --------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------
    pred_encoded = model.predict(input_df)[0]
    pred_label = target_encoder.inverse_transform([pred_encoded])[0]

    st.subheader("✅ Prediction Result")
    st.success(f"**Predicted Kidney Disease Status:** {pred_label}")

    # --------------------------------------------------
    # PREDICTION PROBABILITIES
    # --------------------------------------------------
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        prob_df = pd.DataFrame({
            "Risk Category": target_encoder.classes_,
            "Probability (%)": np.round(probs * 100, 2)
        }).sort_values("Probability (%)", ascending=False)

        st.subheader("📊 Prediction Probabilities")
        st.dataframe(prob_df, use_container_width=True)
