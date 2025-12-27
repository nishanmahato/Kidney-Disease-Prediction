import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt


# PAGE CONFIG (Mobile Friendly)
st.set_page_config(
    page_title="Kidney Disease Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# GLOBAL STYLES 
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    div[data-testid="stForm"] {
        background-color: white;
        padding: 1.5rem;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# HEADER
st.title("🩺 Kidney Disease Risk Prediction")
st.caption("Machine-learning based clinical decision support system")

BASE_DIR = Path(__file__).resolve().parent


# LOAD MODEL ARTIFACTS
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "rf_kidney_disease_model.pkl")
    target_encoder = joblib.load(BASE_DIR / "target_label_encoder.pkl")
    scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
    return model, target_encoder, scaler, feature_columns

model, target_encoder, scaler, feature_columns = load_artifacts()


# BINARY FEATURES (EDIT IF NEEDED)
binary_features = [
    "hypertension",
    "diabetes_mellitus",
    "coronary_artery_disease",
    "appetite",
    "anemia",
    "pedal_edema"
]

def yes_no_to_numeric(value):
    return 1 if value == "Yes" else 0

# INPUT FORM
st.subheader("Patient Information")

input_data = {}

with st.form("prediction_form"):
    cols = st.columns(2)

    for i, col_name in enumerate(feature_columns):
        label = col_name.replace("_", " ").title()

        with cols[i % 2]:
            if col_name in binary_features:
                choice = st.selectbox(
                    label,
                    ["No", "Yes"]
                )
                input_data[col_name] = yes_no_to_numeric(choice)
            else:
                input_data[col_name] = st.number_input(
                    label,
                    value=0.0,
                    format="%.2f"
                )

    submitted = st.form_submit_button("🔍 Predict Risk")


# PREDICTION & VISUALIZATION
if submitted:
    input_df = pd.DataFrame([input_data])

    scaler_features = list(scaler.feature_names_in_)
    input_df[scaler_features] = scaler.transform(input_df[scaler_features])
    input_df = input_df[feature_columns]

    with st.spinner("Analyzing patient data..."):
        pred_encoded = model.predict(input_df)[0]
        pred_label = target_encoder.inverse_transform([pred_encoded])[0]

    st.subheader("Prediction Result")

    if str(pred_label).lower() in ["ckd", "chronic kidney disease", "yes"]:
        st.error("High Risk of Kidney Disease")
    else:
        st.success("Low Risk of Kidney Disease")

    st.metric("Predicted Status", pred_label)


    # PROBABILITIES
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0] * 100

        prob_df = pd.DataFrame({
            "Category": target_encoder.classes_,
            "Probability (%)": np.round(probs, 2)
        }).sort_values("Probability (%)", ascending=False)

        st.subheader("📊 Risk Probability Distribution")

        # -------- BAR CHART --------
        fig, ax = plt.subplots()
        ax.barh(
            prob_df["Category"],
            prob_df["Probability (%)"]
        )
        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        st.pyplot(fig, use_container_width=True)

        # -------- GAUGE STYLE (TOP RISK) --------
        top = prob_df.iloc[0]

        st.subheader("🎯 Highest Risk Confidence")
        st.write(f"**{top['Category']}**")
        st.progress(int(top["Probability (%)"]))

        st.dataframe(prob_df, use_container_width=True)

    st.balloons()
