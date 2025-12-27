import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import altair as alt

# --------------------------------------------------
# PAGE CONFIG (Mobile Optimized)
# --------------------------------------------------
st.set_page_config(
    page_title="Kidney Disease Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# --------------------------------------------------
# GLOBAL STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stForm"] {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🩺 Kidney Disease Risk Prediction System")
st.caption("Machine Learning–based clinical decision support")

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# LOAD MODEL ARTIFACTS
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
# BINARY FEATURES (MATCH TRAINING FEATURES EXACTLY)
# --------------------------------------------------
binary_feature_names = {
    "hypertension",
    "diabetes_mellitus",
    "coronary_artery_disease",
    "appetite",
    "anemia",
    "pedal_edema"
}

binary_features = [
    col for col in feature_columns if col.lower() in binary_feature_names
]

def yes_no_to_numeric(value: str) -> int:
    return 1 if value == "Yes" else 0

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.subheader("🧾 Patient Information")

input_data = {}

with st.form("prediction_form"):
    cols = st.columns(2)

    for i, col_name in enumerate(feature_columns):
        label = col_name.replace("_", " ").title()

        with cols[i % 2]:
            if col_name in binary_features:
                choice = st.selectbox(
                    label,
                    options=["No", "Yes"],
                    index=0
                )
                input_data[col_name] = yes_no_to_numeric(choice)
            else:
                input_data[col_name] = st.number_input(
                    label,
                    value=0.0,
                    format="%.2f"
                )

    submitted = st.form_submit_button("🔍 Predict Risk")

# --------------------------------------------------
# PREDICTION LOGIC
# --------------------------------------------------
if submitted:
    input_df = pd.DataFrame([input_data])

    scaler_features = list(scaler.feature_names_in_)
    input_df[scaler_features] = scaler.transform(
        input_df[scaler_features]
    )

    input_df = input_df[feature_columns]

    with st.spinner("Analyzing patient data..."):
        pred_encoded = model.predict(input_df)[0]
        pred_label = target_encoder.inverse_transform([pred_encoded])[0]

    # --------------------------------------------------
    # RESULT DISPLAY
    # --------------------------------------------------
    st.subheader("✅ Prediction Result")

    if str(pred_label).lower() in ["ckd", "chronic kidney disease", "yes"]:
        st.error("⚠️ High Risk of Kidney Disease")
    else:
        st.success("✅ Low Risk of Kidney Disease")

    st.metric("Predicted Status", pred_label)

    # --------------------------------------------------
    # PROBABILITY VISUALIZATION
    # --------------------------------------------------
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0] * 100

        prob_df = pd.DataFrame({
            "Category": target_encoder.classes_,
            "Probability (%)": np.round(probs, 2)
        }).sort_values("Probability (%)", ascending=False)

        # ---------- HORIZONTAL BAR CHART ----------
        st.subheader("📊 Risk Probability Distribution")

        bar_chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Probability (%):Q",
                    scale=alt.Scale(domain=[0, 100]),
                    title="Probability (%)"
                ),
                y=alt.Y(
                    "Category:N",
                    sort="-x",
                    title=None
                ),
                tooltip=["Category", "Probability (%)"]
            )
        )

        st.altair_chart(bar_chart, use_container_width=True)

        # ---------- HORIZONTAL GAUGE ----------
        top = prob_df.iloc[0]

        st.subheader("🎯 Highest Risk Confidence")

        col1, col2 = st.columns([1, 4])

        with col1:
            st.markdown(f"**{top['Category']}**")

        with col2:
            st.progress(int(top["Probability (%)"]))

        st.dataframe(prob_df, use_container_width=True)
