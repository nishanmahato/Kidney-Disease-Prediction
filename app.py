import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import altair as alt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Kidney Disease Risk Dashboard",
    page_icon="🩺",
    layout="centered"
)

# --------------------------------------------------
# STYLING (CARD UI)
# --------------------------------------------------
st.markdown(
    """
    <style>
    .card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        text-align: center;
    }
    .card-title {
        font-size: 0.85rem;
        color: #6b7280;
    }
    .card-value {
        font-size: 1.4rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD ARTIFACTS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "rf_kidney_disease_model.pkl")
    encoder = joblib.load(BASE_DIR / "target_label_encoder.pkl")
    scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")
    features = joblib.load(BASE_DIR / "feature_columns.pkl")
    return model, encoder, scaler, features

model, target_encoder, scaler, feature_columns = load_artifacts()

# --------------------------------------------------
# EXPLICIT UI FEATURE DEFINITION (CRITICAL FIX)
# --------------------------------------------------
ui_fields = {
    "age": {"type": "number", "label": "Age (years)"},
    "blood_pressure": {"type": "number", "label": "Blood Pressure"},
    "hypertension": {"type": "yesno", "label": "Hypertension"},
    "diabetes_mellitus": {"type": "yesno", "label": "Diabetes"},
    "anemia": {"type": "yesno", "label": "Anemia"},
    "pedal_edema": {"type": "yesno", "label": "Pedal Edema"}
}

def yes_no(val):
    return 1 if val == "Yes" else 0

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.title("🩺 Kidney Disease Risk Prediction")
st.subheader("Patient Information")

input_data = {}

with st.form("patient_form"):
    cols = st.columns(2)

    for i, (feature, config) in enumerate(ui_fields.items()):
        with cols[i % 2]:
            if config["type"] == "yesno":
                choice = st.selectbox(config["label"], ["No", "Yes"])
                input_data[feature] = yes_no(choice)
            else:
                input_data[feature] = st.number_input(
                    config["label"], value=0.0, format="%.2f"
                )

    submit = st.form_submit_button("🔍 Predict Risk")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if submit:
    df = pd.DataFrame([input_data])

    # Fill missing features with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]
    df[scaler.feature_names_in_] = scaler.transform(df[scaler.feature_names_in_])

    pred = model.predict(df)[0]
    label = target_encoder.inverse_transform([pred])[0]

    probs = model.predict_proba(df)[0] * 100

    prob_df = pd.DataFrame({
        "Category": target_encoder.classes_,
        "Probability": np.round(probs, 2)
    }).sort_values("Probability", ascending=False)

    top = prob_df.iloc[0]

    # --------------------------------------------------
    # RESULT DASHBOARD (LIKE YOUR IMAGE)
    # --------------------------------------------------
    st.subheader("Review Your Prediction Outcome")

    c1, c2, c3 = st.columns(3)

    c1.markdown(
        f"<div class='card'><div class='card-title'>Risk Level</div>"
        f"<div class='card-value'>{top['Category']}</div></div>",
        unsafe_allow_html=True
    )

    c2.markdown(
        f"<div class='card'><div class='card-title'>Risk Probability</div>"
        f"<div class='card-value'>{top['Probability']}%</div></div>",
        unsafe_allow_html=True
    )

    c3.markdown(
        f"<div class='card'><div class='card-title'>Model Confidence</div>"
        f"<div class='card-value'>High</div></div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------
    # HORIZONTAL BAR CHART
    # --------------------------------------------------
    st.subheader("Risk Probability Distribution")

    chart = (
        alt.Chart(prob_df)
        .mark_bar()
        .encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Category:N", sort="-x"),
            tooltip=["Category", "Probability"]
        )
    )

    st.altair_chart(chart, use_container_width=True)

    # --------------------------------------------------
    # HORIZONTAL CONFIDENCE BAR
    # --------------------------------------------------
    st.subheader("Highest Risk Confidence")

    colA, colB = st.columns([1, 4])
    colA.markdown(f"**{top['Category']}**")
    colB.progress(int(top["Probability"]))

    st.dataframe(prob_df, use_container_width=True)
