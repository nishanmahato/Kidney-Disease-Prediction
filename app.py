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
# STYLES (REALISTIC CLINICAL UI)
# --------------------------------------------------
st.markdown(
    """
    <style>
    .card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
    }
    .card-title {
        font-size: 0.8rem;
        color: #6b7280;
    }
    .card-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #111827;
    }
    .risk-high { color: #b91c1c; }
    .risk-low { color: #047857; }
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
# AUTO FEATURE TYPE DETECTION
# --------------------------------------------------
YES_NO_KEYWORDS = ["hypertension", "diabetes", "anemia", "edema", "cad"]
GOOD_POOR_KEYWORDS = ["appetite"]

def feature_type(col):
    c = col.lower()
    if any(k in c for k in GOOD_POOR_KEYWORDS):
        return "good_poor"
    if any(k in c for k in YES_NO_KEYWORDS):
        return "yes_no"
    return "number"

def yes_no(v): return 1 if v == "Yes" else 0
def good_poor(v): return 1 if v == "Poor" else 0

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.title("Kidney Disease Risk Prediction")
st.caption("Clinical decision support system")

st.subheader("Patient Information")

input_data = {}

with st.form("patient_form"):
    cols = st.columns(2)

    for i, col in enumerate(feature_columns):
        label = col.replace("_", " ").title()
        ftype = feature_type(col)

        with cols[i % 2]:
            if ftype == "yes_no":
                val = st.selectbox(label, ["No", "Yes"])
                input_data[col] = yes_no(val)

            elif ftype == "good_poor":
                val = st.selectbox(label, ["Good", "Poor"])
                input_data[col] = good_poor(val)

            else:
                input_data[col] = st.number_input(label, value=0.0, format="%.2f")

    submit = st.form_submit_button("Predict Risk")

# --------------------------------------------------
# PREDICTION & DASHBOARD
# --------------------------------------------------
if submit:
    df = pd.DataFrame([input_data])

    df[scaler.feature_names_in_] = scaler.transform(
        df[scaler.feature_names_in_]
    )
    df = df[feature_columns]

    with st.spinner("Analyzing patient data..."):
        pred = model.predict(df)[0]
        label = target_encoder.inverse_transform([pred])[0]
        probs = model.predict_proba(df)[0] * 100

    prob_df = pd.DataFrame({
        "Category": target_encoder.classes_,
        "Probability": np.round(probs, 2)
    }).sort_values("Probability", ascending=False)

    top = prob_df.iloc[0]

    # --------------------------------------------------
    # REVIEW YOUR PREDICTION OUTCOME
    # --------------------------------------------------
    st.subheader("Review Your Prediction Outcome")

    risk_class = "risk-high" if top["Category"].lower() in ["ckd", "yes", "high"] else "risk-low"

    c1, c2, c3 = st.columns(3)

    c1.markdown(
        f"<div class='card'><div class='card-title'>Clinical Risk Assessment</div>"
        f"<div class='card-value {risk_class}'>{top['Category']}</div></div>",
        unsafe_allow_html=True
    )

    c2.markdown(
        f"<div class='card'><div class='card-title'>Estimated Risk Probability</div>"
        f"<div class='card-value'>{top['Probability']}%</div></div>",
        unsafe_allow_html=True
    )

    c3.markdown(
        f"<div class='card'><div class='card-title'>Model Confidence</div>"
        f"<div class='card-value'>High</div></div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------
    # PIE + BAR CHART (SINGLE ROW)
    # --------------------------------------------------
    st.subheader("Risk Probability Visualization")

    col_left, col_right = st.columns(2)

    with col_left:
        pie = (
            alt.Chart(prob_df)
            .mark_arc(innerRadius=50)
            .encode(
                theta="Probability:Q",
                color="Category:N",
                tooltip=["Category", "Probability"]
            )
        )
        st.altair_chart(pie, use_container_width=True)

    with col_right:
        bar = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Category:N", sort="-x"),
                tooltip=["Category", "Probability"]
            )
        )
        st.altair_chart(bar, use_container_width=True)

    # --------------------------------------------------
    # CONFIDENCE BAR
    # --------------------------------------------------
    st.subheader("Prediction Confidence")

    conf1, conf2 = st.columns([1, 4])
    conf1.markdown("**Confidence Score**")
    conf2.progress(int(top["Probability"]))

    st.dataframe(prob_df, use_container_width=True)
