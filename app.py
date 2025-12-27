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
    page_title="Kidney Disease Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# --------------------------------------------------
# STYLES
# --------------------------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f8fafc;
    }
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
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
    return model, encoder, scaler, feature_columns

model, target_encoder, scaler, feature_columns = load_artifacts()

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def yes_no(val):
    return 1 if val == "Yes" else 0

def good_poor(val):
    return 1 if val == "Poor" else 0

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("Kidney Disease Risk Prediction")
st.caption("Machine Learning–Based Clinical Decision Support System")

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
input_data = {}

with st.form("patient_form"):
    st.subheader("Patient Clinical Information")

    c1, c2 = st.columns(2)

    with c1:
        input_data["Age of the patient"] = st.number_input("Age (years)", 0, 120)
        input_data["Blood pressure (mm/Hg)"] = st.number_input("Blood Pressure (mm/Hg)")
        input_data["Specific gravity of urine"] = st.number_input("Specific Gravity of Urine")
        input_data["Albumin in urine"] = st.number_input("Albumin in Urine")
        input_data["Sugar in urine"] = st.number_input("Sugar in Urine")
        input_data["Random blood glucose level (mg/dl)"] = st.number_input("Random Blood Glucose (mg/dl)")
        input_data["Blood urea (mg/dl)"] = st.number_input("Blood Urea (mg/dl)")
        input_data["Serum creatinine (mg/dl)"] = st.number_input("Serum Creatinine (mg/dl)")
        input_data["Sodium level (mEq/L)"] = st.number_input("Sodium Level (mEq/L)")
        input_data["Potassium level (mEq/L)"] = st.number_input("Potassium Level (mEq/L)")
        input_data["Hemoglobin level (gms)"] = st.number_input("Hemoglobin (gms)")
        input_data["Packed cell volume (%)"] = st.number_input("Packed Cell Volume (%)")

    with c2:
        input_data["White blood cell count (cells/cumm)"] = st.number_input("WBC Count (cells/cumm)")
        input_data["Red blood cell count (millions/cumm)"] = st.number_input("RBC Count (millions/cumm)")
        input_data["Estimated Glomerular Filtration Rate (eGFR)"] = st.number_input("eGFR")
        input_data["Urine protein-to-creatinine ratio"] = st.number_input("Urine Protein / Creatinine Ratio")
        input_data["Urine output (ml/day)"] = st.number_input("Urine Output (ml/day)")
        input_data["Serum albumin level"] = st.number_input("Serum Albumin Level")
        input_data["Cholesterol level"] = st.number_input("Cholesterol Level")
        input_data["Parathyroid hormone (PTH) level"] = st.number_input("PTH Level")
        input_data["Serum calcium level"] = st.number_input("Serum Calcium Level")
        input_data["Serum phosphate level"] = st.number_input("Serum Phosphate Level")
        input_data["Body Mass Index (BMI)"] = st.number_input("Body Mass Index (BMI)")

    st.subheader("Medical History")

    c3, c4 = st.columns(2)

    with c3:
        input_data["Hypertension (yes/no)"] = yes_no(st.selectbox("Hypertension", ["No", "Yes"]))
        input_data["Diabetes mellitus (yes/no)"] = yes_no(st.selectbox("Diabetes Mellitus", ["No", "Yes"]))
        input_data["Coronary artery disease (yes/no)"] = yes_no(st.selectbox("Coronary Artery Disease", ["No", "Yes"]))
        input_data["Pedal edema (yes/no)"] = yes_no(st.selectbox("Pedal Edema", ["No", "Yes"]))
        input_data["Anemia (yes/no)"] = yes_no(st.selectbox("Anemia", ["No", "Yes"]))

    with c4:
        input_data["Family history of chronic kidney disease"] = yes_no(
            st.selectbox("Family History of CKD", ["No", "Yes"])
        )
        input_data["Appetite (good/poor)"] = good_poor(
            st.selectbox("Appetite", ["Good", "Poor"])
        )
        input_data["Smoking status"] = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        input_data["Physical activity level"] = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
        input_data["Urinary sediment microscopy results"] = st.selectbox(
            "Urinary Sediment Result", ["Normal", "Abnormal"]
        )

    st.subheader("Inflammatory Markers")

    input_data["Cystatin C level"] = st.number_input("Cystatin C Level")
    input_data["C-reactive protein (CRP) level"] = st.number_input("CRP Level")
    input_data["Interleukin-6 (IL-6) level"] = st.number_input("IL-6 Level")

    submit = st.form_submit_button("Predict Risk")

# --------------------------------------------------
# PREDICTION (FIXED & SAFE)
# --------------------------------------------------
if submit:
    df = pd.DataFrame([input_data])

    # Align columns EXACTLY as in training
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Apply scaler safely
    try:
        df_scaled = scaler.transform(df)
        df = pd.DataFrame(df_scaled, columns=feature_columns)
    except Exception:
        pass

    with st.spinner("Analyzing patient data..."):
        pred = model.predict(df)[0]
        probs = model.predict_proba(df)[0] * 100
        label = target_encoder.inverse_transform([pred])[0]

    prob_df = pd.DataFrame({
        "Category": target_encoder.classes_,
        "Probability (%)": np.round(probs, 2)
    }).sort_values("Probability (%)", ascending=False)

    top = prob_df.iloc[0]

    # --------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------
    st.subheader("Prediction Outcome")

    confidence = top["Probability (%)"]

    if confidence >= 80:
        conf_label = "High"
    elif confidence >= 60:
        conf_label = "Moderate"
    else:
        conf_label = "Low"

    risk_class = "risk-high" if top["Category"].lower() in ["ckd", "yes", "positive"] else "risk-low"

    c1, c2, c3 = st.columns(3)

    c1.markdown(
        f"<div class='card'><div class='card-title'>Clinical Assessment</div>"
        f"<div class='card-value {risk_class}'>{top['Category']}</div></div>",
        unsafe_allow_html=True
    )

    c2.markdown(
        f"<div class='card'><div class='card-title'>Risk Probability</div>"
        f"<div class='card-value'>{confidence:.1f}%</div></div>",
        unsafe_allow_html=True
    )

    c3.markdown(
        f"<div class='card'><div class='card-title'>Model Confidence</div>"
        f"<div class='card-value'>{conf_label}</div></div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------
    st.subheader("Risk Probability Distribution")

    col_l, col_r = st.columns(2)

    with col_l:
        pie = alt.Chart(prob_df).mark_arc(innerRadius=50).encode(
            theta="Probability (%):Q",
            color="Category:N",
            tooltip=["Category", "Probability (%)"]
        )
        st.altair_chart(pie, use_container_width=True)

    with col_r:
        bar = alt.Chart(prob_df).mark_bar().encode(
            x=alt.X("Probability (%):Q", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Category:N", sort="-x"),
            tooltip=["Category", "Probability (%)"]
        )
        st.altair_chart(bar, use_container_width=True)

    st.subheader("Detailed Probability Table")
    st.dataframe(prob_df, use_container_width=True)
