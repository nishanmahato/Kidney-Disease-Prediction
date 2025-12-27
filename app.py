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
# GLOBAL STYLE
# --------------------------------------------------
st.markdown(
    """
    <style>
    body { background-color: #f8fafc; }
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
# LOAD MODEL ARTIFACTS
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
# ENCODING FUNCTIONS
# --------------------------------------------------
def yes_no(val): return 1 if val == "Yes" else 0
def good_poor(val): return 1 if val == "Poor" else 0
def encode_smoking(val): return {"Never": 0, "Former": 1, "Current": 2}[val]
def encode_activity(val): return {"Low": 0, "Moderate": 1, "High": 2}[val]
def encode_sediment(val): return {"Normal": 0, "Abnormal": 1}[val]

# --------------------------------------------------
# APP HEADER
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
        input_data["Specific gravity of urine"] = st.number_input("Specific Gravity")
        input_data["Albumin in urine"] = st.number_input("Albumin in Urine")
        input_data["Sugar in urine"] = st.number_input("Sugar in Urine")
        input_data["Random blood glucose level (mg/dl)"] = st.number_input("Random Blood Glucose")
        input_data["Blood urea (mg/dl)"] = st.number_input("Blood Urea")
        input_data["Serum creatinine (mg/dl)"] = st.number_input("Serum Creatinine")
        input_data["Sodium level (mEq/L)"] = st.number_input("Sodium Level")
        input_data["Potassium level (mEq/L)"] = st.number_input("Potassium Level")
        input_data["Hemoglobin level (gms)"] = st.number_input("Hemoglobin")
        input_data["Packed cell volume (%)"] = st.number_input("Packed Cell Volume")

    with c2:
        input_data["White blood cell count (cells/cumm)"] = st.number_input("WBC Count")
        input_data["Red blood cell count (millions/cumm)"] = st.number_input("RBC Count")
        input_data["Estimated Glomerular Filtration Rate (eGFR)"] = st.number_input("eGFR")
        input_data["Urine protein-to-creatinine ratio"] = st.number_input("Protein/Creatinine Ratio")
        input_data["Urine output (ml/day)"] = st.number_input("Urine Output")
        input_data["Serum albumin level"] = st.number_input("Serum Albumin")
        input_data["Cholesterol level"] = st.number_input("Cholesterol")
        input_data["Parathyroid hormone (PTH) level"] = st.number_input("PTH")
        input_data["Serum calcium level"] = st.number_input("Calcium")
        input_data["Serum phosphate level"] = st.number_input("Phosphate")
        input_data["Body Mass Index (BMI)"] = st.number_input("BMI")

    st.subheader("Medical History")

    c3, c4 = st.columns(2)

    with c3:
        input_data["Hypertension (yes/no)"] = yes_no(st.selectbox("Hypertension", ["No", "Yes"]))
        input_data["Diabetes mellitus (yes/no)"] = yes_no(st.selectbox("Diabetes", ["No", "Yes"]))
        input_data["Coronary artery disease (yes/no)"] = yes_no(st.selectbox("CAD", ["No", "Yes"]))
        input_data["Pedal edema (yes/no)"] = yes_no(st.selectbox("Pedal Edema", ["No", "Yes"]))
        input_data["Anemia (yes/no)"] = yes_no(st.selectbox("Anemia", ["No", "Yes"]))

    with c4:
        input_data["Family history of chronic kidney disease"] = yes_no(
            st.selectbox("Family History of CKD", ["No", "Yes"])
        )
        input_data["Appetite (good/poor)"] = good_poor(
            st.selectbox("Appetite", ["Good", "Poor"])
        )
        input_data["Smoking status"] = encode_smoking(
            st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        )
        input_data["Physical activity level"] = encode_activity(
            st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
        )
        input_data["Urinary sediment microscopy results"] = encode_sediment(
            st.selectbox("Urinary Sediment", ["Normal", "Abnormal"])
        )

    st.subheader("Inflammatory Markers")
    input_data["Cystatin C level"] = st.number_input("Cystatin C")
    input_data["C-reactive protein (CRP) level"] = st.number_input("CRP")
    input_data["Interleukin-6 (IL-6) level"] = st.number_input("IL-6")

    submit = st.form_submit_button("Predict Risk")

# --------------------------------------------------
# PREDICTION PIPELINE
# --------------------------------------------------
if submit:
    df = pd.DataFrame([input_data])
    df = df.reindex(columns=feature_columns, fill_value=0)

    try:
        df = pd.DataFrame(scaler.transform(df), columns=feature_columns)
    except Exception:
        pass

    with st.spinner("Analyzing patient data..."):
        pred = model.predict(df)[0]
        label = target_encoder.inverse_transform([pred])[0]

    # --------------------------------------------------
    # RESULT DASHBOARD
    # --------------------------------------------------
    st.subheader("Prediction Outcome")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0] * 100
        top_idx = np.argmax(probs)
        top_category = target_encoder.classes_[top_idx]
        top_prob = round(probs[top_idx], 2)
    else:
        top_category = label
        top_prob = 0

    risk_class = (
        "risk-high"
        if top_category.lower() in ["ckd", "yes", "positive"]
        else "risk-low"
    )

    c1, c2, c3 = st.columns(3)

    c1.markdown(
        f"<div class='card'><div class='card-title'>Clinical Assessment</div>"
        f"<div class='card-value {risk_class}'>{top_category}</div></div>",
        unsafe_allow_html=True
    )

    c2.markdown(
        f"<div class='card'><div class='card-title'>Risk Probability</div>"
        f"<div class='card-value'>{top_prob}%</div></div>",
        unsafe_allow_html=True
    )

    conf_label = "High" if top_prob >= 80 else "Moderate" if top_prob >= 60 else "Low"

    c3.markdown(
        f"<div class='card'><div class='card-title'>Model Confidence</div>"
        f"<div class='card-value'>{conf_label}</div></div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------
    # PREDICTION PROBABILITIES (AFTER DASHBOARD)
    # --------------------------------------------------
    if hasattr(model, "predict_proba"):
        prob_df = pd.DataFrame({
            "Risk Category": target_encoder.classes_,
            "Probability (%)": np.round(probs, 2)
        }).sort_values("Probability (%)", ascending=False)

        st.subheader("📊 Prediction Probabilities")
        st.dataframe(prob_df, use_container_width=True)

        # --------------------------------------------------
        # VISUALIZATION
        # --------------------------------------------------
        st.subheader("📈 Risk Probability Distribution")

        col_l, spacer, col_r = st.columns([1, 0.15, 1])

        with col_l:
            pie_chart = (
                alt.Chart(prob_df)
                .mark_arc(innerRadius=60)
                .encode(
                    theta="Probability (%):Q",
                    color="Risk Category:N",
                    tooltip=["Risk Category", "Probability (%)"]
                )
                .properties(width=300, height=300)
            )
            st.altair_chart(pie_chart, use_container_width=False)

        with col_r:
            bar_chart = (
    alt.Chart(prob_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "Risk Category:N",
            title="Risk Category",
            axis=alt.Axis(
                domain=True,        # x-axis line ON
                ticks=True,
                labelAngle=0
            )
        ),
        y=alt.Y(
            "Probability (%):Q",
            title="Probability (%)",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(
                domain=True,        # y-axis line ON
                ticks=True,
                grid=True
            )
        ),
        color=alt.Color("Risk Category:N", legend=None),
        tooltip=["Risk Category", "Probability (%)"]
    )
    .properties(
        width=420,   # ⬅ increased width
        height=380   # ⬅ increased height
    )
    .configure_axis(
        labelFontSize=12,
        titleFontSize=14,
        domainWidth=2     # ⬅ thicker axis lines
    )
)

            st.altair_chart(bar_chart, use_container_width=False)

