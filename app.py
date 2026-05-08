"""
ICU Mortality Prediction - Streamlit Demo
Based on the eICU Demo dataset.
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="ICU Mortality Prediction",
    page_icon="H",
    layout="wide",
)

@st.cache_resource
def load_model():
    return joblib.load("models/demo_model.pkl")

@st.cache_data
def load_demo_features():
    with open("data/demo_features.json") as f:
        return json.load(f)

model = load_model()
demo_info = load_demo_features()
DEMO_FEATURES = demo_info["features"]

st.title("ICU Mortality Prediction")
st.markdown(
    "Predict in-hospital mortality risk from first-24-hour ICU data. "
    "Trained on the eICU Collaborative Research Database Demo (1,424 patients, 186 hospitals)."
)

st.info(
    "**About this demo:** This interactive form uses a simplified 14-feature logistic regression "
    "(test ROC-AUC = 0.85) trained specifically for form-based input. The full project model "
    "(102-feature LightGBM, ROC-AUC = 0.83 in-distribution, 0.79 across hospitals) is documented "
    "in the [project notebooks](https://github.com/a-sereshki/eicu-mortality-prediction). "
    "Educational purposes only - not validated for clinical use."
)

with st.sidebar:
    st.header("About this demo")
    st.markdown("**Demo model:** Logistic regression (14 features)")
    st.markdown("**Test ROC-AUC:** 0.85")
    st.markdown("**Test PR-AUC:** 0.36")
    st.markdown("**Cohort prevalence:** 8.3%")
    st.markdown("---")
    st.markdown("**Full model (in notebooks):**")
    st.markdown("LightGBM, 102 features")
    st.markdown("ROC-AUC 0.83 (within-distribution)")
    st.markdown("ROC-AUC 0.79 (cross-hospital)")
    st.markdown("---")
    st.markdown("[GitHub repo](https://github.com/a-sereshki/eicu-mortality-prediction)")

st.header("Patient information")
st.markdown("Enter values from the patient\'s first 24 hours in the ICU.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics & Severity")
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=65)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI (kg/m2)", min_value=12.0, max_value=60.0, value=28.0, step=0.5)
    gcs = st.slider("Glasgow Coma Scale (mean)", min_value=3, max_value=15, value=14)

with col2:
    st.subheader("Vital signs (24h means)")
    hr_mean = st.number_input("Heart rate (bpm)", min_value=30, max_value=200, value=85)
    sbp_mean = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=120)
    map_mean = st.number_input("Mean arterial pressure (mmHg)", min_value=40, max_value=180, value=80)
    rr_mean = st.number_input("Respiratory rate (breaths/min)", min_value=4, max_value=50, value=18)
    spo2_mean = st.number_input("SpO2 (%)", min_value=70, max_value=100, value=97)
    temp_mean = st.number_input("Temperature (C)", min_value=32.0, max_value=42.0, value=37.0, step=0.1)

with col3:
    st.subheader("Laboratory values")
    bun_max = st.number_input("BUN max (mg/dL)", min_value=1, max_value=200, value=20)
    creatinine_max = st.number_input("Creatinine max (mg/dL)", min_value=0.1, max_value=15.0, value=1.0, step=0.1)
    glucose_min = st.number_input("Glucose min (mg/dL)", min_value=20, max_value=500, value=100)
    wbc_mean = st.number_input("WBC (x10^3/uL)", min_value=0.5, max_value=80.0, value=10.0, step=0.5)

st.markdown("---")
if st.button("Predict mortality risk", type="primary"):
    feature_values = {
        "age": age,
        "gender_male": 1 if gender == "Male" else 0,
        "bmi": bmi,
        "gcs_mean": gcs,
        "hr_mean": hr_mean,
        "sbp_mean": sbp_mean,
        "map_mean": map_mean,
        "rr_mean": rr_mean,
        "spo2_mean": spo2_mean,
        "temp_mean": temp_mean,
        "bun_max": bun_max,
        "creatinine_max": creatinine_max,
        "glucose_min": glucose_min,
        "wbc_mean": wbc_mean,
    }
    X_input = pd.DataFrame([[feature_values[f] for f in DEMO_FEATURES]], columns=DEMO_FEATURES)

    proba = model.predict_proba(X_input)[0, 1]

    st.header("Prediction")
    col_a, col_b = st.columns([1, 2])

    with col_a:
        if proba < 0.10:
            risk_label = "Low risk"
        elif proba < 0.25:
            risk_label = "Moderate risk"
        else:
            risk_label = "High risk"

        st.metric(
            label="Predicted mortality probability",
            value=f"{proba*100:.1f}%",
            delta=risk_label,
            delta_color="off",
        )
        st.markdown("**Reference points:**")
        st.markdown("- Cohort prevalence: 8.3%")
        st.markdown("- Low risk: < 10%")
        st.markdown("- High risk: > 25%")

    with col_b:
        st.subheader("Why this prediction?")

        imputer = model.named_steps["imputer"]
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]

        X_imputed = imputer.transform(X_input)
        X_scaled = scaler.transform(X_imputed)
        contributions = clf.coef_[0] * X_scaled[0]

        contrib_df = pd.DataFrame({
            "feature": DEMO_FEATURES,
            "contribution": contributions,
        })
        contrib_df["abs_contribution"] = np.abs(contrib_df["contribution"])
        contrib_df = contrib_df.sort_values("abs_contribution").reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#D85A30" if x > 0 else "#185FA5" for x in contrib_df["contribution"]]
        ax.barh(contrib_df["feature"], contrib_df["contribution"], color=colors, edgecolor="white")
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Contribution to log-odds of mortality")
        ax.set_title("Feature contributions to this prediction")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        st.pyplot(fig)

        st.caption("Red bars increase mortality risk - Blue bars decrease risk")
