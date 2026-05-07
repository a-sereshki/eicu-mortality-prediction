"""
ICU Mortality Prediction - Streamlit App
A clinical decision support demonstration based on the eICU Demo dataset.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

# ===== Page config =====
st.set_page_config(
    page_title="ICU Mortality Prediction",
    page_icon="H",
    layout="wide",
)

# ===== Load model and reference data (cached) =====
@st.cache_resource
def load_model():
    return joblib.load("models/lightgbm_model.pkl")

@st.cache_data
def load_defaults():
    with open("data/feature_defaults.json") as f:
        return json.load(f)

@st.cache_data
def load_modes():
    with open("data/feature_modes.json") as f:
        return json.load(f)

@st.cache_data
def load_template():
    with open("data/feature_template.json") as f:
        return json.load(f)

model = load_model()
defaults = load_defaults()
modes = load_modes()
template = load_template()
feature_order = template["column_order"]
categorical_cols = ["unittype", "numbedscategory", "region"]

# ===== Header =====
st.title("ICU Mortality Prediction")
st.markdown(
    "Predict in-hospital mortality risk from first-24-hour ICU data. "
    "Model trained on the eICU Collaborative Research Database Demo (1,424 patients, 186 hospitals)."
)

st.warning(
    "**Demonstration limitation.** This interactive demo uses cohort-level defaults "
    "for measurement frequency and missingness features that the underlying model "
    "relies on. Predictions for individual patients should be interpreted as illustrative "
    "of the model's logic rather than precise risk assessments. The full model performance "
    "(ROC-AUC = 0.83) is reported on properly-formed test data in the project notebooks."
)

# ===== Sidebar with model info =====
with st.sidebar:
    st.header("About this model")
    st.markdown("**Model:** LightGBM gradient boosted classifier")
    st.markdown("**Features:** 102 (vitals, labs, demographics, severity)")
    st.markdown("**Performance:** ROC-AUC = 0.829 (test), 0.791 (cross-hospital)")
    st.markdown("**Mortality prevalence:** 8.3%")
    st.markdown("---")
    st.markdown("[GitHub repo](https://github.com/a-sereshki/eicu-mortality-prediction)")

# ===== Input form =====
st.header("Patient information")
st.markdown("Fill in the fields below. Unspecified features will use cohort medians.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=65)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi_input = st.number_input("BMI (kg/m2)", min_value=12.0, max_value=60.0, value=28.0, step=0.5)
    
    st.subheader("Severity")
    gcs = st.slider("Glasgow Coma Scale (GCS)", min_value=3, max_value=15, value=14)

with col2:
    st.subheader("Vital signs (means over 24h)")
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
    
    st.subheader("Hospital")
    unittype = st.selectbox("ICU type", ["Med-Surg ICU", "MICU", "SICU", "Cardiac ICU", "CCU-CTICU", "Neuro ICU", "CTICU"])

# ===== Predict button =====
st.markdown("---")
if st.button("Predict mortality risk", type="primary"):
    
    features = {col: defaults.get(col, 0) for col in feature_order if col not in categorical_cols}
    
    features["age"] = age
    features["gender_male"] = 1 if gender == "Male" else 0
    features["bmi"] = bmi_input
    features["bmi_missing"] = False
    features["gcs_min"] = gcs
    features["gcs_max"] = gcs
    features["gcs_mean"] = gcs
    features["gcs_missing"] = False
    features["hr_mean"] = hr_mean
    features["sbp_mean"] = sbp_mean
    features["sbp_max"] = sbp_mean + 20
    features["sbp_min"] = sbp_mean - 20
    features["map_mean"] = map_mean
    features["rr_mean"] = rr_mean
    features["rr_min"] = max(4, rr_mean - 5)
    features["spo2_mean"] = spo2_mean
    features["spo2_min"] = max(70, spo2_mean - 3)
    features["temp_mean"] = temp_mean
    features["temp_max"] = temp_mean + 0.3
    features["bun_max"] = bun_max
    features["creatinine_max"] = creatinine_max
    features["glucose_min"] = glucose_min
    features["wbc_mean"] = wbc_mean
    
    X_input = pd.DataFrame([features])
    X_input["unittype"] = unittype
    X_input["numbedscategory"] = modes["numbedscategory"]
    X_input["region"] = modes["region"]
    X_input = X_input[feature_order]
    
    for col in categorical_cols:
        X_input[col] = X_input[col].astype("category")
    
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
        st.markdown("- Low risk threshold: < 10%")
        st.markdown("- High risk threshold: > 25%")
    
    with col_b:
        st.subheader("Why this prediction?")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        
        feature_impact = pd.DataFrame({
            "feature": X_input.columns,
            "value": X_input.iloc[0].values,
            "shap": shap_values[0],
        })
        feature_impact["abs_shap"] = np.abs(feature_impact["shap"])
        top_features = feature_impact.nlargest(10, "abs_shap").iloc[::-1]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#D85A30" if x > 0 else "#185FA5" for x in top_features["shap"]]
        ax.barh(top_features["feature"], top_features["shap"], color=colors, edgecolor="white")
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("SHAP value (impact on prediction)")
        ax.set_title("Top 10 features driving this prediction")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("Red bars increase mortality risk - Blue bars decrease risk")
