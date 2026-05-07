# Predicting ICU Mortality Across 186 Hospitals

A clinical machine learning project that predicts in-hospital mortality from the first 24 hours of ICU data, with a focus on cross-hospital generalization.

**🚀 [Try the live interactive demo](https://eicu-mortality-prediction-cfyzvung7fmczatg9dknhp.streamlit.app/)** — enter patient vitals and labs, see model predictions with SHAP explanations.

**Headline result: ROC-AUC 0.83 within distribution, 0.79 ± 0.01 across held-out hospital size categories — quantifying the real-world deployment gap most published clinical ML models leave unexamined.**

## Why this project

Most clinical ML papers report performance on a single hospital's data. Real-world deployment requires the model to work at hospitals it wasn't trained on. This project trains a mortality prediction model on the [eICU Collaborative Research Database Demo v2.0.1](https://physionet.org/content/eicu-crd-demo/2.0.1/) (1,424 ICU patients, 186 US hospitals) and explicitly quantifies how performance changes when forced to generalize across hospital size categories.

## Dataset and cohort

The eICU Demo provides ICU stays from 186 hospitals across 4 US census regions and 4 hospital bed-size categories. After applying inclusion criteria (first ICU stay per patient, age ≥ 18, non-missing discharge status, ICU stay ≥ 24 hours), the analytic cohort contains 1,424 patients with an 8.29% in-hospital mortality rate. The full consort flow is in `src/sql/03_cohort_definition.sql`.

| Step | Criterion | N | Dropped |
|------|-----------|---|---------|
| 0 | All patient stays | 2,520 | — |
| 1 | First ICU stay per patient | 2,174 | 346 |
| 2 | Adult patients (age ≥ 18) | 2,166 | 8 |
| 3 | Non-missing discharge status | 2,141 | 25 |
| 4 | ICU stay ≥ 24 hours | 1,424 | 717 |

## Feature engineering

102 features per patient, derived from the first 24 hours of ICU stay:

| Category | Variables | Source tables |
|----------|-----------|---------------|
| Demographics | Age, gender, BMI, ICU type, hospital characteristics | patient, hospital |
| Vital signs | HR, SBP, DBP, MAP, RR, SpO2, temperature (min/max/mean/count + missingness flag each) | vitalperiodic, vitalaperiodic, nursecharting |
| Laboratory values | Sodium, potassium, glucose, creatinine, BUN, hemoglobin, WBC, platelets, bicarbonate, total bilirubin (min/max/mean/count + missingness flag each) | lab |
| Severity | Glasgow Coma Scale total | nursecharting |

Two methodological notes worth flagging:

**Combined measurement sources.** Blood pressure was pulled from both continuous arterial monitoring (`vitalperiodic.systemic*`) and intermittent cuff measurements (`vitalaperiodic.noninvasive*`), expanding coverage from 17% to 96% of the cohort. Temperature was similarly combined from continuous and nursing-chart sources, expanding coverage from 8% to 94%.

**Missingness as signal.** Each variable has a binary missingness indicator alongside imputed values. ICU patients without continuous heart rate monitoring had 3.7% mortality vs. 8.5% for monitored patients — missingness itself encodes acuity, and the model can use that.

## Modeling

| Model | ROC-AUC | PR-AUC | Notes |
|-------|---------|--------|-------|
| Logistic regression (baseline) | **0.845** | 0.257 | Median imputation, L2 regularization, numeric features only |
| LightGBM | 0.829 | **0.314** | Native categorical handling and missing-value support |

The two models showed complementary strengths. Logistic regression had marginally better overall discrimination (ROC-AUC); LightGBM was substantially better at high-precision identification of the highest-risk patients (PR-AUC). For clinical deployment scenarios prioritizing the top-N most-at-risk patients, LightGBM is the more useful model despite slightly lower overall AUC.

## Cross-hospital generalization (the headline analysis)

Leave-one-bed-category-out validation: the model was trained on 3 of 4 hospital bed-size categories and tested on the held-out 4th, rotated through all categories.

| Held-out category | N test | Test deaths | Test mortality | ROC-AUC | PR-AUC |
|-------------------|--------|-------------|----------------|---------|--------|
| 100–249 beds | 465 | 32 | 6.88% | 0.807 | 0.254 |
| 250–499 beds | 264 | 33 | 12.50% | 0.797 | 0.405 |
| < 100 beds | 277 | 15 | 5.42% | 0.784 | 0.166 |
| ≥ 500 beds | 207 | 13 | 6.28% | 0.775 | 0.309 |

**Mean held-out ROC-AUC: 0.791 ± 0.014**

Compared to within-distribution performance (ROC-AUC = 0.829), the model loses approximately 4 percentage points when forced to generalize to a hospital size category not seen during training. The gap is consistent across categories (std = 0.014), suggesting the model has learned both generalizable physiologic patterns and some site-specific patterns. Real-world deployment to a new hospital should anticipate this small but consistent performance penalty.

## Explainability

SHAP analysis of the LightGBM model identified clinically coherent top features:

1. `bun_max` (kidney injury marker)
2. `spo2_count` (monitoring frequency as acuity proxy)
3. `gcs_max` (consciousness level)
4. `age`
5. `temp_max` (infection / hyperthermia)

Notable: no hospital characteristic features (region, unit type, bed category) appeared in the top 20 by importance, supporting that the model relies primarily on patient-level physiology rather than site-specific patterns. The 4-percentage-point cross-hospital generalization gap appears to come from the way physiologic features are *measured and recorded* across hospitals, not from the model directly using site identifiers.

## Repository structure
## Reproducing this work

Requirements: Python 3.11, PostgreSQL 16, the eICU Demo dataset (publicly available from PhysioNet). Python dependencies in `requirements.txt`. Database credentials configured via `.env` (template in `.env.example`). Per the eICU data use agreement, no patient data is committed to this repository.

## Honest limitations

The eICU Demo contains 1,424 patients across 186 hospitals. With only ~30 deaths per hospital size category, statistical power for cross-hospital comparisons is limited. The findings here should be regarded as illustrative of the methodology rather than definitive estimates of generalization gap. The full eICU database (200,000+ patients) would enable substantially more robust analysis.

## Author

**Azadeh Sereshki** — MASc Biomedical Engineering, University of Toronto
