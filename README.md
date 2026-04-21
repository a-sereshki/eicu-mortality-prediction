# Predicting ICU Mortality Across 186 Hospitals

A cross-site generalization study using the eICU Collaborative Research Database Demo (v2.0.1). This project trains an ICU mortality prediction model on data from 186 US hospitals and explicitly evaluates how well it generalizes across hospital size categories.

## Status

**Week 2 in progress.** Cohort defined and characterized; exploratory analysis complete. Feature engineering begins next.

**Completed:**
- PostgreSQL database loaded with the full eICU Demo (31 tables, 5M+ records)
- Analytic cohort defined: 1,424 adult ICU patients across 186 hospitals (8.29% mortality)
- Hospital characterization: viable stratification by bed count, teaching status, and region
- Cohort characterization: demographics, age-stratified mortality, Apache severity gradient
- First exploratory notebook with age-mortality visualization

**In progress:**
- Feature engineering (first-24-hour vitals, labs, Apache components)
- Baseline and gradient-boosted mortality models
- Cross-hospital generalization analysis (primary: bed size; secondary: teaching status, region)
- Streamlit deployment with SHAP explainability

## Problem

Most clinical machine learning models are trained and evaluated on data from a single hospital, which inflates reported performance and obscures real-world deployment challenges. This project trains an ICU mortality prediction model on the eICU Demo dataset and explicitly evaluates how well it generalizes across hospitals of different sizes and types.

Unadjusted mortality varies substantially across hospital size categories in this dataset (5.4–12.5%), with mid-large hospitals (250–499 beds) showing roughly double the mortality of other size categories. This raises a concrete question: do models trained on one hospital size category accurately predict mortality at others?

## Data

The eICU Collaborative Research Database Demo v2.0.1, a publicly available subset of the full eICU database. The demo contains data from 2,520 ICU stays across 186 hospitals, with vital signs, laboratory results, admission diagnoses, severity scores, and outcomes. Per the data use agreement, no patient-level data is stored in this repository.

## Cohort

The analytic cohort includes 1,424 adult ICU patients after applying these inclusion criteria:

| Step | Criterion | N | Dropped |
|------|-----------|---|---------|
| 0 | All patient stays | 2,520 | — |
| 1 | First ICU stay per patient | 2,174 | 346 |
| 2 | Adult patients (age ≥ 18) | 2,166 | 8 |
| 3 | Non-missing discharge status | 2,141 | 25 |
| 4 | ICU stay ≥ 24 hours | 1,424 | 717 |

Overall mortality rate: 8.29% (118 deaths). See `src/sql/03_cohort_definition.sql` for the full specification.

## Key findings (exploratory)

Mortality rises monotonically with age, from 1.8% in patients under 40 to 16.7% in patients aged 85+ — a 9-fold gradient.

The Apache IVa severity score shows a 28-fold mortality gradient (1.3% at scores <30 to 37.0% at scores ≥90), validating it as a powerful univariate predictor. 16% of cohort patients are missing Apache scores and will require imputation or alternative feature construction.

Mortality varies non-monotonically across hospital size categories, with mid-large hospitals (250–499 beds) as an outlier — motivating the cross-hospital generalization analysis.

## Approach

The prediction task uses only the first 24 hours of ICU data to predict in-hospital mortality. Features will include demographics, aggregated vital signs, laboratory values, and Apache severity components. Baseline logistic regression is compared against a LightGBM gradient boosting model, with SHAP values for feature importance and individual prediction explanations. Cross-hospital generalization is evaluated by training on a subset of hospital size categories and testing on held-out ones.

## Repository structure

- `notebooks/` — exploratory and analysis notebooks
- `src/sql/` — documented SQL queries for cohort definition and characterization
- `src/db.py` — reusable database connection utility
- `results/` — figures and model performance outputs (coming in Week 4)
- `docs/` — supplementary documentation

## Reproducing this work

Requires Python 3.11, PostgreSQL 16, and the eICU Demo dataset (publicly available from PhysioNet). See `requirements.txt` for Python dependencies and `.env.example` for required environment variables. Setup instructions will be consolidated in Week 5.

## Author

Azadeh Sereshki — MASc Biomedical Engineering, University of Toronto.

## License

MIT
