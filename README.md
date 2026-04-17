# Predicting ICU Mortality Across 200+ Hospitals

A cross-site generalization study using the eICU Collaborative Research Database (200,000+ ICU admissions, 335 hospitals).

## Status

Work in progress. Expected completion: [June 1, 2026].

## Problem

Most clinical machine learning models are trained and evaluated on data from a single hospital, which inflates reported performance and obscures real-world deployment challenges. This project trains an ICU mortality prediction model on eICU and explicitly evaluates how well it generalizes across hospitals of different sizes, teaching status, and regions.

## Data

The eICU Collaborative Research Database (v2.0), accessed through PhysioNet under credentialed access. Data includes vital signs, laboratory results, admission diagnoses, and outcomes for adult ICU patients across 335 US hospitals. Per the data use agreement, no patient-level data is stored in this repository.

## Approach

The prediction task uses only the first 24 hours of ICU data to predict in-hospital mortality. Features include demographics, aggregated vital signs, laboratory values, and Apache severity components. Baseline logistic regression is compared against a LightGBM gradient boosting model, with SHAP values used for feature importance and individual prediction explanations. Cross-hospital generalization is evaluated by training on a subset of hospitals and testing on held-out sites.

## Repository structure

- notebooks/ — exploratory data analysis and experiments
- src/ — reusable Python modules for data processing and modeling
- results/ — figures and performance metrics
- docs/ — methods diagram and supplementary documentation

## Reproducing this work

Requires credentialed access to eICU via PhysioNet. Setup and environment instructions will be added as the project progresses.

## Author

Azadeh Sereshki — MASc Biomedical Engineering, University of Toronto | Research Assistant, University of Alberta. https://www.linkedin.com/in/azadeh-sereshki/ · azadeh.a.sereshki@gmail.com

## License

MIT
