"""
Train a simplified logistic regression for the Streamlit demo.

Uses only the 14 features the demo form collects, so predictions respond
meaningfully to user input. Performance is lower than the full LightGBM
model (which uses 102 features) but the demo behavior is honest.
"""

import joblib
import json
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

DEMO_FEATURES = [
    "age", "gender_male", "bmi",
    "gcs_mean",
    "hr_mean", "sbp_mean", "map_mean", "rr_mean", "spo2_mean", "temp_mean",
    "bun_max", "creatinine_max", "glucose_min", "wbc_mean",
]


def main():
    features = pd.read_parquet("data/features_final.parquet")
    X = features[DEMO_FEATURES]
    y = features["in_hospital_mortality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)),
    ])

    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred)
    test_pr = average_precision_score(y_test, y_pred)

    print(f"Demo model trained on {len(DEMO_FEATURES)} features")
    print(f"  Train CV AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")
    print(f"  Test ROC-AUC: {test_auc:.3f}")
    print(f"  Test PR-AUC:  {test_pr:.3f}")

    # Save the imputer median for reference (used by the app)
    feature_medians = {
        feat: float(med)
        for feat, med in zip(DEMO_FEATURES, pipeline.named_steps["imputer"].statistics_)
    }

    joblib.dump(pipeline, "models/demo_model.pkl")
    with open("data/demo_features.json", "w") as f:
        json.dump({"features": DEMO_FEATURES, "medians": feature_medians}, f, indent=2)

    print(f"\nSaved models/demo_model.pkl")
    print(f"Saved data/demo_features.json")


if __name__ == "__main__":
    main()
