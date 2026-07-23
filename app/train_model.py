"""
train_model.py
----------------
Trains a diabetes prediction model on the Pima Indians Diabetes dataset.

HOW TO GET REAL DATA (recommended for a real project submission):
    1. Go to: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    2. Download "diabetes.csv"
    3. Place it inside the /data folder of this project as: data/diabetes.csv
    4. Re-run this script.

If data/diabetes.csv is NOT found, this script automatically generates a
realistic synthetic dataset with the same 8 clinical features so the whole
project still runs end-to-end for development/demo purposes.

Run:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib

DATA_PATH = os.path.join("data", "diabetes.csv")
MODEL_DIR = "model"
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COLUMN = "Outcome"


def load_dataset(n_samples: int = 900, random_state: int = 42) -> pd.DataFrame:
    """Load the real Kaggle CSV if present, otherwise build a synthetic
    but clinically-plausible stand-in dataset with the same columns."""

    if os.path.exists(DATA_PATH):
        print(f"Loading real dataset from {DATA_PATH} ...")
        return pd.read_csv(DATA_PATH)

    print("data/diabetes.csv not found -> generating a synthetic demo "
          "dataset instead. Swap in the real Kaggle CSV any time for "
          "better real-world accuracy (see instructions at the top of "
          "this file).")

    rng = np.random.default_rng(random_state)
    n_pos = n_samples // 3  # roughly matches the real dataset's ~35% positive rate

    def make_group(n, diabetic: bool):
        pregnancies = rng.poisson(4 if diabetic else 2.5, n).clip(0, 17)
        glucose = rng.normal(142 if diabetic else 110, 25, n).clip(60, 200)
        blood_pressure = rng.normal(74 if diabetic else 68, 12, n).clip(40, 122)
        skin_thickness = rng.normal(29 if diabetic else 24, 10, n).clip(0, 60)
        insulin = rng.normal(140 if diabetic else 80, 90, n).clip(0, 600)
        bmi = rng.normal(34 if diabetic else 28, 6, n).clip(15, 60)
        pedigree = rng.gamma(2.0, 0.25 if diabetic else 0.18, n).clip(0.05, 2.5)
        age = rng.normal(42 if diabetic else 30, 11, n).clip(21, 81)
        outcome = np.full(n, 1 if diabetic else 0)
        return pd.DataFrame({
            "Pregnancies": pregnancies.round().astype(int),
            "Glucose": glucose.round().astype(int),
            "BloodPressure": blood_pressure.round().astype(int),
            "SkinThickness": skin_thickness.round().astype(int),
            "Insulin": insulin.round().astype(int),
            "BMI": bmi.round(1),
            "DiabetesPedigreeFunction": pedigree.round(3),
            "Age": age.round().astype(int),
            "Outcome": outcome,
        })

    df = pd.concat([
        make_group(n_pos, diabetic=True),
        make_group(n_samples - n_pos, diabetic=False),
    ], ignore_index=True)

    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """The Pima dataset uses 0 as a placeholder for missing values in
    several clinical columns. Replace those with the column median."""
    df = df.copy()
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_as_missing:
        if col in df.columns:
            median_val = df.loc[df[col] != 0, col].median()
            df[col] = df[col].replace(0, median_val)
    return df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_dataset()
    df = clean_data(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000),
            {"C": [0.1, 1.0, 10.0]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [200, 400], "max_depth": [4, 6, None]},
        ),
    }

    best_model = None
    best_score = -1
    best_name = None

    for name, (estimator, param_grid) in candidates.items():
        grid = GridSearchCV(estimator, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        score = grid.best_score_
        print(f"{name}: best CV ROC-AUC = {score:.4f} | params = {grid.best_params_}")
        if score > best_score:
            best_score = score
            best_model = grid.best_estimator_
            best_name = name

    print(f"\nSelected model: {best_name}")

    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Test set performance ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(best_model, os.path.join(MODEL_DIR, "diabetes_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_DIR, "feature_columns.pkl"))

    print(f"\nSaved model artifacts to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
