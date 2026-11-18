"""
Train a credit-card fraud classifier and save it for the Streamlit app.

Steps:
1. Load data/creditcard.csv
2. Undersample legitimate rows so Class 0 and Class 1 have equal counts
3. Train/test split (stratified)
4. Fit LogisticRegression
5. Print metrics and save model + metrics to models/
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"
RANDOM_STATE = 2


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}\n"
            "Download creditcard.csv from Kaggle and place it in data/"
        )
    return pd.read_csv(path)


def undersample(data: pd.DataFrame) -> pd.DataFrame:
    """Match legit and fraud counts so the model is not biased toward Class 0."""
    legit = data[data[TARGET_COLUMN] == 0]
    fraud = data[data[TARGET_COLUMN] == 1]
    legit_sample = legit.sample(n=len(fraud), random_state=RANDOM_STATE)
    balanced = pd.concat([legit_sample, fraud], axis=0)
    return balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def train(data: pd.DataFrame):
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_fraud_total": int((data[TARGET_COLUMN] == 1).sum()),
    }
    return model, metrics, y_test, y_pred


def main() -> None:
    print("Loading dataset...")
    raw = load_data(DATA_PATH)
    print(f"  rows={len(raw):,}  fraud={(raw[TARGET_COLUMN] == 1).sum():,}")

    print("Undersampling to balance classes...")
    balanced = undersample(raw)
    print(f"  balanced rows={len(balanced):,}")

    print("Training LogisticRegression...")
    model, metrics, y_test, y_pred = train(balanced)

    print("\nMetrics")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": FEATURE_COLUMNS},
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model  -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
