"""
Streamlit UI for credit-card fraud prediction.

Run after training:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATA_PATH = ROOT / "data" / "creditcard.csv"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text())


@st.cache_data
def load_sample_rows(n: int = 5) -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    fraud = df[df["Class"] == 1].sample(n=min(n, (df["Class"] == 1).sum()), random_state=2)
    legit = df[df["Class"] == 0].sample(n=min(n, (df["Class"] == 0).sum()), random_state=2)
    return pd.concat([legit, fraud]).sample(frac=1, random_state=2).reset_index(drop=True)


def predict(bundle, values: list[float]) -> int:
    features = np.array(values, dtype=np.float64).reshape(1, -1)
    return int(bundle["model"].predict(features)[0])


def main() -> None:
    st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
    st.title("Credit Card Fraud Detection")
    st.caption("Undersampled logistic regression on the Kaggle creditcard.csv dataset.")

    bundle = load_model()
    metrics = load_metrics()

    if bundle is None:
        st.error(
            "No trained model found. Download `data/creditcard.csv`, then run:\n\n"
            "`python train.py`"
        )
        st.stop()

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy", f"{metrics['test_accuracy']:.2%}")
        c2.metric("Precision", f"{metrics['precision']:.2%}")
        c3.metric("Recall", f"{metrics['recall']:.2%}")
        c4.metric("F1 score", f"{metrics['f1']:.2%}")

    st.subheader("Classify a transaction")
    mode = st.radio(
        "Input mode",
        ["Comma-separated values", "Individual fields", "Pick a sample row"],
        horizontal=True,
    )

    feature_columns = bundle["feature_columns"]
    values: list[float] | None = None

    if mode == "Comma-separated values":
        raw = st.text_input(
            f"Paste {len(feature_columns)} numbers (Time, V1…V28, Amount)",
            placeholder="0,-1.3598,…,149.62",
        )
        if st.button("Predict", type="primary") and raw.strip():
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if len(parts) != len(feature_columns):
                st.warning(
                    f"Expected {len(feature_columns)} values, got {len(parts)}."
                )
            else:
                try:
                    values = [float(p) for p in parts]
                except ValueError:
                    st.error("All values must be numbers.")

    elif mode == "Individual fields":
        cols = st.columns(3)
        entered: list[float] = []
        for i, name in enumerate(feature_columns):
            with cols[i % 3]:
                entered.append(float(st.number_input(name, value=0.0, format="%.6f")))
        if st.button("Predict", type="primary"):
            values = entered

    else:
        samples = load_sample_rows()
        if samples is None:
            st.info("Add `data/creditcard.csv` to enable sample rows.")
        else:
            display = samples[feature_columns + ["Class"]].copy()
            display["Class"] = display["Class"].map({0: "Legit", 1: "Fraud"})
            st.dataframe(display, use_container_width=True)
            idx = st.selectbox("Row index", options=list(range(len(samples))))
            if st.button("Predict selected row", type="primary"):
                values = samples.loc[idx, feature_columns].astype(float).tolist()
                st.write(f"True label: **{display.loc[idx, 'Class']}**")

    if values is not None:
        label = predict(bundle, values)
        if label == 0:
            st.success("Legitimate transaction")
        else:
            st.error("Fraudulent transaction")


if __name__ == "__main__":
    main()
