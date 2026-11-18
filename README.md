# Credit Card Fraud Detection

A beginner-friendly ML project: train a model on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), then predict legitimate vs fraudulent transactions in a Streamlit app.

Inspired by [Creditcard-Fruad-Detection](https://github.com/Mansi-0111/Creditcard-Fruad-Detection).

## What you will learn

1. Load and explore an imbalanced dataset
2. Balance classes with undersampling
3. Train a logistic regression classifier
4. Evaluate with accuracy, precision, recall, and F1
5. Save the model and serve predictions in a web UI

## Dataset

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place it at `data/creditcard.csv`

Columns: `Time`, `V1`–`V28` (PCA features), `Amount`, `Class` (`0` = legit, `1` = fraud).

## Setup

```bash
cd ~/Personal/creditcard-fraud-detection
python3 -m venv .venv
source .venv/bin/activate
# Use public PyPI (this machine may default to a work-internal index)
pip install --index-url https://pypi.org/simple/ -r requirements.txt
```

## Train

```bash
python train.py
```

This undersamples legitimate transactions to match the fraud count, trains the model, prints metrics, and saves:

- `models/fraud_model.joblib`
- `models/metrics.json`

## Run the app

```bash
streamlit run app.py
```

Enter feature values (comma-separated or one per field) to classify a transaction.

## Project layout

```
creditcard-fraud-detection/
├── app.py              # Streamlit UI
├── train.py            # Train + evaluate + save model
├── data/creditcard.csv # Dataset (not committed)
├── models/             # Saved model + metrics
├── requirements.txt
└── README.md
```

## Suggested learning order

1. Open `train.py` and read top to bottom
2. Run training and study the printed metrics (why recall matters for fraud)
3. Run the Streamlit app and try sample rows from the CSV
4. Experiment: change the model, try no undersampling, or add scaling
