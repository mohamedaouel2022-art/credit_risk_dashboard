"""Train and compare baseline credit-risk models, then save a reusable pipeline.

Usage:
    python train_model.py
    python train_model.py --data /path/to/credit_risk_dataset.csv

Target convention expected by this project: loan_status=1 means default; 0 means non-default.
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
TARGET = "loan_status"


def parse_args():
    parser = argparse.ArgumentParser(description="Train CrediScope credit risk models.")
    parser.add_argument("--data", default=str(ROOT / "credit_risk_dataset.csv"), help="Path to CSV dataset.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Classification threshold for reporting metrics.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Holdout share.")
    return parser.parse_args()


def make_one_hot_encoder():
    # Supports both newer and older scikit-learn releases.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(X: pd.DataFrame):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ])
        transformers.append(("numeric", numeric_pipe, numeric))
    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ])
        transformers.append(("categorical", categorical_pipe, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop"), numeric, categorical


def build_models(preprocessor):
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", preprocessor),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=350,
                max_depth=14,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            )),
        ]),
    }


def main():
    args = parse_args()
    data_path = Path(args.data).expanduser()
    if not data_path.exists():
        raise SystemExit(
            f"\nDataset not found: {data_path}\n"
            "Place credit_risk_dataset.csv in this folder or run:\n"
            "python train_model.py --data \"/full/path/to/credit_risk_dataset.csv\"\n"
        )
    if not 0 < args.test_size < 0.5:
        raise SystemExit("--test-size must be between 0 and 0.5.")
    if not 0 < args.threshold < 1:
        raise SystemExit("--threshold must be between 0 and 1.")

    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise SystemExit(f"Target column '{TARGET}' was not found. Columns: {list(df.columns)}")

    df = df.copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    if len(df) < 50:
        raise SystemExit("At least 50 rows with a valid binary loan_status target are recommended.")
    if df[TARGET].nunique() != 2:
        raise SystemExit("loan_status must contain both classes 0 and 1.")
    if df[TARGET].value_counts().min() < 2:
        raise SystemExit("Each target class needs at least two examples for a stratified split.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    # Remove completely empty columns; the app should not depend on unusable features.
    empty_cols = X.columns[X.isna().all()].tolist()
    if empty_cols:
        X = X.drop(columns=empty_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    # Clone separate preprocessing pipelines through fresh model builders to avoid sharing fitted state.
    candidates = {}
    scores = {}
    for model_name in ["Logistic Regression", "Random Forest"]:
        preprocessor, _, _ = build_preprocessor(X_train)
        candidate = build_models(preprocessor)[model_name]
        candidate.fit(X_train, y_train)
        probabilities = candidate.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, probabilities))
        candidates[model_name] = candidate
        scores[model_name] = auc
        print(f"{model_name}: holdout ROC-AUC = {auc:.4f}")

    best_name = max(scores, key=scores.get)
    best_model = candidates[best_name]
    probabilities = best_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= args.threshold).astype(int)

    report = classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["Non-default", "Default"],
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    _, numeric, categorical = build_preprocessor(X_train)
    metrics = {
        "model_name": best_name,
        "model_scores": scores,
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "default_precision": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "default_recall": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "default_f1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "decision_threshold": float(args.threshold),
        "confusion_matrix": cm,
        "class_metrics": {
            "Non-default": {k: float(report["Non-default"][k]) for k in ["precision", "recall", "f1-score"]},
            "Default": {k: float(report["Default"][k]) for k in ["precision", "recall", "f1-score"]},
        },
        "feature_columns": list(X.columns),
        "feature_types": {
            col: ("numeric" if col in numeric else "categorical")
            for col in X.columns
        },
        "dropped_all_missing_columns": empty_cols,
        "rows_used": int(len(df)),
        "training_rows": int(len(X_train)),
        "holdout_rows": int(len(X_test)),
        "default_rate": float(y.mean()),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "dataset_file": data_path.name,
        "target_definition": "loan_status=1 means default; loan_status=0 means non-default",
        "limitations": [
            "Holdout performance is an initial internal evaluation, not external validation.",
            "Predicted probabilities are not guaranteed to be calibrated probabilities of default.",
            "The classification threshold is configurable and is not a lending policy.",
            "Feature leakage, fairness, stability, and local legal requirements require separate review.",
        ],
    }

    model_path = ROOT / "credit_risk_model.joblib"
    metrics_path = ROOT / "model_metrics.json"
    joblib.dump(best_model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print("\nSelected model:", best_name)
    print("Classification report at threshold", args.threshold)
    print(classification_report(y_test, predictions, target_names=["Non-default", "Default"], zero_division=0))
    print("Confusion matrix [rows=actual, columns=predicted]:")
    print(np.array(cm))
    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("\nNext step: streamlit run app.py")


if __name__ == "__main__":
    main()
