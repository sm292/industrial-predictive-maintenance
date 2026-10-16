
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = Path("data/ai4i2020.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Machine failure"

# Identifiers and failure-mode labels must not be predictors.
EXCLUDE = {
    TARGET,
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
}


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found. Run python download_data.py first."
        )

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    if TARGET not in df.columns:
        raise ValueError(
            f"Missing target '{TARGET}'. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=[TARGET]).copy()

    X = df.drop(
        columns=[c for c in EXCLUDE if c in df.columns]
    )
    y = pd.to_numeric(df[TARGET], errors="raise").astype(int)

    # Detect data types rather than hardcoding sensor column names.
    categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical = [
        c for c in X.columns if c not in categorical
    ]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numerical),
        ("categorical", categorical_pipe, categorical),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    fitted_models = {}

    for name, classifier in candidates.items():
        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.50).astype(int)

        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "pr_auc": float(
                average_precision_score(y_test, probabilities)
            ),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "confusion_matrix": confusion_matrix(
                y_test, predictions
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        }

        results.append(metrics)
        fitted_models[name] = pipeline

        print(f"\n{name}")
        print(json.dumps(metrics, indent=2))

    # Select by recall to prioritise identifying failures.
    # Later, optimise the decision threshold based on business costs.
    best = max(results, key=lambda item: item["recall"])
    best_name = best["model"]

    bundle = {
        "pipeline": fitted_models[best_name],
        "feature_columns": X.columns.tolist(),
        "target": TARGET,
        "selected_model": best_name,
        "threshold": 0.50,
    }

    joblib.dump(bundle, MODEL_DIR / "failure_model.joblib")

    with open(MODEL_DIR / "metrics.json", "w") as file:
        json.dump(
            {
                "selected_model": best_name,
                "results": results,
            },
            file,
            indent=2,
        )

    print(f"\nSelected model: {best_name}")
    print(f"Recall: {best['recall']:.4f}")
    print("Saved model to models/failure_model.joblib")


if __name__ == "__main__":
    main()
