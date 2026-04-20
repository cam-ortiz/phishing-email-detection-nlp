# Entry point for the phishing/spam email classification pipeline.
# Compatible with two types of datasets:
#   1. Pre-processed CSV  (--csv path/to/file.csv)
#   2. Raw Enron email dirs (--enron path/to/enron_root)
# Default: looks for data/processed/enron_clean.csv, then
# falls back to data/raw/enron/ for raw parsing.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.tfidf import build_tfidf, fit_transform_tfidf, transform_tfidf
from src.models.baseline import (
    train_logistic_regression,
    train_naive_bayes,
    train_svm,
    evaluate_model,
)
from src.pipeline.data_pipeline import (
    build_enron_modeling_dataframe,
    print_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "processed" / "enron_clean.csv"
DEFAULT_ENRON_RAW = PROJECT_ROOT / "data" / "raw" / "enron"


# Resolve the data source and return a DataFrame with 'text' and 'label' columns.
def load_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        csv_path = Path(args.csv)
        print(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    elif args.enron:
        enron_path = Path(args.enron)
        print(f"Parsing raw Enron emails from: {enron_path}")
        _, _, summary, df = build_enron_modeling_dataframe(enron_path)
        print_summary(summary)
    elif DEFAULT_CSV.exists():
        print(f"Loading CSV: {DEFAULT_CSV}")
        df = pd.read_csv(DEFAULT_CSV)
    elif DEFAULT_ENRON_RAW.exists():
        print(f"Parsing raw Enron emails from: {DEFAULT_ENRON_RAW}")
        _, _, summary, df = build_enron_modeling_dataframe(DEFAULT_ENRON_RAW)
        print_summary(summary)
    else:
        raise FileNotFoundError(
            "No data found. Provide --csv or --enron, or place data in "
            f"{DEFAULT_CSV} or {DEFAULT_ENRON_RAW}"
        )

    # Drop rows with missing or empty text
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().astype(bool)]
    print(f"\nDataset: {len(df)} samples  |  label distribution:\n{df['label'].value_counts().to_string()}\n")
    return df


# TF-IDF vectorization, train/test split, train three baselines, and evaluate.
def run_pipeline(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> None:
    # Split data into training and test sets, stratified by label
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=test_size, random_state=random_state, stratify=df["label"],
    )

    print(f"Train: {len(X_train_text)}  |  Test: {len(X_test_text)}\n")

    # Build TF-IDF features — fit on train, transform both
    vectorizer = build_tfidf()
    X_train = fit_transform_tfidf(X_train_text.tolist(), vectorizer)
    X_test = transform_tfidf(X_test_text.tolist(), vectorizer)

    # Train and evaluate each baseline model
    models = {
        "Logistic Regression": train_logistic_regression,
        "Naive Bayes": train_naive_bayes,
        "SVM (LinearSVC)": train_svm,
    }

    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 67)

    for name, train_fn in models.items():
        model = train_fn(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        print(
            f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f}"
        )

    print()


# Parse CLI arguments and run the full pipeline.
def main() -> None:
    parser = argparse.ArgumentParser(description="Phishing/Spam Email Classifier")
    parser.add_argument("--csv", type=str, help="Path to a pre-processed CSV with 'text' and 'label' columns")
    parser.add_argument("--enron", type=str, help="Path to raw Enron dataset root (expects ham/ and spam/ subdirs)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    df = load_dataframe(args)
    run_pipeline(df, test_size=args.test_size, random_state=args.seed)


if __name__ == "__main__":
    main()
