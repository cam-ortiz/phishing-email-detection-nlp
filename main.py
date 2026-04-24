# Entry point for the phishing/spam email classification pipeline.
# Compatible with two types of datasets:
#   1. Pre-processed CSV  (--csv path/to/file.csv)
#   2. Raw Enron email dirs (--enron path/to/enron_root)
# Default: looks for data/processed/enron_clean.csv, then
# falls back to data/raw/enron/ for raw parsing.

from __future__ import annotations
import matplotlib.pyplot as plt
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
    build_nazario_modeling_dataframe,
    print_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "processed" / "enron_clean.csv"
DEFAULT_ENRON_RAW = PROJECT_ROOT / "data" / "raw" / "enron"

DEFAULT_NAZARIO_CSV = PROJECT_ROOT / "data" / "processed" / "nazario_clean_dedup.csv"
DEFAULT_NAZARIO_RAW = PROJECT_ROOT / "data" / "raw" / "nazario"
DEFAULT_BALANCED_CSV = PROJECT_ROOT / "data" / "processed" / "phishing_legit_balanced.csv"


def clean_loaded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().astype(bool)]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


def maybe_balance_dataframe(
    df: pd.DataFrame,
    balanced: bool,
    random_state: int,
) -> pd.DataFrame:
    if not balanced:
        return df

    counts = df["label"].value_counts()
    min_class_size = counts.min()

    balanced_parts = []

    for label_value in counts.index:
        label_df = df[df["label"] == label_value]
        sampled_df = label_df.sample(
            n=min_class_size,
            random_state=random_state,
        )
        balanced_parts.append(sampled_df)

    df_balanced = (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    return df_balanced


def load_enron_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.enron:
        enron_path = Path(args.enron)
        print(f"Parsing raw Enron emails from: {enron_path}")
        _, _, summary, df = build_enron_modeling_dataframe(enron_path)
        print_summary(summary)
        return df

    elif DEFAULT_CSV.exists():
        print(f"Loading CSV: {DEFAULT_CSV}")
        return pd.read_csv(DEFAULT_CSV)

    elif DEFAULT_ENRON_RAW.exists():
        print(f"Parsing raw Enron emails from: {DEFAULT_ENRON_RAW}")
        _, _, summary, df = build_enron_modeling_dataframe(DEFAULT_ENRON_RAW)
        print_summary(summary)
        return df

    raise FileNotFoundError(
        "No Enron data found. Provide --enron or place data in "
        f"{DEFAULT_CSV} or {DEFAULT_ENRON_RAW}"
    )


def load_nazario_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.nazario:
        nazario_path = Path(args.nazario)
        print(f"Parsing raw Nazario emails from: {nazario_path}")
        _, _, summary, df = build_nazario_modeling_dataframe(nazario_path)
        print_summary(summary)
        return df

    elif DEFAULT_NAZARIO_CSV.exists():
        print(f"Loading CSV: {DEFAULT_NAZARIO_CSV}")
        return pd.read_csv(DEFAULT_NAZARIO_CSV)

    elif DEFAULT_NAZARIO_RAW.exists():
        print(f"Parsing raw Nazario emails from: {DEFAULT_NAZARIO_RAW}")
        _, _, summary, df = build_nazario_modeling_dataframe(DEFAULT_NAZARIO_RAW)
        print_summary(summary)
        return df

    raise FileNotFoundError(
        "No Nazario data found. Provide --nazario or place data in "
        f"{DEFAULT_NAZARIO_CSV} or {DEFAULT_NAZARIO_RAW}"
    )
    
    
def build_phishing_dataframe(
    enron_df: pd.DataFrame,
    nazario_df: pd.DataFrame,
) -> pd.DataFrame:
    # Keep only legitimate Enron emails
    enron_ham_df = enron_df[enron_df["label"] == 0].copy()
    
    print("Enron columns:", enron_ham_df.columns.tolist())
    print(enron_ham_df.head())

    print("nazario columns:", nazario_df.columns.tolist())
    print(nazario_df.head())

    # Nazario should already be phishing label 1, but copy to be safe
    nazario_df = nazario_df.copy()
    nazario_df["label"] = 1

    combined_df = pd.concat(
        [enron_ham_df[["text", "label"]], nazario_df[["text", "label"]]],
        ignore_index=True,
    )

    return combined_df


# Resolve the data source and return a DataFrame with 'text' and 'label' columns.
def load_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        csv_path = Path(args.csv)
        print(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
    elif args.dataset == "enron":
        df = load_enron_dataframe(args)

    elif args.dataset == "phishing":
        enron_df = load_enron_dataframe(args)
        nazario_df = load_nazario_dataframe(args)
        df = build_phishing_dataframe(enron_df, nazario_df)

    else:
        raise ValueError(f"Unsupported dataset option: {args.dataset}")

    print("Before clean:", df.columns.tolist())
    df = clean_loaded_dataframe(df)
    
    print("After clean:", df.columns.tolist())
    df = maybe_balance_dataframe(
        df,
        balanced=args.balanced,
        random_state=args.seed,
    )
    print("After balance:", df.columns.tolist())

    return df


# TF-IDF vectorization, train/test split, train three baselines, and evaluate.
def run_pipeline(
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        random_state: int = 42
) -> None:
    
    if df["label"].nunique() < 2:
        raise ValueError(
            "The dataset must contain at least two classes for training."
        )
        
    # Split data into training and test sets, stratified by label
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], 
        df["label"], 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df["label"],
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

    results = []

    for name, train_fn in models.items():
        model = train_fn(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        results.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
        })
        print(
            f"{name:<25} {metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f}"
        )

    print()

    results_df = pd.DataFrame(results)

    results_dir = PROJECT_ROOT / "reports" / "results"
    figures_dir = PROJECT_ROOT / "reports" / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    results_df.to_csv(results_dir / "baseline_model_comparison.csv", index=False)

    # Create chart
    ax = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar")
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(figures_dir / "baseline_model_comparison.png")
    plt.close()


# Parse CLI arguments and run the full pipeline.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phishing/Spam Email Classifier",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        choices=["enron", "phishing", "csv"],
        default="enron",
        help=(
            "Dataset/task to run:\n"
            "  enron     ham vs spam baseline\n"
            "  phishing  Enron ham + Nazario phishing\n"
            "  csv       use only the CSV passed with --csv"
        ),
    )

    parser.add_argument(
        "--balanced",
        action="store_true",
        help=(
            "Apply class balancing (downsample majority class).\n"
            "Useful for imbalanced datasets like phishing detection."
        ),
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        help=(
            "Path to a pre-processed CSV with 'text' and 'label' columns.\n"
            "Required when using --dataset csv."
        ),
    )

    parser.add_argument(
        "--enron",
        type=str,
        help=(
            "Path to raw Enron dataset root (expects ham/ and spam/ subdirs)."
        ),
    )

    parser.add_argument(
        "--nazario",
        type=str,
        help=(
            "Path to raw Nazario dataset root (expects .mbox files)."
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.dataset == "csv" and not args.csv:
        parser.error("--dataset csv requires --csv")

    if args.dataset == "enron" and args.nazario:
        print("Warning: --nazario is ignored when using --dataset enron")

    if args.dataset == "csv" and args.balanced:
        print("Warning: --balanced is ignored when using --dataset csv")

    # Load dataset
    df = load_dataframe(args)

    # Run pipeline
    run_pipeline(df, test_size=args.test_size, random_state=args.seed)


if __name__ == "__main__":
    main()
