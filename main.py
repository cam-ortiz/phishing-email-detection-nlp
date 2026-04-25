"""
Command-line entry point for the phishing/spam classification pipeline.

This module supports three dataset modes:

1. Enron ham/spam classification.
2. Phishing detection using Enron legitimate emails and Nazario phishing emails.
3. A custom preprocessed CSV containing ``text`` and ``label`` columns.

The pipeline loads data, validates the modeling dataframe, optionally balances
classes, builds TF-IDF features, trains baseline models, evaluates them, and
saves comparison results.
"""


from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split

from src.features.tfidf import build_tfidf, fit_transform_tfidf, transform_tfidf
from src.models.baseline import (
    train_logistic_regression,
    train_naive_bayes,
    train_svm,
)
from src.models.evaluate import (
    evaluate_model,
    save_confusion_matrix_image,
    build_classification_report_df,
    save_classification_report_csv,
)
from src.pipeline.data_pipeline import (
    build_enron_modeling_dataframe,
    build_nazario_modeling_dataframe,
    print_summary,
)
from src.features.linguistic_features import (
    add_linguistic_features,
    summarize_linguistic_features,
)


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CSV = PROJECT_ROOT / "data" / "processed" / "enron_clean.csv"
DEFAULT_ENRON_RAW = PROJECT_ROOT / "data" / "raw" / "enron"

DEFAULT_NAZARIO_CSV = PROJECT_ROOT / "data" / "processed" / "nazario_clean.csv"
DEFAULT_NAZARIO_RAW = PROJECT_ROOT / "data" / "raw" / "nazario"

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

LABEL_NAMES = {
    "enron": ["Ham", "Spam"],
    "phishing": ["Legitimate", "Phishing"],
}

REQUIRED_COLUMNS = {"text", "label"}

logger = logging.getLogger(__name__)

ModelTrainer = Callable[[spmatrix, pd.Series], ClassifierMixin]


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure application logging.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
    )    
    

def print_section(title: str) -> None:
    print(f"\n=== {title} ===")
    

def validate_modeling_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that a dataframe can be used for model training.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to validate.

    Raises
    ------
    ValueError
        If required columns are missing or the label column has fewer than two
        unique classes.
    """
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    
    if missing_columns:
        raise ValueError(
            "Dataframe is missing required columns: "
            f"{sorted(missing_columns)}. Expected columns: {sorted(REQUIRED_COLUMNS)}"
        )

    if df["label"].nunique() < 2:
        raise ValueError(
            "The dataset must contain at least two classes for training."
        )
    
    
def prepare_modeling_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize a modeling dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing at least ``text`` and ``label`` columns.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe containing only non-empty, deduplicated text rows.

    Raises
    ------
    ValueError
        If the dataframe does not contain the required columns.
    """
    validate_modeling_dataframe(df)
    
    cleaned_df = df.copy()
    
    cleaned_df = cleaned_df.dropna(subset=["text", "label"])
    cleaned_df["text"] = cleaned_df["text"].astype(str)
    cleaned_df = cleaned_df[cleaned_df["text"].str.strip().astype(bool)]

    cleaned_df = (
        cleaned_df.drop_duplicates(subset=["text"])
        .reset_index(drop=True)
    )
    return cleaned_df


def balance_classes(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """
    Downsample all classes to match the smallest class size.

    Parameters
    ----------
    df : pandas.DataFrame
        Modeling dataframe containing ``text`` and ``label`` columns.
    random_state : int
        Random seed used for reproducible sampling.

    Returns
    -------
    pandas.DataFrame
        Class-balanced dataframe.
    """
    counts = df["label"].value_counts()
    min_class_size = counts.min()
    
    logger.info("Balancing classes to %s rows per class.", min_class_size)

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
    """
    Load the Enron dataset from raw files or a default processed CSV.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    pandas.DataFrame
        Enron modeling dataframe.

    Raises
    ------
    FileNotFoundError
        If no Enron source can be found.
    """
    if args.enron:
        enron_path = Path(args.enron)
        logger.info("Parsing raw Enron emails from: %s", enron_path)
        
        _, _, summary, df = build_enron_modeling_dataframe(enron_path)
        print_summary(summary)
        
        return df

    elif DEFAULT_CSV.exists():
        logger.info("Loading Enron CSV: %s", DEFAULT_CSV)
        return pd.read_csv(DEFAULT_CSV)

    elif DEFAULT_ENRON_RAW.exists():
        logger.info("Parsing raw Enron emails from: %s", DEFAULT_ENRON_RAW)

        _, _, summary, df = build_enron_modeling_dataframe(DEFAULT_ENRON_RAW)
        print_summary(summary)

        return df

    raise FileNotFoundError(
        "No Enron data found. Provide --enron or place data in "
        f"{DEFAULT_CSV} or {DEFAULT_ENRON_RAW}"
    )


def load_nazario_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load the Nazario phishing dataset from raw files or processed CSV.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    pandas.DataFrame
        Nazario modeling dataframe.

    Raises
    ------
    FileNotFoundError
        If no Nazario source can be found.
    """
    if args.nazario:
        nazario_path = Path(args.nazario)
        logger.info("Parsing raw Nazario emails from: %s", nazario_path)
        
        _, _, summary, df = build_nazario_modeling_dataframe(nazario_path)
        print_summary(summary)
        
        return df

    elif DEFAULT_NAZARIO_CSV.exists():
        logger.info("Loading Nazario CSV: %s", DEFAULT_NAZARIO_CSV)
        return pd.read_csv(DEFAULT_NAZARIO_CSV)

    elif DEFAULT_NAZARIO_RAW.exists():
        logger.info("Parsing raw Nazario emails from: %s", DEFAULT_NAZARIO_RAW)

        _, _, summary, df = build_nazario_modeling_dataframe(DEFAULT_NAZARIO_RAW)
        print_summary(summary)

        return df

    raise FileNotFoundError(
        "No Nazario data found. Provide --nazario or place data in "
        f"{DEFAULT_NAZARIO_CSV} or {DEFAULT_NAZARIO_RAW}"
    )
    
    
def build_phishing_vs_legit_dataframe(
    enron_df: pd.DataFrame,
    nazario_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a phishing detection dataframe from Enron and Nazario data.

    Enron emails labeled ``0`` are treated as legitimate emails. Nazario emails
    are treated as phishing emails and assigned label ``1``.

    Parameters
    ----------
    enron_df : pandas.DataFrame
        Enron dataframe containing ``text`` and ``label`` columns.
    nazario_df : pandas.DataFrame
        Nazario dataframe containing a ``text`` column.

    Returns
    -------
    pandas.DataFrame
        Combined dataframe with ``text`` and ``label`` columns.
    """
    validate_modeling_dataframe(enron_df)
    
    if "text" not in nazario_df.columns:
        raise ValueError("Nazario dataframe must contain a 'text' column.")
        
    # Use only legitimate Enron messages as the non-phishing class.
    enron_ham_df = enron_df.loc[enron_df["label"] == 0, ["text", "label"]].copy()

    # Nazario records are phishing records for this experiment.
    nazario_phishing_df = nazario_df[["text"]].copy()
    nazario_phishing_df["label"] = 1
    
    combined_df = pd.concat(
        [enron_ham_df, nazario_phishing_df],
        ignore_index=True,
    )

    logger.info(
        "Built phishing dataframe with %s Enron ham rows and %s Nazario phishing rows.",
        len(enron_ham_df),
        len(nazario_phishing_df),
    )

    return combined_df


def load_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load and prepare a dataframe based on CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    pandas.DataFrame
        Cleaned and optionally balanced modeling dataframe.
    """
    print_section("Loading Data")
    
    if args.csv:
        csv_path = Path(args.csv)
        logger.info("Loading CSV: %s", csv_path)
        df = pd.read_csv(csv_path)
        
    elif args.dataset == "enron":
        df = load_enron_dataframe(args)

    elif args.dataset == "phishing":
        enron_df = load_enron_dataframe(args)
        nazario_df = load_nazario_dataframe(args)
        
        df = build_phishing_vs_legit_dataframe(
            enron_df=enron_df,
            nazario_df=nazario_df,
        )

    else:
        raise ValueError(f"Unsupported dataset option: {args.dataset}")

    df = prepare_modeling_dataframe(df)
    
    print_section("Preprocessing")
    
    logger.info("After cleaning: %s rows.", len(df))
    logger.info(
        "Class counts before balancing: %s",
        df["label"].value_counts().to_dict(),
    )

    if args.balanced:
        logger.info("Applying class balancing...")
        df = balance_classes(df, random_state=args.seed)

    logger.info("Final dataset size: %s rows.", len(df))
    logger.info(
        "Final class counts: %s",
        df["label"].value_counts().to_dict(),
    )

    return df


def save_results(
    results_df: pd.DataFrame,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """
    Save model comparison metrics to CSV.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Dataframe containing model evaluation metrics.
    results_dir : pathlib.Path, default=DEFAULT_RESULTS_DIR
        Directory where the results CSV should be saved.

    Returns
    -------
    pathlib.Path
        Path to the saved CSV file.
    """

    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "baseline_model_comparison.csv"
    results_df.to_csv(output_path, index=False)

    return output_path


def save_model_comparison_plot(
    results_df: pd.DataFrame,
    dataset: str,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> Path:
    """
    Save a bar chart comparing baseline model performance.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Dataframe containing model evaluation metrics.
    dataset : str
        Dataset identifier used to map labels for evaluation outputs.
    figures_dir : pathlib.Path, default=DEFAULT_FIGURES_DIR
        Directory where the chart should be saved.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """

    figures_dir.mkdir(parents=True, exist_ok=True)

    output_path = figures_dir / "baseline_model_comparison.png"

    ax = results_df.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1"]
    ].plot(kind="bar")

    label_names = LABEL_NAMES.get(dataset, ["Class 0", "Class 1"])
    ax.set_title(f"Model Comparison ({label_names[0]} vs {label_names[1]})")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def run_baseline_experiment(
        df: pd.DataFrame, 
        dataset: str,
        balanced: bool,
        test_size: float = 0.2, 
        random_state: int = 42,
        results_dir: Path = DEFAULT_RESULTS_DIR,
        figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> pd.DataFrame:
    """
    Train and evaluate baseline classifiers using TF-IDF features.

    Parameters
    ----------
    df : pandas.DataFrame
        Modeling dataframe containing ``text`` and ``label`` columns.
    dataset : str
        Dataset name used to determine evaluation label names (e.g., ``"enron"``,
        ``"phishing"``). Defaults to generic labels for custom datasets.
    balanced: bool
        Whether class balancing was applied to the dataset.
    test_size : float, default=0.2
        Fraction of the dataset reserved for testing.
    random_state : int, default=42
        Random seed used for reproducible splitting.
    results_dir : pathlib.Path, default=DEFAULT_RESULTS_DIR
        Directory where the metrics CSV should be saved.
    figures_dir : pathlib.Path, default=DEFAULT_FIGURES_DIR
        Directory where the comparison chart should be saved.

    Returns
    -------
    pandas.DataFrame
        Model evaluation results with accuracy, precision, recall, and F1.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_id}_{dataset}"
    if balanced:
        run_name += "_balanced"
    
    results_dir = results_dir / run_name
    figures_dir = figures_dir / run_name
    
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    validate_modeling_dataframe(df)
    
    print_section("Training")
    
    # Split data into training and test sets, stratified by label
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], 
        df["label"], 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df["label"],
    )

    logger.info("Train size: %s | Test size: %s", len(X_train_text), len(X_test_text))

    label_names = LABEL_NAMES.get(dataset, ["Class 0", "Class 1"])
    
    # Get linguistic feature count for each class
    df_with_linguistic_features = add_linguistic_features(df)

    linguistic_summary_df = summarize_linguistic_features(
        df_with_linguistic_features,
        label_names=label_names,
    )

    linguistic_summary_path = results_dir / "linguistic_feature_summary.csv"
    linguistic_summary_path.parent.mkdir(parents=True, exist_ok=True)
    linguistic_summary_df.to_csv(linguistic_summary_path, index=False)
    
    # Build TF-IDF features — fit on train, transform both
    vectorizer = build_tfidf()
    X_train = fit_transform_tfidf(X_train_text.tolist(), vectorizer)
    X_test = transform_tfidf(X_test_text.tolist(), vectorizer)

    # Train and evaluate each baseline model
    models: dict[str, ModelTrainer] = {
        "Logistic Regression": train_logistic_regression,
        "Naive Bayes": train_naive_bayes,
        "SVM (LinearSVC)": train_svm,
    }
    
    print_section("Model Evaluation")
    print(f"Classes: {label_names[0]} (0) vs {label_names[1]} (1)\n")
    
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 67)

    results: list[dict[str, float | str]] = []

    for model_name, train_fn in models.items():
        model = train_fn(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
                
        safe_model_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        
        confusion_matrix_path = figures_dir / f"{safe_model_name}_confusion_matrix.png"

        title = f"{model_name} Confusion Matrix ({label_names[0]} vs {label_names[1]})"

        save_confusion_matrix_image(
            model=model,
            X_test=X_test,
            y_test=y_test,
            output_path=confusion_matrix_path,
            display_labels=label_names,
            title=title,
        )
        
        report_df = build_classification_report_df(
            model=model,
            X_test=X_test,
            y_test=y_test,
            target_names=label_names,
        )
        
        report_path = results_dir / f"{safe_model_name}_classification_report.csv"

        save_classification_report_csv(
            report_df,
            report_path,
        )
        
        results.append(
            {
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
            }
        )
        
        print(
            f"{model_name:<25} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f}"
        )

    print()

    results_df = pd.DataFrame(results)
    
    print_section("Outputs")

    results_path = save_results(results_df, results_dir=results_dir)
    figure_path = save_model_comparison_plot(
        results_df,
        dataset=dataset,
        figures_dir=figures_dir,
    )

    logger.info("Saved results to: %s", results_path)
    logger.info("Saved model comparison figure to: %s", figure_path)

    return results_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Phishing/Spam Email Classifier",
        formatter_class=argparse.RawTextHelpFormatter,
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
            "Apply class balancing by downsampling majority classes.\n"
            "Useful for imbalanced datasets like phishing detection."
        ),
    )

    parser.add_argument(
        "--csv",
        type=str,
        help=(
            "Path to a preprocessed CSV with 'text' and 'label' columns.\n"
            "Required when using --dataset csv."
        ),
    )

    parser.add_argument(
        "--enron",
        type=str,
        help="Path to raw Enron dataset root.",
    )

    parser.add_argument(
        "--nazario",
        type=str,
        help="Path to raw Nazario dataset root.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing. Default: 0.2.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    args = parser.parse_args()

    if args.dataset == "csv" and not args.csv:
        parser.error("--dataset csv requires --csv")

    if args.dataset == "enron" and args.nazario:
        logger.warning("--nazario is ignored when using --dataset enron.")

    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1.")

    return args


def main() -> None:
    """
    Run the full command-line pipeline.
    """
    configure_logging(logging.INFO)
    
    args = parse_args()
    
    # Load dataset
    df = load_dataframe(args)

    # Run pipeline
    run_baseline_experiment(
        df=df,
        dataset=args.dataset,
        balanced=args.balanced,
        test_size=args.test_size,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
