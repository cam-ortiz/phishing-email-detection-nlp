"""
Standalone DistilBERT experiment runner.

Mirrors the train/test split used by the baseline pipeline in `main.py`
(stratified, seed=42, test_size=0.2) so metrics are directly comparable
to `reports/results/baseline_model_comparison.csv`.

Supported dataset modes (--dataset):
  enron     - Enron ham vs spam (default)
  phishing  - Enron ham (legitimate) vs Nazario phishing, class-balanced

Defaults are tuned for laptop-local training on Apple Silicon (MPS):
- train on a stratified 8,000-row subsample of the training split,
- evaluate on the FULL held-out test set,
- sequence length 128, batch size 16, 2 epochs.

Artifacts written per run:
- reports/results/transformer_model_comparison.csv  (appended)
- reports/figures/transformer_confusion_matrix.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

from src.models.transformer import (
    TransformerConfig,
    evaluate_distilbert,
    get_device,
    train_distilbert,
)

DEFAULT_ENRON_CSV = PROJECT_ROOT / "data" / "processed" / "enron_clean.csv"
DEFAULT_NAZARIO_CSV = PROJECT_ROOT / "data" / "processed" / "nazario_clean.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

LABEL_NAMES = {
    "enron": ("ham (0)", "spam (1)"),
    "phishing": ("legitimate (0)", "phishing (1)"),
}

MODEL_DISPLAY_NAME = "DistilBERT (fine-tuned)"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DistilBERT fine-tuning experiment runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        choices=["enron", "phishing"],
        default="enron",
        help=(
            "Dataset/task to run:\n"
            "  enron     Enron ham vs spam (default)\n"
            "  phishing  Enron ham + Nazario phishing, class-balanced"
        ),
    )
    parser.add_argument(
        "--enron-csv",
        type=Path,
        default=DEFAULT_ENRON_CSV,
        help=f"Enron processed CSV. Default: {DEFAULT_ENRON_CSV}",
    )
    parser.add_argument(
        "--nazario-csv",
        type=Path,
        default=DEFAULT_NAZARIO_CSV,
        help=f"Nazario processed CSV. Default: {DEFAULT_NAZARIO_CSV}",
    )
    parser.add_argument(
        "--train-subsample",
        type=int,
        default=8000,
        help=(
            "Stratified subsample size drawn from the training split.\n"
            "Use 0 to train on the full training split (slow on laptop)."
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
    )

    return parser.parse_args()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().astype(bool)]
    return df.drop_duplicates(subset=["text"]).reset_index(drop=True)


def load_enron_dataset(csv_path: Path) -> pd.DataFrame:
    logger.info("Loading Enron CSV: %s", csv_path)
    df = _clean(pd.read_csv(csv_path))
    logger.info("Enron: %d rows | %s", len(df), df["label"].value_counts().to_dict())
    return df


def load_phishing_dataset(enron_csv: Path, nazario_csv: Path, seed: int) -> pd.DataFrame:
    """
    Build a balanced phishing detection dataset: Enron ham (label=0) vs
    Nazario phishing (label=1). Downsamples Enron ham to match Nazario size.
    """
    enron_df = _clean(pd.read_csv(enron_csv))
    nazario_df = _clean(pd.read_csv(nazario_csv))

    enron_ham = enron_df[enron_df["label"] == 0][["text"]].copy()
    enron_ham["label"] = 0

    nazario_phishing = nazario_df[["text"]].copy()
    nazario_phishing["label"] = 1

    min_size = min(len(enron_ham), len(nazario_phishing))
    enron_ham = enron_ham.sample(n=min_size, random_state=seed)
    nazario_phishing = nazario_phishing.sample(n=min_size, random_state=seed)

    df = pd.concat([enron_ham, nazario_phishing], ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    logger.info(
        "Phishing dataset: %d rows | %s",
        len(df),
        df["label"].value_counts().to_dict(),
    )
    return df


def stratified_subsample(
    texts: pd.Series,
    labels: pd.Series,
    n: int,
    seed: int,
) -> tuple[pd.Series, pd.Series]:
    if n <= 0 or n >= len(texts):
        return texts, labels

    # train_test_split conveniently gives us a stratified sample.
    _, sub_texts, _, sub_labels = train_test_split(
        texts,
        labels,
        test_size=n,
        random_state=seed,
        stratify=labels,
    )
    return sub_texts, sub_labels


def append_metrics_row(
    results_path: Path,
    row: dict,
) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)

    df_row = pd.DataFrame([row])
    header = not results_path.exists()
    df_row.to_csv(results_path, mode="a", header=header, index=False)


def save_confusion_matrix(
    y_true,
    y_pred,
    output_path: Path,
    model_name: str,
    label_names: tuple[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_names,
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    print(f"\n=== DistilBERT Experiment ===")
    print(f"Device:              {get_device()}")
    print(f"Dataset:             {args.dataset}")
    print(f"Model:               {args.model_name}")
    print(f"Train subsample:     {args.train_subsample or 'FULL'}")
    print(f"Epochs:              {args.epochs}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Max sequence length: {args.max_length}")

    if args.dataset == "phishing":
        df = load_phishing_dataset(args.enron_csv, args.nazario_csv, seed=args.seed)
    else:
        df = load_enron_dataset(args.enron_csv)

    label_names = LABEL_NAMES[args.dataset]

    print(f"\n=== Splitting ===")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"],
    )
    logger.info("Full train: %d | Full test: %d", len(X_train_full), len(X_test))

    X_train, y_train = stratified_subsample(
        X_train_full,
        y_train_full,
        n=args.train_subsample,
        seed=args.seed,
    )
    logger.info(
        "Training subsample size: %d | class counts: %s",
        len(X_train),
        y_train.value_counts().to_dict(),
    )

    config = TransformerConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_labels=2,
        seed=args.seed,
    )

    print(f"\n=== Training ===")
    model, tokenizer = train_distilbert(
        texts=X_train.tolist(),
        labels=y_train.tolist(),
        config=config,
    )

    print(f"\n=== Evaluation (full test set: {len(X_test)} rows) ===")
    metrics = evaluate_distilbert(
        model=model,
        tokenizer=tokenizer,
        texts=X_test.tolist(),
        labels=y_test.tolist(),
        config=config,
    )

    print(
        f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    )
    print("-" * 72)
    print(
        f"{MODEL_DISPLAY_NAME:<30} "
        f"{metrics['accuracy']:>10.4f} "
        f"{metrics['precision']:>10.4f} "
        f"{metrics['recall']:>10.4f} "
        f"{metrics['f1']:>10.4f}"
    )

    print(f"\n=== Outputs ===")
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    row = {
        "run_id": run_id,
        "dataset": args.dataset,
        "Model": MODEL_DISPLAY_NAME,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"],
    }
    results_path = args.results_dir / "transformer_model_comparison.csv"
    append_metrics_row(results_path, row)
    logger.info("Appended metrics to %s", results_path)

    confusion_path = args.figures_dir / f"transformer_{args.dataset}_confusion_matrix.png"
    save_confusion_matrix(
        y_true=y_test.tolist(),
        y_pred=metrics["y_pred"],
        output_path=confusion_path,
        model_name=f"{MODEL_DISPLAY_NAME} ({args.dataset})",
        label_names=label_names,
    )
    logger.info("Saved confusion matrix to %s", confusion_path)


if __name__ == "__main__":
    main()
