"""
Model evaluation utilities for email classification models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    model: Any,
    X_test,
    y_test,
    positive_label: int = 1,
) -> dict[str, float]:
    """
    Evaluate a trained classifier using common classification metrics.
    """
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, pos_label=positive_label, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, pos_label=positive_label, zero_division=0
        ),
        "f1": f1_score(
            y_test, y_pred, pos_label=positive_label, zero_division=0
        ),
    }


def build_classification_report_df(
    model,
    X_test,
    y_test,
    target_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a classification report as a DataFrame.
    """
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report).transpose()


def save_metrics_csv(
    results_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Save model comparison metrics to CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    return output_path


def save_classification_report_csv(
    report_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Save a classification report DataFrame to CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path)
    return output_path


def save_confusion_matrix_image(
    model: Any,
    X_test,
    y_test,
    output_path: Path,
    display_labels: list[str] | None = None,
    title: str | None = None,
) -> Path:
    """
    Save a confusion matrix image for a trained model.
    """
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_labels,
    )

    display.plot(values_format="d")
    plt.title(title or "Confusion Matrix")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return output_path