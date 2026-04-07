"""
Dataset preparation utilities for email classification workflows.

This module provides helpers for summarizing parsed email records,
filtering low-quality data, and converting records into pandas DataFrames
ready for modeling.

It builds on the preprocessing and data loading layers to support
end-to-end dataset preparation, including basic quality checks,
length-based filtering, and final dataset construction.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from ..data.load_data import build_enron_records
from ..data.preprocess import EmailRecord, PreprocessingConfig


def summarize_records(records: list[EmailRecord]) -> dict:
    """
    Compute summary statistics for parsed email records.

    Parameters
    ----------
    records : list[EmailRecord]
        Parsed email records.

    Returns
    -------
    dict
        Summary statistics for record counts, labels, fallback usage,
        and text lengths.
    """
    nonempty_lengths = [len(r.text) for r in records if r.text.strip()]

    return {
        "total": len(records),
        "label_counts": Counter(r.label_name for r in records),
        "empty_combined_text": sum(1 for r in records if not r.text.strip()),
        "empty_subject": sum(1 for r in records if not r.subject.strip()),
        "empty_body": sum(1 for r in records if not r.body.strip()),
        "parse_failures": sum(1 for r in records if not r.parse_success),
        "used_fallback": sum(1 for r in records if r.used_fallback),
        "full_fallback": sum(1 for r in records if r.fallback_type == "full_from_raw"),
        "body_fallback": sum(1 for r in records if r.fallback_type == "body_from_raw"),
        "min_length": min(nonempty_lengths) if nonempty_lengths else 0,
        "max_length": max(nonempty_lengths) if nonempty_lengths else 0,
        "avg_length": (
            sum(nonempty_lengths) / len(nonempty_lengths)
            if nonempty_lengths
            else 0.0
        ),
    }


def print_summary(summary: dict) -> None:
    """
    Print record summary statistics in a readable format.

    Parameters
    ----------
    summary : dict
        Summary dictionary returned by summarize_records.
    """
    print("Total:", summary["total"])
    print("Label counts:", summary["label_counts"])
    print("Empty combined text:", summary["empty_combined_text"])
    print("Empty subject:", summary["empty_subject"])
    print("Empty body:", summary["empty_body"])
    print("Parse failures:", summary["parse_failures"])
    print("Used fallback:", summary["used_fallback"])
    print("Full fallback:", summary["full_fallback"])
    print("Body fallback:", summary["body_fallback"])
    print("Min length:", summary["min_length"])
    print("Max length:", summary["max_length"])
    print("Avg length:", summary["avg_length"])


def filter_records(
    records: list[EmailRecord],
    min_length: int = 10,
    drop_empty: bool = True,
) -> list[EmailRecord]:
    """
    Filter parsed email records for downstream modeling.

    Parameters
    ----------
    records : list[EmailRecord]
        Parsed email records.
    min_length : int, default=10
        Minimum text length required to keep a record.
    drop_empty : bool, default=True
        Whether to drop records with empty combined text.

    Returns
    -------
    list[EmailRecord]
        Filtered records.
    """
    filtered: list[EmailRecord] = []

    for record in records:
        text = (record.text or "").strip()

        if drop_empty and not text:
            continue

        if len(text) < min_length:
            continue

        filtered.append(record)

    return filtered


def records_to_dataframe(
    records: list[EmailRecord],
    modeling_only: bool = True,
) -> pd.DataFrame:
    """
    Convert parsed email records to a pandas DataFrame.

    Parameters
    ----------
    records : list[EmailRecord]
        Parsed email records.
    modeling_only : bool, default=True
        Whether to keep only the columns needed for modeling.

    Returns
    -------
    pd.DataFrame
        DataFrame representation of the records.
    """
    df = pd.DataFrame([record.to_dict() for record in records])

    if modeling_only:
        return df[["text", "label"]]

    return df


def build_enron_modeling_dataframe(
    data_root: Path,
    config: Optional[PreprocessingConfig] = None,
    min_length: int = 10,
    drop_empty: bool = True,
    modeling_only: bool = True,
) -> tuple[list[EmailRecord], list[EmailRecord], dict, pd.DataFrame]:
    """
    Build, summarize, filter, and convert Enron email records for modeling.

    Parameters
    ----------
    data_root : Path
        Root directory containing the Enron ham/ and spam/ folders.
    config : Optional[PreprocessingConfig], default=None
        Preprocessing configuration.
    min_length : int, default=10
        Minimum text length required after parsing.
    drop_empty : bool, default=True
        Whether to drop records with empty text.
    modeling_only : bool, default=True
        Whether to keep only modeling columns in the returned DataFrame.

    Returns
    -------
    tuple[list[EmailRecord], list[EmailRecord], dict, pd.DataFrame]
        A tuple containing:
        - all parsed records
        - filtered records
        - summary statistics computed before filtering
        - final DataFrame
    """
    records = build_enron_records(data_root, config=config)
    summary = summarize_records(records)
    filtered_records = filter_records(
        records,
        min_length=min_length,
        drop_empty=drop_empty,
    )
    df = records_to_dataframe(filtered_records, modeling_only=modeling_only)

    return records, filtered_records, summary, df


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save a DataFrame to CSV, creating parent directories if needed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    output_path : Path
        Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
