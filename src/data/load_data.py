"""
Utilities for loading and labeling raw email datasets.

This module supports recursive file discovery, record construction from
directory-based datasets, and conversion of parsed email records into
pandas DataFrames for modeling or analysis.
"""


from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from src.data.preprocess import PreprocessingConfig, parse_email_file, EmailRecord


def iter_email_files(root: Path) -> Iterator[Path]:
    """
    Recursively yield all files under a directory.

    Parameters
    ----------
    root : Path
        Root directory containing raw email files.

    Yields
    ------
    Path
        Each file path discovered under the root.
    """
    yield from (path for path in root.rglob("*") if path.is_file())


def build_labeled_records_from_dir(
    root: Path,
    label: int,
    label_name: str,
    relative_to: Optional[Path] = None,
    config: Optional[PreprocessingConfig] = None,
) -> list[dict]:
    """
    Parse all email files under a directory and assign a label.

    Parameters
    ----------
    root : Path
        Directory containing raw email files.
    label : int
        Numeric label for the class.
    label_name : str
        Human-readable class label.
    relative_to : Optional[Path], default=None
        Base path used for shorter relative path storage.
    config : Optional[PreprocessingConfig], default=None
        Email preprocessing configuration.

    Returns
    -------
    list[dict]
        Parsed records as dictionaries for convenient DataFrame creation.
    """
    records: list[EmailRecord] = []

    for path in iter_email_files(root):
        record = parse_email_file(path, config=config)
    
        if relative_to is not None:
            record.path = str(path.relative_to(relative_to))
        else:
            record.path = str(path)
    
        record.label = label
        record.label_name = label_name
        records.append(record)
    
    return records


def build_enron_records(
    data_root: Path,
    config: Optional[PreprocessingConfig] = None,
) -> list[EmailRecord]:
    """
    Build labeled ham/spam records from the Enron dataset layout.

    Parameters
    ----------
    data_root : Path
        Root directory containing `ham/` and `spam/`.
    config : Optional[PreprocessingConfig], default=None
        Email preprocessing configuration.

    Returns
    -------
    list[dict]
        Combined list of ham and spam records.
    """
    ham_root = data_root / "ham"
    spam_root = data_root / "spam"

    ham_records = build_labeled_records_from_dir(
        root=ham_root,
        label=0,
        label_name="ham",
        relative_to=data_root,
        config=config,
    )
    spam_records = build_labeled_records_from_dir(
        root=spam_root,
        label=1,
        label_name="spam",
        relative_to=data_root,
        config=config,
    )

    return ham_records + spam_records


def records_to_dataframe(records: list[dict], modeling_only: bool = True) -> pd.DataFrame:
    """
    Convert parsed records to a pandas DataFrame.

    Parameters
    ----------
    records : list[dict]
        Parsed email records.
    modeling_only : bool, default=True
        If True, keep only `text` and `label`.

    Returns
    -------
    pd.DataFrame
        DataFrame representation of the records.
    """
    df = pd.DataFrame(records)

    if modeling_only:
        return df[["text", "label"]]

    return df
