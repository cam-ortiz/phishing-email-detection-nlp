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
import mailbox

from src.data.preprocess import (
    PreprocessingConfig, 
    parse_email_file, 
    EmailRecord,
    extract_visible_body_from_message,
    strip_quoted_reply_chain,
    normalize_whitespace,
    finalize_email_text,
   )


def parse_email_message(
    msg,
    path: str,
    config: Optional[PreprocessingConfig] = None,
) -> EmailRecord:
    """
    Parse a single mailbox message into an EmailRecord directly from the
    mailbox message object.
    """
    cfg = config or PreprocessingConfig()

    try:
        subject = msg.get("Subject", "") or ""
        subject = normalize_whitespace(str(subject))

        body = extract_visible_body_from_message(
            msg,
            convert_html=cfg.convert_html,
        )

        parse_success = True
        error = ""
        fallback_type = "none"

    except Exception as exc:
        subject = ""
        body = ""
        parse_success = False
        error = f"parse_error: {exc}"
        fallback_type = "message_parse_error"

    if cfg.strip_reply_chains:
        stripped_body = strip_quoted_reply_chain(body)
        if stripped_body.strip():
            body = stripped_body

    body = normalize_whitespace(body)

    combined_text = normalize_whitespace(
        f"{subject}\n\n{body}".strip()
    )
    combined_text = finalize_email_text(
        combined_text,
        min_length=cfg.min_text_length,
        max_length=cfg.max_text_length,
    ) or ""

    return EmailRecord(
        path=path,
        subject=subject,
        body=body,
        text=combined_text,
        parse_success=parse_success,
        used_fallback=False,
        error=error,
        raw_text=None,
        fallback_type=fallback_type,
    )


def build_nazario_records(
    data_root: Path,
    config: Optional[PreprocessingConfig] = None,
) -> list[EmailRecord]:
    """
    Build labeled phishing records from Nazario .mbox files.
    """
    records: list[EmailRecord] = []

    for mbox_path in sorted(data_root.glob("*.mbox")):
        mbox = mailbox.mbox(mbox_path)

        for i, msg in enumerate(mbox):
            record = parse_email_message(
                msg,
                path=f"{mbox_path.name}::message_{i}",
                config=config,
            )
            record.label = 1
            record.label_name = "phishing"
            records.append(record)

    return records


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
