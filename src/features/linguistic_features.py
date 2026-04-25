"""
Linguistic feature analysis for phishing/social-engineering emails.
"""

from __future__ import annotations

import re

import pandas as pd


LINGUISTIC_PATTERNS = {
    "urgency": [
        "urgent",
        "immediately",
        "act now",
        "right away",
        "as soon as possible",
        "within 24 hours",
        "limited time",
        "expires",
        "deadline",
        "final notice",
        "do not delay",
    ],
    "threat_fear": [
        "suspended",
        "locked",
        "security alert",
        "account closed",
        "unauthorized",
        "risk",
        "fraud",
        "suspicious activity",
        "terminated",
        "deactivated",
        "failure to comply",
        "legal action",
    ],
    "authority": [
        "bank",
        "it department",
        "administrator",
        "support team",
        "security team",
        "help desk",
        "billing department",
        "customer service",
        "microsoft",
        "paypal",
        "amazon",
        "irs",
    ],
    "sensitive_request": [
        "password",
        "verify",
        "confirm",
        "account information",
        "personal information",
        "credentials",
        "social security",
        "ssn",
        "credit card",
        "bank account",
        "security question",
        "date of birth",
    ],
    "action_request": [
        "click",
        "download",
        "login",
        "log in",
        "open attachment",
        "update",
        "reset",
        "verify account",
        "confirm details",
        "follow the link",
        "submit",
        "complete the form",
    ],
}


def pattern_to_regex(pattern: str) -> str:
    """
    Convert a phrase into a regex that supports flexible whitespace.
    """
    words = pattern.lower().split()
    escaped_words = [re.escape(word) for word in words]
    return r"\b" + r"\s+".join(escaped_words) + r"\b"


def count_pattern_matches(text: str, patterns: list[str]) -> int:
    """
    Count how many pattern terms appear in a text string.
    """
    if not isinstance(text, str):
        return 0

    text_lower = text.lower()
    count = 0

    for pattern in patterns:
        regex = pattern_to_regex(pattern)
        count += len(re.findall(regex, text_lower))

    return count


def add_linguistic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add linguistic pattern count columns to a modeling dataframe.
    """
    featured_df = df.copy()

    for category, patterns in LINGUISTIC_PATTERNS.items():
        column_name = f"{category}_count"
        featured_df[column_name] = featured_df["text"].apply(
            lambda text: count_pattern_matches(text, patterns)
        )

    featured_df["total_linguistic_count"] = featured_df[
        [f"{category}_count" for category in LINGUISTIC_PATTERNS]
    ].sum(axis=1)

    return featured_df


def summarize_linguistic_features(
    df: pd.DataFrame,
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Summarize average linguistic feature counts by class label.
    """
    feature_columns = [
        f"{category}_count" for category in LINGUISTIC_PATTERNS
    ] + ["total_linguistic_count"]

    summary_df = (
        df.groupby("label")[feature_columns]
        .mean()
        .reset_index()
    )

    if label_names:
        label_map = {
            label_value: label_names[index]
            for index, label_value in enumerate(sorted(df["label"].unique()))
            if index < len(label_names)
        }

        summary_df["label_name"] = summary_df["label"].map(label_map)

        ordered_columns = ["label", "label_name"] + feature_columns
        summary_df = summary_df[ordered_columns]

    return summary_df