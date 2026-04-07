"""
Email preprocessing utilities for NLP classification tasks.

This module provides a configurable pipeline for parsing raw email files into
clean text suitable for modeling (e.g., spam or phishing detection).

Features include:
- MIME-based subject and body extraction
- HTML-to-text conversion
- Trimming of quoted reply/forward chains (preserving content when stripping 
  would remove all text)
- Whitespace normalization
- Fallback parsing for malformed emails
- Text length validation and truncation

The main entry point is `parse_email_file`, which returns an `EmailRecord`
containing structured and cleaned text fields.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from typing import Optional
from collections.abc import Mapping
import re

from bs4 import BeautifulSoup


HTML_MARKERS: tuple[str, ...] = (
    "<html",
    "<body",
    "<div",
    "<p",
    "<br",
    "<table",
    "<a ",
    "<!doctype html",
)

CHARSET_ALIASES: Mapping[str, str] = {
    "default": "latin-1",
    "iso-1885-1": "iso-8859-1",
    "iso-5480-9": "iso-8859-9",
    "iso-7404-7": "iso-8859-7",
    "7bit": "ascii",
    "8bit": "latin-1",
}

HEADER_PREFIXES: tuple[str, ...] = (
    "content-type:",
    "content-transfer-encoding:",
    "content-description:",
    "content-disposition:",
    "content-id:",
    "content-location:",
    "mime-version:",
    "charset=",
)

# Patterns that commonly indicate the start of a quoted reply/forward block.
REPLY_CHAIN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?im)^-+\s*original message\s*-+\s*$"),
    re.compile(r"(?im)^-+\s*forwarded by.*$"),
    re.compile(
        r"(?im)^from:\s+.+\n^sent:\s+.+\n^to:\s+.+\n^subject:\s+.+", 
        re.MULTILINE
    ),
    re.compile(r"(?im)^on .+ wrote:\s*$"),
)


@dataclass(slots=True)
class PreprocessingConfig:
    """
    Configuration for email parsing and text cleanup.

    Parameters
    ----------
    strip_reply_chains : bool, default=True
        Whether to remove quoted reply/forward chains from message bodies.
        Stripping is only applied when it preserves meaningful content; 
        otherwise, the original body is retained.
    convert_html : bool, default=True
        Whether to convert HTML content to visible text.
    include_raw_text : bool, default=False
        Whether to retain raw decoded email text in the returned record.
        Helpful for debugging but memory-intensive at scale.
    min_text_length : int, default=10
        Minimum allowed length of final combined text.
    max_text_length : int, default=20000
        Maximum allowed length of final combined text. Longer text is truncated.
    """
    strip_reply_chains: bool = True
    convert_html: bool = True
    include_raw_text: bool = False
    min_text_length: int = 10
    max_text_length: int = 20_000


@dataclass(slots=True)
class EmailRecord:
    """
    Parsed email record for downstream NLP workflows.

    Parameters
    ----------
    path : str
        Path to the source email file.
    subject : str
        Extracted email subject.
    body : str
        Extracted visible email body.
    text : str
        Final combined text used for modeling.
    parse_success : bool
        Whether structured MIME parsing succeeded.
    used_fallback : bool
        Whether fallback extraction logic was used.
    error : str, default=""
        Error message or parse note.
    raw_text : Optional[str], default=None
        Raw decoded email text when debugging is enabled.
    """
    path: str
    subject: str
    body: str
    text: str
    parse_success: bool
    used_fallback: bool
    error: str = ""
    raw_text: Optional[str] = None
    fallback_type: str = "none"
    label: Optional[int] = None
    label_name: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert the record to a plain dictionary."""
        return asdict(self)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving readable structure.

    Parameters
    ----------
    text : str
        Input text to normalize.

    Returns
    -------
    str
        Text with normalized line endings, collapsed runs of spaces,
        and reduced excessive blank lines.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[ \xa0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(html: str) -> str:
    """
    Convert HTML markup into visible text.

    Parameters
    ----------
    html : str
        Raw HTML string.

    Returns
    -------
    str
        Visible text extracted from the HTML document.
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # Remove non-visible content that adds noise for NLP.
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    return normalize_whitespace(text)


def looks_like_html(text: str) -> bool:
    """
    Heuristically determine whether a string likely contains HTML.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    bool
        True if common HTML markers are detected, otherwise False.
    """
    if not text:
        return False

    lowered = text.lower()
    return any(marker in lowered for marker in HTML_MARKERS)


def strip_quoted_reply_chain(text: str) -> str:
    """
    Remove quoted reply or forward chains from a message body.

    Parameters
    ----------
    text : str
        Email body text.

    Returns
    -------
    str
        Body text with common quoted reply/forward sections removed.

    Notes
    -----
    This uses heuristic patterns and may need dataset-specific tuning.
    It is intentionally conservative compared with naive substring matching
    on tokens such as 'from:' or 'subject:'.
    """
    if not text:
        return ""

    earliest_match_start: Optional[int] = None

    for pattern in REPLY_CHAIN_PATTERNS:
        match = pattern.search(text)
        if match:
            if (
                earliest_match_start is None
                or match.start() < earliest_match_start
            ):
                earliest_match_start = match.start()

    if earliest_match_start is None:
        return text

    return text[:earliest_match_start].rstrip()


def _decode_part_to_text(part: Message) -> str:
    """
    Decode a text-bearing MIME part into a Python string.

    Parameters
    ----------
    part : Message
        MIME part that is expected to contain textual content.

    Returns
    -------
    str
        Decoded text content, or an empty string if the part cannot be
        reasonably decoded.

    Notes
    -----
    This function first tries the high-level email API (`get_content()`).
    If that fails or yields no usable text, it falls back to lower-level
    payload decoding via `get_payload(decode=True)`. This is helpful for
    malformed corpora where transfer encodings or charsets are inconsistent.
    """
    try:
        content = part.get_content()
        if isinstance(content, str):
            return content
    except Exception:
        pass

    try:
        payload = part.get_payload(decode=True)
    except Exception:
        payload = None

    if payload is None:
        # Sometimes decode=True returns None for already-decoded or oddly
        # formatted payloads. Try raw payload as a last resort.
        try:
            raw_payload = part.get_payload()
        except Exception:
            return ""

        if isinstance(raw_payload, str):
            return raw_payload
        if isinstance(raw_payload, list):
            return ""
        return str(raw_payload)

    if isinstance(payload, str):
        return payload

    if not isinstance(payload, (bytes, bytearray)):
        return ""

    charset = (part.get_content_charset() or "").strip().lower()
    charset = CHARSET_ALIASES.get(charset, charset)

    if charset:
        try:
            return payload.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            pass

    # Pragmatic fallback for noisy legacy email corpora.
    try:
        return payload.decode("utf-8", errors="replace")
    except Exception:
        return payload.decode("latin-1", errors="replace")


def _is_attachment_like(part: Message) -> bool:
    """
    Heuristically determine whether a MIME part should be treated as an
    attachment rather than visible message body text.

    Parameters
    ----------
    part : Message
        MIME part to inspect.

    Returns
    -------
    bool
        True if the part appears to be an attachment or non-body file.
    """
    content_disposition = str(part.get("Content-Disposition", "")).lower()
    filename = part.get_filename()
    content_type = part.get_content_type()

    if "attachment" in content_disposition:
        return True

    # Many file attachments expose a filename even if disposition is weak.
    if filename and not content_type.startswith("text/"):
        return True

    # Skip obvious non-text payloads.
    if not content_type.startswith("text/"):
        return True

    return False


def _append_visible_part(
    part: Message,
    plain_parts: list[str],
    html_parts: list[str],
) -> None:
    """
    Extract visible content from a MIME part and append it to the appropriate
    accumulator.

    Parameters
    ----------
    part : Message
        Email message part to inspect.
    plain_parts : list[str]
        Accumulator for plain-text content.
    html_parts : list[str]
        Accumulator for HTML content.
    """
    if _is_attachment_like(part):
        return

    content_type = part.get_content_type()
    content_text = _decode_part_to_text(part)

    if not content_text:
        return

    content_text = content_text.strip()

    # If declared as text but decode returned empty, try raw payload fallback.
    if not content_text and content_type.startswith("text/"):
        raw_payload = part.get_payload()
        if isinstance(raw_payload, str):
            content_text = raw_payload.strip()

    if not content_text:
        return

    if content_type.startswith("text/"):
        if content_type == "text/html":
            html_parts.append(content_text)
        else:
            plain_parts.append(content_text)


def _collect_visible_body_parts(msg: Message) -> tuple[list[str], list[str]]:
    """
    Collect visible plain-text and HTML body candidates from a parsed email.

    Parameters
    ----------
    msg : Message
        Parsed email message object.

    Returns
    -------
    tuple[list[str], list[str]]
        Plain-text parts and HTML parts collected from the message.
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            # Skip container parts and focus on leaf payloads.
            if part.is_multipart():
                continue
            _append_visible_part(part, plain_parts, html_parts)
        return plain_parts, html_parts

    _append_visible_part(msg, plain_parts, html_parts)
    return plain_parts, html_parts


def extract_visible_body_from_message(
    msg: Message,
    convert_html: bool = True,
) -> str:
    """
    Extract the most useful visible body text from a parsed email message.

    Preference order
    ----------------
    1. text/plain parts
    2. text/html parts converted to visible text

    Parameters
    ----------
    msg : Message
        Parsed email message object.
    convert_html : bool, default=True
        Whether HTML parts should be converted to visible text.

    Returns
    -------
    str
        Extracted visible message body.
    """
    plain_parts, html_parts = _collect_visible_body_parts(msg)

    plain_text = normalize_whitespace("\n\n".join(part for part in plain_parts if part))
    if plain_text:
        return plain_text

    if convert_html and html_parts:
        html_text = normalize_whitespace(
            "\n\n".join(html_to_text(part) for part in html_parts if part)
        )
        if html_text:
            return html_text

    return ""


def strip_mime_boilerplate_from_fallback(body: str) -> str:
    """
    Remove common MIME boundary and part-header boilerplate from fallback body
    text while preserving the visible payload.

    Parameters
    ----------
    body : str
        Raw fallback body text.

    Returns
    -------
    str
        Body text with common multipart boundary/header boilerplate removed.
    """
    if not body:
        return ""

    text = body.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    cleaned_lines: list[str] = []

    in_part_headers = False

    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()

        # Skip multipart boundary markers like
        if stripped.startswith("--") or stripped.startswith("----"):
            in_part_headers = True
            continue

        # Skip MIME part headers that usually appear immediately after a boundary.
        if in_part_headers and lowered.startswith(HEADER_PREFIXES):
            continue

        # Blank line after MIME part headers means actual payload starts next.
        if in_part_headers and stripped == "":
            in_part_headers = False
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    return normalize_whitespace(cleaned)


def looks_like_mime_part_boilerplate(text: str) -> bool:
    """
    Heuristically determine whether fallback body text appears to contain
    MIME boundary/header boilerplate.
    """
    lowered = text.lower()
    return (
        "content-transfer-encoding:" in lowered
        or "content-type:" in lowered
        or re.search(r"(?m)^--[-=_a-zA-Z0-9]+$", text) is not None
    )


def extract_subject_body_fallback(raw_text: str) -> tuple[str, str]:
    """
    Fallback extraction of subject and body from raw email text.

    Parameters
    ----------
    raw_text : str
        Raw decoded email message text.

    Returns
    -------
    tuple[str, str]
        Extracted subject and body text.

    Notes
    -----
    This fallback uses the first blank line as the header/body separator
    and manually searches for a Subject header.
    """
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    subject_match = re.search(
        r"^Subject:\s*(.*)$",
        normalized,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    subject = subject_match.group(1).strip() if subject_match else ""

    parts = normalized.split("\n\n", 1)
    body = parts[1] if len(parts) > 1 else normalized

    # Clean multipart fallback boilerplate before HTML detection / normalization.
    if looks_like_mime_part_boilerplate(body):
        body = strip_mime_boilerplate_from_fallback(body)

    if looks_like_html(body):
        body = html_to_text(body)

    body = normalize_whitespace(body)
    return subject, body


def finalize_email_text(
    text: str,
    min_length: int = 10,
    max_length: int = 20_000,
) -> Optional[str]:
    """
    Validate and truncate final email modeling text.

    Parameters
    ----------
    text : str
        Final combined email text.
    min_length : int, default=10
        Minimum allowed text length.
    max_length : int, default=20000
        Maximum allowed text length.

    Returns
    -------
    Optional[str]
        Cleaned text if it passes validation, otherwise None.
    """
    cleaned = text.strip()

    if len(cleaned) < min_length:
        return None

    return cleaned[:max_length]


@dataclass(slots=True, frozen=True)
class ParsedEmailContent:
    """
    Intermediate representation of parsed email content before final normalization.

    Attributes
    ----------
    subject : str
        Extracted subject line.
    body : str
        Extracted body text.
    parse_success : bool
        Whether structured parsing succeeded.
    error : str
        Error message or note if parsing failed.
    fallback_type : str
        Type of fallback used (if any).
    """
    subject: str
    body: str
    parse_success: bool
    error: str
    fallback_type: str
    
    
def _parse_email_content(
    raw_bytes: bytes,
    raw_text: str,
    cfg: PreprocessingConfig,
) -> ParsedEmailContent:
    """
    Parse structured email content and apply fallback extraction when needed.

    Parameters
    ----------
    raw_bytes : bytes
        Raw email bytes read from disk.
    raw_text : str
        Raw email text decoded for fallback extraction.
    cfg : PreprocessingConfig
        Preprocessing configuration.

    Returns
    -------
    ParsedEmailContent
        Intermediate parse result including subject, body, parse status,
        error message, and fallback classification.
    """
    fallback_type = "none"

    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
        subject = msg.get("Subject", "") or ""
        body = extract_visible_body_from_message(
            msg,
            convert_html=cfg.convert_html,
        )
        parse_success = True
        error = ""
    except Exception as exc:
        # Full fallback when MIME parsing itself fails.
        subject, body = extract_subject_body_fallback(raw_text)
        parse_success = False
        error = f"parse_error: {exc}"
        fallback_type = "full_from_raw"

    subject = normalize_whitespace(str(subject))

    # If parsing succeeded but produced an empty body, use fallback extraction.
    if parse_success and not body.strip():
        fallback_subject, fallback_body = (
            extract_subject_body_fallback(raw_text)
        )

        if not subject:
            subject = normalize_whitespace(fallback_subject)

        body = fallback_body
        fallback_type = "body_from_raw"

    return ParsedEmailContent(
        subject=subject,
        body=body,
        parse_success=parse_success,
        error=error,
        fallback_type=fallback_type,
    )
    
def parse_email_file(
    path: Path,
    config: Optional[PreprocessingConfig] = None,
) -> EmailRecord:
    """
    Parse a raw email file into a structured record for NLP use.

    Parameters
    ----------
    path : Path
        Path to a raw email file.
    config : Optional[PreprocessingConfig], default=None
        Preprocessing configuration. If None, defaults are used.

    Returns
    -------
    EmailRecord
        Parsed email record containing subject, body, and combined text.
    """
    cfg = config or PreprocessingConfig()

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        return EmailRecord(
            path=str(path),
            subject="",
            body="",
            text="",
            parse_success=False,
            used_fallback=False,
            error=f"read_error: {exc}",
            raw_text=None,
            fallback_type="read_error",
        )

    raw_text = raw_bytes.decode("latin-1", errors="ignore")

    parsed = _parse_email_content(raw_bytes, raw_text, cfg)
    body = parsed.body

    # Apply reply-chain stripping.
    if cfg.strip_reply_chains:
        stripped_body = strip_quoted_reply_chain(body)

        # Keep stripped result only if it preserves enough content.
        if stripped_body.strip():
            body = stripped_body

    body = normalize_whitespace(body)

    combined_text = normalize_whitespace(
        f"{parsed.subject}\n\n{body}".strip()
    )
    combined_text = finalize_email_text(
        combined_text,
        min_length=cfg.min_text_length,
        max_length=cfg.max_text_length,
    ) or ""

    return EmailRecord(
        path=str(path),
        subject=parsed.subject,
        body=body,
        text=combined_text,
        parse_success=parsed.parse_success,
        used_fallback=parsed.fallback_type != "none",
        error=parsed.error,
        raw_text=raw_text if cfg.include_raw_text else None,
        fallback_type=parsed.fallback_type,
    )
