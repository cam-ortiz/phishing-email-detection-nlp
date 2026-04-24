"""
DistilBERT-based email classifier.

This module is a pure modeling primitive: it exposes `train_distilbert`,
`predict_distilbert`, and `evaluate_distilbert` functions that mirror the
baseline model interface in `src/models/baseline.py` but operate on raw
text instead of TF-IDF features.

Device selection order: MPS (Apple Silicon) -> CUDA -> CPU.

No file I/O, no plotting, no CLI logic lives here — see
`scripts/run_transformer.py` for the end-to-end experiment runner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


logger = logging.getLogger(__name__)


DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 2
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_NUM_LABELS = 2
DEFAULT_SEED = 42


@dataclass
class TransformerConfig:
    """
    Hyperparameters for fine-tuning and evaluating a transformer classifier.
    """

    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = DEFAULT_BATCH_SIZE
    num_epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    num_labels: int = DEFAULT_NUM_LABELS
    seed: int = DEFAULT_SEED


def get_device() -> torch.device:
    """
    Return the best available torch device, preferring MPS on Apple Silicon.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


class EmailTextDataset(Dataset):
    """
    Minimal torch Dataset that tokenizes email text on demand.

    Keeps memory use low because tokenization happens per-item, which is
    important when we only subsample a few thousand training rows but
    still want lazy behaviour on the full test set.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[int]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.texts: list[str] = [str(t) for t in texts]
        self.labels: Optional[list[int]] = (
            [int(label) for label in labels] if labels is not None else None
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoded.items()}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_distilbert(
    texts: Iterable[str],
    labels: Iterable[int],
    config: Optional[TransformerConfig] = None,
    device: Optional[torch.device] = None,
    log_every: int = 50,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Fine-tune a DistilBERT (or compatible) classifier on labeled text.

    Parameters
    ----------
    texts, labels : sequences
        Training text and integer labels.
    config : TransformerConfig, optional
        Hyperparameters. Defaults to module-level defaults.
    device : torch.device, optional
        Device to train on. Auto-detected if omitted.
    log_every : int, default=50
        How often (in batches) to emit a progress log line.

    Returns
    -------
    tuple
        Fine-tuned model and tokenizer.
    """
    cfg = config or TransformerConfig()
    device = device or get_device()
    _set_seed(cfg.seed)

    logger.info("Device: %s", device)
    logger.info(
        "Model: %s | max_length=%d | batch=%d | epochs=%d | lr=%g",
        cfg.model_name,
        cfg.max_length,
        cfg.batch_size,
        cfg.num_epochs,
        cfg.learning_rate,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
    ).to(device)

    dataset = EmailTextDataset(
        texts=list(texts),
        labels=list(labels),
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    model.train()

    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % log_every == 0:
                running = epoch_loss / (batch_idx + 1)
                logger.info(
                    "epoch %d | batch %d/%d | running loss=%.4f",
                    epoch + 1,
                    batch_idx + 1,
                    len(loader),
                    running,
                )

        avg_loss = epoch_loss / max(len(loader), 1)
        logger.info("epoch %d done | avg loss=%.4f", epoch + 1, avg_loss)

    return model, tokenizer


def predict_distilbert(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Iterable[str],
    config: Optional[TransformerConfig] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Run inference and return integer class predictions.
    """
    cfg = config or TransformerConfig()
    device = device or get_device()

    dataset = EmailTextDataset(
        texts=list(texts),
        labels=None,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    model.eval()
    batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            batches.append(preds)

    return np.concatenate(batches) if batches else np.array([], dtype=np.int64)


def evaluate_distilbert(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Iterable[str],
    labels: Iterable[int],
    config: Optional[TransformerConfig] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Predict on `texts` and compute classification metrics against `labels`.

    Returns
    -------
    dict
        Keys: 'accuracy', 'precision', 'recall', 'f1', 'y_pred'.
        'y_pred' is kept so callers can build confusion matrices without
        re-running inference.
    """
    y_true = np.asarray(list(labels))
    y_pred = predict_distilbert(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        config=config,
        device=device,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "y_pred": y_pred,
    }
