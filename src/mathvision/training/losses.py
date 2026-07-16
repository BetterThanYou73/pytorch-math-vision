"""Class-weight helpers for cross-entropy.

Inverse-sqrt weighting is gentler than pure inverse-frequency: it lifts
under-represented classes without flattening them to equal weight, which
would over-emphasize rare noisy classes and hurt overall accuracy.
"""

from __future__ import annotations

import torch
from torch.utils.data import ConcatDataset, Dataset

from ..data.synthetic import SyntheticCharacters


def compute_class_counts(dataset: Dataset, num_classes: int) -> torch.Tensor:
    """Return a `[num_classes]` tensor of expected per-class counts.

    Delegates to `dataset.labels` when available (EMNIST wrapper exposes
    this) so we do not load images just to count labels. For
    SyntheticCharacters the count is theoretical since class draw is uniform.
    """
    if isinstance(dataset, ConcatDataset):
        total = torch.zeros(num_classes)
        for d in dataset.datasets:
            total += compute_class_counts(d, num_classes)
        return total

    if isinstance(dataset, SyntheticCharacters):
        per = len(dataset) / num_classes
        return torch.full((num_classes,), per, dtype=torch.float32)

    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise TypeError(
            f"cannot count labels for {type(dataset).__name__}: "
            "expose a `.labels` attribute or add a branch here"
        )
    counts = torch.zeros(num_classes)
    for y in labels:
        counts[int(y)] += 1
    return counts


def inverse_sqrt_weights(counts: torch.Tensor, min_count: float = 1.0) -> torch.Tensor:
    """Turn per-class counts into normalized cross-entropy weights."""
    w = 1.0 / torch.sqrt(counts.clamp(min=min_count))
    return w / w.mean()
