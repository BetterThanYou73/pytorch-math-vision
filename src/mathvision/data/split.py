"""Stratified train/val split for label-in-hand datasets."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from torch.utils.data import Dataset, Subset


def stratified_split(
    dataset: Dataset,
    labels: Sequence[int],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Return (train_subset, val_subset) with per-class ratios preserved.

    `labels[i]` must equal `dataset[i]`'s label. Callers pass a precomputed
    label list to avoid iterating the full dataset here.
    """
    if len(labels) != len(dataset):
        raise ValueError(
            f"labels ({len(labels)}) and dataset ({len(dataset)}) length mismatch"
        )
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    rng = random.Random(seed)
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        buckets[int(y)].append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for idxs in buckets.values():
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_fraction)))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
