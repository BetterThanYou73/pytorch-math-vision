"""EMNIST byclass wrapper with the rotation fix and canonical class remap.

torchvision.datasets.EMNIST returns images in the source NIST orientation:
rotated 90 degrees clockwise and horizontally flipped. Callers must undo
this before use, which this wrapper does.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST as _TVEMNIST

from .classes import canonicalize, char_to_idx

_EMNIST_BYCLASS_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class EMNISTByClass(Dataset):
    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform=None,
        download: bool = True,
    ):
        self._base = _TVEMNIST(
            root=str(root), split="byclass", train=train, download=download
        )
        self._transform = transform
        self._remap = [char_to_idx[canonicalize(ch)] for ch in _EMNIST_BYCLASS_CHARS]

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        img, emnist_label = self._base[idx]
        img = img.transpose(Image.TRANSPOSE)
        label = self._remap[int(emnist_label)]
        if self._transform is not None:
            img = self._transform(img)
        return img, label

    @property
    def labels(self) -> list[int]:
        """All labels in canonical index space. O(N) — cache the result."""
        targets = self._base.targets.tolist()
        return [self._remap[t] for t in targets]
