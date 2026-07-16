"""Synthetic character renderer as a torch Dataset.

Renders single characters from .ttf/.otf font files on the fly with light
augmentation (rotation, shear, tight crop). Distilled from the legacy
`synthetic_image.py` but returns tensors from a Dataset - no disk writes.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

import torch
from PIL import Image, ImageChops, ImageDraw, ImageFont
from torch.utils.data import Dataset, get_worker_info

from .classes import CLASSES, char_to_idx


def _list_fonts(font_dir: Path) -> list[Path]:
    fonts = sorted(font_dir.glob("*.ttf")) + sorted(font_dir.glob("*.otf"))
    if not fonts:
        raise FileNotFoundError(f"no .ttf/.otf fonts found in {font_dir}")
    return fonts


def _tight_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    ref = Image.new(image.mode, image.size, 0)
    diff = ImageChops.difference(image, ref)
    bbox = diff.getbbox()
    if bbox is None:
        return image
    pad = rng.randint(4, 12)
    return image.crop(
        (
            max(bbox[0] - pad, 0),
            max(bbox[1] - pad, 0),
            min(bbox[2] + pad, image.width),
            min(bbox[3] + pad, image.height),
        )
    )


def _render_char(char: str, font: ImageFont.FreeTypeFont, rng: random.Random) -> Image.Image:
    canvas = 256
    img = Image.new("L", (canvas, canvas), color=0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas - tw) // 2 - bbox[0]
    y = (canvas - th) // 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=255)

    angle = rng.uniform(-6, 6)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=0)

    shear = rng.uniform(-0.15, 0.15)
    w, h = img.size
    dx = int(abs(shear) * w)
    img = img.transform(
        (w + dx, h),
        Image.AFFINE,
        (1, shear, -dx if shear > 0 else 0, 0, 1, 0),
        resample=Image.BICUBIC,
        fillcolor=0,
    )
    return _tight_crop(img, rng)


class SyntheticCharacters(Dataset):
    def __init__(
        self,
        font_dir: str | Path,
        samples_per_epoch: int = 20_000,
        classes: Iterable[str] = CLASSES,
        transform=None,
        seed: int = 0,
    ):
        self._fonts = _list_fonts(Path(font_dir))
        self._classes = list(classes)
        self._samples_per_epoch = int(samples_per_epoch)
        self._transform = transform
        self._seed = seed

    def __len__(self) -> int:
        return self._samples_per_epoch

    def __getitem__(self, idx: int):
        rng = _rng_for(idx, self._seed)
        ch = rng.choice(self._classes)
        font_path = rng.choice(self._fonts)
        size = rng.randint(90, 150)
        font = ImageFont.truetype(str(font_path), size)
        img = _render_char(ch, font, rng)
        label = char_to_idx[ch]
        if self._transform is not None:
            img = self._transform(img)
        return img, label


def _rng_for(idx: int, seed: int) -> random.Random:
    """Deterministic per-sample RNG that also folds in the worker id.

    Without this, all DataLoader workers would draw from the same sequence
    and produce duplicates.
    """
    info = get_worker_info()
    worker_id = info.id if info is not None else 0
    salt = (seed * 1_000_003) ^ (worker_id * 2_654_435_761) ^ idx
    return random.Random(salt & 0xFFFF_FFFF)


def seed_worker_torch(worker_id: int) -> None:
    """Standard worker_init_fn to reseed torch/numpy per worker."""
    info = get_worker_info()
    base = (info.seed if info is not None else 0) & 0xFFFF_FFFF
    torch.manual_seed(base)
