"""Smoke test for the data pipeline.

Loads one batch from EMNIST and one from the synthetic renderer, saves a
side-by-side grid to outputs/smoke/data_grid.png so you can eyeball that
the class labels line up with what you see.

Run:
    python scripts/smoke_data.py --emnist-root data/emnist
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from mathvision.data import (
    CLASSES,
    NUM_CLASSES,
    EMNISTByClass,
    SyntheticCharacters,
    train_transforms,
)
from mathvision.data.synthetic import seed_worker_torch


def _grid(images: torch.Tensor, labels: list[int], title: str, ax_row) -> None:
    for ax, img, lbl in zip(ax_row, images, labels, strict=False):
        arr = img.squeeze(0).numpy() * 0.5 + 0.5
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(CLASSES[int(lbl)], fontsize=10)
        ax.axis("off")
    ax_row[0].set_ylabel(title, fontsize=12, rotation=0, ha="right", va="center")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--emnist-root", type=Path, default=Path("data/emnist"))
    p.add_argument("--font-dir", type=Path, default=Path("fonts"))
    p.add_argument("--out", type=Path, default=Path("outputs/smoke/data_grid.png"))
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()

    print(f"Canonical class count: {NUM_CLASSES}")
    print(f"First 20 classes: {CLASSES[:20]}")

    tfm = train_transforms()
    emnist = EMNISTByClass(root=args.emnist_root, train=True, transform=tfm, download=True)
    synth = SyntheticCharacters(font_dir=args.font_dir, samples_per_epoch=1024, transform=tfm)

    emnist_loader = DataLoader(emnist, batch_size=args.n, shuffle=True, num_workers=0)
    synth_loader = DataLoader(
        synth,
        batch_size=args.n,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker_torch,
    )

    e_imgs, e_lbls = next(iter(emnist_loader))
    s_imgs, s_lbls = next(iter(synth_loader))
    print(f"EMNIST batch:    imgs={tuple(e_imgs.shape)}, labels={e_lbls.tolist()}")
    print(f"Synthetic batch: imgs={tuple(s_imgs.shape)}, labels={s_lbls.tolist()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, args.n, figsize=(args.n * 1.2, 3.2))
    _grid(e_imgs, e_lbls.tolist(), "EMNIST", axes[0])
    _grid(s_imgs, s_lbls.tolist(), "synthetic", axes[1])
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Wrote grid to {args.out}")


if __name__ == "__main__":
    main()
