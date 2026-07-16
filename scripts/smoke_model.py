"""Smoke test for the model.

Builds the model at the canonical class count, runs a random batch through
it, prints the output shape and parameter count.

Run:
    python scripts/smoke_model.py
"""

from __future__ import annotations

import torch

from mathvision.data import NUM_CLASSES
from mathvision.data.transforms import IMAGE_SIZE
from mathvision.models import build_resnet18
from mathvision.models.resnet import count_parameters


def main() -> None:
    model = build_resnet18(NUM_CLASSES).eval()
    n_params = count_parameters(model)
    print(f"num_classes = {NUM_CLASSES}")
    print(f"parameters  = {n_params:,} ({n_params / 1e6:.2f}M)")

    x = torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        y = model(x)
    print(f"input       = {tuple(x.shape)}")
    print(f"output      = {tuple(y.shape)}")

    assert y.shape == (2, NUM_CLASSES), f"unexpected output shape {y.shape}"
    print("shape check OK")


if __name__ == "__main__":
    main()
