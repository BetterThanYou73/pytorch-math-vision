"""Grayscale ResNet-18 for character classification.

Starts from torchvision's ResNet-18, then:
- Replaces the RGB stem conv with a single-channel version (kept 7x7 stride
  2 so downstream feature-map math is unchanged from the reference model).
- Swaps the 1000-class ImageNet head for a `num_classes` linear layer.

The classification head is the stock ResNet head: adaptive average pool ->
flatten -> linear. No 1024/512 dense stack sitting on top of a Flatten -
that head is what tanked the legacy TF model's variance.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18(num_classes: int) -> nn.Module:
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    nn.init.normal_(model.fc.weight, std=0.01)
    nn.init.zeros_(model.fc.bias)

    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
