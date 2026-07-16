"""Image transforms used across data sources.

All datasets in this package produce single-channel PIL images with the
character drawn in a light tone on a dark background (matching EMNIST).
`RandomInvert` in training gives the model both polarities so it works on
real-world black-on-white handwriting at inference time.
"""

from __future__ import annotations

from torchvision import transforms as T

IMAGE_SIZE = 64


def train_transforms(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.Resize((image_size, image_size)),
            T.RandomAffine(degrees=8, translate=(0.06, 0.06), shear=6, fill=0),
            T.RandomInvert(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def eval_transforms(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
