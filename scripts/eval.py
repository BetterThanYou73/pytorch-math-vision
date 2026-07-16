"""Evaluation CLI.

    python scripts/eval.py --ckpt checkpoints/overnight/best.pt

Writes report.md, confusion.png, confusion.npy under
outputs/eval/{timestamp}/ (override with --output-dir).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from mathvision.eval.report import build_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--emnist-root", type=Path, default=Path("data/emnist"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None, help="cuda / cpu / auto (default: auto)")
    args = p.parse_args()

    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = args.output_dir or Path("outputs/eval") / time.strftime("%Y%m%d-%H%M%S")

    print(f"checkpoint: {args.ckpt}")
    print(f"device:     {device}")
    print(f"output:     {output_dir}")

    summary = build_report(
        ckpt_path=args.ckpt,
        emnist_root=args.emnist_root,
        output_dir=output_dir,
        device=device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print()
    print(f"top-1:                       {summary['top1'] * 100:.2f}%")
    print(f"top-3:                       {summary['top3'] * 100:.2f}%")
    print(
        f"macro F1 (present {summary['n_classes_present']} classes): "
        f"{summary['macro_f1_present']:.4f}"
    )
    print(f"macro F1 (all 80 classes):    {summary['macro_f1']:.4f}")
    print()
    print("worst 10:")
    for c, a in summary["worst_10"]:
        print(f"  {c!r:>4}  {a * 100:.1f}%")
    print()
    print("top confusions (true -> predicted, count):")
    for t, p, c in summary["top_pairs"][:10]:
        print(f"  {t!r:>4} -> {p!r:<4}  {c}")
    print()
    print(f"full report: {output_dir}/report.md")


if __name__ == "__main__":
    main()
