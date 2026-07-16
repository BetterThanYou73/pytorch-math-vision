"""Evaluation report.

Given a checkpoint and a dataset, computes top-1, top-3, macro F1, and
per-class accuracy, and writes:
- `report.md`: summary, worst/best 10 classes, top-20 confusion pairs
- `confusion.png`: row-normalized 80x80 heatmap
- `confusion.npy`: raw confusion matrix for downstream analysis

Read-only: nothing here touches training state.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from ..data import CLASSES, NUM_CLASSES, EMNISTByClass, eval_transforms
from ..data.transforms import IMAGE_SIZE
from ..models import build_resnet18


def _load_state_dict(ckpt_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels_all: list[np.ndarray] = []
    top3_all: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        top3 = torch.topk(logits, k=3, dim=1).indices
        top3_all.append(top3.cpu().numpy())
        labels_all.append(y.numpy())
    labels = np.concatenate(labels_all)
    top3 = np.concatenate(top3_all)
    preds = top3[:, 0]
    return labels, preds, top3


def _top_confusion_pairs(cm: np.ndarray, k: int) -> list[tuple[int, int, int]]:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = off.flatten()
    idx = np.argsort(-flat)[:k]
    pairs: list[tuple[int, int, int]] = []
    for i in idx:
        if flat[i] == 0:
            break
        t, p = int(i // cm.shape[1]), int(i % cm.shape[1])
        pairs.append((t, p, int(cm[t, p])))
    return pairs


def _render_confusion_png(cm: np.ndarray, classes: list[str], out: Path, title: str) -> None:
    n = len(classes)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(max(12.0, n * 0.18), max(12.0, n * 0.18)))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(classes, fontsize=6, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(
    labels: np.ndarray,
    preds: np.ndarray,
    top3: np.ndarray,
    cm: np.ndarray,
    ckpt_path: Path,
    out: Path,
    title: str,
) -> dict:
    n = len(labels)
    top1 = float((preds == labels).mean())
    top3_acc = float(np.any(top3 == labels[:, None], axis=1).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))

    supports = cm.sum(axis=1)
    present = np.where(supports > 0)[0]
    macro_f1_present = float(
        f1_score(labels, preds, labels=present.tolist(), average="macro", zero_division=0)
    )

    row_sums = supports.clip(min=1)
    per_class_acc = np.diag(cm) / row_sums
    # Rank only classes that actually appear in the eval set - a class with
    # zero support is a reporting artefact, not a model failure.
    ranked = sorted(present, key=lambda i: per_class_acc[i])
    worst = [(CLASSES[i], float(per_class_acc[i]), int(supports[i])) for i in ranked[:10]]
    best = [(CLASSES[i], float(per_class_acc[i]), int(supports[i])) for i in ranked[::-1][:10]]
    absent = [CLASSES[i] for i in range(len(CLASSES)) if supports[i] == 0]
    pairs = _top_confusion_pairs(cm, k=20)

    lines: list[str] = []
    lines.append(f"# Eval report - {title}")
    lines.append("")
    lines.append(f"- Checkpoint: `{ckpt_path.as_posix()}`")
    lines.append(f"- Samples: {n}")
    lines.append(f"- Top-1: **{top1 * 100:.2f}%**")
    lines.append(f"- Top-3: **{top3_acc * 100:.2f}%**")
    lines.append(f"- Macro F1 (all {NUM_CLASSES} classes): {macro_f1:.4f}")
    lines.append(
        f"- Macro F1 (present {len(present)} classes): **{macro_f1_present:.4f}**"
    )
    if absent:
        lines.append(
            f"- Classes not present in eval set ({len(absent)}): "
            + " ".join(f"`{c}`" for c in absent)
        )
    lines.append("")
    lines.append("## Worst 10 classes")
    lines.append("")
    lines.append("| class | accuracy | support |")
    lines.append("|---|---|---|")
    for c, acc, sup in worst:
        lines.append(f"| `{c}` | {acc * 100:.1f}% | {sup} |")
    lines.append("")
    lines.append("## Best 10 classes")
    lines.append("")
    lines.append("| class | accuracy | support |")
    lines.append("|---|---|---|")
    for c, acc, sup in best:
        lines.append(f"| `{c}` | {acc * 100:.1f}% | {sup} |")
    lines.append("")
    lines.append("## Top-20 confusion pairs")
    lines.append("")
    lines.append("| true | predicted | count |")
    lines.append("|---|---|---|")
    for t, p, c in pairs:
        lines.append(f"| `{CLASSES[t]}` | `{CLASSES[p]}` | {c} |")

    out.write_text("\n".join(lines), encoding="utf-8")

    return {
        "top1": top1,
        "top3": top3_acc,
        "macro_f1": macro_f1,
        "macro_f1_present": macro_f1_present,
        "n_samples": n,
        "n_classes_present": int(len(present)),
        "worst_10": [(c, a) for c, a, _ in worst],
        "best_10": [(c, a) for c, a, _ in best],
        "top_pairs": [(CLASSES[t], CLASSES[p], c) for t, p, c in pairs],
    }


def build_report(
    ckpt_path: Path,
    emnist_root: Path,
    output_dir: Path,
    device: torch.device,
    image_size: int = IMAGE_SIZE,
    batch_size: int = 256,
    num_workers: int = 4,
) -> dict:
    ckpt_path = Path(ckpt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_resnet18(NUM_CLASSES).to(device)
    model.load_state_dict(_load_state_dict(ckpt_path, device))

    val_ds = EMNISTByClass(
        emnist_root,
        train=False,
        transform=eval_transforms(image_size),
        download=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    labels, preds, top3 = _collect_predictions(model, val_loader, device)
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

    ts = time.strftime("%Y%m%d-%H%M%S")
    title = f"{ckpt_path.stem} @ {ts}"

    _render_confusion_png(cm, CLASSES, output_dir / "confusion.png", title)
    np.save(output_dir / "confusion.npy", cm)
    summary = _write_markdown(labels, preds, top3, cm, ckpt_path, output_dir / "report.md", title)

    return summary
