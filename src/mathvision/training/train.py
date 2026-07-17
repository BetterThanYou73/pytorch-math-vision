"""Training loop.

- AdamW + cosine LR schedule
- Class-weighted cross-entropy (inverse-sqrt weights) with label smoothing
- torch.amp autocast + GradScaler on CUDA, plain fp32 on CPU
- Gradient clipping at `grad_clip`
- Best checkpoint tracked by val loss, plus `last.pt` every epoch
- Per-epoch metrics written to a CSV under runs/{run_name}/metrics.csv
"""

from __future__ import annotations

import csv
import dataclasses
import json
import time
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

from ..data import (
    NUM_CLASSES,
    EMNISTByClass,
    SyntheticCharacters,
    eval_transforms,
    train_transforms,
)
from ..data.synthetic import seed_worker_torch
from ..data.transforms import IMAGE_SIZE
from ..models import build_resnet18
from .losses import compute_class_counts, inverse_sqrt_weights


@dataclasses.dataclass
class TrainConfig:
    seed: int = 42
    image_size: int = IMAGE_SIZE
    sources: tuple[str, ...] = ("emnist", "synthetic")
    emnist_root: str = "data/emnist"
    font_dir: str = "fonts"
    synthetic_samples_per_epoch: int = 50_000
    batch_size: int = 256
    num_workers: int = 4
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints"
    runs_dir: str = "runs"
    run_name: str = "default"
    device: str = "auto"
    max_train_batches: int | None = None
    max_val_batches: int | None = None


def _pick_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _build_datasets(cfg: TrainConfig) -> tuple[Dataset, Dataset]:
    train_tfm = train_transforms(cfg.image_size)
    eval_tfm = eval_transforms(cfg.image_size)
    train_parts: list[Dataset] = []
    if "emnist" in cfg.sources:
        train_parts.append(
            EMNISTByClass(cfg.emnist_root, train=True, transform=train_tfm, download=True)
        )
    if "synthetic" in cfg.sources:
        train_parts.append(
            SyntheticCharacters(
                font_dir=cfg.font_dir,
                samples_per_epoch=cfg.synthetic_samples_per_epoch,
                transform=train_tfm,
                seed=cfg.seed,
            )
        )
    if not train_parts:
        raise ValueError(f"no training sources selected: {cfg.sources}")
    train_ds = ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]

    val_ds = EMNISTByClass(cfg.emnist_root, train=False, transform=eval_tfm, download=True)
    return train_ds, val_ds


def _limited(loader: Iterable, cap: int | None) -> Iterable:
    if cap is None:
        yield from loader
        return
    for i, batch in enumerate(loader):
        if i >= cap:
            return
        yield batch


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None,
    total_batches: int | None = None,
    desc: str = "val",
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    bar = tqdm(
        _limited(loader, max_batches),
        total=total_batches,
        desc=desc,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for x, y in bar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += y.size(0)
    bar.close()
    return loss_sum / max(total, 1), correct / max(total, 1)


def _write_metrics_row(path: Path, row: dict) -> None:
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def train(cfg: TrainConfig) -> Path:
    """Run training end to end. Returns path to the best checkpoint."""
    torch.manual_seed(cfg.seed)
    device = _pick_device(cfg.device)
    print(f"device: {device}")

    train_ds, val_ds = _build_datasets(cfg)
    print(f"train samples: {len(train_ds)}    val samples: {len(val_ds)}")

    counts = compute_class_counts(train_ds, NUM_CLASSES)
    class_weights = inverse_sqrt_weights(counts).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker_torch,
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.num_workers > 0,
    )

    model = build_resnet18(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    ckpt_dir = Path(cfg.checkpoint_dir) / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(cfg.runs_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    metrics_path = run_dir / "metrics.csv"

    best_val_loss = float("inf")
    best_ckpt = ckpt_dir / "best.pt"

    total_train_batches = cfg.max_train_batches or (len(train_ds) // cfg.batch_size)
    total_val_batches = cfg.max_val_batches or ((len(val_ds) + cfg.batch_size - 1) // cfg.batch_size)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        train_bar = tqdm(
            _limited(train_loader, cfg.max_train_batches),
            total=total_train_batches,
            desc=f"epoch {epoch:>3}/{cfg.epochs} train",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for x, y in train_bar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            batch_size = y.size(0)
            loss_val = float(loss.item())
            epoch_loss += loss_val * batch_size
            batch_correct = int((logits.argmax(dim=1) == y).sum().item())
            epoch_correct += batch_correct
            epoch_total += batch_size
            train_bar.set_postfix(
                loss=f"{epoch_loss / epoch_total:.3f}",
                acc=f"{epoch_correct / epoch_total:.3f}",
            )
        train_bar.close()

        scheduler.step()

        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        val_loss, val_acc = _evaluate(
            model,
            val_loader,
            criterion,
            device,
            cfg.max_val_batches,
            total_batches=total_val_batches,
            desc=f"epoch {epoch:>3}/{cfg.epochs} val",
        )
        lr_now = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        print(
            f"epoch {epoch:>3}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
            f"lr={lr_now:.2e}  time={dt:.1f}s"
        )
        _write_metrics_row(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
                "lr": f"{lr_now:.6e}",
                "time_s": f"{dt:.2f}",
            },
        )

        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                best_ckpt,
            )
            print(f"  new best val_loss - saved {best_ckpt}")

    return best_ckpt
