"""Training CLI.

    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/smoke.yaml --set epochs=1

--set key=value overrides map into TrainConfig fields. Values are parsed
as YAML so `--set sources=[emnist]` and `--set max_train_batches=10` do
what you would expect.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import yaml

from mathvision.training import TrainConfig, train


def _load_config(path: Path, overrides: list[str]) -> TrainConfig:
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, _, value = item.partition("=")
        raw[key.strip()] = yaml.safe_load(value)
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(raw) - fields
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    if isinstance(raw.get("sources"), list):
        raw["sources"] = tuple(raw["sources"])
    return TrainConfig(**raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = _load_config(args.config, args.__dict__["set"])
    print(f"config: {dataclasses.asdict(cfg)}")
    best = train(cfg)
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
