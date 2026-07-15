# pytorch-math-vision

Single-character classifier for handwritten and printed math symbols. Successor to
[tensorflow-math-vision](https://github.com/BetterThanYou73/tensorflow-math-vision),
rebuilt in PyTorch. Intended to feed a downstream equation OCR and solver (separate repo).

## Status

Early scaffold. Work happens on feature branches; `main` only accepts merged features.

## Class set

Roughly 100 classes covering digits (0-9), letters (A-Z, a-z), and math punctuation
(`+ - * / = ( ) [ ] { } . , : ; < > ? ! @ # $ % ^ & _ | ~`). Visually ambiguous case pairs
(C/c, O/o, S/s, U/u, V/v, W/w, X/x, Z/z, K/k, P/p) are merged into a single class -
you cannot tell them apart from a tight character crop, and pretending you can just
poisons the labels.

The canonical class list lives in `src/mathvision/data/classes.py` (added in the
data-pipeline feature branch).

## Data sources

| Source | Access | Notes |
| --- | --- | --- |
| EMNIST byclass | `torchvision.datasets.EMNIST(split="byclass", download=True)` | 62 classes, 814k images, auto-downloaded |
| Handwritten math symbols | Kaggle `xainano/handwrittenmathsymbols` | 82 classes, needs `~/.kaggle/kaggle.json` |
| Synthetic | rendered on the fly in a `Dataset` from `.ttf` fonts | Ported from legacy `synthetic_image.py` |
| HASYv2 (optional) | Zenodo download | 369 LaTeX symbols, for a later expansion pass |

## Development

Requires Python 3.11 or 3.12.

```
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -e ".[dev]"
```

## Roadmap

Tracked in GitHub Issues. High level:

1. Data pipeline (EMNIST + class merge + synthetic renderer)
2. Kaggle symbols loader
3. Grayscale ResNet-18 backbone with GAP head
4. Training loop with mixed precision and stratified split
5. Evaluation, confusion matrix, per-class accuracy report
6. ONNX export for downstream deployment

## License

MIT. See `LICENSE`.
