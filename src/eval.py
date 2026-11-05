"""Per-class and per-condition evaluation.

    python -m src.eval models/<run>/best.pt [--split test]

prints the metrics and writes confusion_<split>.png and errors_<split>.csv
(misclassified images, label = "true->predicted") next to the checkpoint;
`python tools/preview.py models/<run>/errors_test.csv` shows them as a grid.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.datasets import ManifestDataset, build_transforms, load_splits
from src.models import create_model


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(labels)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def metrics(preds, targets, classes):
    """Accuracy plus class-balanced scores: with imbalanced classes, plain accuracy hides weak classes."""
    labels = list(range(len(classes)))
    return {
        "accuracy": accuracy_score(targets, preds),
        "balanced_accuracy": balanced_accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, labels=labels, average="macro", zero_division=0),
        "report": classification_report(targets, preds, labels=labels, target_names=classes,
                                        digits=3, zero_division=0),
        "confusion": confusion_matrix(targets, preds, labels=labels).tolist(),
    }


def metrics_by_condition(rows, preds, targets, classes=None, columns=("lighting", "weather", "surface")):
    """Accuracy and macro F1 per value of each condition column (rows without a value are skipped).

    With `classes`, every value also gets per-class precision and recall: the brief asks for
    precision and recall per paving type. None means undefined (no such images, or never predicted).
    """
    out = {}
    for column in columns:
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            if row.get(column):
                groups[row[column]].append(i)
        out[column] = {}
        for value, idx in sorted(groups.items()):
            t, p = targets[idx], preds[idx]
            entry = {"n": len(idx),
                     "accuracy": accuracy_score(t, p),
                     "macro_f1": f1_score(t, p, average="macro", zero_division=0)}
            if classes:
                entry["per_class"] = {name: precision_recall(t, p, k) for k, name in enumerate(classes)}
            out[column][value] = entry
    return out


def precision_recall(targets, preds, k):
    """Precision and recall of class k (None when undefined) and how many images truly are k."""
    hits = int(((preds == k) & (targets == k)).sum())
    n_true, n_pred = int((targets == k).sum()), int((preds == k).sum())
    return {"n": n_true,
            "precision": hits / n_pred if n_pred else None,
            "recall": hits / n_true if n_true else None}


def condition_table(by_condition, focus="weed"):
    """One line per condition value, with precision/recall of the `focus` class when available."""
    def fmt(x):
        return "  -  " if x is None else f"{x:.3f}"

    lines = []
    for column, values in by_condition.items():
        for value, m in values.items():
            line = f"  {column}={value:<10} n={m['n']:<5} acc {m['accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}"
            pr = m.get("per_class", {}).get(focus)
            if pr:
                line += f"  {focus} precision {fmt(pr['precision'])} recall {fmt(pr['recall'])}"
            lines.append(line)
    return "\n".join(lines) or "  no lighting/weather/surface tags in this split yet, so no per-condition results"


def plot_confusion(confusion, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(confusion)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = create_model(ckpt["model_name"], len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    rows = [r for r in load_splits()[1][args.split] if r["label"] in classes]
    dataset = ManifestDataset(rows, classes, build_transforms(False, ckpt["img_size"]))
    preds, targets = predict(model, DataLoader(dataset, batch_size=args.batch_size), device)
    result = metrics(preds, targets, classes)
    result["by_condition"] = metrics_by_condition(rows, preds, targets, classes)

    out = args.checkpoint.parent
    plot_confusion(result["confusion"], classes, out / f"confusion_{args.split}.png")
    (out / f"metrics_{args.split}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(out / f"errors_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows((r["path"], f"{classes[t]}->{classes[p]}") for r, p, t in zip(rows, preds, targets) if p != t)

    print(f"{args.split.upper()}  accuracy {result['accuracy']:.3f}  balanced accuracy "
          f"{result['balanced_accuracy']:.3f}  macro-F1 {result['macro_f1']:.3f}\n{result['report']}")
    print(f"confusion matrix (rows = true, columns = predicted, order {classes}):")
    for row in result["confusion"]:
        print("  ", row)
    print("per condition:\n" + condition_table(result["by_condition"]))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
