"""Training loop and experiment tracking.

Fine-tunes an ImageNet-pretrained timm backbone on every manifest in manifests/,
keeps the epoch with the best validation macro-F1 and reports on the test split.
Results go to models/<run>/: best.pt, last.pt, metrics.json, confusion_test.png.

    python -m src.train                                # efficientnet_b0, 15 epochs
    python -m src.train --model convnext_tiny --epochs 20
    python -m src.train --resume models/<run>/last.pt  # continue an interrupted run
    python -m src.train --limit 200 --epochs 1         # quick smoke test
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import SPLITS, ManifestDataset, build_transforms, load_splits
from src.eval import metrics, metrics_by_condition, pick_device, plot_confusion, predict
from src.models import create_model

try:
    import mlflow  # optional: when installed, every run is also logged to ./mlruns
except ImportError:
    mlflow = None

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0", help="any timm model name")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1, help="softens the effect of noisy auto-labels")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--patience", type=int, default=5, help="stop after N epochs without val improvement")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="use only N images per split (smoke test)")
    ap.add_argument("--name", help="run folder under models/ (default: model name + time)")
    ap.add_argument("--resume", type=Path, help="last.pt of an interrupted run")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def class_weights(labels, n):
    """Inverse square-root frequency: helps rare classes without overfitting their few images."""
    counts = np.bincount(labels, minlength=n).astype(float)
    w = 1 / np.sqrt(counts.clip(min=1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    use_amp = device.type == "cuda"

    classes, splits = load_splits(args.limit, args.seed)
    print(f"device: {device}  classes: {classes}")
    for s in SPLITS:
        print(f"  {s}: {len(splits[s])} images")

    datasets = {s: ManifestDataset(splits[s], classes, build_transforms(s == "train", args.img_size)) for s in SPLITS}
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                     persistent_workers=args.workers > 0, pin_memory=use_amp)
    train_loader = DataLoader(datasets["train"], shuffle=True, drop_last=True, **loader_kw)
    eval_loaders = {s: DataLoader(datasets[s], shuffle=False, **loader_kw) for s in ("val", "test")}

    model = create_model(args.model, len(classes)).to(device)
    weights = class_weights(np.array([y for _, y in datasets["train"].items]), len(classes))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup = len(train_loader)  # one epoch of linear warm-up, then cosine decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_f1, stale, history = 0, -1.0, 0, []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_f1, stale, history = ckpt["epoch"] + 1, ckpt["best_f1"], ckpt["stale"], ckpt["history"]
        run_dir = args.resume.parent
        print(f"resuming {run_dir.name} from epoch {start_epoch + 1}")
    else:
        run_dir = ROOT / "models" / (args.name or f"{args.model}_{time.strftime('%Y%m%d_%H%M')}")
        run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"classes": classes, "model_name": args.model, "img_size": args.img_size}

    if mlflow:
        mlflow.set_experiment("weed-robot")
        mlflow.start_run(run_name=run_dir.name)
        mlflow.log_params({k: str(v) for k, v in vars(args).items()})

    for epoch in range(start_epoch, args.epochs):
        model.train()
        start, running = time.time(), 0.0
        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device.type, enabled=use_amp):
                loss = criterion(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 50 == 0:
                left = (time.time() - start) / step * (len(train_loader) - step)
                print(f"  epoch {epoch + 1} step {step}/{len(train_loader)} loss {running / step:.3f} (~{left / 60:.0f} min left)")

        val = metrics(*predict(model, eval_loaders["val"], device), classes)
        record = {"epoch": epoch + 1, "train_loss": running / len(train_loader),
                  "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                  "minutes": (time.time() - start) / 60}
        history.append(record)
        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1, stale = val["macro_f1"], 0
            torch.save({"model": model.state_dict(), **meta}, run_dir / "best.pt")
        else:
            stale += 1
        print(f"epoch {epoch + 1}/{args.epochs}: loss {record['train_loss']:.3f}  val acc {val['accuracy']:.3f}  "
              f"val macro-F1 {val['macro_f1']:.3f}{'  * best' if improved else ''}  ({record['minutes']:.1f} min)")
        if mlflow:
            mlflow.log_metrics({k: v for k, v in record.items() if k != "epoch"}, step=epoch + 1)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_f1": best_f1, "stale": stale, "history": history, **meta},
                   run_dir / "last.pt")
        if stale >= args.patience:
            print(f"no improvement for {args.patience} epochs, stopping")
            break

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=False)["model"])
    preds, targets = predict(model, eval_loaders["test"], device)
    test = metrics(preds, targets, classes)
    test["by_condition"] = metrics_by_condition(splits["test"], preds, targets)
    plot_confusion(test["confusion"], classes, run_dir / "confusion_test.png")
    (run_dir / "metrics.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "classes": classes,
         "history": history, "best_val_macro_f1": best_f1, "test": test}, indent=2), encoding="utf-8")

    print(f"\nTEST  accuracy {test['accuracy']:.3f}  balanced accuracy {test['balanced_accuracy']:.3f}  "
          f"macro-F1 {test['macro_f1']:.3f}\n{test['report']}")
    print(f"saved to {run_dir}")
    if mlflow:
        mlflow.log_metrics({f"test_{k}": test[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")})
        mlflow.log_artifact(str(run_dir / "metrics.json"))
        mlflow.log_artifact(str(run_dir / "confusion_test.png"))
        mlflow.end_run()


if __name__ == "__main__":
    main()
