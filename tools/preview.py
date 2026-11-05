"""Save a grid of random images per class from a manifest, to eyeball label quality.

Usage: python tools/preview.py [manifest.csv] [per_class] [out.jpg]
"""
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
TILE = 200

if __name__ == "__main__":
    manifest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "manifests" / "inaturalist.csv"
    per_class = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "docs" / f"preview_{manifest.stem}.jpg"

    by_label = defaultdict(list)
    with open(manifest, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_label[row["label"]].append(row["path"])
    labels = sorted(by_label)

    rng = random.Random(0)
    grid = Image.new("RGB", (per_class * TILE, len(labels) * (TILE + 24)), "white")
    draw = ImageDraw.Draw(grid)
    for row, label in enumerate(labels):
        y = row * (TILE + 24)
        draw.text((4, y + 4), f"{label} ({len(by_label[label])})", fill="black")
        for col, path in enumerate(rng.sample(by_label[label], min(per_class, len(by_label[label])))):
            image = Image.open(ROOT / "data" / path).convert("RGB")
            image.thumbnail((TILE, TILE))
            grid.paste(image, (col * TILE, y + 24))
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out, quality=90)
    print(out)
