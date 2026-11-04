"""Pack the photos for notebooks/06_detection_colab.ipynb.

    python tools/pack_detection.py

Takes every whole photo in manifests/inaturalist.csv (weed, moss, desired_plant)
plus the clean_surface crops (background images without plants), with the same
photographer split as the classifier. Writes data/processed/detection/
(images/ + photos.csv, hard links, no extra disk space) and
data/processed/detection.zip to upload to Google Drive.
"""
import csv
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "processed" / "detection"


def link(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:  # another drive or a filesystem without hard links
        shutil.copy2(src, dst)


def main():
    with open(ROOT / "manifests" / "inaturalist.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["path"].startswith("raw/") or r["label"] == "clean_surface"]
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for r in rows:
        link(DATA / r["path"], OUT / "images" / Path(r["path"]).name)
    with open(OUT / "photos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "split", "session", "licence"])
        w.writerows((Path(r["path"]).name, r["label"], r["split"], r["session"], r["licence"]) for r in rows)

    zip_path = OUT.with_suffix(".zip")
    # JPEGs are already compressed: store them as they are
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        z.write(OUT / "photos.csv", "detection/photos.csv")
        for r in rows:
            name = Path(r["path"]).name
            z.write(OUT / "images" / name, f"detection/images/{name}")

    for split in ("train", "val", "test"):
        print(f"  {split}: {dict(Counter(r['label'] for r in rows if r['split'] == split))}")
    print(f"{len(rows)} images -> {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
