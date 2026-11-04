"""Turn the scored iNaturalist photos into manifests/inaturalist.csv.

* Whole photos that pass the scene filter get their group's class:
  weed -> weed, moss -> moss, desired -> desired_plant.
* Each kept photo is also cut into a 2x2 grid of crops, scored by CLIP as
  "plant" or "bare surface". The most plant-like crop gets the photo's class;
  from pavement weed photos up to two bare crops with (almost) no green pixels
  become clean_surface. Crops look like the tiles the robot will classify, and
  they keep the classes from differing only in zoom level.
* Split 80/10/10 by photographer (session), so one person's streets never land
  in two splits, while keeping every class close to 80/10/10 too.

Columns follow manifests/manifest_schema.md. lighting, weather and surface are
left empty: they are unknown for iNaturalist photos, and CLIP's surface guess
proved wrong on close-ups (weeds on concrete came out as "soil"). Attribution
(author, licence, link) goes to manifests/inaturalist_attribution.csv.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from score_scenes import Clip, p_first

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw" / "inaturalist"
CROPS = DATA / "interim" / "inaturalist_crops"
MANIFESTS = ROOT / "manifests"
LABEL = {"weed": "weed", "moss": "moss", "desired": "desired_plant"}
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

CROP_PLANT = [
    "a photo of a green plant",
    "a photo of a weed",
    "a photo of moss",
    "a photo of grass",
    "a photo of a flower",
]
CROP_BARE = [
    "a photo of bare paving stones",
    "a photo of an empty concrete sidewalk",
    "a photo of bare brick pavers",
    "a photo of gravel",
    "a photo of asphalt",
    "a photo of fallen dry leaves on pavement",
    "a photo of litter on the pavement",
    "a photo of a shadow on paving stones",
    "a photo of bare soil",
]


def grid_crops(image):
    w, h = image.size
    return [image.crop((x, y, x + w // 2, y + h // 2)) for y in (0, h // 2) for x in (0, w // 2)]


def green_fraction(image):
    """Share of clearly green pixels (excess-green index), to catch plants CLIP missed."""
    a = np.asarray(image, dtype=np.int16)
    return float(((2 * a[..., 1] - a[..., 0] - a[..., 2]) > 30).mean())


def assign_splits(rows):
    """Biggest photographers first, each to the split whose per-class shortfall
    their photos fill best (shortfalls relative to class size, so small classes count too)."""
    per_user = defaultdict(Counter)
    for r in rows:
        per_user[r["user_id"]][r["group"]] += 1
    total = Counter(r["group"] for r in rows)
    filled = {s: Counter() for s in SPLITS}
    split_of = {}
    for user, counts in sorted(per_user.items(), key=lambda kv: -sum(kv[1].values())):
        def gain(s):
            return sum(n * (SPLITS[s] - filled[s][g] / total[g]) for g, n in counts.items())
        split = max(SPLITS, key=gain)
        split_of[user] = split
        filled[split].update(counts)
    return split_of


def score_crops(rows):
    """p_plant for every crop of every kept photo, cached in crop_scores.csv."""
    CROPS.mkdir(parents=True, exist_ok=True)
    cache_path = CROPS / "crop_scores.csv"
    cache = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cache = {(r["photo_id"], r["k"]): float(r["p_plant"]) for r in csv.DictReader(f)}
    todo = [r for r in rows if (r["photo_id"], "0") not in cache]
    if todo:
        clip = Clip()
        text = clip.text(CROP_PLANT + CROP_BARE)
        print(f"scoring crops of {len(todo)} photos")
        for i in range(0, len(todo), 8):
            batch = todo[i:i + 8]
            crops = [c for r in batch for c in grid_crops(Image.open(RAW / "images" / f"{r['photo_id']}.jpg").convert("RGB"))]
            p_plant = p_first(clip.images(crops), text, len(CROP_PLANT)).tolist()
            for j, r in enumerate(batch):
                for k in range(4):
                    cache[(r["photo_id"], str(k))] = p_plant[4 * j + k]
            if (i // 8) % 100 == 0:
                print(f"  {min(i + 8, len(todo))}/{len(todo)}")
                save_crop_scores(cache_path, cache)
        save_crop_scores(cache_path, cache)
    return cache


def save_crop_scores(path, cache):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["photo_id", "k", "p_plant"])
        w.writerows((pid, k, f"{p:.4f}") for (pid, k), p in cache.items())


def main():
    ap = argparse.ArgumentParser()
    # Scene thresholds per group, picked by eyeballing score bins: below 0.7 weed
    # photos are mostly lawns and wide street shots, desired ones greenhouses and
    # trees; moss is fine from 0.5
    ap.add_argument("--min-weed", type=float, default=0.7)
    ap.add_argument("--min-moss", type=float, default=0.5)
    ap.add_argument("--min-desired", type=float, default=0.7)
    ap.add_argument("--min-crop", type=float, default=0.9, help="CLIP confidence for a crop to be used")
    ap.add_argument("--max-green", type=float, default=0.01, help="max green-pixel share of a clean_surface crop")
    ap.add_argument("--limit", type=int, default=0, help="use only N photos (quick test)")
    args = ap.parse_args()
    min_keep = {"weed": args.min_weed, "moss": args.min_moss, "desired": args.min_desired}

    with open(RAW / "scores.csv", encoding="utf-8") as f:
        scores = {r["photo_id"]: r for r in csv.DictReader(f)}
    with open(RAW / "meta.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["photo_id"] in scores and float(scores[r["photo_id"]]["p_keep"]) >= min_keep[r["group"]]]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} photos pass the scene filter: {dict(Counter(r['group'] for r in rows))}")

    split_of = assign_splits(rows)
    crop_scores = score_crops(rows)

    entries = []
    for r in rows:
        pid = r["photo_id"]
        base = {
            "source": "inaturalist",
            "licence": r["license"],
            "lighting": "",
            "weather": "",
            "surface": "",
            "session": f"inat_{r['user_id']}",
            "split": split_of[r["user_id"]],
        }
        entries.append({"path": f"raw/inaturalist/images/{pid}.jpg", "label": LABEL[r["group"]], **base})

        p = [crop_scores[(pid, str(k))] for k in range(4)]
        picks = []
        best_plant = max(range(4), key=lambda k: p[k])
        if p[best_plant] >= args.min_crop:
            picks.append((best_plant, LABEL[r["group"]]))
        # Only weed photos: moss covers most of a moss photo, and CLIP takes dry brown moss for bare stone
        bare = [k for k in sorted(range(4), key=lambda k: p[k]) if 1 - p[k] >= args.min_crop]
        if (r["group"] == "weed" and bare) or picks:
            crops = grid_crops(Image.open(RAW / "images" / f"{pid}.jpg").convert("RGB"))
            if r["group"] == "weed":
                clean = [k for k in bare if green_fraction(crops[k]) < args.max_green]
                picks += [(k, "clean_surface") for k in clean[:2]]
            for k, label in picks:
                out = CROPS / f"{pid}_{k}.jpg"
                if not out.exists():
                    crops[k].save(out, quality=92)
                entries.append({"path": f"interim/inaturalist_crops/{pid}_{k}.jpg", "label": label, **base})

    fields = ["path", "label", "source", "licence", "lighting", "weather", "surface", "session", "split"]
    with open(MANIFESTS / "inaturalist.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(entries)
    with open(MANIFESTS / "inaturalist_attribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["photo_id", "author", "licence", "observation"])
        w.writerows((r["photo_id"], r["user_login"], r["license"],
                     f"https://www.inaturalist.org/observations/{r['obs_id']}") for r in rows)

    print(f"{len(entries)} manifest rows")
    for split in SPLITS:
        print(f"  {split}: {dict(Counter(e['label'] for e in entries if e['split'] == split))}")


if __name__ == "__main__":
    main()
