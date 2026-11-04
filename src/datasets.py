"""Dataset classes and transforms (shared by notebooks and app)."""
import csv
import random
from pathlib import Path

import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFESTS = ROOT / "manifests"
REQUIRED = {"path", "label", "split"}
SPLITS = ("train", "val", "test")

# ImageNet statistics: every timm backbone we use was pretrained with them
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def class_names():
    """Class names in id order, from configs/classes.yaml."""
    cfg = yaml.safe_load((ROOT / "configs" / "classes.yaml").read_text(encoding="utf-8"))
    return [c["name"] for c in sorted(cfg["classes"], key=lambda c: c["id"])]


def load_manifests(paths=None):
    """Rows of every manifest CSV (files without path/label/split columns are skipped)."""
    rows = []
    for path in paths or sorted(MANIFESTS.glob("*.csv")):
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if REQUIRED <= set(reader.fieldnames or []):
                rows += list(reader)
    return rows


def load_splits(limit=0, seed=0):
    """Classes that have training images, and the manifest rows of each split."""
    names = class_names()
    rows = load_manifests()
    unknown = {r["label"] for r in rows} - set(names)
    if unknown:
        raise ValueError(f"labels not in configs/classes.yaml: {sorted(unknown)}")
    present = {r["label"] for r in rows if r["split"] == "train"}
    classes = [c for c in names if c in present]
    missing = [c for c in names if c not in present]
    if missing:
        print(f"no training images for {missing}: left out of this model")
    splits = {s: [r for r in rows if r["split"] == s and r["label"] in present] for s in SPLITS}
    if limit:
        rng = random.Random(seed)
        splits = {s: rng.sample(v, min(limit, len(v))) for s, v in splits.items()}
    return classes, splits


def build_transforms(train, size=224):
    if not train:
        return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.35, 1.0)),
        # The robot camera looks straight down, so any orientation is valid
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(90)], p=0.5),
        # Sun, overcast and wet pavement change brightness and contrast a lot
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.03),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25),
    ])


class ManifestDataset(Dataset):
    def __init__(self, rows, classes, transform):
        index = {name: i for i, name in enumerate(classes)}
        self.items = [(DATA / r["path"], index[r["label"]]) for r in rows]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label
