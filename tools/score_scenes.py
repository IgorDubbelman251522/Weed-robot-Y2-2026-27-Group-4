"""Score every downloaded photo with CLIP zero-shot: is it the scene we want?

Each group (see inat_fetch.py) has prompts for the scenes to keep and the
scenes to drop; p_keep is the summed probability of the "keep" prompts.
Writes data/raw/inaturalist/scores.csv. Safe to re-run: scored photos are skipped.
"""
import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "inaturalist"
MODEL = "openai/clip-vit-base-patch16"
BATCH = 32

NOT_PAVEMENT = [
    "a photo of a plant growing in garden soil",
    "a photo of a plant in a grassy meadow",
    "a close-up photo of a flower",
    "a photo of a plant in a forest",
    "a photo of a lawn",
    "a photo of a plant in a flower pot",
    "a photo of a plant growing on a wall",
    "a photo of a pressed plant specimen on paper",
    "a photo of a plant growing in water",
    "a photo of a leaf held in a hand",
    # wide shots: the robot camera looks down at the ground from close by
    "a wide photo of a street with cars and buildings",
    "a photo of a tree or a large shrub",
]
PROMPTS = {  # group: (keep, drop)
    "weed": ([
        "a photo of a weed growing between paving stones",
        "a photo of a plant growing in a crack in the pavement",
        "a photo of a weed on a sidewalk",
        "a photo of a plant growing at the edge of a curb on the street",
        "a photo of weeds in the joints of a brick road",
        "a photo of a plant growing through asphalt",
    ], NOT_PAVEMENT),
    "moss": ([
        "a photo of moss growing between paving stones",
        "a photo of moss on a concrete sidewalk",
        "a photo of moss on stone tiles",
        "a photo of moss between bricks on the ground",
    ], [
        "a photo of moss on a tree trunk",
        "a photo of moss in a forest",
        "a photo of moss on a roof",
        "a photo of moss on a vertical wall",
        "a photo of moss held in a hand",
        "a microscope photo of moss",
    ]),
    "desired": ([
        "a photo of a flower bed in a garden",
        "a photo of planted flowers in a border",
        "a photo of ornamental plants in a park",
        "a photo of a mown lawn",
        "a photo of a hedge next to a sidewalk",
        "a photo of plants in a street planter",
    ], [
        "a photo of a houseplant indoors",
        "a photo of a plant in a greenhouse",
        "a photo of cut flowers in a vase",
        "a photo of a tree trunk",
        "a photo of fruit or vegetables",
        "a photo of a plant label or sign",
    ]),
}


def normalized(features):
    # transformers 5 returns an output object; older versions return the tensor itself
    emb = features if torch.is_tensor(features) else features.pooler_output
    return emb / emb.norm(dim=-1, keepdim=True)


class Clip:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(MODEL).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL)

    @torch.no_grad()
    def text(self, prompts):
        return normalized(self.model.get_text_features(
            **self.processor(text=prompts, return_tensors="pt", padding=True)))

    @torch.no_grad()
    def images(self, images):
        return normalized(self.model.get_image_features(
            **self.processor(images=images, return_tensors="pt")))


def p_first(img_emb, text_emb, n_first):
    """Probability mass of the first n_first prompts."""
    return (100 * img_emb @ text_emb.T).softmax(dim=-1)[:, :n_first].sum(dim=-1)


def main():
    clip = Clip()
    group_text = {g: (clip.text(keep + drop), len(keep)) for g, (keep, drop) in PROMPTS.items()}

    with open(RAW / "meta.csv", encoding="utf-8") as f:
        group_of = {r["photo_id"]: r["group"] for r in csv.DictReader(f)}
    out = RAW / "scores.csv"
    scored = {}
    if out.exists():
        with open(out, encoding="utf-8") as f:
            scored = {r["photo_id"]: r for r in csv.DictReader(f)}
    todo = [p for p in sorted((RAW / "images").glob("*.jpg")) if p.stem in group_of and p.stem not in scored]
    print(f"{len(todo)} photos to score")

    for i in range(0, len(todo), BATCH):
        batch, ids = [], []
        for p in todo[i:i + BATCH]:
            try:
                batch.append(Image.open(p).convert("RGB"))
                ids.append(p.stem)
            except Exception:
                pass  # unreadable file: left out of scores.csv, so never used
        if not batch:
            continue
        img_emb = clip.images(batch)
        for j, pid in enumerate(ids):
            text_emb, n_keep = group_text[group_of[pid]]
            p_keep = p_first(img_emb[j:j + 1], text_emb, n_keep).item()
            scored[pid] = {"photo_id": pid, "p_keep": f"{p_keep:.4f}"}
        if (i // BATCH) % 20 == 0:
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)}")
            save(out, scored)
    save(out, scored)
    print(f"done: {len(scored)} photos scored")


def save(path, scored):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["photo_id", "p_keep"], extrasaction="ignore")
        w.writeheader()
        w.writerows(scored.values())


if __name__ == "__main__":
    main()
