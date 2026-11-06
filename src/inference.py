"""Run a trained model on one photo (used by app/streamlit_app.py).

Two kinds of weights in models/ are recognised:
  * detector: YOLO weights from notebooks/06_detection_colab.ipynb -> boxes plus aim points
  * classifier: best.pt / last.pt from `python -m src.train` -> a class for every tile of the photo
"""
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.datasets import build_transforms
from src.eval import pick_device
from src.models import create_model

# Same colours as the detection notebook, plus the classes it does not have
COLOURS = {
    "weed": (255, 40, 40),
    "moss": (255, 220, 0),
    "desired_plant": (40, 160, 255),
    "seedling": (255, 140, 0),
    "clean_surface": (160, 160, 160),
}
SPARE = [(170, 80, 255), (0, 200, 140), (255, 90, 190), (120, 120, 255)]


def colour(name, classes):
    return COLOURS.get(name, SPARE[classes.index(name) % len(SPARE)] if name in classes else SPARE[0])


def find_checkpoints(models_dir):
    """Every .pt file under models/, newest first, with training-resume files (last.pt) at the end."""
    return sorted(Path(models_dir).rglob("*.pt"), key=lambda p: (p.name == "last.pt", -p.stat().st_mtime))


def load_model(path):
    """(kind, model, info) for a checkpoint, where kind is "classifier" or "detector"."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_name" in ckpt:  # saved by src/train.py
        device = pick_device()
        model = create_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False)
        model.load_state_dict(ckpt["model"])
        return "classifier", model.to(device).eval(), {
            "classes": list(ckpt["classes"]), "img_size": ckpt["img_size"], "device": device}

    from ultralytics import YOLO

    model = YOLO(str(path))
    train_args = (getattr(model, "ckpt", None) or {}).get("train_args") or {}
    return "detector", model, {"classes": list(model.names.values()), "imgsz": train_args.get("imgsz", 640)}


def open_photo(file):
    """RGB image, turned upright using the camera's EXIF orientation (phone photos need this)."""
    return ImageOps.exif_transpose(Image.open(file)).convert("RGB")


def detect(model, info, image, conf=0.25):
    """Boxes in the image, strongest first, each with its aim point (box centre, in pixels)."""
    result = model.predict(image, imgsz=info["imgsz"], conf=conf, verbose=False)[0]
    found = [{"class": model.names[int(c)], "confidence": round(score, 3),
              "aim_x": round((x1 + x2) / 2), "aim_y": round((y1 + y2) / 2), "box": (x1, y1, x2, y2)}
             for (x1, y1, x2, y2), c, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                                   result.boxes.conf.tolist())]
    found.sort(key=lambda f: -f["confidence"])
    return draw_boxes(image, found, info["classes"]), found


@torch.no_grad()
def classify_tiles(model, info, image, per_side=2):
    """Class and confidence for every tile of a per_side x per_side grid (2 matches the training crops)."""
    boxes = tile_boxes(*image.size, per_side)
    transform = build_transforms(False, info["img_size"])
    batch = torch.stack([transform(image.crop(box)) for box in boxes]).to(info["device"])
    conf, idx = model(batch).softmax(dim=1).max(dim=1)
    tiles = [{"tile": i + 1, "class": info["classes"][k], "confidence": round(c, 3), "box": box}
             for i, (box, k, c) in enumerate(zip(boxes, idx.tolist(), conf.tolist()))]
    return draw_tiles(image, tiles, info["classes"]), tiles


def tile_boxes(width, height, per_side):
    """Grid boxes row by row, the same order as the crops in tools/make_manifest.py."""
    xs = [round(i * width / per_side) for i in range(per_side + 1)]
    ys = [round(j * height / per_side) for j in range(per_side + 1)]
    return [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for j in range(per_side) for i in range(per_side)]


def line_width(image):
    return max(2, round(max(image.size) / 300))


def font_for(image):
    size = max(14, round(max(image.size) / 45))
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1 has a single fixed-size default font
        return ImageFont.load_default()


def draw_label(draw, xy, text, col, font):
    left, top, right, bottom = draw.textbbox(xy, text, font=font)
    pad = 3
    draw.rectangle((left - pad, top - pad, right + pad, bottom + pad), fill=col)
    draw.text(xy, text, fill=(0, 0, 0) if sum(col) > 400 else (255, 255, 255), font=font)
    return bottom - top + 2 * pad


def draw_boxes(image, found, classes):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width, font = line_width(out), font_for(out)
    for f in found:
        col = colour(f["class"], classes)
        x1, y1, x2, y2 = f["box"]
        draw.rectangle((x1, y1, x2, y2), outline=col, width=width)
        cx, cy, r = f["aim_x"], f["aim_y"], 4 * width  # aim point: a cross at the box centre
        draw.line((cx - r, cy, cx + r, cy), fill=col, width=width)
        draw.line((cx, cy - r, cx, cy + r), fill=col, width=width)
        text = f"{f['class']} {f['confidence']:.2f}"
        height = draw.textbbox((0, 0), text, font=font)[3] + 8
        draw_label(draw, (x1 + 3, y1 - height if y1 >= height else y1 + 3), text, col, font)
    return out


def draw_tiles(image, tiles, classes):
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, font = line_width(base), font_for(base)
    for t in tiles:
        col = colour(t["class"], classes)
        draw.rectangle(t["box"], fill=col + (55,), outline=col + (255,), width=width)
        draw_label(draw, (t["box"][0] + 2 * width, t["box"][1] + 2 * width),
                   f"{t['class']} {t['confidence']:.2f}", col, font)
    return Image.alpha_composite(base, overlay).convert("RGB")
