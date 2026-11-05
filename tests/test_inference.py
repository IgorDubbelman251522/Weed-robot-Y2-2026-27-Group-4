import os

from PIL import Image

from src.inference import draw_boxes, draw_tiles, find_checkpoints, tile_boxes


def test_best_checkpoints_come_before_last(tmp_path):
    for i, name in enumerate(["run/best.pt", "run/last.pt", "yolo26n_weeds.pt"]):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"")
        os.utime(path, (1000 + i, 1000 + i))
    assert [p.name for p in find_checkpoints(tmp_path)] == ["yolo26n_weeds.pt", "best.pt", "last.pt"]


def test_tiles_cover_the_image_row_by_row():
    assert tile_boxes(101, 50, 2) == [(0, 0, 50, 25), (50, 0, 101, 25), (0, 25, 50, 50), (50, 25, 101, 50)]


def test_drawing_keeps_the_photo_size():
    image = Image.new("RGB", (640, 480), "grey")
    found = [{"class": "weed", "confidence": 0.91, "aim_x": 100, "aim_y": 100, "box": (50, 50, 150, 150)},
             {"class": "pothole", "confidence": 0.4, "aim_x": 300, "aim_y": 10, "box": (250, 0, 350, 20)}]
    assert draw_boxes(image, found, ["weed", "pothole"]).size == (640, 480)
    tiles = [{"tile": 1, "class": "moss", "confidence": 0.7, "box": (0, 0, 320, 240)}]
    assert draw_tiles(image, tiles, ["moss"]).size == (640, 480)
