"""Weed robot demo: run our detector or classifier on your own photos.

    streamlit run app/streamlit_app.py

Weights go in models/ (any .pt file, subfolders are fine):
  * detector: best.pt from My Drive/weed-robot/runs/yolo26n_v1/weights/ (notebooks/06_detection_colab.ipynb)
  * classifier: models/<run>/best.pt from `python -m src.train`
"""
import io
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # lets `streamlit run app/streamlit_app.py` import src/

from src.inference import classify_tiles, detect, find_checkpoints, load_model, open_photo  # noqa: E402

st.set_page_config(page_title="Weed robot", page_icon="🌱", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def cached_model(path, modified):  # `modified` makes a replaced weights file load again
    return load_model(path)


st.title("Weed robot: what does the camera see?")

checkpoints = find_checkpoints(ROOT / "models")
if not checkpoints:
    st.info(
        "There are no model weights in `models/` yet.\n\n"
        "**Detector:** download `best.pt` from `My Drive/weed-robot/runs/yolo26n_v1/weights/` "
        "(made by `notebooks/06_detection_colab.ipynb`) and save it as `models/yolo26n_weeds.pt`.\n\n"
        "**Classifier:** run `python -m src.train`, which saves `models/<run>/best.pt`.\n\n"
        "Then reload this page."
    )
    st.stop()

with st.sidebar:
    path = st.selectbox("Model", checkpoints, format_func=lambda p: p.relative_to(ROOT / "models").as_posix())
    try:
        kind, model, info = cached_model(str(path), path.stat().st_mtime)
    except Exception as e:  # e.g. a half-downloaded file
        st.error(f"Could not load {path.name}: {e}")
        st.stop()
    st.caption(f"{kind.capitalize()} for {', '.join(info['classes'])}")
    if kind == "detector":
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                         help="Lower finds more plants, and more false alarms.")
    else:
        per_side = st.slider("Tiles per side", 1, 6, 2,
                             help="The photo is cut into a grid and every tile is classified. "
                                  "2 matches the crops the classifier was trained on.")
    st.divider()
    st.caption("Our test scores so far are measured against automatic labels, not hand-checked ones, "
               "so treat this as a first impression rather than a verdict.")

photos = st.file_uploader("Add photos of paving, gravel or planted borders",
                          type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
if not photos:
    st.write("Photograph the ground looking straight down, the way the robot's camera would see it, "
             "then add the photo here.")
    st.stop()

for i, photo in enumerate(photos):
    image = open_photo(photo)
    start = time.perf_counter()
    if kind == "detector":
        shown, found = detect(model, info, image, conf)
        columns = ["class", "confidence", "aim_x", "aim_y"]
    else:
        shown, found = classify_tiles(model, info, image, per_side)
        columns = ["tile", "class", "confidence"]
    ms = (time.perf_counter() - start) * 1000

    left, right = st.columns([3, 2])
    left.image(shown, caption=photo.name)
    with right:
        counts = pd.Series([f["class"] for f in found], dtype=object).value_counts()
        if kind == "detector" and not found:
            st.write("Nothing found above this confidence threshold.")
        else:
            label = "Found" if kind == "detector" else "Tiles"
            st.markdown(f"**{label}:** " + ", ".join(f"{n} {c}" for c, n in counts.items()))
            st.dataframe(pd.DataFrame([{k: f[k] for k in columns} for f in found]), hide_index=True)
        if kind == "detector" and found:
            st.caption("aim_x and aim_y are the box centres in pixels: where the robot would aim.")
        st.caption(f"Model time: {ms:.0f} ms")
        buffer = io.BytesIO()
        shown.save(buffer, format="JPEG", quality=90)
        st.download_button("Download annotated photo", buffer.getvalue(), key=f"download-{i}",
                           file_name=f"{Path(photo.name).stem}_annotated.jpg", mime="image/jpeg")
    st.divider()
