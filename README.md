# Weed Robot: can a camera spot a young weed in a paving joint?

Group 4 · Applied Data Science & AI, BUas · Year 2, Block A challenge (2026-27)

This repository is our answer to the **Weed Robot** challenge set by Uptime Robotics (kickoff brief:
[`The Weed Robot — BUas Challenge Kickoff.pdf`](<The Weed Robot — BUas Challenge Kickoff.pdf>)).
Since the 2016 ban on chemical weed control, Dutch municipalities have to remove weeds from paving by
other means. Uptime Robotics wants a robot that passes lightly and often and ends weeds while they are
still seedlings. Whether that works depends on one thing: **can it see?**

We built a pipeline that collects plant, moss and pavement photos, trains an image classifier and a
box detector on them, and measures both per class, so that Uptime Robotics can judge whether a camera is
good enough for this job.

> "Weed" here means *onkruid*: unwanted plants growing in paving, not cannabis.

## Results at a glance

**Detector** (YOLO26n, test split, photographers never seen in training):

| class | precision | recall | mAP50 |
|---|---|---|---|
| weed | 0.811 | 0.731 | 0.832 |
| moss | 0.861 | 0.733 | 0.836 |
| desired_plant | 0.808 | 0.767 | 0.842 |
| **all** | | | **0.837** (mAP50-95: 0.726) |

Inference takes about 4 ms per image on a Colab T4 GPU. These scores measure agreement with
*automatically generated* boxes (Grounding DINO), not with hand-drawn ones; see
[Limitations](#limitations).

**Classifier**: trained and evaluated with `python -m src.train`. Checkpoints and metrics are not
committed (`models/` is gitignored), so run it to get the numbers for your machine. Each run writes
`metrics.json` and a confusion matrix to `models/<run>/`.

## How it works

```
iNaturalist (CC-licensed photos)
   │  tools/inat_fetch.py       download ~25k candidate photos
   │  tools/score_scenes.py     CLIP keeps pavement-like scenes, drops gardens, meadows, herbarium sheets
   │  tools/make_manifest.py    crops, labels, split by photographer  ->  manifests/inaturalist.csv
   ▼
   ├── Classifier   src/train.py        timm backbone (EfficientNet-B0 by default), fine-tuned
   └── Detector     tools/pack_detection.py  ->  notebooks/06_detection_colab.ipynb
                    Grounding DINO draws the boxes, YOLO26n learns them
```

The dataset has four classes with training data (13,571 images; split 80/10/10 by photographer so that
one person's streets never end up in two splits):

| class | train | val | test | what it is |
|---|---|---|---|---|
| weed | 5,594 | 696 | 698 | unwanted plant in a joint or crack |
| moss | 1,472 | 182 | 178 | moss and liverworts on stone |
| desired_plant | 3,379 | 427 | 424 | planted, cultivated plants the robot must leave alone |
| clean_surface | 436 | 44 | 41 | bare paving (crops of weed photos with almost no green) |

`configs/classes.yaml` also defines a fifth class, `seedling`, but there is no training data for it yet
(see [Limitations](#limitations)). `src/datasets.py` notices this and trains on the classes that have data.

## Repository layout

```
configs/      classes.yaml (class definitions, conditions), label_mapping.yaml (draft, see note below)
manifests/    inaturalist.csv (13,571 labelled images), inaturalist_attribution.csv (authors, licences),
              manifest_schema.md (column definitions)
src/          datasets.py, models.py, train.py, eval.py: the classifier pipeline
tools/        dataset builder scripts (see tools/README.md)
notebooks/    06_detection_colab.ipynb: the detector (01-05 are empty placeholders)
app/          streamlit_app.py: upload page (no model inference yet)
tests/        test_config.py
data/raw/     DeepWeeds and the sesame crop/weed set (Git LFS), plus iNaturalist once you download it
data/interim/ crops made by tools/make_manifest.py (not in git)
data/processed/ detection.zip for Colab (not in git)
models/       trained weights and metrics (not in git)
```

## Getting started

### Prerequisites

Git with [Git LFS](https://git-lfs.com), conda, and Python 3.11. A GPU is optional but makes training much
faster (the code picks CUDA, Apple MPS or CPU automatically). Run all commands from the repository root.

### 1. Clone

The repository stores the DeepWeeds and sesame datasets with Git LFS (about 1.1 GB: each is kept both as
a zip and unzipped). **The current models do not use them**, so you can skip that download:

```powershell
git lfs install
$env:GIT_LFS_SKIP_SMUDGE = "1"          # bash: export GIT_LFS_SKIP_SMUDGE=1
git clone https://github.com/IgorDubbelman251522/Weed-robot-Y2-2026-27-Group-4.git
cd Weed-robot-Y2-2026-27-Group-4
```

To fetch the LFS files later: `git lfs pull`.

### 2. Environment

```
conda env create -f environment.yml
conda activate weed-robot
```

For an NVIDIA GPU, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org) instead of the
default one.

### 3. Build the dataset

The iNaturalist photos are not in the repository (3.4 GB), only the manifests are. Rebuild the photos and
crops with the scripts in `tools/`. Each step can be re-run and continues where it stopped:

```
python tools/inat_fetch.py      # ~25k photos -> data/raw/inaturalist        (~1 h)
python tools/score_scenes.py    # CLIP scene filter                          (~30 min on CPU)
python tools/make_manifest.py   # crops + manifests/inaturalist.csv          (~40 min on CPU)
python tools/preview.py         # grid of random examples per class -> docs/ (to eyeball label quality)
```

Details on how the labels are made are in [`tools/README.md`](tools/README.md).

### 4. Train and evaluate the classifier

```
python -m src.train                                  # EfficientNet-B0, 15 epochs
python -m src.train --model convnext_tiny --epochs 20
python -m src.train --limit 200 --epochs 1           # quick smoke test
python -m src.train --resume models/<run>/last.pt    # continue an interrupted run

python -m src.eval models/<run>/best.pt --split test
python tools/preview.py models/<run>/errors_test.csv # look at the misclassified images
```

Training keeps the epoch with the best validation macro-F1 and reports on the test split. Runs go to
`models/<run>/` (`best.pt`, `last.pt`, `metrics.json`, `confusion_test.png`) and are also logged to MLflow
(`mlflow ui`, then open http://localhost:5000). The metrics include accuracy, balanced accuracy, macro-F1,
per-class precision and recall, and a breakdown by lighting, weather and surface wherever those columns are filled in.

### 5. Train the detector (Colab)

```
python tools/pack_detection.py   # -> data/processed/detection.zip (1.25 GB)
```

Upload the zip to `My Drive/weed-robot/`, open `notebooks/06_detection_colab.ipynb` in Google Colab,
choose a **T4 GPU** runtime and *Run all*. Auto-labelling takes about 30 minutes and training 1.5–2 hours;
everything is saved to Drive and resumes if the session drops. Download `runs/yolo26n_v1/weights/best.pt`
into `models/`.

### 6. Tests and app

```
python -m pytest
streamlit run app/streamlit_app.py
```

The app currently shows an upload page only; model inference is not connected.

## Using the trained models

Detector: boxes plus an aim point (box centre) per plant.

```python
from ultralytics import YOLO

model = YOLO("models/yolo26n_weeds.pt")
for box in model("frame.jpg", conf=0.25)[0].boxes:
    x, y = box.xywh[0][:2].tolist()
    print(model.names[int(box.cls)], round(float(box.conf), 2), x, y)
```

Classifier: class probabilities for one image or tile.

```python
import torch
from PIL import Image
from src.datasets import build_transforms
from src.models import create_model

ckpt = torch.load("models/<run>/best.pt", map_location="cpu", weights_only=False)
model = create_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False)
model.load_state_dict(ckpt["model"])
model.eval()

x = build_transforms(False, ckpt["img_size"])(Image.open("tile.jpg").convert("RGB")).unsqueeze(0)
probs = model(x).softmax(dim=1)[0]
print({c: round(float(p), 3) for c, p in zip(ckpt["classes"], probs)})
```

## Data and licences

| dataset | used for | licence |
|---|---|---|
| iNaturalist photos | all current models | mostly CC BY-NC (**non-commercial use only**); authors and links in `manifests/inaturalist_attribution.csv` |
| [DeepWeeds](https://github.com/AlexOlsen/DeepWeeds) | kept in `data/raw/deepweeds`, not used yet | CC BY 4.0. Olsen et al., *DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning*, Scientific Reports, 2019 |
| Crop and Weed Detection Data with Bounding Boxes (sesame, Kaggle) | kept in `data/raw/crop_weed_sesame`, not used yet | CC0. Panara Utsav, Pandya Raviraj, Mohit Rayja |

Because of the non-commercial iNaturalist photos, models trained here are for research and education.
A commercial product would need new, properly licensed data.

## Limitations

Read these before trusting any number above.

- **No photos of our own.** The brief asks teams to collect and label their own data. We used iNaturalist
  photos instead, which mostly show plants photographed by hobbyists, not a robot camera looking down at a
  street. Expect a domain gap on real robot footage.
- **Lighting, weather and surface are unknown.** Those columns in the manifest are empty because
  iNaturalist photos carry no such information. Night, rain, dusk and per-paving-type results therefore
  cannot be measured yet, even though `src/eval.py` can break results down by them once the columns are filled.
- **No seedling class.** `seedling` is defined but has no data, so the models cannot yet tell a seedling
  from an established weed, which is the question the challenge is about.
- **Noisy labels.** All labels are automatic (CLIP scene filter, taxon-based classes); roughly 1 in 5 is
  expected to be wrong. `clean_surface` is the smallest class (521 images) and is cut from weed photos
  rather than photographed separately.
- **The detector is scored against automatic boxes.** The reference boxes come from Grounding DINO, so
  mAP measures agreement with that model, not with a human. Correct the test boxes by hand for an honest number.
- **DeepWeeds and sesame are not used** by the current models. DeepWeeds comes from Australian rangeland
  and sesame from a single crop, so they are a poor match for Dutch paving.
- **`configs/label_mapping.yaml` is an unused draft** from the planning stage and has not been checked
  against the real label files.
- **`app/` and `notebooks/01-05` are placeholders.** The working code is in `src/`, `tools/` and notebook 06.

## Team

Igor Dubbelman (251522), Pranav Kumar (252142), Oleksandr Karlov (250149)

Challenge set by Peter Brouwers and Jeroen Griffioen of Uptime Robotics, with the BUas Academy of AI,
Games and Media.
