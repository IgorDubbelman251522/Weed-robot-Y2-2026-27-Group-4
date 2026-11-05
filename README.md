# Weed robot

BUas Applied Data Science & AI, Block A 2026–27, group 4. A challenge from
[Uptime Robotics](https://uptimerobotics.nl), introduced in the
[kickoff deck](The%20Weed%20Robot%20%E2%80%94%20BUas%20Challenge%20Kickoff.pdf).

Since the Dutch ban on chemical weed control on pavements in 2016, weeds are removed by hand and machine in a
few heavy passes a year. Uptime Robotics wants to know whether a robot could instead pass lightly and often,
ending weeds while they are still seedlings. That only works if a camera can tell a young weed in a paving
joint from moss, gravel, cracks, leaves and planted green, on any paving and in any light.

The brief asks for four things: a dataset (no ready one exists), a model, an honest measurement (precision and
recall per paving type, with a real error analysis) and a verdict on whether it could go down a street
unsupervised.

## Status (5 November 2026)

| Part | State |
|---|---|
| Dataset | 13,571 automatically labelled images from iNaturalist, split by photographer |
| Detector | YOLO26n trained on Colab, test mAP50 0.837 measured against automatic boxes |
| Classifier | Training and evaluation pipeline in `src/`, no results yet |
| Demo | `app/streamlit_app.py` runs the detector or a classifier on uploaded photos |
| Own photos | Not collected yet |

## Detector results so far

YOLO26n (2.4M parameters), trained for up to 60 epochs at 512 px on a Colab T4, where it takes about 4 ms per
image. The test set is 816 images (41 of them without plants) from photographers the model never saw in
training.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| weed | 0.811 | 0.731 | 0.832 | 0.697 |
| moss | 0.861 | 0.733 | 0.836 | 0.719 |
| desired_plant | 0.808 | 0.767 | 0.842 | 0.762 |
| all | 0.826 | 0.744 | 0.837 | 0.726 |

The reference boxes were drawn by Grounding DINO, not by a person, so these numbers measure agreement with the
automatic labels. They are not yet an accuracy we can stand behind.

## Dataset

There is no public dataset of weeds in paving, so `tools/` builds one from Creative Commons photos on
iNaturalist (details in [`tools/README.md`](tools/README.md)):

1. `inat_fetch.py` downloads about 25,000 photos: pavement-crack projects, Dutch observations of typical
   paving-joint weeds and stone mosses, and Dutch observations of planted (cultivated) plants.
2. `score_scenes.py` uses CLIP to drop the wrong kinds of scene, such as meadows, forests, potted plants and
   wide street shots.
3. `make_manifest.py` cuts each kept photo into 2×2 crops, which look more like the robot's close-up view, and
   writes `manifests/inaturalist.csv`. Bare crops with almost no green become `clean_surface`.

Splits are 80/10/10 by photographer, so one person's streets never appear in two splits.

| Class | Images (whole photos + crops) |
|---|---|
| weed | 6,988 |
| desired_plant | 4,230 |
| moss | 1,832 |
| clean_surface | 521 |
| seedling | 0 |

Every label is automatic; expect roughly one in five to be wrong. About 80% of the images have a
non-commercial licence (mostly CC BY-NC). Authors and links are in `manifests/inaturalist_attribution.csv`.

## What is missing before we can give a verdict

- **Seedlings.** The brief is about young weeds, but the `seedling` class has no images, so neither model can
  find them yet. iNaturalist mostly shows grown plants.
- **Our own photos.** None yet. iNaturalist photos are handheld and taken from every angle, while the robot
  looks straight down from close by.
- **Conditions.** `lighting`, `weather` and `surface` are empty for every image, so there are no results per
  paving type yet. `src/eval.py` reports precision and recall per condition as soon as photos are tagged.
- **Hand-checked test labels.** Needed for both models before any number is one we can trust.
- **Licences.** We need to check with Uptime Robotics whether non-commercial training images are a problem
  for how they would use the model.

## Running it

```bash
conda env create -f environment.yml
conda activate weed-robot

# Dataset: about 2 hours on a CPU and 3.4 GB; every step resumes where it stopped
python tools/inat_fetch.py
python tools/score_scenes.py
python tools/make_manifest.py
python tools/preview.py                     # grid of examples per class in docs/

# Classifier
python -m src.train                         # efficientnet_b0, results in models/<run>/
python -m src.eval models/<run>/best.pt     # metrics, confusion matrix, misclassified images

# Detector: see tools/README.md and notebooks/06_detection_colab.ipynb (Colab T4)

# Demo
streamlit run app/streamlit_app.py

# Tests
pytest
```

For the demo, put weights in `models/`: the detector's `best.pt` from
`My Drive/weed-robot/runs/yolo26n_v1/weights/` (save it as `models/yolo26n_weeds.pt`), and/or a classifier run
from `python -m src.train`. The app finds every `.pt` file there.

## Layout

```
app/          Streamlit demo
configs/      class definitions and label mapping (drafts)
manifests/    one CSV per dataset: path, label, licence, conditions, split
notebooks/    01–05 planned analysis steps, 06 detector training on Colab
src/          datasets, models, training, evaluation, inference
tests/        pytest
tools/        iNaturalist dataset builder and detection packer
data/, models/  local only, not in git
```
