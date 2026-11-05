# iNaturalist dataset builder

Builds `manifests/inaturalist.csv` (classes `weed`, `moss`, `desired_plant`,
`clean_surface`) from Creative Commons photos on iNaturalist. Needs `torch`,
`transformers`, `pillow`, `numpy`. Every step can be re-run and continues
where it stopped.

```
python tools/inat_fetch.py      # ~25k photos -> data/raw/inaturalist (3.4 GB, ~1 h)
python tools/score_scenes.py    # CLIP: is this the scene we want? (~30 min on CPU)
python tools/make_manifest.py   # crops + manifest (~40 min on CPU the first time)
python tools/preview.py         # grid of random examples per class -> docs/
```

How labels are made (all automatic, so expect some noise, roughly 1 in 5):

| class | source |
|---|---|
| weed | pavement-crack projects + NL observations of typical pavement-joint species |
| moss | mosses and liverworts from the same projects + NL mosses that grow on stone |
| desired_plant | NL observations marked as cultivated (planted) |
| clean_surface | crops of weed photos that CLIP calls bare and that have <1% green pixels |

Scene thresholds (`--min-weed 0.7 --min-moss 0.5 --min-desired 0.7`) were set by
looking at sample grids per score range. The split is by photographer, so one
person's streets never appear in two splits. `lighting`, `weather` and
`surface` are empty because they are unknown for these photos.

Licences: most photos are CC BY-NC (non-commercial only). Authors and links
are in `manifests/inaturalist_attribution.csv`.

## Detector (boxes)

```
python tools/pack_detection.py  # whole photos + clean crops -> data/processed/detection.zip (1.25 GB)
```

Upload the zip to `My Drive/weed-robot/` and run
`notebooks/06_detection_colab.ipynb` on a Colab T4: Grounding DINO draws the
boxes, YOLO26n learns them. Results stay on Drive; copy `best.pt` to `models/`.
