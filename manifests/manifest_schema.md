# Manifest schema

One CSV per dataset in this folder, merged later into `all.csv`.

| column | meaning |
|---|---|
| path | image path relative to `data/` |
| label | one of the 5 class names in `configs/classes.yaml` |
| source | dataset name (deepweeds, own, inaturalist, ...) |
| licence | licence string of the image or dataset |
| lighting | day_sun / overcast / dusk / night |
| weather | dry / wet / rain |
| surface | tiles / brick / gravel / soil |
| session | collection session or location ID (split by this, not randomly) |
| split | train / val / test |