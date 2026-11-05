"""Download candidate photos from iNaturalist for the weed / moss / desired_plant classes.

Sources (group in brackets):
  * projects about plants in pavement cracks, worldwide         [weed, moss]
  * Netherlands observations of typical pavement-joint species  [weed]
  * Netherlands observations of mosses that grow on stone       [moss]
  * Netherlands observations of cultivated (planted) plants     [desired]

Only Plantae observations whose photos have a Creative Commons licence are used.
Writes data/raw/inaturalist/images/<photo_id>.jpg and data/raw/inaturalist/meta.csv
(the CSV keeps author + licence for attribution). Safe to re-run: an existing
meta.csv is reused and already downloaded photos are skipped.
"""
import csv
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "inaturalist"
API = "https://api.inaturalist.org/v1/observations"
UA = {"User-Agent": "Weed-robot student project (dataset builder)"}
LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa,cc-by-nd,cc-by-nc-nd"

NETHERLANDS = 7506
MOSS_TAXA = "311249,64615"  # Bryophyta (mosses), Marchantiophyta (liverworts)

PROJECTS = {
    82227: "Flora of sidewalk cracks",
    198083: "The Hidden Life of Urban Pavements Cracks",
    165254: "Poison Free Pavements",
    84960: "Urban Weeds of the World",
}
NL_WEEDS = {
    55667: "Sagina procumbens",
    52808: "Poa annua",
    47602: "Taraxacum officinale",
    58961: "Plantago major",
    78678: "Polygonum aviculare",
    56121: "Capsella bursa-pastoris",
    53211: "Senecio vulgaris",
    55829: "Cardamine hirsuta",
    76907: "Erigeron canadensis",
    53168: "Oxalis corniculata",
    57036: "Euphorbia peplus",
    53294: "Sonchus oleraceus",
    58127: "Chenopodium album",
    53050: "Epilobium ciliatum",
    52992: "Matricaria discoidea",
    53298: "Stellaria media",
    55666: "Veronica persica",
    52801: "Lolium perenne",
}
NL_MOSSES = {
    123390: "Bryum argenteum",
    56406: "Marchantia polymorpha",
    123498: "Tortula muralis",
    123117: "Ceratodon purpureus",
    56117: "Grimmia pulvinata",
}

# (source, API filters, group, max photos). Moss queries come first: a photo found
# by several sources keeps the group of the first one.
SOURCES = (
    [(f"project-moss:{n}", {"project_id": p, "taxon_id": MOSS_TAXA}, "moss", None) for p, n in PROJECTS.items()]
    + [(f"nl-moss:{n}", {"taxon_id": t, "place_id": NETHERLANDS}, "moss", None) for t, n in NL_MOSSES.items()]
    + [(f"project:{n}", {"project_id": p}, "weed", None) for p, n in PROJECTS.items()]
    + [(f"nl:{n}", {"taxon_id": t, "place_id": NETHERLANDS, "captive": "false"}, "weed", None)
       for t, n in NL_WEEDS.items()]
    + [("nl-cultivated", {"place_id": NETHERLANDS, "captive": "true"}, "desired", 4000)]
)

FIELDS = ["photo_id", "obs_id", "user_id", "user_login", "license", "taxon", "place", "source", "group", "url"]


def get_json(params):
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception as e:
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def observations(**filters):
    """Yield matching observations, newest first, paging by id (the API caps page numbers)."""
    below = None
    while True:
        params = {
            **filters,
            "photos": "true",
            "iconic_taxa": "Plantae",
            "photo_license": LICENSES,
            "order_by": "id",
            "order": "desc",
            "per_page": 200,
        }
        if below:
            params["id_below"] = below
        results = get_json(params)["results"]
        if not results:
            return
        yield from results
        below = results[-1]["id"]
        time.sleep(1)  # iNaturalist asks for at most ~1 request per second


def to_row(obs, source, group):
    # First photo only: it is usually the overview shot, later ones are close-ups
    photo = obs["photos"][0]
    if not photo.get("license_code"):
        return None
    return {
        "photo_id": photo["id"],
        "obs_id": obs["id"],
        "user_id": obs["user"]["id"],
        "user_login": obs["user"]["login"],
        "license": photo["license_code"],
        "taxon": (obs.get("taxon") or {}).get("name", ""),
        "place": obs.get("place_guess") or "",
        "source": source,
        "group": group,
        "url": photo["url"].replace("/square.", "/medium."),
    }


def collect():
    rows, seen = [], set()
    for source, filters, group, limit in SOURCES:
        added = 0
        for obs in observations(**filters):
            row = to_row(obs, source, group)
            if row and row["photo_id"] not in seen:
                seen.add(row["photo_id"])
                rows.append(row)
                added += 1
                if limit and added >= limit:
                    break
        print(f"{source}: +{added}  (total {len(rows)})")

    with open(OUT / "meta.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def download(row):
    path = OUT / "images" / f"{row['photo_id']}.jpg"
    if path.exists():
        return True
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(row["url"], headers=UA), timeout=60) as r:
                data = r.read()
            path.write_bytes(data)
            return True
        except Exception:
            time.sleep(3 * (attempt + 1))
    return False


def download_all(rows):
    print(f"downloading {len(rows)} photos...")
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, ok in enumerate(pool.map(download, rows), 1):
            done += ok
            if i % 500 == 0:
                print(f"  {i}/{len(rows)}")
    print(f"done: {done}/{len(rows)} photos in {OUT / 'images'}")


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    if (OUT / "meta.csv").exists():
        with open(OUT / "meta.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"reusing meta.csv ({len(rows)} photos)")
    else:
        rows = collect()
    download_all(rows)


if __name__ == "__main__":
    main()
