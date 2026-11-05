# The weed robot: speaker notes

Final presentation, 6 November 2026. One section per slide, in the order of the deck.

## 1. The weed robot

Our final presentation for the one-week challenge from Uptime Robotics. In ten minutes: what they asked for, the dataset and detector we built, what it can see, what we can't claim, and our verdict on sending it down a street unsupervised.

## 2. What Uptime Robotics asked for

The brief has four parts. The key point for us is the last line: Uptime Robotics would rather have a modest score with a real error analysis than an impressive number on pictures they can't trust. Everything that follows is framed that way.

## 3. We built a dataset from iNaturalist

No public dataset of weeds in paving exists, so we mined iNaturalist: pavement-crack projects, Dutch observations of typical paving-joint weeds and stone mosses, and planted garden plants. CLIP, a zero-shot image model, throws out meadows, forests, potted plants and wide street shots. Each kept photo is cut into four crops, which look more like the robot's close view. Splitting by photographer stops the same street leaking from training into test. The labels are automatic, so they are noisy.

## 4. What the dataset holds

Plenty of grown weeds, desired plants and moss, but zero seedlings, and seedlings are the whole point of the brief: the robot should end weeds while they are a few days old. iNaturalist users photograph plants they can identify, which means grown plants. The licence point matters because Uptime Robotics is a company: most of these photos are licensed for non-commercial use only.

## 5. Our first detector: YOLO26n

There are no hand-drawn boxes, so Grounding DINO, an open-vocabulary detector, drew them automatically and YOLO26n learned from those. On photographers it never saw, it matches the automatic boxes with mAP50 0.84 for all three classes. Be clear that this is agreement with another model, not accuracy against a person. The size and speed matter for a robot: 2.4 million parameters and about 4 ms per image on a T4.

## 6. What the detector sees

Real output on the held-out test set. It finds the plants well, but look at what the photos are: flowers, shrubs and weeds photographed by people who wanted to identify them. The robot will look straight down at paving from close by, often at seedlings a few pixels wide. Credits for every photo are in manifests/inaturalist_attribution.csv.

## 7. Live demo

Before presenting: in the repo folder run "conda activate weed-robot" and then "streamlit run app/streamlit_app.py". The detector weights must be in models/yolo26n_weeds.pt. Use a photo of a real paving joint taken from above, ideally one with moss and a weed in it. Show the confidence slider: it is the trade-off between a missed weed and a false alarm, which leads into our question for Uptime Robotics about which mistake costs more.

## 8. What we can't claim

This is the honest list. Our code already reports precision and recall per surface, lighting and weather, but it needs tagged photos to have anything to report. Without our own photos and a hand-checked test set, the 0.84 on the earlier slides is not a number we can stand behind.

## 9. Our verdict

This is the yes-or-no the brief asked for, and our answer is no. The three reasons are the gaps that matter most: no seedlings, the wrong viewpoint, and scores nobody has checked. The positive finding is real but limited: the model separates the three plant classes well against automatic labels, and it is small and fast enough to run on a robot.

## 10. What would change our answer

These four things would let anyone test the camera approach properly before the December decision. The last question sets the detector's confidence threshold: a missed weed grows back, a false alarm damages paving or planted green, and only Uptime Robotics can say which is worse for their customers.
