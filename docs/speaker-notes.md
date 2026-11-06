# The weed robot: speaker notes

Final presentation, 6 November 2026. One section per slide, in the order of the deck.

## 1. The weed robot

Kickoff was on Monday and this is our final presentation. We cover their question, the data and model we built, what it gets right, what we can't claim, and our answer.

## 2. What Uptime Robotics asked

This is the question from their kickoff. They asked for a dataset, a model, precision and recall per paving type with an error analysis, and a yes or no on sending it down a street unsupervised. They also said a modest score that is well explained beats a high score on pictures nobody trusts.

## 3. No dataset existed, so we built one

There is no public dataset of weeds in paving, so we used iNaturalist, where people share plant photos. We downloaded about 25,000 photos from pavement-crack projects and Dutch sightings of common paving weeds, mosses and planted plants. An image model (CLIP) removed meadows, forests, potted plants and street views, which left 7,762. Cutting each photo into four close-up crops gave 13,571 images. We split by photographer, so the same street never appears in both training and test.

## 4. Lots of weeds, no seedlings

Plenty of grown weeds, planted plants and moss, but no seedlings, and seedlings are what the robot is meant to remove. People photograph plants they can identify, which means grown plants. Counts include the close-up crops.

## 5. It finds about 3 in 4 plants

There were no hand-drawn boxes, so another model (Grounding DINO) drew them, and our detector (YOLO26n: 2.4 million parameters, about 4 ms per image on a Colab T4 GPU) learned from those. On 816 test photos from photographers it never saw, it finds about three in four plants, and about four in five of its detections are right. Say clearly that this compares it with another model's boxes, not with a person's.

## 6. Close-ups, not the robot's view

Output on test photos the model never saw. It finds the plants, but look at the photos: flowers, shrubs and weeds, shot up close by people who wanted to identify them. The robot will look straight down at paving, often at seedlings only a few pixels wide.

## 7. Live demo

Before presenting: open a terminal in the repo folder, run conda activate weed-robot, then streamlit run app/streamlit_app.py. The detector file must be in models/yolo26n_weeds.pt. Use a photo of a paving joint taken from above. Move the confidence slider to show the trade-off between missed weeds and false alarms.

## 8. What's missing

These gaps are behind our answer. Our code can already report precision and recall per surface, light and weather, but none of the photos have those tags. About 80% of the images are licensed for non-commercial use only, which matters because Uptime Robotics is a company.

## 9. Verdict: No.

This is the yes or no they asked for. Our answer is no, for three reasons: no seedlings, the wrong viewpoint, and scores no person has checked. What does work: it tells weeds, moss and planted plants apart well against automatic labels, and it is small and fast enough to run on a robot.

## 10. What would change our answer

With these, someone could test a camera properly before the December decision. The last question is theirs to answer: a missed weed grows back, while a false alarm can damage paving or plants. Their answer sets the detector's confidence threshold.
