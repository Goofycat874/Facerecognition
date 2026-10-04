# Facerecognition

Accurate face recognition for photos, video files and webcams, in Python.

It uses the same detector and recognition model as InsightFace's `buffalo_l` pack, one of the most
accurate face models you can download for free:

| Stage | What it does | Why it matters for accuracy |
|---|---|---|
| **Detect**: SCRFD-10GF | Finds faces plus 5 landmarks (eyes, nose, mouth corners) | Finds small, tilted and partly hidden faces that Haar cascades and HOG miss |
| **Align** | Similarity-warps each face onto the canonical ArcFace 112×112 template | The recognition model was trained on aligned faces. Unaligned crops lose a lot of accuracy |
| **Quality gate** | Checks face size, blur, detector confidence and head turn | Bad samples get rejected instead of producing confident wrong answers |
| **Embed**: ArcFace ResNet-50 (WebFace600K) | Turns each face into a 512-d identity vector | Margin-based training separates identities far better than older 128-d models |
| **Flip TTA** | Averages the embedding of the face and its mirror image | Steadier under side lighting and slight head turns |
| **Open-set matching** | Scores each person against several samples; requires a threshold *and* a lead over the runner-up | Strangers come back as `Unknown` instead of "closest person", and look-alikes stop getting mixed up |
| **Video fusion** | Tracks faces and averages their embeddings over recent frames | Several frames beat one. A blurry frame keeps its name |
| **Calibration** | Measures accuracy on your own labelled photos and picks the threshold | Tunes the model for your cameras and your people |

Everything runs through ONNX Runtime on CPU, or on GPU when one is available. No dlib, no
TensorFlow, and nothing to compile.

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
# NVIDIA GPU (optional): pip install onnxruntime-gpu
```

The first run downloads the models (~290 MB, SHA-256 verified) to `~/.facerec/models/`.
To store them somewhere else, set `FACEREC_HOME` or pass `--model-dir`.

## Quick start

```bash
# 1. Enroll people: give each person a few photos (files or folders)
facerec enroll Alice photos/alice/
facerec enroll Bob bob1.jpg bob2.jpg bob3.jpg

# ...or enroll everyone at once from people/<name>/*.jpg
facerec enroll-dir people/

# 2. Recognize
facerec recognize group_photo.jpg -o annotated/      # prints names and saves boxed images
facerec video                                        # webcam (press q to quit)
facerec video clip.mp4 -o clip_labeled.mp4 --no-display

# 3. Check two photos directly
facerec verify a.jpg b.jpg
```

If you don't have the console script, `python -m facerec ...` works the same way.

Example output:

```
group_photo.jpg: 3 face(s)
  Alice                score 0.81  box=(412,88,530,240)
  Bob                  score 0.74  box=(102,95,215,251)
  Unknown              score 0.21, closest: Alice (below threshold)  box=(700,120,790,236)
```

## Getting the best accuracy

1. **Enroll 3–10 photos per person**, with different angles, lighting, glasses on and off, and
   expressions. One photo works, but more variety is the biggest accuracy boost you control.
2. **Let enrollment reject bad photos.** It refuses photos that are blurry, tiny, strongly turned
   or have several faces. It also refuses photos that don't look like the person's other photos, or
   that look like someone else who is already enrolled. Use `--force` only when you're sure.
3. **Calibrate on your own photos.** Put a few labelled photos per person in `people/<name>/`
   (the more people, the better the estimate), then run:
   ```bash
   facerec calibrate people/          # shows accuracy at the current vs. recommended threshold
   facerec calibrate people/ --save   # stores the recommended threshold in face_db/config.json
   ```
4. **Tiny faces in big photos:** raise the detector resolution, for example `--det-size 960` or
   `1280`.
5. **Video is more accurate than single photos** because embeddings are fused across frames.
   For identity checks, use video when you can.

### Tuning knobs

| Option | Default | Effect |
|---|---|---|
| `--threshold` | `0.40` | Minimum similarity to name someone. Raise it for fewer false matches, lower it to recognize more |
| `--margin` | `0.06` | How far the best person must beat the second best |
| `--det-size` | `640` | Detector input size. Bigger finds smaller faces, smaller is faster |
| `--db` | `./face_db` | Where enrolled faces are stored |
| `--cpu` | off | Ignore any GPU |

`calibrate --save` stores the threshold and margin, so you don't have to pass them each time.

## Python API

```python
import cv2
from facerec import FaceRecognizer, FaceTracker

rec = FaceRecognizer(db="face_db")

rec.enroll("Alice", cv2.imread("alice.jpg"))
rec.gallery.save()

for face in rec.identify(cv2.imread("group.jpg")):
    print(face.name, round(face.match.score, 2), face.bbox, face.quality.issues)

# Video: reuse one tracker so identities are fused over time
tracker = FaceTracker()
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    faces = rec.identify_frame(frame, tracker)
```

## Project layout

```
facerec/
  models.py      model download + ONNX Runtime sessions (GPU if available)
  detector.py    SCRFD face + landmark detector
  align.py       5-point similarity alignment (Umeyama)
  embedder.py    ArcFace embeddings with flip TTA
  quality.py     size / blur / pose / confidence checks
  gallery.py     face database + open-set matching
  tracker.py     IoU tracker with temporal embedding fusion
  recognizer.py  the full pipeline and enrollment safeguards
  evaluate.py    leave-one-out accuracy and threshold calibration
  images.py      image I/O (unicode-safe) and drawing
  cli.py         `facerec` command
tests/           unit tests (pytest)
```

Run the tests with `pip install pytest && pytest`. The model test only runs when the models are
already downloaded.

## Privacy and licensing

- Face embeddings are biometric data. `face_db/` stays on your machine and is git-ignored. Only
  enroll people who have agreed to it.
- The code in this repo is yours to use. The **pretrained InsightFace models are released for
  non-commercial research use only**. For commercial use, check the
  [InsightFace license terms](https://github.com/deepinsight/insightface#license).
