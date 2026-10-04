"""High-level face recognition pipeline: detect -> align -> quality check -> embed -> match."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .align import align_face
from .detector import SCRFD, Detection
from .embedder import ArcFace, l2_normalize
from .gallery import UNKNOWN, Gallery, Match
from .models import DETECTOR_FILE, EMBEDDER_FILE, ensure_models
from .quality import ENROLL_THRESHOLDS, RECOGNIZE_THRESHOLDS, Quality, assess
from .tracker import FaceTracker

# Below this similarity to a person's existing samples, a new enrollment photo is almost
# certainly someone else (wrong file, or the wrong face picked in a group shot).
ENROLL_CONSISTENCY_MIN = 0.30
# Above this similarity the new sample adds nothing over one already stored.
ENROLL_DUPLICATE_MAX = 0.97


@dataclass
class Face:
    detection: Detection
    aligned: np.ndarray
    quality: Quality
    embedding: np.ndarray | None
    match: Match | None = None
    track_id: int | None = None

    @property
    def bbox(self) -> np.ndarray:
        return self.detection.bbox

    @property
    def name(self) -> str:
        return self.match.name if self.match else UNKNOWN


@dataclass
class EnrollResult:
    added: bool
    message: str


class FaceRecognizer:
    def __init__(
        self,
        db: str | Path = "face_db",
        model_dir: str | Path | None = None,
        det_size: int = 640,
        det_thresh: float = 0.5,
        threshold: float | None = None,
        margin: float | None = None,
        providers: list[str] | None = None,
    ):
        model_dir = ensure_models(model_dir)
        self.detector = SCRFD(model_dir / DETECTOR_FILE, input_size=det_size, providers=providers)
        self.embedder = ArcFace(model_dir / EMBEDDER_FILE, providers=providers)
        self.det_thresh = det_thresh
        self.gallery = Gallery(db, threshold=threshold, margin=margin)

    # ---- core --------------------------------------------------------------------
    def _detect(self, img: np.ndarray, max_faces: int, enroll: bool) -> list[Face]:
        dets = self.detector.detect(img, score_thresh=self.det_thresh, max_faces=max_faces)
        thresholds = ENROLL_THRESHOLDS if enroll else RECOGNIZE_THRESHOLDS
        faces = []
        for d in dets:
            aligned = align_face(img, d.kps)
            faces.append(Face(d, aligned, assess(d, aligned, thresholds), None))
        return faces

    def analyze(self, img: np.ndarray, max_faces: int = 0, enroll: bool = False) -> list[Face]:
        """Detect, align, quality-check and embed every face in a BGR image."""
        faces = self._detect(img, max_faces, enroll)
        embs = self.embedder.embed([f.aligned for f in faces])
        for face, emb in zip(faces, embs):
            face.embedding = emb
        return faces

    def _match(self, face: Face, embedding: np.ndarray | None = None, trusted: bool = False) -> Match:
        emb = face.embedding if embedding is None else embedding
        if emb is None:
            return Match(UNKNOWN, 0.0, None, None, -1.0)
        match = self.gallery.match(emb)
        if not (face.quality.ok or trusted) and match.known:
            # Too small / blurry / turned away to trust: report it, but don't name anyone.
            match = Match(UNKNOWN, match.score, match.best_name, match.runner_up, match.runner_up_score)
        return match

    def identify(self, img: np.ndarray) -> list[Face]:
        faces = self.analyze(img)
        for face in faces:
            face.match = self._match(face)
        return faces

    def identify_frame(self, img: np.ndarray, tracker: FaceTracker) -> list[Face]:
        """Video variant: follows faces across frames and matches their fused embedding.

        Averaging several frames of the same face beats any single frame, so this is both
        steadier and more accurate than per-image identify(). Settled tracks are only
        re-embedded every few frames, which keeps it fast.
        """
        faces = self._detect(img, 0, enroll=False)
        tracks = tracker.update([f.bbox for f in faces])
        todo = [
            i for i, (f, t) in enumerate(zip(faces, tracks))
            if f.quality.ok and tracker.needs_embedding(t)
        ]
        # Temporal fusion replaces flip TTA here, so skip it for speed.
        embs = self.embedder.embed([faces[i].aligned for i in todo], flip_tta=False)
        for i, emb in zip(todo, embs):
            faces[i].embedding = emb
            tracks[i].add_embedding(emb, faces[i].quality.weight)
        for face, track in zip(faces, tracks):
            face.track_id = track.track_id
            # The history only holds good-quality frames, so a momentary blur keeps its name.
            face.match = self._match(face, track.fused_embedding(), trusted=bool(track.history))
        return faces

    def verify(self, img_a: np.ndarray, img_b: np.ndarray) -> tuple[float | None, str]:
        """Cosine similarity between the largest faces of two images."""
        fa = self.analyze(img_a, max_faces=1)
        fb = self.analyze(img_b, max_faces=1)
        if not fa or not fb:
            return None, "no face found in " + ("first" if not fa else "second") + " image"
        return float(fa[0].embedding @ fb[0].embedding), ""

    # ---- enrollment --------------------------------------------------------------
    def enroll(self, name: str, img: np.ndarray, source: str = "", force: bool = False) -> EnrollResult:
        """Add one photo of `name` to the gallery (call gallery.save() afterwards)."""
        faces = self.analyze(img, enroll=True)
        if not faces:
            return EnrollResult(False, "no face found")
        faces.sort(key=lambda f: f.detection.width * f.detection.height, reverse=True)
        face = faces[0]
        if len(faces) > 1 and not force:
            area = lambda f: f.detection.width * f.detection.height  # noqa: E731
            if area(faces[1]) > 0.4 * area(face):
                return EnrollResult(
                    False, f"{len(faces)} similar-sized faces, can't tell which is {name} "
                    "(crop the photo or use --force to take the largest)"
                )
        if not face.quality.ok and not force:
            return EnrollResult(False, "rejected: " + "; ".join(face.quality.issues))

        if not force:
            own = self.gallery.embeddings_of(name)
            if len(own):
                sims = own @ face.embedding
                if sims.max() > ENROLL_DUPLICATE_MAX:
                    return EnrollResult(False, "skipped: near-duplicate of an existing sample")
                centroid_sim = float(l2_normalize(own.sum(axis=0)) @ face.embedding)
                if centroid_sim < ENROLL_CONSISTENCY_MIN:
                    return EnrollResult(
                        False,
                        f"rejected: doesn't look like the other {name} photos "
                        f"(similarity {centroid_sim:.2f}); use --force if this is really them",
                    )
            match = self.gallery.match(face.embedding)
            if match.best_name not in (None, name) and match.score >= self.gallery.threshold:
                return EnrollResult(
                    False,
                    f"rejected: this face matches '{match.best_name}' ({match.score:.2f}); "
                    "use --force if they really are different people",
                )

        self.gallery.add(name, face.embedding, source)
        q = face.quality
        return EnrollResult(
            True, f"ok (face {q.face_px:.0f}px, sharpness {q.sharpness:.0f}, frontal {q.frontalness:.2f})"
        )
