"""Cheap face-quality checks used to reject bad enrollment samples and flag shaky matches."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import Detection


@dataclass
class QualityThresholds:
    min_face_px: float = 24.0  # shorter bbox side, in pixels
    min_det_score: float = 0.5
    min_sharpness: float = 15.0  # variance of Laplacian on the aligned 112x112 crop
    min_frontalness: float = 0.12  # 1.0 = looking straight at the camera, 0 = full profile


# Enrollment samples define an identity, so they are held to a much higher bar.
ENROLL_THRESHOLDS = QualityThresholds(
    min_face_px=60.0, min_det_score=0.7, min_sharpness=40.0, min_frontalness=0.35
)
RECOGNIZE_THRESHOLDS = QualityThresholds()


@dataclass
class Quality:
    face_px: float
    det_score: float
    sharpness: float
    frontalness: float
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def weight(self) -> float:
        """A 0..1 weight for averaging embeddings over several frames."""
        size = min(self.face_px / 112.0, 1.0)
        sharp = min(self.sharpness / 100.0, 1.0)
        return float(max(0.05, self.det_score * size * sharp * self.frontalness))


def frontalness(kps: np.ndarray) -> float:
    """Estimate how frontal a face is from its 5 landmarks (1 = frontal, 0 = profile).

    As the head turns, the nose slides toward one eye along the eye axis, so the ratio of
    the nose's distances to each eye is a cheap, stable yaw estimate.
    """
    le, re, nose = np.asarray(kps, dtype=np.float64)[:3]
    eye_vec = re - le
    eye_dist = np.linalg.norm(eye_vec)
    if eye_dist < 1e-6:
        return 0.0
    ux = eye_vec / eye_dist
    d_left = np.dot(nose - le, ux)
    d_right = np.dot(re - nose, ux)
    if d_left <= 0 or d_right <= 0:
        return 0.0
    return float(min(d_left, d_right) / max(d_left, d_right))


def sharpness(aligned: np.ndarray) -> float:
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if aligned.ndim == 3 else aligned
    # Only look at the inner face so background texture doesn't count as "sharp".
    inner = gray[16:104, 20:92]
    return float(cv2.Laplacian(inner, cv2.CV_64F).var())


def assess(det: Detection, aligned: np.ndarray, thresholds: QualityThresholds) -> Quality:
    q = Quality(
        face_px=min(det.width, det.height),
        det_score=det.score,
        sharpness=sharpness(aligned),
        frontalness=frontalness(det.kps),
    )
    if q.face_px < thresholds.min_face_px:
        q.issues.append(f"face too small ({q.face_px:.0f}px < {thresholds.min_face_px:.0f}px)")
    if q.det_score < thresholds.min_det_score:
        q.issues.append(f"low detection confidence ({q.det_score:.2f})")
    if q.sharpness < thresholds.min_sharpness:
        q.issues.append(f"blurry ({q.sharpness:.0f} < {thresholds.min_sharpness:.0f})")
    if q.frontalness < thresholds.min_frontalness:
        q.issues.append(f"head turned too far ({q.frontalness:.2f})")
    return q
