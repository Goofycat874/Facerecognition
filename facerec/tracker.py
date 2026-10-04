"""Lightweight IoU tracker that fuses embeddings over time for steadier video recognition."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .embedder import l2_normalize

# A new embedding this far from a track's history means a different person took over the box.
SAME_PERSON_MIN_SIM = 0.2


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    history: deque = field(default_factory=lambda: deque(maxlen=15))
    missed: int = 0
    frames_since_embed: int = 0

    def add_embedding(self, embedding: np.ndarray, weight: float) -> None:
        if self.history and float(self.fused_embedding() @ embedding) < SAME_PERSON_MIN_SIM:
            self.history.clear()
        self.history.append((embedding, weight))
        self.frames_since_embed = 0

    def fused_embedding(self) -> np.ndarray | None:
        """Quality-weighted mean of the recent embeddings for this face."""
        if not self.history:
            return None
        embs = np.stack([e for e, _ in self.history])
        weights = np.array([w for _, w in self.history], dtype=np.float32)
        return l2_normalize((embs * weights[:, None]).sum(axis=0))


class FaceTracker:
    def __init__(
        self, iou_threshold: float = 0.3, max_missed: int = 10, window: int = 15, embed_every: int = 3
    ):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.window = window
        self.embed_every = embed_every
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, bboxes: list[np.ndarray]) -> list[Track]:
        """Associate this frame's boxes with tracks. Returns one track per box, in order."""
        pairs = sorted(
            (
                (iou(t.bbox, b), ti, di)
                for ti, t in enumerate(self.tracks)
                for di, b in enumerate(bboxes)
            ),
            key=lambda p: p[0],
            reverse=True,
        )
        assigned: dict[int, Track] = {}
        used: set[int] = set()
        for overlap, ti, di in pairs:
            if overlap < self.iou_threshold:
                break
            if ti in used or di in assigned:
                continue
            used.add(ti)
            assigned[di] = self.tracks[ti]

        for ti, track in enumerate(self.tracks):
            if ti not in used:
                track.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        result = []
        for di, bbox in enumerate(bboxes):
            track = assigned.get(di)
            if track is None:
                track = Track(self._next_id, bbox, deque(maxlen=self.window))
                self._next_id += 1
                self.tracks.append(track)
            track.bbox = bbox
            track.missed = 0
            track.frames_since_embed += 1
            result.append(track)
        return result

    def needs_embedding(self, track: Track) -> bool:
        """Embedding is the slow step, so refresh a settled track only every few frames."""
        return len(track.history) < 3 or track.frames_since_embed >= self.embed_every
