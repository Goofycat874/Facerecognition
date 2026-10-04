"""On-disk face database and open-set identity matching."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedder import l2_normalize

UNKNOWN = "Unknown"

# Cosine-similarity defaults for ArcFace w600k_r50 with flip TTA. Same-person pairs typically
# land at 0.5-0.95, different people at -0.1-0.25. Run `facerec calibrate` to tune for your data.
DEFAULT_THRESHOLD = 0.40
DEFAULT_MARGIN = 0.06


def identity_score(samples: np.ndarray, query: np.ndarray) -> float:
    """Blend of the closest enrolled sample and the identity's mean template."""
    best = float((samples @ query).max())
    mean = float(l2_normalize(samples.sum(axis=0)) @ query)
    return 0.6 * best + 0.4 * mean


@dataclass
class Match:
    name: str  # UNKNOWN when rejected
    score: float  # similarity to the best identity
    best_name: str | None  # best identity even if rejected (for debugging)
    runner_up: str | None
    runner_up_score: float

    @property
    def known(self) -> bool:
        return self.name != UNKNOWN


class Gallery:
    """Stores several embeddings per person and matches queries against them.

    Each identity is scored as a blend of its single closest sample (handles pose
    variety) and its normalized mean template (robust to one bad sample). A query is
    accepted only if that score clears the threshold AND beats the runner-up identity by
    a margin, which cuts down confident mix-ups between look-alikes.
    """

    def __init__(self, path: str | Path, threshold: float | None = None, margin: float | None = None):
        self.dir = Path(path)
        self._names: list[str] = []
        self._embs = np.zeros((0, 512), dtype=np.float32)
        self._sources: list[str] = []
        self.config = {"threshold": DEFAULT_THRESHOLD, "margin": DEFAULT_MARGIN}
        self._load()
        # Per-run overrides; unlike `config`, these are never written to disk.
        self._threshold_override = threshold
        self._margin_override = margin
        self._index: dict[str, np.ndarray] | None = None

    # ---- persistence -------------------------------------------------------------
    @property
    def _db_file(self) -> Path:
        return self.dir / "embeddings.npz"

    @property
    def _config_file(self) -> Path:
        return self.dir / "config.json"

    def _load(self) -> None:
        if self._db_file.is_file():
            with np.load(self._db_file, allow_pickle=False) as data:
                self._embs = data["embeddings"].astype(np.float32)
                self._names = [str(n) for n in data["names"]]
                self._sources = [str(s) for s in data["sources"]]
        if self._config_file.is_file():
            self.config.update(json.loads(self._config_file.read_text()))

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".npz")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(
                    f,
                    embeddings=self._embs,
                    names=np.array(self._names, dtype=str),
                    sources=np.array(self._sources, dtype=str),
                )
            os.replace(tmp, self._db_file)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self._config_file.write_text(json.dumps(self.config, indent=2) + "\n")

    # ---- editing -----------------------------------------------------------------
    @property
    def threshold(self) -> float:
        if self._threshold_override is not None:
            return float(self._threshold_override)
        return float(self.config["threshold"])

    @property
    def margin(self) -> float:
        if self._margin_override is not None:
            return float(self._margin_override)
        return float(self.config["margin"])

    def identities(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for n in self._names:
            counts[n] = counts.get(n, 0) + 1
        return dict(sorted(counts.items()))

    def embeddings_of(self, name: str) -> np.ndarray:
        return self._embs[[i for i, n in enumerate(self._names) if n == name]]

    def add(self, name: str, embedding: np.ndarray, source: str = "") -> None:
        if name == UNKNOWN:
            raise ValueError(f"'{UNKNOWN}' is reserved")
        emb = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))
        self._embs = np.vstack([self._embs, emb])
        self._names.append(name)
        self._sources.append(source)
        self._index = None

    def remove(self, name: str) -> int:
        keep = [i for i, n in enumerate(self._names) if n != name]
        removed = len(self._names) - len(keep)
        self._embs = self._embs[keep]
        self._names = [self._names[i] for i in keep]
        self._sources = [self._sources[i] for i in keep]
        self._index = None
        return removed

    def __len__(self) -> int:
        return len(self._names)

    # ---- matching ----------------------------------------------------------------
    def scores(self, query: np.ndarray) -> dict[str, float]:
        """Similarity of a query embedding to every identity."""
        if self._index is None:
            self._index = {name: self.embeddings_of(name) for name in self.identities()}
        q = l2_normalize(np.asarray(query, dtype=np.float32).reshape(-1))
        return {name: identity_score(samples, q) for name, samples in self._index.items()}

    def match(self, query: np.ndarray) -> Match:
        return decide(self.scores(query), self.threshold, self.margin)


def decide(scores: dict[str, float], threshold: float, margin: float) -> Match:
    """Accept the best identity only if it clears the threshold and beats the runner-up by margin."""
    if not scores:
        return Match(UNKNOWN, 0.0, None, None, -1.0)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best = ranked[0]
    runner_name, runner = ranked[1] if len(ranked) > 1 else (None, -1.0)
    accepted = best >= threshold and (best - runner) >= margin
    return Match(best_name if accepted else UNKNOWN, best, best_name, runner_name, runner)
