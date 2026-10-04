"""Measure accuracy on a labelled folder of faces and pick the best threshold for it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gallery import decide, identity_score


@dataclass
class OperatingPoint:
    threshold: float
    margin: float
    known_accuracy: float  # known people named correctly
    unknown_rejection: float  # strangers correctly reported as Unknown
    misidentified: float  # known people given someone else's name

    @property
    def balanced_accuracy(self) -> float:
        return 0.5 * (self.known_accuracy + self.unknown_rejection)


@dataclass
class EvalReport:
    num_people: int
    num_images: int
    genuine: np.ndarray  # cosine similarity of same-person pairs
    impostor: np.ndarray  # cosine similarity of different-person pairs
    rank1: float  # closed-set identification accuracy
    best: OperatingPoint
    current: OperatingPoint


def _open_set_trials(embs: np.ndarray, labels: list[str]):
    """Leave-one-out trials.

    Known trial: query image i against everyone, with image i removed.
    Unknown trial: query image i against everyone except its own identity.
    Yields (kind, true_label, scores_by_identity).
    """
    names = sorted(set(labels))
    idx_of = {n: np.array([i for i, l in enumerate(labels) if l == n]) for n in names}
    for i, label in enumerate(labels):
        q = embs[i]
        known_scores, unknown_scores = {}, {}
        for n in names:
            idx = idx_of[n]
            if n == label:
                idx = idx[idx != i]
                if idx.size:
                    known_scores[n] = identity_score(embs[idx], q)
            else:
                s = identity_score(embs[idx], q)
                known_scores[n] = s
                unknown_scores[n] = s
        if label in known_scores:
            yield "known", label, known_scores
        if unknown_scores:
            yield "unknown", label, unknown_scores


def _operating_point(trials, threshold: float, margin: float) -> OperatingPoint:
    known = correct = wrong = unknown = rejected = 0
    for kind, label, scores in trials:
        m = decide(scores, threshold, margin)
        if kind == "known":
            known += 1
            correct += m.name == label
            wrong += m.known and m.name != label
        else:
            unknown += 1
            rejected += not m.known
    return OperatingPoint(
        threshold, margin,
        correct / known if known else 0.0,
        rejected / unknown if unknown else 0.0,
        wrong / known if known else 0.0,
    )


def evaluate(embs: np.ndarray, labels: list[str], threshold: float, margin: float) -> EvalReport:
    if len(set(labels)) < 2:
        raise ValueError("need at least 2 different people to evaluate")
    lab = np.array(labels)
    sims = embs @ embs.T
    iu = np.triu_indices(len(labels), k=1)
    same = (lab[:, None] == lab[None, :])[iu]
    genuine, impostor = sims[iu][same], sims[iu][~same]

    trials = list(_open_set_trials(embs, labels))
    known_trials = [t for t in trials if t[0] == "known"]
    rank1 = (
        float(np.mean([max(s, key=s.get) == label for _, label, s in known_trials]))
        if known_trials else 0.0
    )

    candidates = [
        _operating_point(trials, t, m)
        for t in np.round(np.arange(0.20, 0.70, 0.01), 2)
        for m in (0.0, 0.03, 0.06, 0.10)
    ]
    # Best balanced accuracy, then fewest misidentifications. Usually a whole range of
    # thresholds ties; take the middle of it so the setting has headroom on both sides.
    top = max((round(p.balanced_accuracy, 4), -p.misidentified) for p in candidates)
    tied = [p for p in candidates if (round(p.balanced_accuracy, 4), -p.misidentified) == top]
    mid_t = float(np.median([p.threshold for p in tied]))
    mid_m = float(np.median([p.margin for p in tied]))
    best = min(tied, key=lambda p: (abs(p.threshold - mid_t), abs(p.margin - mid_m)))
    current = _operating_point(trials, threshold, margin)
    return EvalReport(len(set(labels)), len(labels), genuine, impostor, rank1, best, current)
