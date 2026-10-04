import numpy as np
import pytest

from facerec.embedder import l2_normalize
from facerec.evaluate import evaluate
from facerec.gallery import UNKNOWN, Gallery
from facerec.tracker import FaceTracker

rng = np.random.default_rng(0)


def _person(n, spread=0.35):
    """n unit embeddings clustered around a random identity direction."""
    center = l2_normalize(rng.normal(size=512))
    return l2_normalize(center + spread * l2_normalize(rng.normal(size=(n, 512)))), center


def test_gallery_identifies_and_rejects(tmp_path):
    g = Gallery(tmp_path / "db")
    alice, a_center = _person(4)
    bob, _ = _person(4)
    for e in alice:
        g.add("alice", e)
    for e in bob:
        g.add("bob", e)

    probe = l2_normalize(a_center + 0.35 * l2_normalize(rng.normal(size=512)))
    m = g.match(probe)
    assert m.name == "alice" and m.known
    stranger = l2_normalize(rng.normal(size=512))
    assert g.match(stranger).name == UNKNOWN


def test_margin_rejects_ambiguous_matches(tmp_path):
    g = Gallery(tmp_path / "db", threshold=0.3, margin=0.1)
    base = l2_normalize(rng.normal(size=512))
    g.add("twin_a", base)
    g.add("twin_b", l2_normalize(base + 0.05 * l2_normalize(rng.normal(size=512))))
    m = g.match(base)
    assert m.score > 0.9
    assert m.name == UNKNOWN and m.best_name in ("twin_a", "twin_b")


def test_gallery_persists(tmp_path):
    g = Gallery(tmp_path / "db")
    g.config["threshold"] = 0.5
    e, _ = _person(2)
    g.add("Zoë", e[0], "a.jpg")
    g.add("Zoë", e[1], "b.jpg")
    g.save()
    g2 = Gallery(tmp_path / "db")
    assert g2.identities() == {"Zoë": 2}
    assert g2.threshold == 0.5
    np.testing.assert_allclose(g2.embeddings_of("Zoë"), e, atol=1e-6)
    assert g2.remove("Zoë") == 2 and len(g2) == 0


def test_threshold_override_is_not_saved(tmp_path):
    g = Gallery(tmp_path / "db", threshold=0.9)
    assert g.threshold == 0.9
    g.save()
    assert Gallery(tmp_path / "db").threshold != 0.9


def test_unknown_name_is_reserved(tmp_path):
    with pytest.raises(ValueError):
        Gallery(tmp_path).add(UNKNOWN, np.ones(512))


def test_tracker_keeps_ids_and_detects_identity_swap():
    t = FaceTracker(embed_every=3)
    box = np.array([10, 10, 50, 50], dtype=np.float32)
    first = t.update([box])[0]
    assert t.needs_embedding(first)
    alice, _ = _person(3, spread=0.1)
    for e in alice:
        first.add_embedding(e, 1.0)
    second = t.update([box + 2])[0]
    assert second.track_id == first.track_id
    assert not t.needs_embedding(second)
    t.update([box + 3])
    assert t.needs_embedding(t.update([box + 4])[0])

    other = l2_normalize(rng.normal(size=512))
    second.add_embedding(other, 1.0)  # someone else stepped into the box
    assert len(second.history) == 1

    far = t.update([box + 300])[0]
    assert far.track_id != first.track_id


def test_tracker_drops_stale_tracks():
    t = FaceTracker(max_missed=2)
    t.update([np.array([0, 0, 10, 10], dtype=np.float32)])
    for _ in range(3):
        t.update([])
    assert t.tracks == []


def test_evaluate_finds_working_threshold():
    embs, labels = [], []
    for i in range(6):
        e, _ = _person(4)
        embs.extend(e)
        labels += [f"p{i}"] * 4
    report = evaluate(np.stack(embs), labels, threshold=0.4, margin=0.06)
    assert report.rank1 == 1.0
    assert report.genuine.min() > report.impostor.max()
    assert report.best.known_accuracy == 1.0 and report.best.unknown_rejection == 1.0
    assert report.genuine.min() > report.best.threshold > report.impostor.max()
