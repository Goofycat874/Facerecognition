"""Runs against the real ONNX models when they are already downloaded; skipped otherwise."""

import numpy as np
import pytest

from facerec.models import DETECTOR_FILE, EMBEDDER_FILE, default_model_dir

MODEL_DIR = default_model_dir()
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / DETECTOR_FILE).is_file(), reason="models not downloaded"
)


def test_detector_and_embedder_run():
    from facerec.detector import SCRFD
    from facerec.embedder import ArcFace

    det = SCRFD(MODEL_DIR / DETECTOR_FILE)
    assert det.detect(np.zeros((480, 640, 3), np.uint8)) == []

    emb = ArcFace(MODEL_DIR / EMBEDDER_FILE)
    crops = [np.random.default_rng(1).integers(0, 255, (112, 112, 3), dtype=np.uint8)] * 2
    out = emb.embed(crops)
    assert out.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
