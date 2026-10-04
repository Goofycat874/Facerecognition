"""ArcFace embedding extractor (InsightFace w600k_r50)."""

from __future__ import annotations

import cv2
import numpy as np

from .models import create_session


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


class ArcFace:
    def __init__(self, model_path, providers=None):
        self.session = create_session(model_path, providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        batch_dim = self.session.get_inputs()[0].shape[0]
        self.dynamic_batch = not isinstance(batch_dim, int)

    def _run(self, crops: list[np.ndarray]) -> np.ndarray:
        blob = cv2.dnn.blobFromImages(
            crops, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )
        if self.dynamic_batch:
            return self.session.run([self.output_name], {self.input_name: blob})[0]
        return np.concatenate(
            [self.session.run([self.output_name], {self.input_name: b[None]})[0] for b in blob]
        )

    def embed(self, crops: list[np.ndarray], flip_tta: bool = True) -> np.ndarray:
        """Embed aligned 112x112 BGR crops into unit-length 512-d vectors.

        With flip_tta, each face is also embedded mirrored and the two features are summed,
        which makes the embedding more stable against pose and lighting asymmetry.
        """
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        if flip_tta:
            feats = self._run(list(crops) + [cv2.flip(c, 1) for c in crops])
            n = len(crops)
            feats = feats[:n] + feats[n:]
        else:
            feats = self._run(list(crops))
        return l2_normalize(feats.astype(np.float32))
