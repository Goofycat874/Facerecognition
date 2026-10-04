"""5-point similarity alignment to the canonical ArcFace 112x112 template."""

from __future__ import annotations

import cv2
import numpy as np

ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation, uniform scale, translation) src -> dst.

    Returns a 2x3 affine matrix.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_c, dst_c = src - src_mean, dst - dst_mean

    cov = dst_c.T @ src_c / n
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[-1] = -1
    u, s, vt = np.linalg.svd(cov)
    r = u @ np.diag(d) @ vt
    var_src = (src_c ** 2).sum() / n
    scale = (s * d).sum() / var_src
    t = dst_mean - scale * r @ src_mean

    m = np.zeros((2, 3), dtype=np.float64)
    m[:, :2] = scale * r
    m[:, 2] = t
    return m


def align_face(img: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    """Warp a face so its landmarks land on the ArcFace template. Returns a size x size crop."""
    dst = ARCFACE_TEMPLATE * (size / 112.0)
    m = umeyama(kps, dst)
    return cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=0.0)
