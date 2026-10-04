"""Image loading and drawing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_image(path: str | Path) -> np.ndarray | None:
    """cv2.imread that also works with non-ASCII paths (e.g. on Windows)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise IOError(f"could not encode {path}")
    buf.tofile(str(path))


def iter_images(paths: Iterable[str | Path]) -> Iterator[Path]:
    """Yield image files from a mix of file and directory paths (directories are searched recursively)."""
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        elif p.is_file():
            yield p


def draw_faces(img: np.ndarray, faces) -> np.ndarray:
    out = img.copy()
    scale = max(out.shape[:2]) / 1000.0
    thickness = max(1, int(round(2 * scale)))
    font_scale = max(0.45, 0.6 * scale)
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        known = face.match is not None and face.match.known
        color = (60, 200, 60) if known else (40, 40, 230)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = face.name
        if face.match is not None:
            label += f" {face.match.score:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = y1 - 4 if y1 - th - base - 4 > 0 else y2 + th + 4
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw + 4, ty + base), color, -1)
        cv2.putText(out, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return out
