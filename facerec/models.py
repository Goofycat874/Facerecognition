"""Model download, caching and ONNX Runtime session creation."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# InsightFace "buffalo_l" pack: SCRFD-10GF detector + ArcFace ResNet-50 trained on WebFace600K.
MODEL_PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
MODEL_PACK_SHA256 = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
DETECTOR_FILE = "det_10g.onnx"
EMBEDDER_FILE = "w600k_r50.onnx"
REQUIRED_FILES = (DETECTOR_FILE, EMBEDDER_FILE)


def default_model_dir() -> Path:
    home = os.environ.get("FACEREC_HOME")
    base = Path(home) if home else Path.home() / ".facerec"
    return base / "models" / "buffalo_l"


def _progress(block_num: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(block_num * block_size, total)
    pct = done * 100 // total
    sys.stderr.write(f"\r  downloading models: {pct:3d}% ({done >> 20}/{total >> 20} MB)")
    if done >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_models(model_dir: str | Path | None = None) -> Path:
    """Return a directory holding the detector and embedder, downloading them if needed."""
    model_dir = Path(model_dir) if model_dir else default_model_dir()
    if all((model_dir / f).is_file() for f in REQUIRED_FILES):
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    sys.stderr.write(f"First run: fetching face models into {model_dir}\n")
    with tempfile.TemporaryDirectory(dir=model_dir) as tmp:
        zip_path = Path(tmp) / "buffalo_l.zip"
        urllib.request.urlretrieve(MODEL_PACK_URL, zip_path, _progress)
        digest = _sha256(zip_path)
        if digest != MODEL_PACK_SHA256:
            raise RuntimeError(
                f"Model download is corrupted (sha256 {digest}, expected {MODEL_PACK_SHA256})."
            )
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if name in REQUIRED_FILES:
                    with zf.open(member) as src, open(model_dir / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)

    missing = [f for f in REQUIRED_FILES if not (model_dir / f).is_file()]
    if missing:
        raise RuntimeError(f"Model pack did not contain {missing}")
    return model_dir


def create_session(model_path: str | Path, providers: list[str] | None = None):
    """Create an ONNX Runtime session, preferring a GPU provider when one is installed."""
    import onnxruntime as ort

    if providers is None:
        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available] or available
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
