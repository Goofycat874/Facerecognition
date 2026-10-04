"""Accurate face recognition: SCRFD detection, ArcFace embeddings, open-set matching."""

from .gallery import UNKNOWN, Gallery, Match
from .recognizer import EnrollResult, Face, FaceRecognizer
from .tracker import FaceTracker

__all__ = ["FaceRecognizer", "Face", "EnrollResult", "Gallery", "Match", "FaceTracker", "UNKNOWN"]
__version__ = "1.0.0"
