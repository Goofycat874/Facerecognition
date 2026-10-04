"""SCRFD face detector (InsightFace det_10g) with 5-point landmarks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import create_session


@dataclass
class Detection:
    bbox: np.ndarray  # (4,) x1, y1, x2, y2 in image pixels
    score: float
    kps: np.ndarray  # (5, 2) left eye, right eye, nose, left mouth, right mouth

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, distance.shape[1], 2):
        out.append(points[:, 0] + distance[:, i])
        out.append(points[:, 1] + distance[:, i + 1])
    return np.stack(out, axis=-1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, highest score first."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


class SCRFD:
    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2

    def __init__(self, model_path, input_size: int = 640, providers=None):
        self.session = create_session(model_path, providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        if len(self.output_names) != 9:
            raise ValueError("Expected an SCRFD model with keypoints (9 outputs)")
        self.input_size = input_size
        self._center_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        centers = self._center_cache.get(key)
        if centers is None:
            centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            centers = np.repeat(centers, self.NUM_ANCHORS, axis=0)
            self._center_cache[key] = centers
        return centers

    def detect(
        self,
        img: np.ndarray,
        score_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        max_faces: int = 0,
    ) -> list[Detection]:
        """Detect faces in a BGR image."""
        size = self.input_size
        h, w = img.shape[:2]
        scale = size / max(h, w)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(img, (new_w, new_h), interpolation=interp)

        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128.0, (size, size), (127.5, 127.5, 127.5), swapRB=True
        )
        outs = self.session.run(self.output_names, {self.input_name: blob})

        all_scores, all_boxes, all_kps = [], [], []
        fmc = len(self.STRIDES)
        for idx, stride in enumerate(self.STRIDES):
            scores = outs[idx].reshape(-1)
            pos = np.where(scores >= score_thresh)[0]
            if pos.size == 0:
                continue
            centers = self._anchor_centers(size // stride, size // stride, stride)[pos]
            boxes = distance2bbox(centers, outs[idx + fmc][pos] * stride)
            kps = distance2kps(centers, outs[idx + 2 * fmc][pos] * stride).reshape(-1, 5, 2)
            all_scores.append(scores[pos])
            all_boxes.append(boxes)
            all_kps.append(kps)

        if not all_scores:
            return []
        scores = np.concatenate(all_scores)
        boxes = np.concatenate(all_boxes) / scale
        kps = np.concatenate(all_kps) / scale

        keep = nms(boxes, scores, nms_thresh)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        dets = [Detection(boxes[i], float(scores[i]), kps[i]) for i in keep]
        if max_faces > 0:
            dets.sort(key=lambda d: d.width * d.height, reverse=True)
            dets = dets[:max_faces]
        return dets
