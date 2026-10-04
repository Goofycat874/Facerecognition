import numpy as np

from facerec.align import ARCFACE_TEMPLATE, align_face, umeyama
from facerec.detector import distance2bbox, distance2kps, nms
from facerec.quality import frontalness


def _similarity(scale, angle, tx, ty):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[scale * c, -scale * s, tx], [scale * s, scale * c, ty]])


def test_umeyama_recovers_similarity_transform():
    m_true = _similarity(2.5, 0.3, 40.0, -12.0)
    src = ARCFACE_TEMPLATE.astype(np.float64)
    dst = src @ m_true[:, :2].T + m_true[:, 2]
    np.testing.assert_allclose(umeyama(src, dst), m_true, atol=1e-6)


def test_align_face_maps_landmarks_onto_template():
    m = _similarity(3.0, -0.4, 200.0, 150.0)
    kps = ARCFACE_TEMPLATE @ m[:, :2].T + m[:, 2]
    img = np.zeros((600, 600, 3), np.uint8)
    for x, y in kps.astype(int):
        img[y - 3:y + 4, x - 3:x + 4] = 255
    crop = align_face(img, kps)
    assert crop.shape == (112, 112, 3)
    for x, y in ARCFACE_TEMPLATE.astype(int):
        assert crop[y, x].max() > 200


def test_distance_decoding():
    pts = np.array([[10.0, 20.0]])
    np.testing.assert_allclose(distance2bbox(pts, np.array([[1, 2, 3, 4.0]])), [[9, 18, 13, 24]])
    np.testing.assert_allclose(distance2kps(pts, np.array([[1, -1, 2, 2.0]])), [[11, 19, 12, 22]])


def test_nms_drops_overlapping_lower_scores():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.4) == [0, 2]


def test_frontalness():
    assert frontalness(ARCFACE_TEMPLATE) > 0.9
    turned = ARCFACE_TEMPLATE.copy()
    turned[2, 0] = turned[1, 0] - 2  # nose almost on the right eye
    assert frontalness(turned) < 0.1
    profile = ARCFACE_TEMPLATE.copy()
    profile[2, 0] = turned[1, 0] + 5  # nose past the eye
    assert frontalness(profile) == 0.0
