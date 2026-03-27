import numpy as np
import cv2
import pytest
from checkers_game.camera.board_detection_1 import BoardDetection

# python

def _make_rect_contour(x, y, w, h):
    # contour must be of shape (N,1,2) like OpenCV contours
    return np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)

def test_get_contours_off_all_rectangles_returns_64_contours(monkeypatch):
    # Prepare a BoardDetection instance without calling __init__ (which opens GUI windows)
    bd = BoardDetection.__new__(BoardDetection)
    # Set a starting param so the method will run its loop and use our patched findContours
    bd.param1ForGetAllContours = 50

    # Create many synthetic contours (more than 64 so method can pick 64)
    synthetic_contours = []
    # create 80 rectangles spaced out (w,h > 20)
    x, y = 0, 0
    for i in range(80):
        # vary position and size a bit
        w = 30 + (i % 3)
        h = 30 + ((i + 1) % 3)
        cx = (i % 10) * 40
        cy = (i // 10) * 40
        synthetic_contours.append(_make_rect_contour(cx, cy, w, h))

    # Patch cv2.findContours to return our synthetic contours regardless of input
    def fake_findContours(img, mode, method):
        return synthetic_contours, None

    monkeypatch.setattr(cv2, "findContours", fake_findContours)

    # Also patch cv2.HoughLines to return a plausible empty/None (not required but safe)
    monkeypatch.setattr(cv2, "HoughLines", lambda *args, **kwargs: None)

    # Call the method with a dummy BGR image
    dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)
    result = bd._get_contours_off_all_rectangles(dummy_img)

    # Expect 64 contours returned
    assert isinstance(result, list)
    assert len(result) == 64

    # Each returned contour should be a numpy ndarray and have bounding box > 20
    for cnt in result:
        assert isinstance(cnt, np.ndarray)
        x, y, w, h = cv2.boundingRect(cnt)
        assert w > 20 and h > 20

def test_get_contours_off_all_rectangles_handles_exactly_64_contours(monkeypatch):
    bd = BoardDetection.__new__(BoardDetection)
    bd.param1ForGetAllContours = 10

    # Create exactly 64 synthetic contours with adequate size
    synthetic_contours = []
    for i in range(64):
        cx = (i % 8) * 50
        cy = (i // 8) * 50
        synthetic_contours.append(_make_rect_contour(cx, cy, 40, 40))

    def fake_findContours(img, mode, method):
        return synthetic_contours, None

    monkeypatch.setattr(cv2, "findContours", fake_findContours)
    monkeypatch.setattr(cv2, "HoughLines", lambda *args, **kwargs: None)

    dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)
    result = bd._get_contours_off_all_rectangles(dummy_img)

    assert isinstance(result, list)
    assert len(result) == 64
    for cnt in result:
        assert isinstance(cnt, np.ndarray)
        x, y, w, h = cv2.boundingRect(cnt)
        assert w >= 40 and h >= 40