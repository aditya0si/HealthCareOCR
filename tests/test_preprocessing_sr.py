import numpy as np
import cv2
from src.preprocessing.super_resolution import SuperResolution


def test_should_upscale_and_upscale_fallback():
    # Create a small synthetic document-like image (A4 portrait scaled down)
    h, w = 300, 200  # small image -> low DPI
    img = np.ones((h, w), dtype=np.uint8) * 255

    sr = SuperResolution(model_path="models/FSRCNN_x2.pb", min_dpi=180)
    est = sr.estimate_dpi(img)
    assert est > 0

    # The synthetic image DPI should be below the min_dpi and trigger upscale
    assert sr.should_upscale(img) == (est < (sr.min_dpi - 5))

    up = sr.upscale(img)
    # Upscaled image should be same or larger than input
    assert up is not None
    assert up.shape[0] >= img.shape[0]
    assert up.shape[1] >= img.shape[1]
