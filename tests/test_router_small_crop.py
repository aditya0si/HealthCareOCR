import numpy as np
import cv2
from src.ocr.router import OCRRouter


def test_classify_tiny_crop_is_printed():
    router = OCRRouter(None, None)
    tiny = np.ones((6, 6), dtype=np.uint8) * 255
    assert router.classify_crop(tiny) == "printed"

    tiny_color = np.ones((6, 6, 3), dtype=np.uint8) * 255
    assert router.classify_crop(tiny_color) == "printed"
