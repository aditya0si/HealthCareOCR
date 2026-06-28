from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor
from surya.settings import settings
from PIL import Image
import numpy as np
import cv2
import os
import logging

logger = logging.getLogger(__name__)

class SuryaEngine:
    def __init__(self, config: dict = None):
        """
        Surya2 printed OCR engine using standard FoundationPredictor (BF16/FP16) for high performance.
        """
        cfg = config if config is not None else {}
        device = cfg.get("device", "cuda")
        checkpoint = cfg.get("checkpoint", settings.RECOGNITION_MODEL_CHECKPOINT)
        self.foundation_predictor = FoundationPredictor(checkpoint=checkpoint, device=device)
        self.rec_predictor = RecognitionPredictor(self.foundation_predictor)

    def recognize(self, image: np.ndarray, bboxes: list[list[int]]) -> list[str]:
        """
        Perform printed text OCR on the specified bounding boxes of the image.
        Returns a list of strings representing the OCR output for each box.
        """
        if not bboxes:
            return []

        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(image)
        else:
            pil_img = image

        valid_bboxes = []
        h_img, w_img = image.shape[:2] if isinstance(image, np.ndarray) else (0, 0)
        for bbox in bboxes:
            if not bbox or len(bbox) < 4:
                continue
            x1 = max(0, int(round(bbox[0])))
            y1 = max(0, int(round(bbox[1])))
            x2 = min(w_img, int(round(bbox[2]))) if w_img else int(round(bbox[2]))
            y2 = min(h_img, int(round(bbox[3]))) if h_img else int(round(bbox[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            # Skip extremely tiny crops that are unlikely to contain readable text
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue
            valid_bboxes.append([x1, y1, x2, y2])

        if not valid_bboxes:
            return []

        try:
            model_device = next(self.foundation_predictor.model.parameters()).device.type
            if model_device == "cpu":
                logger.info("Moving Surya model to CUDA for recognition...")
                self.foundation_predictor.model.to("cuda")
                self.foundation_predictor.device = "cuda"
                if hasattr(self.foundation_predictor, "special_token_ids") and self.foundation_predictor.special_token_ids is not None:
                    try:
                        self.foundation_predictor.special_token_ids = self.foundation_predictor.special_token_ids.to("cuda")
                    except Exception:
                        logger.debug("Couldn't move special_token_ids to CUDA; continuing")
                if hasattr(self.rec_predictor, "device"):
                    self.rec_predictor.device = "cuda"
        except Exception as e:
            logger.exception("Error checking/moving Surya model: %s", e)

        try:
            predictions = self.rec_predictor(
                [pil_img],
                bboxes=[valid_bboxes],
                math_mode=False
            )
            prediction = predictions[0]
            texts = [line.text for line in prediction.text_lines]
            return texts
        except Exception as e:
            logger.exception("Surya recognition failed: %s", e)
            # Return empty strings for each bbox to preserve alignment
            return ["" for _ in valid_bboxes]
