from transformers import VisionEncoderDecoderModel, TrOCRProcessor, BitsAndBytesConfig
import torch
import numpy as np
import cv2
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

class TrOCREngine:
    def __init__(self, config: dict = None):
        """
        TrOCR handwriting recognition engine running in 4-bit quantized mode for high speed and low memory.
        """
        cfg = config if config is not None else {}
        device = cfg.get("device", "cuda")
        # Force loading on CUDA if available since bitsandbytes requires GPU
        load_device = "cuda" if torch.cuda.is_available() else device
        
        # Load default checkpoint from config, falling back to microsoft/trocr-small-handwritten
        model_name = cfg.get("models", {}).get("handwriting_ocr_model", {}).get("name", "microsoft/trocr-small-handwritten")
        
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        
        if load_device == "cuda":
            print(f"[TrOCR] Loading handwriting model: {model_name} in 4-bit NF4 on {load_device}...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            print(f"[TrOCR] Loading handwriting model: {model_name} in float32 on CPU (fallback)...")
            self.model = VisionEncoderDecoderModel.from_pretrained(
                model_name
            ).to(load_device)
            
        self.model.eval()
        self.device = load_device

    def recognize(self, image: np.ndarray, bboxes: list[list[int]], batch_size: int = 32) -> list[str]:
        """
        Crop regions from the image based on bounding boxes, batch them, and run TrOCR.
        """
        if not bboxes:
            return []

        h_img, w_img = image.shape[:2]
        valid_crops = []
        valid_bbox_count = 0

        for bbox in bboxes:
            if not bbox or len(bbox) < 4:
                continue
            x1 = max(0, int(round(bbox[0])))
            y1 = max(0, int(round(bbox[1])))
            x2 = min(w_img, int(round(bbox[2])))
            y2 = min(h_img, int(round(bbox[3])))

            if x2 <= x1 or y2 <= y1:
                continue
            # Skip extremely tiny crops that are unlikely to contain readable text
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue

            crop = image[y1:y2, x1:x2]
            if len(crop.shape) == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            valid_crops.append(crop)
            valid_bbox_count += 1

        if valid_bbox_count == 0:
            return []

        try:
            model_device = next(self.model.parameters()).device.type
            if model_device == "cpu":
                logger.info("Moving TrOCR model to CUDA for recognition...")
                try:
                    self.model.to("cuda")
                    self.device = "cuda"
                except Exception:
                    logger.exception("Failed to move TrOCR model to CUDA; continuing on CPU")
        except Exception as e:
            logger.exception("Error checking/moving TrOCR model: %s", e)

        texts = []
        for i in range(0, len(valid_crops), batch_size):
            batch_crops = valid_crops[i:i+batch_size]
            pixel_values = self.processor(images=batch_crops, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device).half()

            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_new_tokens=64,
                    use_cache=True,
                    num_beams=1
                )

            batch_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            texts.extend(batch_texts)

        try:
            return [t.strip() for t in texts]
        except Exception:
            logger.exception("Error decoding TrOCR outputs; returning empty strings for batch")
            return ["" for _ in range(len(texts))]
