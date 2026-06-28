import numpy as np
import cv2
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class OCRRouter:
    def __init__(self, surya_engine, trocr_engine, config: dict = None):
        """
        Orchestrates routing of document regions to Surya (printed OCR) vs TrOCR (handwritten OCR).
        """
        self.surya = surya_engine
        self.trocr = trocr_engine
        self.config = config if config is not None else {}
        self.parallel_ocr = self.config.get("parallel_ocr", True)
        self.max_workers = min(2, self.config.get("parallel_ocr_workers", 2))
        # Be conservative by default: require a larger proportion to flip routing
        self.classification_threshold = self.config.get("ocr_routing", {}).get("classification_threshold", 0.35)

    def classify_crop(self, crop: np.ndarray) -> str:
        """
        Classifies a crop as 'printed' or 'handwritten'.
        Uses stroke height variance and diagonal gradient ratio metrics.
        """
        if crop is None or crop.size == 0:
            logger.debug("Empty crop passed to classify_crop; defaulting to printed")
            return "printed"
            
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()
            
        h, w = gray.shape[:2]
        if h < 8 or w < 8:
            logger.debug("Tiny crop (%dx%d) treated as printed", w, h)
            return "printed"
            
        # Threshold to extract text mask
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        heights = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 4:
                heights.append(stats[i, cv2.CC_STAT_HEIGHT])
                
        if len(heights) < 3:
            logger.debug("Insufficient connected component heights (%d); classified as printed", len(heights))
            return "printed"
            
        median_h = np.median(heights)
        std_h = np.std(heights)
        height_var = std_h / (median_h + 1e-5)
        
        # Calculate gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx) * 180 / np.pi
        
        strong_gradients = direction[magnitude > 20]
        if len(strong_gradients) > 0:
            diagonals = np.sum((np.abs(strong_gradients) > 30) & (np.abs(strong_gradients) < 60)) + \
                        np.sum((np.abs(strong_gradients) > 120) & (np.abs(strong_gradients) < 150))
            diag_ratio = diagonals / len(strong_gradients)
        else:
            diag_ratio = 0.0
            
        # Printed characters are regular and uniform. Cursive writing varies.
        is_hand = (height_var > 0.5) or (diag_ratio > 0.25 and height_var > 0.35)
        logger.debug("Crop classification: median_h=%.2f std_h=%.2f height_var=%.3f diag_ratio=%.3f -> %s",
                 median_h, std_h, height_var, diag_ratio, "handwritten" if is_hand else "printed")
        return "handwritten" if is_hand else "printed"

    def process_document(self, image: np.ndarray, regions: list[dict]) -> list[dict]:
        """
        Classifies each region, executes dual OCR engines in parallel batches,
        and assigns text outputs while preserving the original reading order.
        """
        if not regions:
            return []
            
        h_img, w_img = image.shape[:2]
        
        # 1. Classify each region crop
        for region in regions:
            bbox = region["bbox"]
            x1 = max(0, int(round(bbox[0])))
            y1 = max(0, int(round(bbox[1])))
            x2 = min(w_img, int(round(bbox[2])))
            y2 = min(h_img, int(round(bbox[3])))
            
            if x2 <= x1 or y2 <= y1:
                region["type"] = "printed"
                region["crop"] = None
                region["text"] = ""
            else:
                crop = image[y1:y2, x1:x2]
                region["type"] = self.classify_crop(crop)
                region["crop"] = crop
                region["text"] = ""

        # 2. Group indices and bounding boxes
        printed_indices = []
        printed_bboxes = []
        hand_indices = []
        hand_bboxes = []
        
        for idx, region in enumerate(regions):
            if region["type"] == "printed":
                printed_indices.append(idx)
                printed_bboxes.append(region["bbox"])
            else:
                hand_indices.append(idx)
                hand_bboxes.append(region["bbox"])

        # Consolidate routing conservatively so small or ambiguous layout noise does not force
        # every region through the wrong engine.
        total_regions = len(regions)
        if total_regions <= 2:
            # Keep the original assignments for tiny region sets.
            pass
        elif len(hand_bboxes) <= 2 or len(hand_bboxes) < self.classification_threshold * total_regions:
            printed_indices = list(range(total_regions))
            printed_bboxes = [r["bbox"] for r in regions]
            for r in regions:
                r["type"] = "printed"
            hand_indices = []
            hand_bboxes = []
        elif len(printed_bboxes) <= 2 or len(printed_bboxes) < self.classification_threshold * total_regions:
            hand_indices = list(range(total_regions))
            hand_bboxes = [r["bbox"] for r in regions]
            for r in regions:
                r["type"] = "handwritten"
            printed_indices = []
            printed_bboxes = []

        # 3. Printed and handwritten OCR batch execution
        printed_texts = []
        hand_texts = []
        if self.parallel_ocr and printed_bboxes and hand_bboxes:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_role = {
                    executor.submit(self.surya.recognize, image, printed_bboxes): "printed",
                    executor.submit(self.trocr.recognize, image, hand_bboxes): "handwritten"
                }
                for future in as_completed(future_to_role):
                    role = future_to_role[future]
                    try:
                        texts = future.result()
                    except Exception as e:
                        logger.exception("Parallel OCR execution error (%s): %s", role, e)
                        texts = []
                    if role == "printed":
                        printed_texts = texts
                    else:
                        hand_texts = texts
        else:
            if printed_bboxes:
                printed_texts = self.surya.recognize(image, printed_bboxes)
            if hand_bboxes:
                hand_texts = self.trocr.recognize(image, hand_bboxes)

        # 5. Re-integrate texts into original region list
        for idx, text in zip(printed_indices, printed_texts):
            if isinstance(text, str) and text.strip():
                regions[idx]["text"] = text

        for idx, text in zip(hand_indices, hand_texts):
            if isinstance(text, str) and text.strip():
                regions[idx]["text"] = text

        # Fallback: if the routed engine produced no useful text for a region, try the other engine.
        for region_idx, region in enumerate(regions):
            if region.get("text", "").strip():
                continue
            bbox = region.get("bbox")
            if not bbox:
                continue
            image_region = image[max(0, int(round(bbox[1]))):min(image.shape[0], int(round(bbox[3]))),
                                  max(0, int(round(bbox[0]))):min(image.shape[1], int(round(bbox[2])))]
            if image_region.size == 0:
                continue
            if region.get("type") == "printed":
                logger.debug("Fallback attempt: trying TrOCR for printed region %d", region_idx)
                fallback_texts = self.trocr.recognize(image, [bbox])
            else:
                logger.debug("Fallback attempt: trying Surya for handwritten region %d", region_idx)
                fallback_texts = self.surya.recognize(image, [bbox])
            if fallback_texts:
                region["text"] = fallback_texts[0]

        # Clean up crop objects to release memory
        for region in regions:
            if "crop" in region:
                del region["crop"]

        return regions
