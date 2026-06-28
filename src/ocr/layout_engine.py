import torch
from surya.detection import DetectionPredictor
from PIL import Image
import numpy as np
import cv2

class LayoutEngine:
    def __init__(self, config: dict = None):
        """
        Layout engine utilizing Surya text line detection model for robust line isolation.
        """
        cfg = config if config is not None else {}
        device = cfg.get("device", "cuda")
        self.detector = DetectionPredictor(device=device)

    def detect_regions(self, image: np.ndarray) -> list[dict]:
        """
        Detect all text line regions in the image.
        Returns a list of dicts:
            {
                "bbox": [x1, y1, x2, y2],
                "polygon": [[x1, y1], ...],
                "confidence": float,
                "label": str
            }
        """
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                # Convert BGR to RGB for PIL Image compatibility
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(image)
        else:
            pil_img = image

        predictions = self.detector([pil_img])
        prediction = predictions[0]

        regions = []
        for poly_box in prediction.bboxes:
            regions.append({
                "bbox": poly_box.bbox,
                "polygon": poly_box.polygon,
                "confidence": poly_box.confidence,
                "label": "text_line"
            })
            
        # Reconstruct reading order by sorting top-to-bottom, then left-to-right
        regions = self._sort_regions(regions)
        return regions

    def _sort_regions(self, regions: list[dict]) -> list[dict]:
        """
        Sort regions in natural reading order (top-to-bottom, left-to-right).
        Uses a standard threshold for vertical grouping (y-coordinate overlap).
        """
        if not regions:
            return []
            
        # Sort primarily by y-coordinate
        sorted_by_y = sorted(regions, key=lambda r: r["bbox"][1])
        
        # Group lines that are on roughly the same horizontal level (y-overlap)
        rows = []
        current_row = [sorted_by_y[0]]
        
        for r in sorted_by_y[1:]:
            prev_r = current_row[-1]
            prev_y1, prev_y2 = prev_r["bbox"][1], prev_r["bbox"][3]
            curr_y1, curr_y2 = r["bbox"][1], r["bbox"][3]
            
            # Check overlap threshold: if overlap is significant, they are on the same row
            prev_h = prev_y2 - prev_y1
            overlap = min(prev_y2, curr_y2) - max(prev_y1, curr_y1)
            
            if overlap > 0.4 * prev_h:
                current_row.append(r)
            else:
                rows.append(current_row)
                current_row = [r]
        rows.append(current_row)
        
        # Sort each row from left to right (by x-coordinate)
        sorted_regions = []
        for row in rows:
            sorted_row = sorted(row, key=lambda r: r["bbox"][0])
            sorted_regions.extend(sorted_row)
            
        return sorted_regions
