import sys
import os
import cv2
import logging
from src.preprocessing import PreprocessorPipeline
from src.ocr import LayoutEngine, SuryaEngine, TrOCREngine, OCRRouter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ocr_diagnostics")

def run(path):
    if not os.path.exists(path):
        print(f"Image not found: {path}")
        return
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to read image: {path}")
        return

    pre = PreprocessorPipeline()
    layout = LayoutEngine()
    surya = SuryaEngine()
    trocr = TrOCREngine()
    router = OCRRouter(surya, trocr)

    print("Running preprocessing...")
    proc_img, metrics, timings = pre.process(img)
    print("Preprocessing metrics:", metrics.__dict__ if hasattr(metrics, '__dict__') else metrics)
    print("Timings:", timings)

    print("Detecting regions...")
    regions = layout.detect_regions(proc_img)
    print(f"Detected {len(regions)} regions")
    for i, r in enumerate(regions[:10]):
        print(i, r.get('bbox'), r.get('confidence'))

    print("Running router...")
    routed = router.process_document(pre.last_ocr_image if getattr(pre, 'last_ocr_image', None) is not None else proc_img, regions)
    for i, r in enumerate(routed[:20]):
        print(i, r.get('type'), 'text_len=', len(r.get('text','')))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/ocr_diagnostics.py <image_path>")
    else:
        run(sys.argv[1])
