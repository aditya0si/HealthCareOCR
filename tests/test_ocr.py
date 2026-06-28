import os
import cv2
import numpy as np
import pytest
import time
import torch
from src.ocr import LayoutEngine, SuryaEngine, TrOCREngine, OCRRouter

KASTOOR_DIR = "Patient_Kastoor/Lab_Report"
WHATSAPP_DIR = "WhatsApp.Unknown.2026-04-27.at.12.10.10/WhatsApp Unknown 2026-04-27 at 12.10.10"

@pytest.fixture(scope="module")
def engines():
    # Load all models once for tests to save time and VRAM reloading
    layout = LayoutEngine()
    surya = SuryaEngine()
    trocr = TrOCREngine()
    router = OCRRouter(surya, trocr)
    
    # Warm up using a real image to ensure shapes and allocations are standard
    import os
    import cv2
    WHATSAPP_DIR = "WhatsApp.Unknown.2026-04-27.at.12.10.10/WhatsApp Unknown 2026-04-27 at 12.10.10"
    files = [f for f in os.listdir(WHATSAPP_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if files:
        img_path = os.path.join(WHATSAPP_DIR, files[0])
        img = cv2.imread(img_path)
        if img is not None:
            try:
                # Run layout detection once to warm up
                regions = layout.detect_regions(img)
                # Take a couple of regions to warm up router and recognition models
                if regions:
                    # Filter for one printed and one handwritten if possible, or just the first two
                    _ = router.process_document(img, regions[:2])
            except Exception as e:
                print(f"Warm-up failed: {e}")
        
    return {
        "layout": layout,
        "surya": surya,
        "trocr": trocr,
        "router": router
    }

def test_layout_engine(engines):
    layout = engines["layout"]
    
    # Load a WhatsApp image
    assert os.path.exists(WHATSAPP_DIR), f"Directory {WHATSAPP_DIR} does not exist"
    files = [f for f in os.listdir(WHATSAPP_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    assert len(files) > 0
    img_path = os.path.join(WHATSAPP_DIR, files[0])
    img = cv2.imread(img_path)
    assert img is not None
    
    regions = layout.detect_regions(img)
    assert isinstance(regions, list)
    assert len(regions) > 0
    
    for r in regions:
        assert "bbox" in r
        assert len(r["bbox"]) == 4
        assert "polygon" in r
        assert "confidence" in r
        assert r["label"] == "text_line"
        
    print(f"\n[Layout Engine Test] Detected {len(regions)} text lines.")

def test_ocr_routing_and_execution(engines):
    layout = engines["layout"]
    router = engines["router"]
    
    # Load a WhatsApp image (e.g. handwritten prescription or any other)
    files = [f for f in os.listdir(WHATSAPP_DIR) if "handwritten_prescription" in f]
    if not files:
        files = [f for f in os.listdir(WHATSAPP_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    assert len(files) > 0
    img_path = os.path.join(WHATSAPP_DIR, files[0])
    img = cv2.imread(img_path)
    assert img is not None
    
    # Measure VRAM before
    torch.cuda.empty_cache()
    vram_before = torch.cuda.memory_allocated() / (1024 ** 2)
    
    # Run layout detection
    start_time = time.time()
    regions = layout.detect_regions(img)
    
    # Run classification, routing, and dual-engine OCR
    processed_regions = router.process_document(img, regions)
    elapsed = time.time() - start_time
    
    # Measure VRAM after
    vram_after = torch.cuda.memory_allocated() / (1024 ** 2)
    
    print(f"\n[OCR Routing Test] Image: {files[0]}")
    print(f"  Processed {len(processed_regions)} regions in {elapsed:.2f} seconds.")
    print(f"  Allocated VRAM: {vram_after:.2f} MB")
    
    # Assertions
    assert len(processed_regions) == len(regions)
    printed_count = 0
    handwritten_count = 0
    for r in processed_regions:
        assert "text" in r
        assert "type" in r
        assert r["type"] in ("printed", "handwritten")
        assert isinstance(r["text"], str)
        if r["type"] == "printed":
            printed_count += 1
        else:
            handwritten_count += 1
            
    print(f"  Printed lines: {printed_count}, Handwritten lines: {handwritten_count}")
    
    # Assertions on performance constraints
    # Pure inference time should easily complete in < 7.0s for heavy handwriting (batched on GPU in FP16)
    assert elapsed < 7.0, f"OCR execution took too long: {elapsed:.2f}s"
    
    # Active weights allocations (Surya + TrOCR + layout) should be < 4.0 GB (4096 MB)
    assert vram_after < 4096.0, f"VRAM allocation is too high: {vram_after:.2f} MB"
