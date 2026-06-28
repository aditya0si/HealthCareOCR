import os
import cv2
import pytest
import torch
import time
from src.pipeline import MedicalOCRPipeline
from src.utils.gpu_manager import GPUManager

WHATSAPP_DIR = "WhatsApp.Unknown.2026-04-27.at.12.10.10/WhatsApp Unknown 2026-04-27 at 12.10.10"

@pytest.fixture(scope="module")
def pipeline():
    return MedicalOCRPipeline()

def test_pipeline_lifecycle_and_swap(pipeline):
    # Verify that a sample image goes end-to-end without VRAM build-up
    assert os.path.exists(WHATSAPP_DIR), f"Directory {WHATSAPP_DIR} does not exist"
    files = [f for f in os.listdir(WHATSAPP_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    assert len(files) > 0, "No image files found in WhatsApp directory"
    
    # Select a sample file
    sample_path = os.path.join(WHATSAPP_DIR, files[0])
    img = cv2.imread(sample_path)
    assert img is not None, f"Failed to load image: {sample_path}"
    
    # Clean and check VRAM before starting
    GPUManager.clean_memory()
    vram_start = GPUManager.get_vram_allocated()
    print(f"\n[Test Pipeline] Starting VRAM: {vram_start:.2f} MB")
    
    start_time = time.time()
    result = pipeline.process_image(img)
    elapsed = time.time() - start_time
    
    vram_end = GPUManager.get_vram_allocated()
    print(f"[Test Pipeline] Ending VRAM: {vram_end:.2f} MB")
    print(f"[Test Pipeline] Total Latency: {elapsed:.2f} seconds")
    
    # Assert result structure
    assert "raw_ocr_text" in result
    assert "corrected_ocr_text" in result
    assert "clinical_summary" in result
    assert "quality_metrics" in result
    assert "timings" in result
    
    # Assert summarizer output details
    summary = result["clinical_summary"]
    assert isinstance(summary, dict), f"Clinical summary is not a dictionary: {summary}"
    # Summarization is bypassed in disintegrated pipeline, so summary is empty
    
    # Assert timing parameters exist for active stages
    timings = result["timings"]
    assert "Preprocessing" in timings
    assert "Layout & OCR" in timings
    
    # Check that VRAM has been unloaded successfully back to system baseline (< 150 MB allocated)
    assert vram_end < 150.0, f"VRAM was not unloaded correctly. Remaining: {vram_end:.2f} MB"
    
    print("\n--- Timings Breakdown ---")
    for stage, t in timings.items():
        if isinstance(t, float):
            print(f"  {stage:<30}: {t:.3f} s")
