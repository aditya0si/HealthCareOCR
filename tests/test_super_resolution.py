import os
import cv2
import numpy as np
import pytest
import time
import torch
from src.preprocessing import PreprocessorPipeline
from src.preprocessing.super_resolution import SuperResolution

KASTOOR_DIR = "Patient_Kastoor/Lab_Report"
WHATSAPP_DIR = "WhatsApp.Unknown.2026-04-27.at.12.10.10/WhatsApp Unknown 2026-04-27 at 12.10.10"

@pytest.fixture
def sr_module():
    return SuperResolution()

def test_dpi_estimation_and_trigger(sr_module):
    # Test a small image (WhatsApp style)
    small_img = np.zeros((800, 600, 3), dtype=np.uint8)
    dpi = sr_module.estimate_dpi(small_img)
    # width / 8.27 = 600/8.27 = 72.5
    # height / 11.69 = 800/11.69 = 68.4
    # avg DPI should be around 70
    assert 65 < dpi < 75
    assert sr_module.should_upscale(small_img) is True

    # Test a landscape small image
    small_landscape = np.zeros((600, 800, 3), dtype=np.uint8)
    dpi_l = sr_module.estimate_dpi(small_landscape)
    # width / 11.69 = 800/11.69 = 68.4
    # height / 8.27 = 600/8.27 = 72.5
    # avg DPI should be around 70
    assert 65 < dpi_l < 75
    assert sr_module.should_upscale(small_landscape) is True

    # Test a large image (Kastoor style)
    large_img = np.zeros((3000, 2200, 3), dtype=np.uint8)
    dpi_lg = sr_module.estimate_dpi(large_img)
    assert dpi_lg > 200
    assert sr_module.should_upscale(large_img) is False  # max_dim >= 2000

def test_fsrcnn_upscale_performance(sr_module):
    # Create a small dummy image for testing
    dummy_img = np.zeros((200, 150, 3), dtype=np.uint8)
    
    # Track VRAM before
    torch.cuda.empty_cache()
    vram_before = torch.cuda.memory_allocated()
    
    start_time = time.time()
    upscaled = sr_module.upscale(dummy_img)
    elapsed = (time.time() - start_time) * 1000
    
    # Track VRAM after
    vram_after = torch.cuda.memory_allocated()
    
    assert upscaled.shape == (400, 300, 3)
    assert elapsed < 150.0  # Small image upscaling should be very fast (<150ms)
    assert (vram_after - vram_before) == 0  # 0 VRAM footprint

def test_pipeline_sr_integration():
    pipeline = PreprocessorPipeline()
    
    # 1. Verify WhatsApp image triggers SR
    assert os.path.exists(WHATSAPP_DIR), f"Directory {WHATSAPP_DIR} does not exist"
    whatsapp_files = [f for f in os.listdir(WHATSAPP_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    assert len(whatsapp_files) > 0
    whatsapp_path = os.path.join(WHATSAPP_DIR, whatsapp_files[0])
    img_wa = cv2.imread(whatsapp_path)
    
    _, _, timings_wa = pipeline.process(img_wa)
    assert timings_wa["super_resolution_run"] is True
    assert timings_wa["super_resolution_ms"] > 0
    
    # 2. Verify Kastoor image skips SR
    assert os.path.exists(KASTOOR_DIR), f"Directory {KASTOOR_DIR} does not exist"
    kastoor_files = [f for f in os.listdir(KASTOOR_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    assert len(kastoor_files) > 0
    kastoor_path = os.path.join(KASTOOR_DIR, kastoor_files[0])
    img_ka = cv2.imread(kastoor_path)
    
    _, _, timings_ka = pipeline.process(img_ka)
    assert timings_ka["super_resolution_run"] is False
    assert timings_ka["super_resolution_ms"] == 0.0
