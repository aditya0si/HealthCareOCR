import os
import cv2
import numpy as np
import pytest
import time
from src.preprocessing import PreprocessorPipeline

# Dataset paths
KASTOOR_DIR = "Patient_Kastoor/Lab_Report"
WHATSAPP_DIR = "WhatsApp.Unknown.2026-04-27.at.12.10.10/WhatsApp Unknown 2026-04-27 at 12.10.10"

@pytest.fixture
def pipeline():
    return PreprocessorPipeline()

def test_pipeline_on_kastoor_sample(pipeline):
    # Retrieve a sample image from Patient_Kastoor
    assert os.path.exists(KASTOOR_DIR), f"Directory {KASTOOR_DIR} does not exist"
    files = [f for f in os.listdir(KASTOOR_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    assert len(files) > 0, "No image files found in Kastoor directory"
    
    sample_path = os.path.join(KASTOOR_DIR, files[0])
    img = cv2.imread(sample_path)
    assert img is not None, f"Failed to load image: {sample_path}"
    
    # Process the image
    processed, metrics, timings = pipeline.process(img)
    
    # Assert processed image properties
    assert isinstance(processed, np.ndarray), "Processed output must be a numpy array"
    assert len(processed.shape) == 2, "Processed output must be grayscale (2D)"
    
    # Assert values are strictly binary (0 or 255)
    unique_vals = np.unique(processed)
    for val in unique_vals:
        assert val in (0, 255), f"Output pixel value {val} is not binary (0 or 255)"
        
    # Assert quality metrics
    assert 0.0 <= metrics.sharpness <= 1.0
    assert 0.0 <= metrics.contrast <= 1.0
    assert 0.0 <= metrics.brightness <= 1.0
    assert 0.0 <= metrics.noise_ratio <= 1.0
    assert isinstance(metrics.skew_angle, float)
    
    # Assert timings exist and are positive
    assert timings["total_preprocessing_ms"] > 0
    print(f"\n[Kastoor Preprocessing Test] Image: {files[0]}")
    print(f"  Shape original: {img.shape} -> Processed: {processed.shape}")
    print(f"  Quality Metrics: {metrics}")
    print(f"  Timings: {timings}")

def test_pipeline_on_whatsapp_sample(pipeline):
    # Retrieve a sample image from WhatsApp dataset
    assert os.path.exists(WHATSAPP_DIR), f"Directory {WHATSAPP_DIR} does not exist"
    files = [f for f in os.listdir(WHATSAPP_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    assert len(files) > 0, "No image files found in WhatsApp directory"
    
    sample_path = os.path.join(WHATSAPP_DIR, files[0])
    img = cv2.imread(sample_path)
    assert img is not None, f"Failed to load image: {sample_path}"
    
    # Process the image
    processed, metrics, timings = pipeline.process(img)
    
    # Assert processed image properties
    assert isinstance(processed, np.ndarray)
    assert len(processed.shape) == 2
    
    # Assert quality metrics
    assert 0.0 <= metrics.sharpness <= 1.0
    assert 0.0 <= metrics.contrast <= 1.0
    assert 0.0 <= metrics.brightness <= 1.0
    assert 0.0 <= metrics.noise_ratio <= 1.0
    
    # Assert timings exist and are positive
    assert timings["total_preprocessing_ms"] > 0
    print(f"\n[WhatsApp Preprocessing Test] Image: {files[0]}")
    print(f"  Shape original: {img.shape} -> Processed: {processed.shape}")
    print(f"  Quality Metrics: {metrics}")
    print(f"  Timings: {timings}")

def test_pipeline_all_whatsapp_images(pipeline):
    # Process all 29 WhatsApp images
    assert os.path.exists(WHATSAPP_DIR), f"Directory {WHATSAPP_DIR} does not exist"
    files = [f for f in os.listdir(WHATSAPP_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n--- Processing All {len(files)} WhatsApp Images ---")
    timing_list = []
    low_quality_count = 0
    
    for f in files:
        sample_path = os.path.join(WHATSAPP_DIR, f)
        img = cv2.imread(sample_path)
        assert img is not None
        
        start = time.time()
        processed, metrics, timings = pipeline.process(img)
        elapsed = (time.time() - start) * 1000
        
        timing_list.append(elapsed)
        if timings.get("enhancement_tier") == "low_medium":
            low_quality_count += 1
            
        print(f"  {f:<40} | Time: {elapsed:.1f}ms | Tier: {timings.get('enhancement_tier')} | Skew: {metrics.skew_angle:.2f}°")
        
    avg_time = sum(timing_list) / len(timing_list)
    max_time = max(timing_list)
    print(f"\nSummary:")
    print(f"  Total processed: {len(files)}")
    print(f"  Average time: {avg_time:.1f}ms")
    print(f"  Max time: {max_time:.1f}ms")
    print(f"  Low-quality path triggered: {low_quality_count} times")
    
    # Assert average time is well under the 2500ms budget (allows for CPU variation in SR and Sauvola)
    assert avg_time < 2500.0, "Average preprocessing time is too high"

def test_low_quality_pipeline_path():
    # Force low-quality path by setting very high quality thresholds on the pipeline
    strict_config = {
        "quality_assessment": {
            "min_sharpness": 0.99,   # Almost impossible to satisfy -> forces low quality
            "min_contrast": 0.99,
            "min_brightness": 0.99,
            "max_brightness": 0.01,
            "max_noise_ratio": 0.01,
            "max_skew_angle": 0.01
        }
    }
    strict_pipeline = PreprocessorPipeline(config=strict_config)
    
    # Run on a WhatsApp sample
    files = [f for f in os.listdir(WHATSAPP_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    sample_path = os.path.join(WHATSAPP_DIR, files[0])
    img = cv2.imread(sample_path)
    
    processed, metrics, timings = strict_pipeline.process(img)
    
    # Assert it routed to low_medium tier
    assert timings["enhancement_tier"] == "low_medium", "Should have routed to low_medium tier due to strict thresholds"
    print(f"\n[Low-Quality Path Verification]")
    print(f"  Timings: {timings}")


def test_preprocessor_exposes_ocr_friendly_image():
    img = np.full((240, 320, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (40, 60), (280, 190), (0, 0, 0), thickness=3)

    pipeline = PreprocessorPipeline()
    processed, metrics, timings = pipeline.process(img)

    assert isinstance(processed, np.ndarray)
    assert len(processed.shape) == 2
    assert hasattr(pipeline, "last_ocr_image")
    assert pipeline.last_ocr_image is not None
    assert pipeline.last_ocr_image.size > 0
    assert timings["total_preprocessing_ms"] > 0

