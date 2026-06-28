"""
Batch Runner — Parallel Preprocessing + Serial GPU OCR Pipeline.

Architecture:
  Phase 1: ThreadPoolExecutor preprocesses ALL images concurrently on CPU.
  Phase 2: Preprocessed images feed serially into GPU OCR (single GPU constraint).

This eliminates CPU wait time from the critical path and keeps the GPU 100% utilized.
"""

import os
import time
import uuid
import cv2
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


@dataclass
class ImageResult:
    """Result container for a single processed image."""
    filename: str
    filepath: str
    relative_path: str
    raw_ocr_text: str = ""
    corrected_ocr_text: str = ""
    quality_metrics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    error: Optional[str] = None
    preprocessing_time_s: float = 0.0
    ocr_time_s: float = 0.0
    total_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "relative_path": self.relative_path,
            "raw_ocr_text": self.raw_ocr_text,
            "corrected_ocr_text": self.corrected_ocr_text,
            "quality_metrics": self.quality_metrics,
            "timings": self.timings,
            "error": self.error,
            "preprocessing_time_s": round(self.preprocessing_time_s, 3),
            "ocr_time_s": round(self.ocr_time_s, 3),
            "total_time_s": round(self.total_time_s, 3),
        }


@dataclass
class BatchStatus:
    """Tracks live batch processing state."""
    batch_id: str
    total_images: int = 0
    processed: int = 0
    preprocessing_done: int = 0
    ocr_done: int = 0
    failed: int = 0
    status: str = "pending"  # pending | preprocessing | ocr | complete | error
    current_image: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    results: list = field(default_factory=list)
    directories: list = field(default_factory=list)

    def to_dict(self) -> dict:
        elapsed = (self.end_time or time.time()) - self.start_time if self.start_time else 0
        return {
            "batch_id": self.batch_id,
            "total_images": self.total_images,
            "processed": self.processed,
            "preprocessing_done": self.preprocessing_done,
            "ocr_done": self.ocr_done,
            "failed": self.failed,
            "status": self.status,
            "current_image": self.current_image,
            "elapsed_seconds": round(elapsed, 2),
            "directories": self.directories,
        }


def discover_images(directories: list[str]) -> list[tuple[str, str]]:
    """
    Recursively discover all image files in the given directories.
    Returns list of (absolute_path, relative_display_path) tuples.
    """
    images = []
    for dir_path in directories:
        if not os.path.isdir(dir_path):
            continue
        base_name = os.path.basename(dir_path)
        for root, _dirs, files in os.walk(dir_path):
            for f in sorted(files):
                if f.lower().endswith(IMAGE_EXTENSIONS) and not f.startswith("."):
                    abs_path = os.path.join(root, f)
                    rel_path = os.path.join(base_name, os.path.relpath(abs_path, dir_path))
                    images.append((abs_path, rel_path))
    return images


def _preprocess_single(preprocessor, img_path: str, rel_path: str) -> dict:
    """
    CPU-bound preprocessing of a single image. Safe for ThreadPoolExecutor.
    Returns dict with preprocessed data or error.
    """
    start = time.time()
    try:
        img = cv2.imread(img_path)
        if img is None:
            return {
                "path": img_path,
                "rel_path": rel_path,
                "error": f"Failed to read image: {img_path}",
                "time": time.time() - start,
            }

        preprocessed_img, metrics, prep_timings = preprocessor.process(img)

        # Store the enhanced gray from preprocessor (needed for OCR)
        enhanced_gray = getattr(preprocessor, "last_enhanced_gray", None)
        if enhanced_gray is not None:
            enhanced_gray = enhanced_gray.copy()

        return {
            "path": img_path,
            "rel_path": rel_path,
            "preprocessed_img": preprocessed_img,
            "enhanced_gray": enhanced_gray,
            "metrics": metrics,
            "prep_timings": prep_timings,
            "error": None,
            "time": time.time() - start,
        }
    except Exception as e:
        return {
            "path": img_path,
            "rel_path": rel_path,
            "error": str(e),
            "time": time.time() - start,
        }


class BatchRunner:
    """
    Parallel batch processor for the Medical OCR Pipeline.

    Strategy:
      1. All CPU preprocessing runs in parallel via ThreadPoolExecutor
      2. GPU OCR runs serially (single GPU constraint)
      3. Models loaded ONCE and kept in VRAM for the entire batch
    """

    def __init__(self, pipeline, max_preprocessing_workers: int = None):
        self.pipeline = pipeline
        self.max_workers = max_preprocessing_workers or min(os.cpu_count() or 4, 8)
        self._batches: dict[str, BatchStatus] = {}
        self._lock = threading.Lock()

    def get_batch_status(self, batch_id: str) -> Optional[BatchStatus]:
        with self._lock:
            return self._batches.get(batch_id)

    def get_all_batches(self) -> list[dict]:
        with self._lock:
            return [b.to_dict() for b in self._batches.values()]

    def run_batch(self, directories: list[str]) -> str:
        """
        Start a batch processing run. Returns batch_id immediately.
        Processing happens in a background thread.
        """
        batch_id = str(uuid.uuid4())[:8]
        images = discover_images(directories)

        batch = BatchStatus(
            batch_id=batch_id,
            total_images=len(images),
            status="pending",
            start_time=time.time(),
            directories=[os.path.basename(d) for d in directories],
        )

        with self._lock:
            self._batches[batch_id] = batch

        # Run in background thread
        thread = threading.Thread(
            target=self._run_batch_sync,
            args=(batch_id, images),
            daemon=True,
        )
        thread.start()

        return batch_id

    def _run_batch_sync(self, batch_id: str, images: list[tuple[str, str]]):
        """
        Synchronous batch execution (runs in background thread).
        Phase 1: Parallel CPU preprocessing
        Phase 2: Serial GPU OCR
        """
        batch = self._batches[batch_id]

        if not images:
            batch.status = "complete"
            batch.end_time = time.time()
            return

        try:
            # ============================================================
            # PHASE 1: PARALLEL CPU PREPROCESSING
            # ============================================================
            batch.status = "preprocessing"
            print(f"\n[Batch {batch_id}] Phase 1: Preprocessing {len(images)} images "
                  f"with {self.max_workers} workers...")

            from src.preprocessing import PreprocessorPipeline

            # Create per-worker preprocessor instances to avoid shared state issues
            preprocessors = [
                PreprocessorPipeline(self.pipeline.config)
                for _ in range(self.max_workers)
            ]

            preprocessed_queue = []  # Collect results in order
            worker_idx = 0

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for img_path, rel_path in images:
                    prep = preprocessors[worker_idx % len(preprocessors)]
                    worker_idx += 1
                    future = executor.submit(
                        _preprocess_single, prep, img_path, rel_path
                    )
                    futures[future] = (img_path, rel_path)

                for future in as_completed(futures):
                    result = future.result()
                    preprocessed_queue.append(result)
                    with self._lock:
                        batch.preprocessing_done += 1
                        batch.current_image = f"Preprocessing: {os.path.basename(result['path'])}"

            print(f"[Batch {batch_id}] Phase 1 complete: {len(preprocessed_queue)} images preprocessed")

            # ============================================================
            # PHASE 2: SERIAL GPU OCR
            # ============================================================
            batch.status = "ocr"
            print(f"[Batch {batch_id}] Phase 2: Serial GPU OCR...")

            # Load OCR engines ONCE, keep them in VRAM for entire batch
            self.pipeline.load_ocr_engines()
            if self.pipeline.router is None:
                raise RuntimeError("OCR router was not initialized before batch OCR execution.")

            for prep_result in preprocessed_queue:
                img_path = prep_result["path"]
                rel_path = prep_result["rel_path"]
                filename = os.path.basename(img_path)
                raw_text = ""

                with self._lock:
                    batch.current_image = f"OCR: {filename}"

                result = ImageResult(
                    filename=filename,
                    filepath=img_path,
                    relative_path=rel_path,
                )

                if prep_result.get("error"):
                    result.error = prep_result["error"]
                    result.preprocessing_time_s = prep_result.get("time", 0)
                    with self._lock:
                        batch.results.append(result)
                        batch.failed += 1
                        batch.processed += 1
                    continue

                result.preprocessing_time_s = prep_result["time"]

                # Build quality metrics dict
                metrics = prep_result["metrics"]
                result.quality_metrics = {
                    "sharpness": metrics.sharpness,
                    "contrast": metrics.contrast,
                    "brightness": metrics.brightness,
                    "noise_ratio": metrics.noise_ratio,
                    "skew_angle": metrics.skew_angle,
                    "is_blurry": metrics.is_blurry,
                    "is_low_contrast": metrics.is_low_contrast,
                    "is_too_dark": metrics.is_too_dark,
                    "is_too_bright": metrics.is_too_bright,
                    "is_noisy": metrics.is_noisy,
                    "is_skewed": metrics.is_skewed,
                }

                # Store preprocessing timings
                prep_timings = prep_result.get("prep_timings", {})

                # Run GPU OCR
                ocr_start = time.time()
                raw_text = ""
                try:
                    preprocessed_img = prep_result["preprocessed_img"]
                    enhanced_gray = prep_result.get("enhanced_gray")

                    regions = self.pipeline.layout.detect_regions(preprocessed_img)
                    ocr_image = enhanced_gray if enhanced_gray is not None else preprocessed_img
                    ocr_results = self.pipeline.router.process_document(ocr_image, regions)
                    raw_text = "\n".join(
                        [r.get("text", "") for r in ocr_results if r.get("text", "")]
                    )
                    result.raw_ocr_text = raw_text
                    result.corrected_ocr_text = raw_text  # No LLM correction active
                    result.ocr_time_s = time.time() - ocr_start

                except Exception as e:
                    raw_text = ""
                    result.error = f"OCR error: {str(e)}"
                    result.ocr_time_s = time.time() - ocr_start
                    with self._lock:
                        batch.failed += 1

                result.total_time_s = result.preprocessing_time_s + result.ocr_time_s

                # Merge timings
                all_timings = {}
                for k, v in prep_timings.items():
                    if isinstance(v, (int, float)):
                        all_timings[f"prep_{k}"] = v
                all_timings["preprocessing_s"] = round(result.preprocessing_time_s, 3)
                all_timings["ocr_s"] = round(result.ocr_time_s, 3)
                all_timings["total_s"] = round(result.total_time_s, 3)
                result.timings = all_timings

                with self._lock:
                    batch.results.append(result)
                    batch.ocr_done += 1
                    batch.processed += 1

                print(f"  [{batch.processed}/{batch.total_images}] {filename} "
                      f"— prep: {result.preprocessing_time_s:.2f}s, "
                      f"ocr: {result.ocr_time_s:.2f}s, "
                      f"text: {len(raw_text)} chars")

            # Unload OCR engines after batch
            self.pipeline.unload_ocr_engines()

            # Run LLM Correction Phase sequentially using VRAM swap
            llm_loaded = False
            try:
                for result in batch.results:
                    if result.error or not result.raw_ocr_text.strip():
                        continue
                    
                    # Run confidence scorer
                    marked_text, flagged_words = self.pipeline.scorer.process_text(result.raw_ocr_text)
                    if len(flagged_words) > 0:
                        if not llm_loaded:
                            print(f"[Batch {batch_id}] Loading LLM for corrections...")
                            self.pipeline.corrector.load_model()
                            llm_loaded = True
                        
                        corr_start = time.time()
                        corrected = self.pipeline.corrector.correct_text(marked_text)
                        result.corrected_ocr_text = corrected
                        corr_time = time.time() - corr_start
                        result.ocr_time_s += corr_time
                        result.total_time_s += corr_time
                        result.timings["llm_correction_s"] = round(corr_time, 3)
                        result.timings["total_s"] = round(result.total_time_s, 3)
            finally:
                if llm_loaded:
                    print(f"[Batch {batch_id}] Unloading LLM after corrections...")
                    self.pipeline.corrector.unload()
                    from src.utils.gpu import GPUManager
                    GPUManager.clean_memory()

            batch.status = "complete"
            batch.end_time = time.time()
            elapsed = batch.end_time - batch.start_time

            print(f"\n[Batch {batch_id}] COMPLETE — "
                  f"{batch.processed} images in {elapsed:.1f}s "
                  f"({batch.failed} failed)")

        except Exception as e:
            batch.status = "error"
            batch.end_time = time.time()
            batch.current_image = f"Error: {str(e)}"
            print(f"[Batch {batch_id}] FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
