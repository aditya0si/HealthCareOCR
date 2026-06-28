import time
import cv2
import numpy as np
from .quality_assessor import QualityAssessor, QualityMetrics
from .paper_detector import PaperDetector
from .geometric import GeometricCorrector
from .enhancement import ImageEnhancer
from .super_resolution import SuperResolution

class PreprocessorPipeline:
    def __init__(self, config: dict = None):
        """
        Orchestrator for the adaptive preprocessing pipeline.
        Parses config dictionaries or uses defaults.
        """
        cfg = config if config is not None else {}
        
        # Configure Quality Assessor
        qa_cfg = cfg.get("quality_assessment", {})
        self.assessor = QualityAssessor(
            min_sharpness=qa_cfg.get("min_sharpness", 0.15),
            min_contrast=qa_cfg.get("min_contrast", 0.20),
            min_brightness=qa_cfg.get("min_brightness", 0.15),
            max_brightness=qa_cfg.get("max_brightness", 0.85),
            max_noise_ratio=qa_cfg.get("max_noise_ratio", 0.25),
            max_skew_angle=qa_cfg.get("max_skew_angle", 1.0)
        )
        
        # Configure Paper Detector
        pd_cfg = cfg.get("paper_detection", {})
        self.detector = PaperDetector(
            processing_max_dim=pd_cfg.get("processing_max_dim", 1024),
            canny_low=pd_cfg.get("canny_low", 50),
            canny_high=pd_cfg.get("canny_high", 150),
            gaussian_blur_ksize=pd_cfg.get("gaussian_blur_ksize", 5),
            enable_grabcut_fallback=pd_cfg.get("enable_grabcut_fallback", True),
            enable_hough_fallback=pd_cfg.get("enable_hough_fallback", True)
        )
        
        # Configure Geometric Corrector & Enhancer
        self.corrector = GeometricCorrector()
        self.enhancer = ImageEnhancer()
        
        # Configure Super Resolution
        sr_cfg = cfg.get("super_resolution", {})
        # Target a conservative OCR-friendly DPI by default (around 180 DPI)
        self.super_res = SuperResolution(
            model_path=sr_cfg.get("model_path", "models/FSRCNN_x2.pb"),
            min_dpi=sr_cfg.get("min_dpi", 180),
            max_dim_for_sr=sr_cfg.get("max_dim_for_sr", 2000)
        )
        
        bin_cfg = cfg.get("binarization", {})
        self.sauvola_ws = bin_cfg.get("window_size", 25)
        self.sauvola_k = bin_cfg.get("k_value", 0.2)
        self.last_enhanced_gray = None
        self.last_ocr_image = None

    def process(self, image: np.ndarray) -> tuple[np.ndarray, QualityMetrics, dict]:
        """
        Process a single raw input image adaptively based on its evaluated quality.
        Returns:
            - processed_image (np.ndarray): Binarized/enhanced final output (grayscale binary).
            - metrics (QualityMetrics): Assessed quality score card of the cropped document.
            - timings (dict): Execution time of each preprocessing stage in milliseconds.
        """
        timings = {}
        
        # 1. Bounding Box detection and Document Isolation (Paper detection)
        start_time = time.time()
        cropped = self.detector.detect_and_crop(image)
        timings["paper_detection_ms"] = (time.time() - start_time) * 1000

        # 1.5. Conditional Super Resolution
        start_time_sr = time.time()
        original_dpi = self.super_res.estimate_dpi(cropped)
        timings["estimated_dpi"] = original_dpi
        
        sr_triggered = self.super_res.should_upscale(cropped)
        if sr_triggered:
            cropped = self.super_res.upscale(cropped)
            timings["super_resolution_ms"] = (time.time() - start_time_sr) * 1000
            timings["super_resolution_run"] = True
        else:
            timings["super_resolution_ms"] = 0.0
            timings["super_resolution_run"] = False

        # Diagnostic logging: DPI and SR decision
        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug("Preprocessor: estimated_dpi=%.2f sr_triggered=%s", original_dpi, timings["super_resolution_run"])
        except Exception:
            pass

        # Convert to gray for quality assessment and binarization steps
        if len(cropped.shape) == 3:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped.copy()

        # 2. Quality Assessment on the cropped (and potentially upscaled) document
        start_time = time.time()
        metrics = self.assessor.assess(gray)
        timings["quality_assessment_ms"] = (time.time() - start_time) * 1000

        # 3. Geometric Correction (Deskew)
        start_time = time.time()
        if metrics.is_skewed:
            # Deskew on grayscale/color image as needed
            gray_straightened = self.corrector.correct_skew(gray, metrics.skew_angle)
            cropped_straightened = self.corrector.correct_skew(cropped, metrics.skew_angle)
            timings["deskew_ms"] = (time.time() - start_time) * 1000
        else:
            gray_straightened = gray
            cropped_straightened = cropped
            timings["deskew_ms"] = 0.0

        # 4. Adaptive Enhancement & Binarization
        start_time = time.time()
        
        # Determine Quality Tier for Adaptive Routing
        is_poor_quality = (metrics.is_blurry or 
                           metrics.is_low_contrast or 
                           metrics.is_noisy or 
                           metrics.is_too_dark)
        
        if is_poor_quality:
            # Low / Medium Quality Path: Requires full binarization & lighting corrections
            no_shadows = self.enhancer.remove_shadows(gray_straightened)
            contrast_enhanced = self.enhancer.apply_clahe(no_shadows)
            if metrics.is_noisy:
                denoised = self.enhancer.denoise(contrast_enhanced)
            else:
                denoised = contrast_enhanced
            final_bin = self.enhancer.binarize_sauvola(denoised, 
                                                        window_size=self.sauvola_ws, 
                                                        k=self.sauvola_k)
            self.last_enhanced_gray = denoised
            timings["enhancement_tier"] = "low_medium"
        else:
            # High Quality Path: Keep an OCR-friendly grayscale image instead of forcing a hard binary mask.
            _, final_bin = cv2.threshold(gray_straightened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.last_enhanced_gray = gray_straightened
            timings["enhancement_tier"] = "high"

        # Preserve a cleaner OCR input that retains stroke detail for the recognizers.
        if len(cropped_straightened.shape) == 3:
            self.last_ocr_image = cropped_straightened.copy()
        else:
            self.last_ocr_image = gray_straightened.copy()

        if self.last_ocr_image is not None and self.last_ocr_image.size > 0:
            if len(self.last_ocr_image.shape) == 2:
                self.last_ocr_image = cv2.cvtColor(self.last_ocr_image, cv2.COLOR_GRAY2BGR)

        timings["enhancement_ms"] = (time.time() - start_time) * 1000
        timings["total_preprocessing_ms"] = sum(v for k, v in timings.items() if k.endswith("_ms") and isinstance(v, float))

        return final_bin, metrics, timings
