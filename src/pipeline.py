import os
import time
import yaml
import torch
import numpy as np

# ==============================================================================
# Compatibility Monkeypatches for Surya & Transformers
# ==============================================================================
try:
    import transformers
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from surya.common.surya.decoder import SuryaDecoderConfig
    from surya.common.surya import SuryaModel
    
    # Patch 1: pad_token_id in SuryaDecoderConfig
    SuryaDecoderConfig.pad_token_id = 1

    # Patch 2: all_tied_weights_keys in PreTrainedModel
    if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
        def get_tied_keys(self):
            if not hasattr(self, "_all_tied_weights_keys_custom"):
                self._all_tied_weights_keys_custom = {}
            return self._all_tied_weights_keys_custom

        def set_tied_keys(self, value):
            self._all_tied_weights_keys_custom = value

        transformers.PreTrainedModel.all_tied_weights_keys = property(get_tied_keys, set_tied_keys)

    # Patch 3: _tie_or_clone_weights in PreTrainedModel
    if not hasattr(transformers.PreTrainedModel, "_tie_or_clone_weights"):
        def _tie_or_clone_weights(self, first_module, second_module):
            first_module.weight = second_module.weight
        transformers.PreTrainedModel._tie_or_clone_weights = _tie_or_clone_weights

    # Patch 4: tie_weights signature in SuryaModel
    old_tie_weights = SuryaModel.tie_weights
    def new_tie_weights(self, *args, **kwargs):
        try:
            return old_tie_weights(self)
        except TypeError:
            return old_tie_weights(self, *args, **kwargs)
    SuryaModel.tie_weights = new_tie_weights

    # Patch 5: ROPE_INIT_FUNCTIONS['default']
    def default_rope_init_fn(config=None, device=None, seq_len=None, layer_type=None):
        if hasattr(config, "standardize_rope_params"):
            try:
                config.standardize_rope_params()
            except Exception:
                pass
        base = 10000.0
        partial_rotary_factor = 1.0
        if hasattr(config, "rope_parameters") and config.rope_parameters:
            try:
                base = config.rope_parameters.get("rope_theta", 10000.0)
                partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
            except Exception:
                pass
        elif hasattr(config, "rope_scaling") and config.rope_scaling:
            try:
                base = config.rope_scaling.get("rope_theta", 10000.0)
                partial_rotary_factor = config.rope_scaling.get("partial_rotary_factor", 1.0)
            except Exception:
                pass
        elif hasattr(config, "rope_theta") and config.rope_theta is not None:
            base = config.rope_theta
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None and hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
            head_dim = config.hidden_size // config.num_attention_heads
        if head_dim is None:
            head_dim = 64
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = default_rope_init_fn
    print("[Surya Patch] Successfully applied compatibility patches for newer transformers version.")
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Failed to apply Surya monkeypatches: %s", e)
# ==============================================================================

from src.preprocessing import PreprocessorPipeline
from src.ocr import LayoutEngine, SuryaEngine, TrOCREngine, OCRRouter, VLMEngine
from src.postprocessing.confidence_scorer import ConfidenceScorer
from src.postprocessing.llm_corrector import LLMCorrector
from src.summarization.summarizer import ClinicalSummarizer
from src.utils.gpu_manager import GPUManager
from src.utils.timer import PipelineTimer
from src.utils.medical_dict import MedicalDictionary
import logging

logger = logging.getLogger(__name__)

class MedicalOCRPipeline:
    def __init__(self, config: dict = None):
        """
        End-to-end Medical OCR + Summarization Pipeline with VRAM Swapping.
        """
        self.config = self._load_config(config)
        device_config = self._get_config("device", "cuda")
        
        # Auto-select best GPU if just "cuda" is specified
        if device_config == "cuda":
            self.device = GPUManager.get_best_gpu_device()
        else:
            self.device = device_config
        
        # Update config with selected device
        if "hardware" not in self.config:
            self.config["hardware"] = {}
        self.config["device"] = self.device
        
        print(f"[Pipeline] Using device: {self.device}")
        
        self.keep_models_loaded = self._get_config("hardware.keep_models_loaded", False)
        self.parallel_ocr = self._get_config("hardware.parallel_ocr", True)
        self.parallel_ocr_workers = self._get_config("hardware.parallel_ocr_workers", 2)
        
        # 1. Preprocessing components
        self.preprocessor = PreprocessorPipeline(self.config)
        
        # 2. Shared Medical Dictionary (loaded once on CPU)
        self.dictionary = MedicalDictionary()
        self.scorer = ConfidenceScorer(self.dictionary)
        
        # 3. OCR Engines (Lazy-loaded during execution)
        self.layout = None
        self.surya = None
        self.trocr = None
        self.router = None
        self.vlm = None
        
        # 4. LLM Components (Lazy-loaded during execution)
        self.corrector = LLMCorrector(device=self.device)
        self.summarizer = ClinicalSummarizer(device=self.device)

    def _load_config(self, config):
        if isinstance(config, str) and os.path.exists(config):
            try:
                with open(config, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    return loaded if isinstance(loaded, dict) else {}
            except Exception as e:
                print(f"Warning: failed to load YAML config {config}: {e}")
                return {}
        return config if isinstance(config, dict) else {}

    def _get_config(self, key, default=None):
        parts = key.split(".") if isinstance(key, str) else []
        value = self.config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value if value is not None else default

    def load_ocr_engines(self):
        """
        Loads Layout, Surya, and TrOCR models, placing only the detector on CUDA initially.
        Recognizer engines are loaded on CPU and move to CUDA dynamically only when needed.
        """
        if self.layout is None:
            print("Loading LayoutLMv3 engine...")
            self.layout = LayoutEngine(self.config)
            
        if self.surya is None or self.trocr is None or self.router is None:
            print("Loading recognizer engines (Surya2, TrOCR)...")
            cpu_config = self.config.copy()
            cpu_config["device"] = "cpu"
            if self.surya is None:
                self.surya = SuryaEngine(cpu_config)
            if self.trocr is None:
                self.trocr = TrOCREngine(cpu_config)
            if self.router is None:
                if self.surya is None or self.trocr is None:
                    raise RuntimeError("Unable to initialize OCR router because one or more OCR engines failed to load.")
                self.router = OCRRouter(self.surya, self.trocr, self.config)

        if self.router is None:
            if self.surya is None or self.trocr is None:
                raise RuntimeError("OCR router could not be created; Surya or TrOCR engine is missing.")
            self.router = OCRRouter(self.surya, self.trocr, self.config)

        if hasattr(self.layout, "detector") and self.layout.detector is not None:
            if hasattr(self.layout.detector, "model") and self.layout.detector.model is not None:
                self.layout.detector.model.to(self.device)

    def unload_ocr_engines(self):
        """
        Moves Layout, Surya, and TrOCR models to CPU memory to free GPU VRAM.
        """
        if self.layout is None:
            return
            
        print("Moving OCR engines to CPU...")
        if hasattr(self.layout, "detector") and self.layout.detector is not None:
            if hasattr(self.layout.detector, "model") and self.layout.detector.model is not None:
                self.layout.detector.model.to("cpu")
        if hasattr(self.surya, "foundation_predictor") and self.surya.foundation_predictor is not None:
            if hasattr(self.surya.foundation_predictor, "model") and self.surya.foundation_predictor.model is not None:
                self.surya.foundation_predictor.model.to("cpu")
        if hasattr(self.trocr, "model") and self.trocr.model is not None:
            try:
                is_quantized = getattr(self.trocr.model, "is_quantized", False) or \
                               (hasattr(self.trocr.model, "config") and getattr(self.trocr.model.config, "quantization_config", None) is not None)
                if not is_quantized:
                    self.trocr.model.to("cpu")
                else:
                    print("[Pipeline] TrOCR is quantized; skipping CPU offloading.")
            except Exception as e:
                print(f"[Pipeline] Failed to move TrOCR model to CPU: {e}")
            
        GPUManager.clean_memory()

    def preprocess_image(self, image: np.ndarray):
        """Run only the preprocessing stage and return results."""
        timer = PipelineTimer()
        with timer("Preprocessing"):
            preprocessed_img, prep_metrics, prep_timings = self.preprocessor.process(image)
        summary = timer.get_summary()
        return preprocessed_img, prep_metrics, prep_timings, summary

    def run_ocr(self, preprocessed_img: np.ndarray):
        """Run layout detection and OCR on preprocessed input."""
        timer = PipelineTimer()
        with timer("Layout & OCR"):
            self.load_ocr_engines()
            if self.router is None:
                raise RuntimeError("OCR router was not initialized after engine loading.")
            regions = self.layout.detect_regions(preprocessed_img)
            ocr_image = self.preprocessor.last_enhanced_gray if getattr(self.preprocessor, "last_enhanced_gray", None) is not None else preprocessed_img
            logger.debug("Selected OCR image: last_enhanced_gray present=%s", 
                         getattr(self.preprocessor, "last_enhanced_gray", None) is not None)
            if ocr_image is None:
                ocr_image = preprocessed_img
            ocr_results = self.router.process_document(ocr_image, regions)
            raw_text = "\n".join([r.get("text", "") for r in ocr_results if r.get("text", "")])
            if not raw_text.strip():
                logger.debug("Initial OCR returned no text; attempting fallback image path")
                fallback_image = self.preprocessor.last_enhanced_gray if getattr(self.preprocessor, "last_enhanced_gray", None) is not None else preprocessed_img
                if fallback_image is not None and fallback_image is not ocr_image:
                    ocr_results = self.router.process_document(fallback_image, regions)
                    raw_text = "\n".join([r.get("text", "") for r in ocr_results if r.get("text", "")])
                    logger.debug("Fallback OCR produced text length=%d", len(raw_text))
        return raw_text, timer.get_summary()

    def process_image(self, image: np.ndarray, skip_summarization: bool = False, unload_after: bool = True, engine_mode: str = None) -> dict:
        """
        Run the complete pipeline on a single input image.
        Supports Track A (VLM) and Track B (Multi-Stage).
        """
        timer = PipelineTimer()
        mode = engine_mode or self._get_config("engine_mode", "track_b")
        
        if mode == "track_a":
            # Track A: Unified VLM Pipeline
            with timer("Layout & OCR"):
                if self.vlm is None:
                    self.vlm = VLMEngine(self.config)
                # Recognizing and summarizing is a single unified pass
                summary_json, vlm_metrics = self.vlm.recognize_and_summarize(image)
                raw_text = summary_json.get("raw_ocr_text", "")
                corrected_text = raw_text

            vlm_unload_time = 0.0
            if unload_after and not self.keep_models_loaded:
                with timer("VRAM Swap (Unload OCR)"):
                    start_unload = time.time()
                    self.vlm.unload()
                    vlm_unload_time = time.time() - start_unload
            
            all_timings = timer.get_summary()
            all_timings["prep_total_preprocessing"] = 0.0
            all_timings["Preprocessing"] = 0.0
            
            return {
                "raw_ocr_text": raw_text,
                "corrected_ocr_text": corrected_text,
                "clinical_summary": summary_json,
                "quality_metrics": {
                    "sharpness": 0.0,
                    "contrast": 0.0,
                    "brightness": 0.0,
                    "noise_ratio": 0.0,
                    "skew_angle": 0.0,
                    "is_blurry": False,
                    "is_low_contrast": False,
                    "is_too_dark": False,
                    "is_too_bright": False,
                    "is_noisy": False,
                    "is_skewed": False
                },
                "timings": all_timings
            }

        # Track B: Existing Multi-stage Pipeline
        # Stage 1: Preprocessing
        with timer("Preprocessing"):
            preprocessed_img, prep_metrics, prep_timings = self.preprocessor.process(image)
            
        # Stage 2: Layout & OCR Execution
        with timer("Layout & OCR"):
            self.load_ocr_engines()
            if self.router is None:
                raise RuntimeError("OCR router was not initialized after engine loading.")
            regions = self.layout.detect_regions(preprocessed_img)
            ocr_image = self.preprocessor.last_enhanced_gray if getattr(self.preprocessor, "last_enhanced_gray", None) is not None else preprocessed_img
            if ocr_image is None:
                ocr_image = preprocessed_img
            ocr_results = self.router.process_document(ocr_image, regions)
            raw_text = "\n".join([r.get("text", "") for r in ocr_results if r.get("text", "")])
            if not raw_text.strip():
                fallback_image = self.preprocessor.last_enhanced_gray if getattr(self.preprocessor, "last_enhanced_gray", None) is not None else preprocessed_img
                if fallback_image is not None and fallback_image is not ocr_image:
                    ocr_results = self.router.process_document(fallback_image, regions)
                    raw_text = "\n".join([r.get("text", "") for r in ocr_results if r.get("text", "")])
            
        # Stage 3: VRAM Swap (Unload OCR models)
        if unload_after and not self.keep_models_loaded:
            with timer("VRAM Swap (Unload OCR)"):
                self.unload_ocr_engines()
            
        # Stage 4: Confidence Scoring
        with timer("Confidence Scoring"):
            marked_text, flagged_words = self.scorer.process_text(raw_text)
            
        # Stage 5: Conditional LLM Correction
        corrected_text = raw_text
        if len(flagged_words) > 0:
            with timer("LLM Correction"):
                # Load Phi-4 model (either from scratch or move CPU -> CUDA)
                self.corrector.load_model()
                corrected_text = self.corrector.correct_text(marked_text)
        else:
            print("No low-confidence words flagged. Skipping LLM correction.")
            
        summary_json = {}
        if not skip_summarization:
            # Stage 6: Clinical Summarization
            with timer("Clinical Summarization"):
                # Always ensure the LLM is loaded/moved to CUDA via corrector
                self.corrector.load_model()
                
                # Share loaded Phi-4 model with summarizer
                self.summarizer.model = self.corrector.model
                self.summarizer.tokenizer = self.corrector.tokenizer
                    
                summary_json = self.summarizer.summarize_text(corrected_text)
                
            # Stage 7: VRAM Swap (Unload LLM)
            with timer("VRAM Swap (Unload LLM)"):
                # Move the shared LLM to CPU
                self.corrector.unload()
                self.summarizer.model = None
                self.summarizer.tokenizer = None
                GPUManager.clean_memory()
        else:
            # If we skipped summarization but loaded the LLM for correction, we must still unload it!
            if len(flagged_words) > 0:
                with timer("VRAM Swap (Unload LLM)"):
                    self.corrector.unload()
                    self.summarizer.model = None
                    self.summarizer.tokenizer = None
                    GPUManager.clean_memory()
            
        # Gather all timings
        all_timings = timer.get_summary()
        # Merge preprocessing internal timings
        for k, v in prep_timings.items():
            if k.endswith("_ms"):
                all_timings[f"prep_{k[:-3]}"] = v / 1000.0  # convert to seconds
            else:
                all_timings[f"prep_{k}"] = v
                
        return {
            "raw_ocr_text": raw_text,
            "corrected_ocr_text": corrected_text,
            "clinical_summary": summary_json,
            "quality_metrics": {
                "sharpness": prep_metrics.sharpness,
                "contrast": prep_metrics.contrast,
                "brightness": prep_metrics.brightness,
                "noise_ratio": prep_metrics.noise_ratio,
                "skew_angle": prep_metrics.skew_angle,
                "is_blurry": prep_metrics.is_blurry,
                "is_low_contrast": prep_metrics.is_low_contrast,
                "is_too_dark": prep_metrics.is_too_dark,
                "is_too_bright": prep_metrics.is_too_bright,
                "is_noisy": prep_metrics.is_noisy,
                "is_skewed": prep_metrics.is_skewed
            },
            "timings": all_timings
        }
