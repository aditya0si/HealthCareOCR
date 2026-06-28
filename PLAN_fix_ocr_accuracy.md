# PLAN: Fix OCR Accuracy

## SECTION A — GOAL DEFINITION

The goal of this task is to fix the severe OCR transcription accuracy bottleneck (currently ~80% Character Error Rate) in the HealthCareOCR pipeline. 

### 1. What is being built or changed?
* Modify `src/preprocessing/__init__.py` to store the intermediate enhanced (shadow-removed, contrast-adjusted, deskewed) grayscale image (`self.last_enhanced_gray`) during the preprocessing pipeline before binarization.
* Modify `src/pipeline.py` to pass this non-binarized `last_enhanced_gray` image to `self.router.process_document` for OCR transcription, while still using the binarized `preprocessed_img` for layout/line detection.
* Update `configs/pipeline_config.yaml` if any hyperparameters need adjustment.

### 2. What does "done" look like?
* Running `scripts/benchmark.py` over the test prescription dataset achieves an average Character Error Rate (CER) of <= 15% (down from ~80%).
* All unit tests (`pytest`) pass successfully.
* Re-run the full evaluation split to confirm no regressions in peak VRAM (<= 4.0 GB) and latency.

### 3. What is explicitly out of scope?
* Modifying the layout detection logic.
* Training new LoRA adapters or models.
* Altering the clinical summarizer LLM prompt or output format.

---

## SECTION B — TECH STACK

* **Runtime & Frameworks**: Python 3.12, PyTorch 2.12.1+cu132, OpenCV (contrib-headless)
* **OCR Engines**: Surya OCR, TrOCR-base
* **Files Modified**:
  * [src/preprocessing/__init__.py](file:///C:/Users/oliad/Desktop/HealthCareOCR/src/preprocessing/__init__.py)
  * [src/pipeline.py](file:///C:/Users/oliad/Desktop/HealthCareOCR/src/pipeline.py)
* **Files Touched/Run**:
  * [scripts/benchmark.py](file:///C:/Users/oliad/Desktop/HealthCareOCR/scripts/benchmark.py)
  * [tests/test_preprocessing.py](file:///C:/Users/oliad/Desktop/HealthCareOCR/tests/test_preprocessing.py)

---

## SECTION C — SESSION MODULARIZATION

### Session 1: Preprocessor & Pipeline Integration
* **Objective**: Store the enhanced non-binarized image in the preprocessor and route it to the OCR engines.
* **Scope**: `src/preprocessing/__init__.py`, `src/pipeline.py`
* **Output**:
  * `PreprocessorPipeline` stores `self.last_enhanced_gray`.
  * `MedicalOCRPipeline` passes `self.preprocessor.last_enhanced_gray` to the OCR router.
* **Connects to**: Session 2 (Verification).
* **Failure Surface**: Mismatch in image size/shape between binary layout detection and the grayscale image if super-resolution or cropping behaves differently (prevented by using the exact same coordinates and deskew path).

### Session 2: Benchmarking & Validation
* **Objective**: Run the test suite and benchmark runner to verify the accuracy improvement.
* **Scope**: `scripts/benchmark.py`, unit tests
* **Output**:
  * Running `.\venv\Scripts\python.exe scripts/benchmark.py --samples 20` reports average CER <= 15% and peak VRAM <= 4.0 GB.
  * All 16 unit tests pass.
* **Failure Surface**: Preprocessing test failures or regression in latency/VRAM constraints.

---

## SECTION D — PROGRESS CHECKLIST

- [x] **Session 1: Preprocessor & Pipeline Integration**
  - [x] Add `self.last_enhanced_gray` assignment to `src/preprocessing/__init__.py` (both poor and high quality paths)
  - [x] Modify `src/pipeline.py` to route `self.preprocessor.last_enhanced_gray` to `self.router.process_document`
- [x] **Session 2: Benchmarking & Validation**
  - [x] Run `pytest` inside the virtual environment to ensure all tests pass without regression
  - [x] Run `scripts/benchmark.py` over 20 test samples and verify average CER <= 15%
  - [x] Confirm Peak VRAM is still <= 4.0 GB and OOM rate is 0.0%
