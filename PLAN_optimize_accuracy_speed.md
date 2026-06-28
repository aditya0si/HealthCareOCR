# PLAN: Optimize OCR Pipeline Accuracy and Speed

## SECTION A — GOAL DEFINITION

1. **What is being built or changed?**
   - **Coordinate Alignment Fix**: Resolve the alignment issue by routing the deskewed and contrast-enhanced grayscale image (`last_enhanced_gray`) to the OCR engines in both the single-image pipeline and the batch runner.
   - **Preprocessing Speed Optimization**:
     - Bypass the slow FSRCNN deep learning upscaling (which takes ~3.5s) with OpenCV's high-quality cubic interpolation (`cv2.resize` with `cv2.INTER_CUBIC`), reducing time to <5ms.
     - Replace the slow scikit-image Sauvola binarization (which takes ~2.4s) with OpenCV's fast adaptive thresholding (`cv2.adaptiveThreshold`), reducing time to <5ms.
   - **LLM Reintegration**: Re-enable the Phi-4 LLM post-correction and clinical summarization stages to restore handwriting spelling corrections and structured summary generation (peak VRAM will be kept under 3.5GB using VRAM swapping, which fits comfortably on the user's 8GB RTX 5060).
2. **What does "done" look like?**
   - The end-to-end processing pipeline runs in **≤ 7 seconds** per image (down from 15-25s).
   - OCR transcription accuracy is significantly improved, producing readable medical terms and structured summaries instead of raw gibberish.
   - Both unit tests and manual execution complete successfully on the GPU without OOM errors.
3. **What is explicitly out of scope for this task?**
   - Fine-tuning the models or training new weights.
   - Modifying layout detection models.

---

## SECTION B — TECH STACK

- **Languages/Runtime**: Python 3.12, PyTorch, CUDA
- **Libraries**: OpenCV, transformers, accelerate, bitsandbytes
- **Files Modified**:
  - [src/pipeline.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/src/pipeline.py)
  - [src/preprocessing/__init__.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/src/preprocessing/__init__.py)
  - [src/preprocessing/super_resolution.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/src/preprocessing/super_resolution.py)
  - [src/batch_runner.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/src/batch_runner.py)
  - [app.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/app.py)

---

## SECTION C — SESSION MODULARIZATION

### Session 1: Alignment & Preprocessing Speed Fix
- **Objective**: Fix the coordinates alignment bug and replace slow CPU preprocessing operations with fast OpenCV alternatives.
- **Scope**: `src/preprocessing/__init__.py`, `src/preprocessing/super_resolution.py`, `src/pipeline.py`, `src/batch_runner.py`
- **Output**: Preprocessing pipeline executes in <200ms (down from 6s) and routes deskewed enhanced images to the OCR engines.
- **Connects to**: Session 2.
- **Failure Surface**: Loss of layout detection accuracy if fast adaptive thresholding behaves differently than Sauvola (prevented by tuning threshold params).

### Session 2: LLM Reintegration
- **Objective**: Restore the Phi-4 LLM corrector and summarizer stage, and update the web UI to re-enable summary/corrected text tabs.
- **Scope**: `src/pipeline.py`, `app.py`
- **Output**: Full end-to-end pipeline running under 7 seconds with structured clinical summaries.
- **Connects to**: Done.
- **Failure Surface**: VRAM OOM error if CUDA memory is not freed cleanly during swaps.

---

## SECTION D — PROGRESS CHECKLIST

- [ ] Session 1: Alignment & Preprocessing Speed Fix
  - [ ] Add config override/parameter to bypass FSRCNN upscaling with `cv2.resize` in `src/preprocessing/super_resolution.py`.
  - [ ] Replace `threshold_sauvola` with `cv2.adaptiveThreshold` in `src/preprocessing/enhancement.py` or `src/preprocessing/__init__.py`.
  - [ ] Ensure `last_enhanced_gray` is correctly used in `src/pipeline.py` and coordinates align.
- [ ] Session 2: LLM Reintegration
  - [ ] Uncomment the LLM stages in `src/pipeline.py` according to `REINTEGRATION_GUIDE.md`.
  - [ ] Re-enable LLM correction and summarizer tabs and JS handlers in `app.py`.
  - [ ] Restart the server, test end-to-end flow, and verify execution time (≤ 7s) and output quality.
