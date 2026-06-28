# PLAN: Fix OCR Engine Lazy-Loading NoneType Error

## SECTION A — GOAL DEFINITION

1. **What is being built or changed?**
   - Modify the lazy-loading mechanism in `src/pipeline.py`'s `load_ocr_engines` function to load layout and recognition models independently. This ensures that if any part of the initialization fails, subsequent calls can recover and fully initialize the remaining engine components (such as `self.router`).
2. **What does "done" look like?**
   - The FastAPI backend process starts and executes batch OCR jobs successfully without throwing `'NoneType' object has no attribute 'process_document'` errors on images.
   - All OCR unit tests pass successfully.
3. **What is explicitly out of scope for this task?**
   - Modifying model architectures, hyperparameters, or routing threshold logic.
   - Modifying preprocessing steps or LLM summarization components.

---

## SECTION B — TECH STACK

- **Languages/Runtime**: Python 3.12, PyTorch, CUDA
- **OCR Libraries**: Surya OCR, TrOCR
- **Files Modified**:
  - [src/pipeline.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/src/pipeline.py)

---

## SECTION C — SESSION MODULARIZATION

### Session 1: Robust Engine Loading Fix
- **Objective**: Refactor `load_ocr_engines` in [src/pipeline.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/src/pipeline.py) to check and load components (`self.layout`, `self.surya`, `self.trocr`, `self.router`) individually rather than all-or-nothing based solely on `self.layout`.
- **Scope**: [src/pipeline.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/src/pipeline.py)
- **Output**: Robust `load_ocr_engines` implementation.
- **Connects to**: Session 2.
- **Failure Surface**: Python syntax errors or variable scope issues.

### Session 2: Verification & Test Execution
- **Objective**: Run automated pytest unit tests and sanity checks to confirm the fix works.
- **Scope**: [tests/test_ocr.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/tests/test_ocr.py) and backend server execution.
- **Output**: Test suite passes cleanly.
- **Connects to**: Done.
- **Failure Surface**: CUDA out-of-memory or model download failures.

---

## SECTION D — PROGRESS CHECKLIST

- [x] Session 1: Robust Engine Loading Fix
  - [x] Edit `load_ocr_engines` in [src/pipeline.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/src/pipeline.py) to load engines independently.
- [x] Session 2: Verification & Test Execution
  - [x] Execute `pytest tests/test_ocr.py` inside the virtual environment.
  - [x] Verify server runs successfully.
