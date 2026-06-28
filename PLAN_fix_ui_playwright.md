# PLAN: Fix UI Issues using Playwright Automation

## SECTION A — GOAL DEFINITION

The goal of this task is to resolve model loading and execution issues in the Healthcare OCR project, verify the web UI flow using Playwright, and verify that the test suite and evaluation scripts run successfully.

### 1. What is being built or changed?
* **Model Initialization Fix**: Resolve the `AttributeError: weight is not an nn.Module` error in `src/ocr/trocr_engine.py` by switching from incompatible `bitsandbytes` 4-bit quantization to standard `float16` precision for the small TrOCR model.
* **Server Verification**: Start the FastAPI server (`app.py`) on port 7860.
* **UI Verification via Playwright**: Use the browser subagent to interact with the local web interface, upload a test image, run the OCR process, and verify that the UI displays results successfully.
* **Evaluation Script and Tests**: Execute the evaluation script `scripts/playwright_ocr_eval.py` and run the `pytest` test suite to ensure complete correctness.

### 2. What does "done" look like?
* The local web app runs on port 7860.
* The Playwright evaluation script completes successfully and generates the evaluation report.
* All tests in the test suite pass.

### 3. What is explicitly out of scope?
* Modifying the post-processing or summarization logic unless direct bugs are found.
* Upgrading dependencies globally.

---

## SECTION B — TECH STACK

* **Python**: 3.12.6
* **Libraries**: Playwright (browser subagent), FastAPI, Uvicorn, PyTorch, Transformers, BitsAndBytes, Pytest
* **Files Modified**:
  * `src/ocr/trocr_engine.py` (TrOCR loading precision)
  * `PLAN_fix_ui_playwright.md` (Project plan)

---

## SECTION C — SESSION MODULARIZATION

### Session 1: Fix Model Loading
* **Objective**: Modify `src/ocr/trocr_engine.py` to use `float16` precision instead of incompatible 4-bit quantization.
* **Scope**: `src/ocr/trocr_engine.py`
* **Output**: Correctly loading `TrOCREngine` without errors.
* **Connects to**: Session 2 (UI verification).
* **Failure Surface**: VRAM limits, though the model is extremely small (~60M parameters, ~120MB).

### Session 2: Web App and UI Verification
* **Objective**: Run `app.py` and use the Playwright browser subagent to verify the UI works end-to-end.
* **Scope**: `app.py` server execution, Playwright browser subagent.
* **Output**: Validated single-image execution through the UI web page.
* **Connects to**: Session 3 (eval script and test suite).
* **Failure Surface**: Port conflicts, UI element selector issues.

### Session 3: Running Scripts and Test Suite
* **Objective**: Run `scripts/playwright_ocr_eval.py` and verify all tests pass.
* **Scope**: `scripts/playwright_ocr_eval.py`, `tests/`
* **Output**: Verified execution of scripts and a fully green test suite.
* **Failure Surface**: Unexpected test assertions.

---

## SECTION D — PROGRESS CHECKLIST

- [ ] **Session 1: Fix Model Loading**
  - [ ] Update [trocr_engine.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/src/ocr/trocr_engine.py) to load the model using standard `torch_dtype=torch.float16` on CUDA (if available).
  - [ ] Verify local model loading using the test load scratch script.
- [ ] **Session 2: Web App and UI Verification**
  - [ ] Start the FastAPI server `app.py` in the background.
  - [ ] Use the browser subagent to navigate to `http://127.0.0.1:7860/`.
  - [ ] Perform a test upload of a prescription/lab report in the UI.
  - [ ] Confirm results (OCR transcription, diffs, timing breakdown) display correctly in the browser.
- [ ] **Session 3: Running Scripts and Test Suite**
  - [ ] Run [playwright_ocr_eval.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/scripts/playwright_ocr_eval.py) while the server is active.
  - [ ] Execute `pytest` and verify that all previously failing tests now pass.
  - [ ] Stop the background server task.
