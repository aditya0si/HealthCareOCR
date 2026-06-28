# PLAN: Disintegrate Phi Correction and Clinical Analysis

## SECTION A — GOAL DEFINITION

The goal of this task is to completely remove/bypass the Phi-4 LLM post-processing correction and clinical summarization stages from both the pipeline execution and the user interface. This is done to eliminate CUDA initialization/memory errors caused by loading the heavy LLM on the user's system, leaving a fast and lightweight layout-aware neural OCR pipeline.

### 1. What is being built or changed?
* **In the pipeline (`src/pipeline.py`)**: Comment out/bypass the conditional LLM correction and clinical summarization steps. Set `corrected_text` to be identical to `raw_text`, and set the clinical summary output to an empty dictionary `{}` without loading the LLM into CUDA.
* **In the web application (`app.py`)**: Update the Fast API `/process` response mapping. Comment out or remove the UI tabs/sections for the LLM Corrected Diff, Structured Summary, and the loader steps associated with LLM loading, keeping only the Raw OCR and Preprocessing/DPI metrics.
* **Model/Files retention**: The unused modules (`src/postprocessing/llm_corrector.py`, `src/summarization/summarizer.py`) will remain intact in the codebase but will not be imported or executed.

### 2. What does "done" look like?
* Launching `app.py` runs successfully on port 7860.
* Uploading an image through the web interface runs successfully in < 5 seconds with zero CUDA/OOM errors.
* The UI displays only the Raw OCR layout text, Preprocessing metrics, and Performance timings (without tabs/sections for correction or clinical summaries).
* All tests (`pytest`) pass successfully (we may adjust or mock LLM test checks if they run pipeline checks).

### 3. What is explicitly out of scope?
* Modifying the preprocessor paper detection, quality assessment, deskew, and binarization steps.
* Modifying the LayoutLMv3 + Surya/TrOCR OCR routing.
* Deleting any code files from the filesystem.

---

## SECTION B — TECH STACK

* **Runtime & Frameworks**: Python 3.12, PyTorch 2.12.1+cu132, FastAPI, Uvicorn, OpenCV
* **Files Modified**:
  * [src/pipeline.py](file:///C:/Users/oliad/Desktop/HealthCareOCR/src/pipeline.py)
  * [app.py](file:///C:/Users/oliad/Desktop/HealthCareOCR/app.py)
* **Files Touched/Run**:
  * [tests/test_pipeline.py](file:///C:/Users/oliad/Desktop/HealthCareOCR/tests/test_pipeline.py)

---

## SECTION C — SESSION MODULARIZATION

### Session 1: Pipeline Disintegration
* **Objective**: Remove LLM corrector and clinical summarization steps from the orchestrator.
* **Scope**: `src/pipeline.py`
* **Output**: `MedicalOCRPipeline` processes images through preprocessor, layout detection, and OCR routing, and returns immediately without invoking confidence scoring, LLM correction, or clinical summarization.
* **Connects to**: Session 2 (UI updates).
* **Failure Surface**: None (simple bypass of downstream steps).

### Session 2: UI Cleanup & Validation
* **Objective**: Modify the FastAPI embedded HTML page to remove/hide summary and correction components.
* **Scope**: `app.py`
* **Output**: Clean UI displaying only Preprocessing metrics, Raw OCR text, and processing timings.
* **Failure Surface**: JavaScript execution errors if HTML components are deleted but script attempts to write to them. (Prevented by commenting out/removing the DOM updates for these components).

---

## SECTION D — PROGRESS CHECKLIST

- [x] **Session 1: Pipeline Disintegration**
  - [x] Modify `src/pipeline.py` to bypass Stages 4, 5, 6, and 7
  - [x] Set `corrected_text = raw_text` and `summary_json = {}`
  - [x] Create `REINTEGRATION_GUIDE.md` detailing how to restore the LLM/Phi correction and summarization stages
- [x] **Session 2: UI Cleanup & Validation**
  - [x] Comment out/remove the "Structured Summary" and "Corrected Transcription (Phi-4 mini LLM)" panels from the HTML in `app.py`
  - [x] Hide/disable the corresponding tabs in the UI
  - [x] Clean up the JS rendering function in `app.py` to prevent DOM errors
  - [x] Verify execution by uploading a document in the browser and confirming timing/accuracy
