# PLAN: Playwright OCR Automation and Quality Evaluation

## SECTION A — GOAL DEFINITION

The goal of this task is to use Playwright to automate the interaction with our local Healthcare OCR web app (running on port 7860), upload a representative selection of images from the WhatsApp and Patient_Kastoor directories, perform OCR, and evaluate both the speed (seconds per page) and transcription accuracy (by cross-checking the actual image content with the OCR output).

### 1. What is being built or changed?
* **Playwright Environment Setup**: Install the `playwright` python library and download required browser binaries (chromium) inside the virtual environment.
* **Automation Script**: Create a script `scripts/playwright_ocr_eval.py` that starts a browser instance, uploads a file, triggers the pipeline, waits for the result, and logs the raw OCR transcription and timing breakdown.
* **Evaluation Data Collection**: Run the script on at least 3 different images: a printed lab report, a handwritten prescription, and a mixed layout document.
* **Crosscheck & Verdict Analysis**: Manually inspect the source images using local viewing, compare them word-for-word with the OCR results, calculate errors, and provide a clear quality verdict and processing speed.

### 2. What does "done" look like?
* Playwright successfully installed and running.
* The script successfully extracts OCR text and timings from the local web app.
* Detailed analysis is provided for the selected documents, showing speed per page and an accuracy review.
* A clear quality verdict is given.

### 3. What is explicitly out of scope?
* Modifying the FastAPI backend code in `app.py` or the pipeline in `src/pipeline.py`.
* Automating user interactions outside of document uploading and result extraction.

---

## SECTION B — TECH STACK

* **Python**: 3.12.6
* **Libraries**: Playwright, asyncio, cv2, numpy, pytest
* **Files Created/Modified**:
  * `scripts/playwright_ocr_eval.py`
  * `PLAN_playwright_ocr_eval.md`

---

## SECTION C — SESSION MODULARIZATION

### Session 1: Playwright Environment Setup
* **Objective**: Install playwright and install browser binaries inside the python venv.
* **Scope**: virtual environment packages
* **Output**: Successfully install playwright package and chromium browser binaries.
* **Failure Surface**: Network timeouts or permission errors during binary download.

### Session 2: Create Playwright Automation Script
* **Objective**: Write `scripts/playwright_ocr_eval.py` to automate document upload and data collection.
* **Scope**: `scripts/playwright_ocr_eval.py`
* **Output**: A script that launches chromium headlessly, navigates to `http://127.0.0.1:7860/`, uploads a given file path, clicks the run button, waits for processing, and prints the OCR text and timing metrics.
* **Failure Surface**: Element selector mismatches in the web UI (FastAPI HTML).

### Session 3: Image Selection and OCR Execution
* **Objective**: Execute the script on 3 test images representing printed, handwritten, and mixed reports.
* **Scope**: Running script on selected images.
* **Output**: Extracted OCR text and processing duration for:
  1. Printed Lab Report (e.g. `Patient_Kastoor/Lab_Report/` or `WhatsApp.Unknown.../lab_report_01.jpeg`)
  2. Handwritten Prescription (e.g. `WhatsApp.Unknown.../handwritten_prescription_01.jpeg`)
  3. Mixed layout report (e.g. `WhatsApp.Unknown.../discharge_summary_case_booklet_01.jpeg`)
* **Failure Surface**: Backend timeouts if an image takes too long to process.

### Session 4: Visual Crosscheck & Quality Verdict
* **Objective**: Compare the raw text on the images with the returned OCR transcriptions.
* **Scope**: Manual inspection of images and analysis of output.
* **Output**: A detailed summary highlighting where the OCR succeeded (e.g. names, dates, key findings) and where it had difficulty (e.g. medical symbols, signature notes, handwritten scribbles). Report speed per page and a final quality verdict.

---

## SECTION D — PROGRESS CHECKLIST

- [x] **Session 1: Playwright Environment Setup**
  - [x] Run `pip install playwright` in the virtual environment.
  - [x] Run `playwright install chromium` to fetch the browser binary.
- [x] **Session 2: Create Playwright Automation Script**
  - [x] Write `scripts/playwright_ocr_eval.py` with page automation steps.
  - [x] Test the script on a single simple image to verify data collection.
- [x] **Session 3: Image Selection and OCR Execution**
  - [x] Run the automation on a printed lab report and save the output.
  - [x] Run the automation on a handwritten prescription and save the output.
  - [x] Run the automation on a mixed case booklet and save the output.
- [x] **Session 4: Visual Crosscheck & Quality Verdict**
  - [x] Inspect the images and cross-check the visual text with OCR results.
  - [x] Calculate the processing speed per page.
  - [x] Provide the final accuracy verdict.
