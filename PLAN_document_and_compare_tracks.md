# PLAN: Document and Compare Tracks

---

## SECTION A — GOAL DEFINITION

1. **What is being built or changed?**
   - We are creating a comprehensive, reference-grade comparison and documentation file `OCR_TRACKS_COMPARISON.md` in the root of the workspace.
   - We are documenting the critical environment discovery regarding Python 3.14 (broken Surya detection due to pre-release Torch/Transformers dependency issues) vs Python 3.12 (fully functional Surya detection).
   - We will provide clear step-by-step instructions for launching, running, and verifying both Tracks (A and B).
2. **What does "done" look like?**
   - The file `OCR_TRACKS_COMPARISON.md` exists in the workspace root.
   - It contains a detailed comparison of Track A vs Track B (resource footprint, accuracy, speed, pipeline stages).
   - It details the environment issue and gives exact command lines to run the pipeline using the local virtual environment (`venv\Scripts\python.exe`) which uses Python 3.12.
3. **What is explicitly out of scope?**
   - Retraining the models.
   - Changing the underlying frontend code unless necessary for porting/fixing path issues.

---

## SECTION B — TECH STACK

- **Languages:** Python (3.12.6 virtual environment vs 3.14.3 system environment).
- **Libraries:**
  - `surya-ocr` (2.2.0)
  - `transformers` (modeling, image processors, pipeline)
  - `torch` (deep learning backend)
  - `qwen-vl` / `trocr` / `layoutlmv3`
- **Orchestration:** FastAPI/Uvicorn server (`app.py`), custom pipelines (`vlm_engine.py`, `layout_engine.py`).

---

## SECTION C — SESSION MODULARIZATION

### Session 1: Create Comparison and Reference Documentation
- **Objective:** Write a high-quality, comprehensive reference document (`OCR_TRACKS_COMPARISON.md`) explaining the technical architectures, execution steps, and environment compatibility finding (Python 3.12 vs 3.14).
- **Scope:** Root workspace documentation files.
- **Output:** [OCR_TRACKS_COMPARISON.md](file:///c:/Users/oliad/Desktop/HealthCareOCR/OCR_TRACKS_COMPARISON.md).
- **Connects to:** Session 2.
- **Failure Surface:** Typographical errors or missing environment steps.

### Session 2: Align and Verify app.py Execution
- **Objective:** Ensure the web application (`app.py`) runs correctly under the Python 3.12 virtual environment and supports switching between Track A and Track B dynamically.
- **Scope:** [app.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/app.py).
- **Output:** Verified execution of Uvicorn server in Python 3.12.
- **Failure Surface:** Address or port conflicts.

---

## SECTION D — PROGRESS CHECKLIST

- [ ] **Session 1: Reference Documentation**
  - [ ] Write [OCR_TRACKS_COMPARISON.md](file:///c:/Users/oliad/Desktop/HealthCareOCR/OCR_TRACKS_COMPARISON.md) with details on Track A, Track B, and Python 3.12 compatibility.
  - [ ] Document the exact commands to run each verification script.
- [ ] **Session 2: Verify app.py Execution**
  - [ ] Test launch the backend using `venv\Scripts\python.exe app.py` or equivalent server command.
  - [ ] Verify both Track A and Track B options are functional and selectable in the UI.
