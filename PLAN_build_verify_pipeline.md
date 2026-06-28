# PLAN: Build and Verify Medical OCR Pipeline

## SECTION A — GOAL DEFINITION

The goal of this task is to ensure the Healthcare OCR pipeline is built, functional, and fully verified by executing tests, resolving environment/library issues (such as the bitsandbytes CUDA version mismatch), and aligning the test suite and pipeline orchestrator with the disintegrated Phi-4 LLM clinical summarization/correction stage.

### 1. What is being built or changed?
* **bitsandbytes DLL resolution**: Resolve the CUDA 13.2 version mismatch in the `bitsandbytes` library inside the python virtual environment.
* **Test Suite Alignment**: Adjust `tests/test_pipeline.py` and `tests/test_postprocessing.py` to assert correct behavior for the disintegrated/bypassed LLM pipeline (e.g. not expecting `Clinical Summarization` in timings or detailed clinical summary fields, or mocking them).
* **Fine-Tuning Tests**: Skip or stub the fine-tuning tests if the `medical-prescription-dataset` is not present in the workspace.
* **Pipeline Validation**: Run benchmarks to verify execution of the layout and OCR engines in < 5 seconds without memory leak or VRAM buildup.

### 2. What does "done" look like?
* All unit tests in the `tests/` directory pass successfully.
* `python -m pytest` executes without errors or collection failures.
* The web app can be launched via `app.py` and runs successfully.
* Benchmarks run without OOM.

### 3. What is explicitly out of scope?
* Restoring/re-integrating the heavy Phi-4 LLM corrector/clinical summarizer (this remains disintegrated per `PLAN_disintegrate_phi_llm.md`).
* Acquiring the external `medical-prescription-dataset` if it's not present.

---

## SECTION B — TECH STACK

* **Python**: 3.12.6
* **Libraries**: PyTorch 2.12.1+cu132, bitsandbytes, transformers, pytest, FastAPI
* **Files Modified**:
  * `tests/test_pipeline.py`
  * `tests/test_postprocessing.py`
  * `tests/test_fine_tuning.py`
  * `venv/Lib/site-packages/bitsandbytes/` (DLL copies)

---

## SECTION C — SESSION MODULARIZATION

### Session 1: Resolve bitsandbytes CUDA Version Mismatch
* **Objective**: Make bitsandbytes compatible with the installed CUDA 13.2 / PyTorch cu132 environment.
* **Scope**: `venv/Lib/site-packages/bitsandbytes/` DLL directory
* **Output**: Copy `libbitsandbytes_cuda130.dll` to `libbitsandbytes_cuda132.dll` to bypass mismatch error.
* **Failure Surface**: If the copy is still incompatible with CUDA 13.2, try forcing `BNB_CUDA_VERSION=130` in the environment.

### Session 2: Align Test Suite with Disintegrated Phi LLM
* **Objective**: Update test assertions to match the disintegrated (bypassed) pipeline.
* **Scope**: `tests/test_pipeline.py`, `tests/test_postprocessing.py`
* **Output**:
  * Skip/mock LLM-based correction checks in `tests/test_postprocessing.py` as the LLM is not loaded.
  * Adjust `tests/test_pipeline.py` to check for empty/disabled clinical summarization timings and summary objects.
* **Failure Surface**: Mismatched assertions on timings or returned fields.

### Session 3: Handle Missing Fine-Tuning Dataset in Tests
* **Objective**: Add conditional skipping to training dataset tests so they pass when running the general test suite.
* **Scope**: `tests/test_fine_tuning.py`
* **Output**: Use `pytest.mark.skipif` to skip dataset-dependent training tests when the `medical-prescription-dataset` directory is missing.

### Session 4: Validate End-to-End Pipeline & Web App
* **Objective**: Run benchmarks and confirm correct web app launch.
* **Scope**: `scripts/benchmark.py`, `app.py`
* **Output**: Verification of pipeline timings and VRAM.

---

## SECTION D — PROGRESS CHECKLIST

- [x] **Session 1: Resolve bitsandbytes CUDA Version Mismatch**
  - [x] Copy `libbitsandbytes_cuda130.dll` to `libbitsandbytes_cuda132.dll` inside virtual environment's bitsandbytes directory.
  - [x] Verify that importing bitsandbytes and running a simple 4bit model load works.
- [x] **Session 2: Align Test Suite with Disintegrated Phi LLM**
  - [x] Modify `tests/test_pipeline.py` to verify bypassed summary stages.
  - [x] Modify `tests/test_postprocessing.py` to bypass/mock the real LLM load test.
- [x] **Session 3: Handle Missing Fine-Tuning Dataset in Tests**
  - [x] Skip tests in `tests/test_fine_tuning.py` if `medical-prescription-dataset` is absent.
- [x] **Session 4: Validate End-to-End Pipeline & Web App**
  - [x] Run the complete pytest test suite.
  - [x] Run the benchmark script.
  - [x] Verify that `app.py` boots up successfully.
