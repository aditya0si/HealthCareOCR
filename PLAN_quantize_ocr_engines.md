# PLAN_quantize_ocr_engines.md

## SECTION A — GOAL DEFINITION
1. **What is being built or changed?**
   - Update **TrOCR** (`src/ocr/trocr_engine.py`) to use dynamic 4-bit `nf4` quantization via Hugging Face `BitsAndBytesConfig` + `transformers` to reduce GPU memory usage.
   - Update **Surya OCR** (`src/ocr/surya_engine.py`) to use the **GGUF** quantized model format to run on low-resource hardware/CPU.
   - Modify the initialization and unloading logic in `src/pipeline.py` to support these quantized engines, ensuring that we handle memory transitions (such as moving the models to/from CPU/GPU) gracefully without crashing.
2. **What does "done" look like?**
   - TrOCR is loaded in 4-bit dynamic `nf4` precision using `bitsandbytes`.
   - Surya OCR is loaded and runs using its GGUF version (via upgraded `surya-ocr` package supporting GGUF/llama.cpp, or custom python integration).
   - The startup warm-up and pipeline run successfully on the backend and return correct text extraction results.
3. **What is explicitly out of scope?**
   - Replacing LayoutLMv3 layout detection models or fine-tuning models.
   - Restructuring the clinical summarizer beyond what is needed to integrate the new OCR formats.

---

## SECTION B — TECH STACK
- Python (current venv environment)
- PyTorch & Transformers
- `bitsandbytes` (for TrOCR 4-bit quantization)
- `accelerate` (for device mapping)
- `surya-ocr` (to be upgraded or configured to use GGUF via llama.cpp/llama-server or custom llama-cpp-python code)
- Python modules: `src/ocr/trocr_engine.py`, `src/ocr/surya_engine.py`, `src/pipeline.py`, requirements.txt.

---

## SECTION C — SESSION MODULARIZATION

### Session 1: TrOCR dynamic 4-bit NF4 Quantization
- **Objective**: Implement 4-bit NF4 loading for TrOCR and update the model unloading to prevent crash on quantized models.
- **Scope**: `src/ocr/trocr_engine.py`, `src/pipeline.py`.
- **Output**: TrOCR loads dynamically in 4-bit and the pipeline is protected against calling `.to("cpu")` on quantized models.
- **Connects to**: Session 2.
- **Failure Surface**: OOM errors, bitsandbytes compatibility issues on Windows, or `ValueError` during model unloading/CPU offloading.

### Session 2: Surya OCR GGUF Integration
- **Objective**: Implement GGUF loading for Surya OCR.
- **Scope**: `src/ocr/surya_engine.py`, `requirements.txt`, and virtual environment packages.
- **Output**: Surya OCR operates using the GGUF model files via upgraded `surya-ocr` or `llama-cpp-python` integration.
- **Connects to**: Verification.
- **Failure Surface**: `llama-server` process execution failures on Windows, compilation failures for `llama-cpp-python` on Windows, or mismatch in Surya OCR API versions.

### Session 3: Validation and Verification
- **Objective**: Verify that both engines execute correctly on test inputs and the app's startup warm-up completes successfully.
- **Scope**: `app.py`, `src/pipeline.py`, verification script.
- **Output**: Passing end-to-end run of the OCR pipeline.
- **Failure Surface**: Text accuracy regression or startup failures.

---

## SECTION E — ASSUMPTIONS & UNCERTAINTIES
- **[ASSUMPTION: TrOCR CUDA requirement]** Dynamic 4-bit quantization with `bitsandbytes` is CUDA-only. We assume the target environment has a CUDA-enabled GPU. If loaded on CPU/MPS, the engine must fall back to FP32/FP16 without `bitsandbytes` to prevent model loading errors.
- **[ASSUMPTION: Surya GGUF backend]** Upgrading `surya-ocr` to v0.20.0 (or v2) changes the API (replacing `FoundationPredictor` with `SuryaInferenceManager` and output schemas from `text_lines` to `blocks`). We assume this upgrade is acceptable and we will adjust `layout_engine.py` and `surya_engine.py` to match the new API. We also assume that `llama-server` / `llama.cpp` can be configured on the host machine or that we can fall back to standard execution if the server fails.

---

## SECTION D — PROGRESS CHECKLIST
- [ ] **Session 1: TrOCR dynamic 4-bit NF4 Quantization**
  - [ ] Import `BitsAndBytesConfig` and configure 4-bit NF4 settings in `src/ocr/trocr_engine.py`.
  - [ ] Load TrOCR model with `quantization_config=bnb_config` and `device_map="auto"` (or specific CUDA map).
  - [ ] Adjust `recognize` in `src/ocr/trocr_engine.py` to skip manual `.to("cuda")` or handle it safely.
  - [ ] Add try-except protection or quantization check in `src/pipeline.py` before calling `self.trocr.model.to("cpu")`.
- [ ] **Session 2: Surya OCR GGUF Integration**
  - [ ] Install/Upgrade necessary dependencies (e.g. `surya-ocr>=0.20.0` or `llama-cpp-python` as required).
  - [ ] Configure `surya_engine.py` to use GGUF files (e.g. from `datalab-to/surya-ocr-2-gguf` or standard Surya llama.cpp backend).
  - [ ] Update `LayoutEngine` and `SuryaEngine` API usage to match version changes if packages are upgraded.
- [ ] **Session 3: Validation and Verification**
  - [ ] Run the start-up warm-up and check for errors.
  - [ ] Run a pipeline test script on sample medical images.
  - [ ] Confirm text outputs are returned correctly.
