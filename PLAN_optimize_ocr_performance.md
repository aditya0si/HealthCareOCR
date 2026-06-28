# PLAN: Optimize OCR Performance and Accuracy (Dual-Track Implementation)

## SECTION A — GOAL DEFINITION

### 1. What is being built or changed?
We are implementing and documenting **both** Tracks (A and B) and creating a side-by-side comparative Web UI in the browser to compare them.
- **Track A (Unified VLM - Recommended)**: Replace the multi-stage pipeline with a single `Qwen2-VL-2B-Instruct` model loaded in 4-bit quantization, running on the raw uploaded image (bypassing pre-processing, segmentation, routing, and post-correction) to directly output structured OCR text + clinical summaries in ≤ 5 seconds.
- **Track B (Streamlined Multi-Model)**: Keep the multi-stage pipeline but optimize it by using 4-bit NF4 quantized Surya and 4-bit NF4 quantized `trocr-small-handwritten` to speed up loading and execution to ≤ 7 seconds.
- **Side-by-Side Comparative UI**: Update `app.py` to allow the user to run either Track A or Track B, and compare their output quality and latency.

### 2. What does "done" look like?
- Both tracks are fully implemented, functional, and benchmarked.
- Track A runs in **≤ 5 seconds** (excluding the first-time model loading latency) with high handwriting accuracy.
- Track B runs in **≤ 7 seconds** with lower memory overhead than the original FP16 pipeline.
- The web app displays execution time and transcribes files cleanly for both modes side-by-side or via selector.
- A perfect reference document `COMPUTATIONAL_TRACKS_EVAL.md` details both approaches, loading instructions, code changes, and performance comparisons.

### 3. What is explicitly out of scope?
- Fine-tuning new custom model weights.
- Integrating external cloud APIs.

---

## SECTION B — TECH STACK

- **Track A Engine**: `Qwen/Qwen2-VL-2B-Instruct` loaded in 4-bit using `bitsandbytes` + HF `transformers`
- **Track B Engine**: `LayoutLMv3` (ONNX INT8) + `Surya` (4-bit/FP16) + `microsoft/trocr-small-handwritten` (4-bit)
- **Frameworks**: PyTorch 2.12.1+cu132, CUDA 12.0
- **Web App**: HTML/CSS + JavaScript, Python Flask/Gradio backend (integrated in `app.py`)

---

## SECTION C — SESSION MODULARIZATION

### Session 1: Track A VLM Implementation
- **OBJECTIVE**: Create the VLM wrapper in `src/ocr/vlm_engine.py` using 4-bit Qwen2-VL-2B-Instruct with pixel bounds to prevent OOMs, and integrate it into `src/pipeline.py`.
- **SCOPE**: `src/ocr/vlm_engine.py`, [pipeline.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/src/pipeline.py), [configs/pipeline_config.yaml](file:///c:/Users/oliad/Desktop/HealthCareOCR/configs/pipeline_config.yaml)
- **OUTPUT**: Working Track A engine that outputs raw OCR + clinical summary in JSON format directly.
- **CONNECTS TO**: Session 2.
- **FAILURE SURFACE**: VRAM OOM if input image resolution exceeds memory limits (mitigated by setting `max_pixels = 768 * 768`).

### Session 2: Track B Multi-Stage Optimization
- **OBJECTIVE**: Update `surya_engine.py` and `trocr_engine.py` to strip out PEFT/LoRA checks and add support for quantized loading. Modify configuration defaults to use `trocr-small-handwritten`.
- **SCOPE**: [surya_engine.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/src/ocr/surya_engine.py), [trocr_engine.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/src/ocr/trocr_engine.py), [configs/pipeline_config.yaml](file:///c:/Users/oliad/Desktop/HealthCareOCR/configs/pipeline_config.yaml)
- **OUTPUT**: Optimized Track B multi-stage pipeline running under 7 seconds.
- **CONNECTS TO**: Session 3.
- **FAILURE SURFACE**: Quantization issues with custom Surya model class.

### Session 3: Web UI Integration, Reference Documentation & Verification
- **OBJECTIVE**: Integrate both engines into `app.py` with an interactive toggle/selector, write `COMPUTATIONAL_TRACKS_EVAL.md` detailing comparison and future references, and verify.
- **SCOPE**: [app.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/app.py), `COMPUTATIONAL_TRACKS_EVAL.md`
- **OUTPUT**: Side-by-side comparative UI in browser, complete reference documentation, passing verification report.
- **CONNECTS TO**: None.
- **FAILURE SURFACE**: Mismatched JSON formats in Flask responses.

---

## SECTION D — PROGRESS CHECKLIST

- [x] Session 1: Track A VLM Implementation
  - [x] Implement `src/ocr/vlm_engine.py` with 4-bit `Qwen2-VL-2B-Instruct` and pixel capping (`max_pixels = 768 * 768`).
  - [x] Integrate VLM execution in `src/pipeline.py` based on config parameter `engine_mode = 'track_a'`.
  - [x] Test VLM path on sample images from the terminal.
- [/] Session 2: Track B Multi-Stage Optimization
  - [x] Remove unused PEFT/LoRA adapters boilerplate from `surya_engine.py` and `trocr_engine.py`.
  - [x] Update `trocr_engine.py` to use `BitsAndBytesConfig` 4-bit quantization.
  - [x] Update `configs/pipeline_config.yaml` default handwriting model to `microsoft/trocr-small-handwritten`.
  - [ ] Validate optimized Track B execution speed and memory footprints.
- [ ] Session 3: Web UI Integration, Reference Documentation & Verification
  - [ ] Modify `app.py` to support comparative engine selection toggle.
  - [ ] Create `COMPUTATIONAL_TRACKS_EVAL.md` with complete implementation details, benchmarks, and future reference guide.
  - [ ] Run benchmark validation scripts and verify all endpoints.
