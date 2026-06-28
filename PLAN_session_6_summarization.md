# PLAN: Session 6 — Summarization + End-to-End Pipeline Assembly + Web App

## SECTION A — GOAL DEFINITION

The goal of Session 6 is to build the summarization stage, assemble all pipeline components into a cohesive orchestrator (`pipeline.py`), add timing instrumentation, and build a local web app deployment for user interaction.

### Observable Outcomes ("Done" Criteria):
1. **Clinical Summarizer (`src/summarization/summarizer.py`)** extracts structured JSON patient summaries from the corrected OCR text using the 4-bit `microsoft/Phi-4-mini-instruct` model.
2. **Timer Utility (`src/utils/timer.py`)** tracks and logs elapsed milliseconds for each pipeline stage, printing a final performance report.
3. **Pipeline Orchestrator (`src/pipeline.py`)** coordinates the entire sequence: Quality Assessment -> Paper Contour Detection & Perspective Crop -> Conditional Super-Resolution -> Layout Analysis & Region Classification -> Printed (Surya2) vs. Handwritten (TrOCR) OCR routing -> Dictionary Confidence Scoring -> Conditional LLM Post-Correction -> Clinical Summarization -> Structured Output.
4. **Local Web App (`app.py` / `src/app/`)** provides a beautiful, modern UI using Gradio or FastAPI+HTML, allowing users to upload a medical document image, see the step-by-step progress, view the corrected OCR text side-by-side, read the structured JSON summary, and inspect timing metrics.
5. **Execution Speed & Reliability**: End-to-end processing on a standard image runs in **≤ 7 seconds** wall-clock time on the RTX 5060 GPU without any OOM failures.

### Out of Scope:
- Production multi-tenant user authentication.
- Cloud hosting or serverless deployments.

---

## SECTION B — TECH STACK

- **Core Framework**: Python 3.12 (system-wide)
- **Deep Learning**: PyTorch + Transformers + BitsAndBytes (4-bit quantization)
- **OCR Engines**: Surya-OCR (layout/printed text) + TrOCR (handwritten text)
- **Web App UI**: Gradio (for maximum reliability, built-in layout, clean styling, and simple local execution via `python app.py`)
- **JSON Utility**: standard `json` with regex fallback for robust markdown JSON codeblock parsing

---

## SECTION C — SESSION MODULARIZATION

### Session 6.1: Clinical Summarizer (`src/summarization/summarizer.py`)
- **Objective:** Format prompts, invoke the 4-bit Phi-4 model, and parse the output into a structured JSON schema.
- **Details:** Includes prompt templates with zero-shot formatting instructions and a robust JSON extraction regex to retrieve code blocks cleanly.

### Session 6.2: Timing Instrument (`src/utils/timer.py`)
- **Objective:** Create a class/context manager to record elapsed milliseconds per stage.
- **Details:** Exposes `__enter__` and `__exit__` context tracking and prints a formatted timing table.

### Session 6.3: Main Pipeline Orchestrator (`src/pipeline.py`)
- **Objective:** Sequence all components from Session 2-5 into an end-to-end runner.
- **Details:** Coordinates loading/unloading models (Surya/TrOCR vs. Phi-4) via `GPUManager` to maintain a tight VRAM footprint (~2.8 GB max peak).

### Session 6.4: Local Web App (`app.py`)
- **Objective:** Build a beautiful local web application to demonstrate the pipeline.
- **Details:** Renders upload box, displays output fields side-by-side, plots per-stage timing breakdown, and runs on `http://127.0.0.1:7860`.

---

## SECTION D — PROGRESS CHECKLIST

- [ ] **Session 6: Summarization + End-to-End Pipeline Assembly + Web App**
  - [ ] Create `src/summarization/summarizer.py` (Structured JSON clinical summary generator)
  - [ ] Create `src/utils/timer.py` (Timing context manager)
  - [ ] Implement `src/pipeline.py` (Main pipeline orchestrator with GPU memory orchestration)
  - [ ] Write `app.py` (Gradio-based local web app interface)
  - [ ] Write `tests/test_pipeline.py` verifying end-to-end execution accuracy, timing (< 7s limit), and VRAM swapping
  - [ ] Run end-to-end verification and confirm all tests pass successfully
