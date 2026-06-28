# PLAN: Batch Parallel Processing — WhatsApp Unknown + Patient Kastoor

## A — GOAL DEFINITION

1. **What is being built?**
   A batch-processing system that runs the existing `MedicalOCRPipeline` across all images in two directories (`WhatsApp.Unknown.2026-04-27.at.12.10.10` and `Patient_Kastoor`) using maximum parallelization, then adds a new "Batch Results" view in the existing FastAPI UI to browse all results.

2. **What does "done" look like?**
   - A new `/batch-process` API endpoint that accepts directory paths, recursively discovers all `.jpg`/`.jpeg`/`.png` images, and processes them through the pipeline.
   - **Parallelization strategy**: CPU-bound preprocessing is parallelized across a `ThreadPoolExecutor` across all images concurrently, while GPU-bound OCR (which must be serial due to single-GPU VRAM contention) is pipelined so the next image's preprocessing runs while the current image's OCR executes.
   - A new **Batch Results** tab in the UI showing a gallery/table of all processed images with their OCR text, quality metrics, and timings.
   - Both directories processed and results visible in the UI.

3. **What is out of scope?**
   - Multi-GPU support (we have one GPU).
   - Changing the existing single-image upload flow.
   - LLM correction / clinical summarization (currently disabled in the pipeline).

## B — TECH STACK

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Backend | FastAPI (existing) | Already in place |
| Pipeline | `MedicalOCRPipeline` (existing) | Core OCR engine |
| Parallelization | `concurrent.futures.ThreadPoolExecutor` for preprocessing + asyncio for I/O | Preprocessing is CPU-bound but uses OpenCV (releases GIL). GPU OCR must be serial. |
| Pipelining | `asyncio.Queue`-based producer-consumer | Overlap CPU preprocessing of image N+1 with GPU OCR of image N |
| Frontend | Vanilla JS + existing CSS design system | Matches existing UI |
| Results storage | In-memory dict keyed by batch ID | Simple, no DB needed |

### Parallelization Architecture

```
Phase 1: PARALLEL CPU PREPROCESSING
ThreadPoolExecutor(max_workers=N)
All images preprocessed concurrently
(paper detection, deskew, enhancement, SR)
N = os.cpu_count() or 8

Phase 2: SERIAL GPU OCR
Single-threaded (GPU can only run one model)
Layout detection + Surya/TrOCR recognition
Models loaded ONCE, kept in VRAM
```

## C — SESSION MODULARIZATION

### Session 1: Batch Processing Backend
- **OBJECTIVE**: Create `POST /batch-process` endpoint and parallel batch runner.
- **SCOPE**: New `src/batch_runner.py` module, modifications to `app.py`.
- **OUTPUT**: Working endpoint that processes directories, returns JSON results with progress streaming.
- **CONNECTS TO**: Session 2 needs the endpoint contract (JSON shape) to render results.
- **FAILURE SURFACE**: GPU OOM on large images; ThreadPool contention with GPU operations.

### Session 2: Batch Results UI
- **OBJECTIVE**: Add "Batch Results" tab to the UI with gallery view, progress indicator, and per-image drill-down.
- **SCOPE**: HTML/CSS/JS additions to the embedded `HTML_CONTENT` in `app.py`.
- **OUTPUT**: Fully functional batch results tab with image thumbnails, OCR text, quality scores, and timing breakdown.
- **CONNECTS TO**: Session 1's endpoint and JSON contract.
- **FAILURE SURFACE**: Large number of results (85+ images) causing DOM performance issues.

### Session 3: Execute & Validate
- **OBJECTIVE**: Run the batch processing on both directories and verify results in the UI.
- **SCOPE**: Run the server, trigger batch processing, verify output.
- **OUTPUT**: Both directories processed, results visible in UI.
- **CONNECTS TO**: N/A — final session.
- **FAILURE SURFACE**: Pipeline errors on specific image types (ultrasound images, blank pages).

## D — PROGRESS CHECKLIST

- [x] Session 1: Batch Processing Backend
  - [x] Create `src/batch_runner.py` with parallel preprocessing + serial OCR pipeline
  - [x] Add `POST /batch-process` endpoint to `app.py`
  - [x] Add `GET /batch-status/{batch_id}` for progress polling
  - [x] Add `GET /batch-results/{batch_id}` for final results
  - [x] Handle recursive image discovery in nested subdirectories
  - [x] Implement polling-based progress reporting

- [x] Session 2: Batch Results UI
  - [x] Add "Batch Processing" tab to the header/sidebar
  - [x] Create batch trigger UI with pre-configured directory buttons
  - [x] Build results gallery with image thumbnails and OCR text preview
  - [x] Add per-image detail modal/expandable view
  - [x] Add batch timing summary and progress bar
  - [x] Pagination for large result sets

- [x] Session 3: Execute & Validate
  - [x] Start server and trigger batch on both directories
  - [x] Verify all images processed without OOM
  - [x] Verify results display correctly in UI
