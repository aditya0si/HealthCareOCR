# PLAN: Session 4 — OCR Engine (Layout + Dual-Engine OCR)

## SECTION A — GOAL DEFINITION

The goal of Session 4 is to build the core document text extraction engine. It takes preprocessed (and potentially upscaled) document images, parses their layout to isolate text regions, classifies each region as printed vs. handwritten, and routes them to the appropriate specialized OCR engine.

### Observable Outcomes ("Done" Criteria):
1. **Layout Engine** loads LayoutLMv3 (or Surya Layout) and detects text regions/bounding boxes.
2. **Text Router** successfully classifies crop segments as `printed` vs `handwritten`.
3. **Surya OCR Engine** runs in NF4 quantization and recognizes printed text regions.
4. **TrOCR Engine** runs in NF4 quantization and recognizes handwritten text lines.
5. **Memory Management**: All active OCR models fit inside the 8GB VRAM footprint simultaneously (< 3GB total VRAM).
6. **Execution Speed**: Merged OCR recognition on a single page takes **< 3.0s** on the RTX 5060 GPU.

### Out of Scope:
- PDF document parsing (focus is solely on image-based document pages).
- Full LLM summarization (which is deferred to Session 5 and 6).

---

## SECTION B — TECH STACK

- **PyTorch (CUDA 13.2)**: GPU compute framework for running deep learning models.
- **bitsandbytes**: NF4 quantization wrapper for loading models in 4-bit mode.
- **ONNX Runtime GPU / Hugging Face Transformers**: For running LayoutLMv3 in INT8.
- **Surya-OCR (`surya`)**: Library containing the layout parsing and printed OCR models (`vikp/surya_det`, `vikp/surya_rec`).
- **TrOCR (`microsoft/trocr-base-handwritten`)**: Encoder-decoder handwriting recognition model.
- **OpenCV & NumPy**: For image cropping, coordinate normalization, and processing.

---

## SECTION C — SESSION MODULARIZATION

### Sub-session 4.1: Bounding Box Detection & Layout Parsing (`layout_engine.py`)
- **Objective:** Detect text region coordinates on the document page.
- **Details:** Use Surya layout or LayoutLMv3 to detect all text bounding boxes/lines on the preprocessed page.
- **Interface:** `LayoutEngine.detect_regions(image: np.ndarray) -> list[dict]` (each region containing coordinates and type).

### Sub-session 4.2: Printed vs. Handwritten Classification & Routing (`router.py`)
- **Objective:** Classify each text line/crop as printed vs. handwritten and route them accordingly.
- **Implementation:**
  - Build a lightweight convolutional classifier (`HandwritingClassifier` or a structural stroke heuristic) to output a binary label (`printed` or `handwritten`).
  - Route printed boxes to Surya2 and handwritten lines to TrOCR.

### Sub-session 4.3: Surya NF4 OCR Loader (`surya_engine.py`)
- **Objective:** Initialize and run Surya printed OCR in NF4.
- **Interface:** `SuryaEngine.recognize(image: np.ndarray, boxes: list) -> list[str]`

### Sub-session 4.4: TrOCR NF4 Loader (`trocr_engine.py`)
- **Objective:** Initialize and run TrOCR handwriting recognition in NF4.
- **Interface:** `TrOCREngine.recognize(image: np.ndarray, boxes: list) -> list[str]`

---

## SECTION D — PROGRESS CHECKLIST

- [ ] **Session 4: OCR Engine — Layout + Dual-Engine OCR**
  - [ ] Implement `src/ocr/layout_engine.py` (Layout parser)
  - [ ] Implement `src/ocr/surya_engine.py` (Printed OCR using Surya in NF4)
  - [ ] Implement `src/ocr/trocr_engine.py` (Handwritten OCR using TrOCR in NF4)
  - [ ] Implement `src/ocr/router.py` (Printed/Handwritten classification & routing)
  - [ ] Implement orchestrating router logic to merge texts in reading order
  - [ ] Create unit tests in `tests/test_ocr.py` verifying detection, routing, and recognition
  - [ ] Assert that VRAM consumption of the combined OCR stage is < 3.0 GB
  - [ ] Assert that single-page OCR recognition completes in < 3.0 seconds
