# PLAN: Session 3 — Super Resolution + DPI Management

## SECTION A — GOAL DEFINITION

The goal of Session 3 is to implement a high-speed, conditional Super Resolution (SR) step in the preprocessing pipeline. Because low-DPI images (e.g. <400 DPI, WhatsApp compressed screenshots, small phone photos) can severely degrade layout parsing and OCR accuracy, this session will estimate image DPI and run a 2x upscaler to recover text details.

### Observable Outcomes ("Done" Criteria):
1. **DPI Calculator** correctly estimates the DPI of cropped images by assuming standard A4/booklet page dimensions.
2. **Conditional Triggering** correctly initiates Super Resolution ONLY when the estimated DPI is < 400 and the maximum image dimension is < 2000 pixels.
3. **OpenCV DNN-based FSRCNN** upscales the image by 2x in **< 150ms** on CPU (or GPU if opencl is enabled), keeping latency well under the initial 500ms budget.
4. **VRAM footprint is 0 MB** for the SR step since we use OpenCV's CPU/OpenCL DNN module instead of loading a heavy 1.5GB PyTorch Real-ESRGAN model, preserving VRAM for OCR and LLM.
5. High-resolution images (e.g. Kastoor images ~4000x3000) correctly skip the SR step, preserving the <= 7s total pipeline latency.

### Out of Scope:
- Running heavy deep learning super resolution models in PyTorch (such as Real-ESRGAN x4) that exceed the VRAM budget or take >500ms.

---

## SECTION B — TECH STACK

- **OpenCV (`cv2.dnn_superres`)**: Built-in OpenCV module for running optimized Super Resolution models.
- **FSRCNN x2 (`models/FSRCNN_x2.pb`)**: Fast Super-Resolution Convolutional Neural Network model (upscale factor of 2, file size 39 KB) trained in TensorFlow and exported to pb format.
- **NumPy**: Assisting with pixel size calculations and shape manipulations.

---

## SECTION C — SESSION MODULARIZATION

### Sub-session 3.1: DPI Calculator & Conditional Trigger (`super_resolution.py`)
- **Objective:** Calculate the physical resolution (DPI) of the cropped image and decide whether to upscale.
- **DPI Estimation Formula:**
  - Assume standard A4 document size: `8.27 x 11.69` inches.
  - `dpi_w = width_pixels / 8.27`
  - `dpi_h = height_pixels / 11.69`
  - `estimated_dpi = (dpi_w + dpi_h) / 2.0`
- **SR Trigger Condition:**
  - `estimated_dpi < 400` AND `max(width, height) < 2000`

### Sub-session 3.2: FSRCNN Inference (`super_resolution.py`)
- **Objective:** Run the 2x upscaler using OpenCV's DNN module.
- **Implementation:**
  - Create super-resolution instance: `cv2.dnn_superres.DnnSuperResImpl_create()`
  - Load `models/FSRCNN_x2.pb`
  - Set model name to `fsrcnn` and scale to `2`.
  - Perform upscaling: `result = sr.upsample(image)`.
- **Latency Optimization:** FSRCNN is selected because it is 10x faster than EDSR while retaining sharp text edges.

---

## SECTION D — PROGRESS CHECKLIST

- [x] **Session 3: Super Resolution + DPI Management**
  - [x] Implement `src/preprocessing/super_resolution.py`
  - [x] Integrate `SuperResolution` class into the main orchestrator (`src/preprocessing/__init__.py`)
  - [x] Create unit tests in `tests/test_super_resolution.py`
  - [x] Verify that low-resolution WhatsApp images trigger SR and upscale successfully
  - [x] Verify that high-resolution Kastoor images correctly skip SR
  - [x] Verify that upscaling takes <150ms and consumes 0 VRAM
