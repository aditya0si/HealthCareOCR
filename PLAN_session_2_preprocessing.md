# PLAN: Session 2 — Adaptive Image Preprocessing Pipeline

## SECTION A — GOAL DEFINITION

The goal of Session 2 is to design and implement a robust, high-performance image preprocessing pipeline that takes a raw, user-uploaded patient report image (with potential distortion, rotation, shadows, or noise) and outputs a clean, deskewed, cropped, and enhanced grayscale/binarized image optimized for layout analysis and OCR.

### Observable Outcomes ("Done" Criteria):
1. **Quality Assessor** runs on an image and produces a standardized quality score card on 5 dimensions (sharpness, contrast, brightness, noise, skew).
2. **Paper Detector** successfully crops and flattens paper documents from background on at least 90% of test images, falling back to GrabCut or Hough lines when standard contours fail.
3. **Deskewing** rotates skewed documents to <1 degree of error.
4. **Sauvola Binarization** removes shadows and balances uneven illumination.
5. **Adaptive Router** correctly bypasses slow preprocessing steps for high-quality input images to minimize latency.
6. The entire preprocessing pipeline runs on a single image in:
   - **<150ms** for High-Quality images (skips heavy steps).
   - **<350ms** for Medium-Quality images (skips super-resolution and heavy denoising).
   - **<900ms** for Low-Quality images (includes all checks and full enhancement).

### Out of Scope:
- Super-resolution using deep learning (delegated to Session 3).
- Layout parsing or OCR engine execution (delegated to Session 4).

---

## SECTION B — TECH STACK

Within the existing system Python 3.12 environment, Session 2 will utilize:
- **OpenCV (`cv2`)**: Primary library for Canny edges, contours, GrabCut, Hough lines, CLAHE, morphological operations, and perspective warping.
- **scikit-image (`skimage`)**: Optimized Sauvola local thresholding filter (`skimage.filters.threshold_sauvola`).
- **`deskew`**: Standard python library for fast skew angle determination (`deskew.determine_skew`).
- **NumPy**: Matrix math for corner sorting, brightness/contrast calculations, and fast image manipulation.

---

## SECTION C — SESSION MODULARIZATION

### Sub-session 2.1: Quality Assessor (`quality_assessor.py`)
- **Objective:** Score images on 5 dimensions to enable adaptive routing.
- **Inputs:** Grayscale image (`np.ndarray`).
- **Outputs:** `QualityMetrics` data structure containing:
  - `sharpness` (Laplacian variance normalized: `var / (var + 100)`).
  - `contrast` (std dev of pixel intensity normalized: `std / 128.0`).
  - `brightness` (mean pixel intensity: `mean / 255.0`).
  - `noise_ratio` (estimated noise variance via local standard deviations).
  - `skew_angle` (detected skew via `deskew.determine_skew`).
- **Failure Surface:** Skew detection might fail or return NaN on non-text regions (e.g. blank sections). *Mitigation:* Fall back to 0.0 degrees if deskew fails.

### Sub-session 2.2: Paper Detector (`paper_detector.py`)
- **Objective:** Locate document boundaries, crop, and apply perspective homography.
- **Algorithm Steps:**
  1. Downscale to max-dim 1024px for latency-sensitive processing.
  2. Gaussian Blur (5x5) + Canny Edge Detection (low=50, high=150).
  3. Contour detection + area sorting. Select largest contour.
  4. Approximate contour to polygon. If it has exactly 4 corners and covers >15% of image area, return corners.
  5. *Fallback 1 (GrabCut):* Initialize GrabCut with a 5% margin bounding box. Run 3 iterations. Find largest contour of foreground mask, approximate to 4 corners.
  6. *Fallback 2 (Hough Lines):* Run HoughLinesP. Group lines into horizontal and vertical. Find intersections of outermost lines to form 4 corners.
  7. *Corner Ordering:* Sort corners to `[top-left, top-right, bottom-right, bottom-left]` using sum and difference of coordinates.
  8. *Perspective Warp:* Calculate target size from width/height of corners. Warp to get flat rectangular crop.
- **Failure Surface:** Extremes of lighting or matching background colors. *Mitigation:* Fall back to full image crop if all detection methods fail.

### Sub-session 2.3: Deskew & Geometric Correction (`geometric.py`)
- **Objective:** Detect and correct rotation of the cropped paper.
- **Implementation:**
  - Call `deskew.determine_skew` on grayscale image.
  - If skew angle is > `max_skew_angle` (default 1.0 deg), rotate image using `cv2.getRotationMatrix2D` + `cv2.warpAffine` with bilinear interpolation and white background border.
- **Failure Surface:** Text-less images or tables with horizontal lines skewing the detector. *Mitigation:* Clip skew angle to max 45 degrees.

### Sub-session 2.4: Image Enhancement & Sauvola Binarization (`enhancement.py`)
- **Objective:** Correct shadow regions, uneven lighting, and generate clean binarized images.
- **Implementation:**
  - **Shadow Removal:** Extract L channel from LAB color space. Apply morphological dilation (kernel size 21) followed by large Gaussian blur (ksize 41) to estimate background illumination. Subtract L channel from background to normalize lighting.
  - **CLAHE:** Apply contrast limited adaptive histogram equalization to L channel (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`) to enhance faded text.
  - **Sauvola Binarization:** Grayscale image -> `skimage.filters.threshold_sauvola(window_size=25, k=0.2)`. Convert threshold mask to binary uint8.
- **Failure Surface:** Low local contrast can cause Sauvola to binarize background noise (speckle noise). *Mitigation:* Clean binarized mask using morphological opening (small 2x2 kernel).

---

## SECTION D — PROGRESS CHECKLIST

- [ ] **Session 2: Adaptive Image Preprocessing Pipeline**
  - [ ] Implement `src/preprocessing/quality_assessor.py`
  - [ ] Implement `src/preprocessing/paper_detector.py`
  - [ ] Implement `src/preprocessing/geometric.py`
  - [ ] Implement `src/preprocessing/enhancement.py`
  - [ ] Create adaptive preprocessing router in `src/preprocessing/__init__.py`
  - [ ] Create preprocessing unit tests in `tests/test_preprocessing.py`
  - [ ] Verify preprocessing on all 37 Lab Report images
  - [ ] Verify preprocessing on all 29 WhatsApp images
  - [ ] Output timing logs for each image class to confirm latency targets
