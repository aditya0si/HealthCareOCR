import cv2
import numpy as np

class PaperDetector:
    def __init__(self, 
                 processing_max_dim: int = 512,  # Reduced from 1024 to 512 for speed
                 canny_low: int = 50,
                 canny_high: int = 150,
                 gaussian_blur_ksize: int = 5,
                 enable_grabcut_fallback: bool = True,
                 enable_hough_fallback: bool = True,
                 grabcut_max_dim: int = 256):    # New parameter for extreme GrabCut speedup
        self.processing_max_dim = processing_max_dim
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.gaussian_blur_ksize = gaussian_blur_ksize
        self.enable_grabcut_fallback = enable_grabcut_fallback
        self.enable_hough_fallback = enable_hough_fallback
        self.grabcut_max_dim = grabcut_max_dim

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Sort coordinates into order: [top-left, top-right, bottom-right, bottom-left]
        """
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()

        ordered = np.zeros((4, 2), dtype="float32")
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]
        ordered[1] = pts[np.argmin(diff)]
        ordered[3] = pts[np.argmax(diff)]
        return ordered

    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Warp perspective of the input image to match the 4 corners pts.
        """
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        return warped

    def detect_and_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Detects paper boundaries and returns warped cropped document.
        Optimized for latency by using low-res images for contour/GrabCut analysis.
        """
        h, w = image.shape[:2]
        
        # Scale down for main processing (Canny, contours, etc.)
        scale = 1.0
        if max(h, w) > self.processing_max_dim:
            scale = self.processing_max_dim / max(h, w)
            proc_w = int(w * scale)
            proc_h = int(h * scale)
            proc_img = cv2.resize(image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            proc_img = image.copy()

        gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY) if len(proc_img.shape) == 3 else proc_img
        blurred = cv2.GaussianBlur(gray, (self.gaussian_blur_ksize, self.gaussian_blur_ksize), 0)
        edged = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # --- STRATEGY 1: CONTOUR DETECTION ---
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            area_ratio = cv2.contourArea(approx) / (proc_img.shape[0] * proc_img.shape[1])
            
            if len(approx) == 4 and area_ratio > 0.15:
                pts = approx.reshape(4, 2) / scale
                return self.four_point_transform(image, pts)

        # --- STRATEGY 2: GRABCUT FALLBACK (at lower resolution for extreme speedup) ---
        if self.enable_grabcut_fallback:
            try:
                # Downscale further for GrabCut (e.g. 256px max dim)
                gc_scale = 1.0
                if max(proc_img.shape[:2]) > self.grabcut_max_dim:
                    gc_scale = self.grabcut_max_dim / max(proc_img.shape[:2])
                    gc_img = cv2.resize(proc_img, (int(proc_img.shape[1] * gc_scale), int(proc_img.shape[0] * gc_scale)), interpolation=cv2.INTER_AREA)
                else:
                    gc_img = proc_img.copy()

                mask = np.zeros(gc_img.shape[:2], np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                # Set bounding box with 5% margin
                margin_x = int(gc_img.shape[1] * 0.05)
                margin_y = int(gc_img.shape[0] * 0.05)
                rect = (margin_x, margin_y, gc_img.shape[1] - 2 * margin_x, gc_img.shape[0] - 2 * margin_y)
                
                # Execute GrabCut with only 2 iterations for speed
                cv2.grabCut(gc_img, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
                bin_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
                
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
                
                gc_contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if gc_contours:
                    largest_c = max(gc_contours, key=cv2.contourArea)
                    perimeter = cv2.arcLength(largest_c, True)
                    approx = cv2.approxPolyDP(largest_c, 0.02 * perimeter, True)
                    
                    # Convert coordinates back to original image space
                    total_scale = scale * gc_scale
                    if len(approx) == 4:
                        pts = approx.reshape(4, 2) / total_scale
                        return self.four_point_transform(image, pts)
                        
                    hull = cv2.convexHull(largest_c)
                    hull_perimeter = cv2.arcLength(hull, True)
                    approx_hull = cv2.approxPolyDP(hull, 0.02 * hull_perimeter, True)
                    if len(approx_hull) == 4:
                        pts = approx_hull.reshape(4, 2) / total_scale
                        return self.four_point_transform(image, pts)
            except Exception:
                pass

        # --- STRATEGY 3: HOUGH LINE FALLBACK ---
        if self.enable_hough_fallback:
            try:
                lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 50, minLineLength=80, maxLineGap=20)
                if lines is not None:
                    ys = []
                    xs = []
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        xs.extend([x1, x2])
                        ys.extend([y1, y2])
                    
                    if len(xs) >= 4 and len(ys) >= 4:
                        xs.sort()
                        ys.sort()
                        min_x = xs[int(len(xs) * 0.05)] / scale
                        max_x = xs[int(len(xs) * 0.95)] / scale
                        min_y = ys[int(len(ys) * 0.05)] / scale
                        max_y = ys[int(len(ys) * 0.95)] / scale
                        
                        pts = np.array([
                            [min_x, min_y],
                            [max_x, min_y],
                            [max_x, max_y],
                            [min_x, max_y]
                        ], dtype="float32")
                        return self.four_point_transform(image, pts)
            except Exception:
                pass

        # --- STRATEGY 4: FULL IMAGE FALLBACK ---
        return image.copy()
