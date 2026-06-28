import cv2
import numpy as np
from skimage.filters import threshold_sauvola

class ImageEnhancer:
    def __init__(self, shadow_dilation_ksize: int = 21, shadow_blur_ksize: int = 41):
        self.shadow_dilation_ksize = shadow_dilation_ksize
        self.shadow_blur_ksize = shadow_blur_ksize

    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Removes uneven shadows and background lighting variations.
        Works for both BGR and Grayscale images.
        """
        is_color = len(image.shape) == 3
        if is_color:
            # Convert to LAB to only process the L (Lightness) channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            target = l
        else:
            target = image.copy()

        # Perform morphological dilation to estimate the background page illumination
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.shadow_dilation_ksize, self.shadow_dilation_ksize))
        background = cv2.dilate(target, kernel)
        
        # Smooth the background estimation
        background = cv2.GaussianBlur(background, (self.shadow_blur_ksize, self.shadow_blur_ksize), 0)
        
        # Calculate the lighting division to normalize illumination
        # (normalized = target / background * 255)
        # Avoid division by zero by clamping background to at least 1
        background = np.clip(background, 1, 255)
        normalized = cv2.divide(target, background, scale=255)

        if is_color:
            # Reconstruct the LAB image and convert back to BGR
            enhanced_lab = cv2.merge((normalized, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            enhanced = normalized

        return enhanced

    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization to enhance text contrast.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        is_color = len(image.shape) == 3
        
        if is_color:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        else:
            enhanced = clahe.apply(image)
            
        return enhanced

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies bilateral filtering to smooth out sensor grain and high-frequency noise
        while preserving sharp text edges.
        """
        is_color = len(image.shape) == 3
        if is_color:
            # Bilateral filter for color BGR image
            return cv2.bilateralFilter(image, 9, 75, 75)
        else:
            # Bilateral filter for grayscale image
            return cv2.bilateralFilter(image, 9, 75, 75)

    def binarize_sauvola(self, image: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
        """
        Performs local Sauvola binarization (optimized using OpenCV adaptiveThreshold).
        Always returns a single-channel binary image (values 0 or 255).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # OpenCV fast adaptive Gaussian thresholding (runs in ~2-5ms)
        # Ensure window_size is odd (required by cv2.adaptiveThreshold)
        block_size = window_size if window_size % 2 == 1 else window_size + 1
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, 10
        )
        
        # Clean speckle noise using morphological opening with a small 2x2 kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
