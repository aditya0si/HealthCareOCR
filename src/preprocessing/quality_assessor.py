import cv2
import numpy as np
from dataclasses import dataclass
from deskew import determine_skew

@dataclass
class QualityMetrics:
    sharpness: float      # Laplacian variance normalized to [0, 1]
    contrast: float       # Standard deviation of pixel intensities normalized to [0, 1]
    brightness: float     # Mean pixel intensity normalized to [0, 1]
    noise_ratio: float    # High-frequency noise estimation in [0, 1]
    skew_angle: float     # Estimated skew angle in degrees
    is_blurry: bool
    is_low_contrast: bool
    is_too_dark: bool
    is_too_bright: bool
    is_noisy: bool
    is_skewed: bool

class QualityAssessor:
    def __init__(self, 
                 min_sharpness: float = 0.15,
                 min_contrast: float = 0.20,
                 min_brightness: float = 0.15,
                 max_brightness: float = 0.85,
                 max_noise_ratio: float = 0.25,
                 max_skew_angle: float = 1.0,
                 assessment_max_dim: int = 512):
        self.min_sharpness = min_sharpness
        self.min_contrast = min_contrast
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_noise_ratio = max_noise_ratio
        self.max_skew_angle = max_skew_angle
        self.assessment_max_dim = assessment_max_dim

    def assess(self, image: np.ndarray) -> QualityMetrics:
        """
        Assess 5 quality dimensions of the input image.
        Downscales the image internally to self.assessment_max_dim to guarantee
        low latency (<50ms) even on high-resolution inputs.
        """
        if len(image.shape) == 3:
            gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_orig = image.copy()

        # Downscale for performance
        h, w = gray_orig.shape[:2]
        if max(h, w) > self.assessment_max_dim:
            scale = self.assessment_max_dim / max(h, w)
            gray = cv2.resize(gray_orig, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            gray = gray_orig.copy()

        # 1. Sharpness (Laplacian Variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = float(laplacian_var / (laplacian_var + 100.0))

        # 2. Contrast (Intensity standard deviation)
        intensity_std = float(gray.std())
        contrast = float(min(1.0, intensity_std / 128.0))

        # 3. Brightness (Mean pixel intensity)
        mean_intensity = float(gray.mean())
        brightness = float(mean_intensity / 255.0)

        # 4. Noise Ratio (Bilateral filter diff std dev on downscaled image)
        # Small diameter, moderate color/space sigma for fast computation
        denoised = cv2.bilateralFilter(gray, 5, 30, 30)
        noise_diff = cv2.absdiff(gray, denoised)
        noise_std = float(noise_diff.std())
        noise_ratio = float(min(1.0, noise_std / 20.0))

        # 5. Skew Angle (Deskew on downscaled image is much faster)
        try:
            skew_angle = determine_skew(gray)
            if skew_angle is None:
                skew_angle = 0.0
            else:
                skew_angle = float(skew_angle)
        except Exception:
            skew_angle = 0.0

        # Assess boolean flags
        is_blurry = sharpness < self.min_sharpness
        is_low_contrast = contrast < self.min_contrast
        is_too_dark = brightness < self.min_brightness
        is_too_bright = brightness > self.max_brightness
        is_noisy = noise_ratio > self.max_noise_ratio
        is_skewed = abs(skew_angle) > self.max_skew_angle

        return QualityMetrics(
            sharpness=sharpness,
            contrast=contrast,
            brightness=brightness,
            noise_ratio=noise_ratio,
            skew_angle=skew_angle,
            is_blurry=is_blurry,
            is_low_contrast=is_low_contrast,
            is_too_dark=is_too_dark,
            is_too_bright=is_too_bright,
            is_noisy=is_noisy,
            is_skewed=is_skewed
        )
