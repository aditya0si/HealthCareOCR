import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SuperResolution:
    def __init__(self, model_path: str = "models/FSRCNN_x2.pb", min_dpi: int = 180, max_dim_for_sr: int = 2400, use_dnn_sr: bool = False):
        """
        Conditional Super Resolution module using OpenCV's DNN Super Resolution.
        """
        self.model_path = model_path
        self.min_dpi = min_dpi
        self.max_dim_for_sr = max_dim_for_sr
        self.sr = None
        self.use_dnn_sr = use_dnn_sr

    def _init_model(self):
        if self.sr is None:
            if not os.path.exists(self.model_path):
                logger.warning(f"Super resolution model file not found at: {self.model_path}; will fallback to cv2.resize")
                self.sr = None
                return

            logger.info(f"Initializing FSRCNN model from {self.model_path}")
            try:
                # Some OpenCV builds do not expose dnn_superres; guard against that
                self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
                self.sr.readModel(self.model_path)
                self.sr.setModel("fsrcnn", 2)
            except Exception:
                logger.exception("OpenCV dnn_superres is not available or failed to initialize; will fallback to cv2.resize upscaling")
                self.sr = None

    def estimate_dpi(self, image: np.ndarray) -> float:
        """
        Estimate the DPI of the image assuming standard A4 dimensions (8.27 x 11.69 inches).
        Accounts for landscape/portrait orientation.
        """
        h, w = image.shape[:2]
        if w >= h:
            # Landscape
            dpi_w = w / 11.69
            dpi_h = h / 8.27
        else:
            # Portrait
            dpi_w = w / 8.27
            dpi_h = h / 11.69
        return (dpi_w + dpi_h) / 2.0

    def should_upscale(self, image: np.ndarray) -> bool:
        """
        Check if the image should be upscaled based on estimated DPI and image dimensions.
        The target is a conservative 180–200 DPI workflow for OCR, so we trigger SR only
        when the estimated resolution is clearly below that band.
        """
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return False

        estimated_dpi = self.estimate_dpi(image)
        max_dim = max(h, w)
        if max_dim >= self.max_dim_for_sr:
            return False
        # Add small hysteresis to avoid frequent up/down scaling
        return estimated_dpi < (self.min_dpi - 5)

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale the image 2x using FSRCNN or cubic interpolation.
        Handles both grayscale and color images.
        """
        # Ensure we have initialized SR resources if using DNN upscaling
        if self.use_dnn_sr:
            self._init_model()

        # Estimate required scale factor to reach target DPI
        est_dpi = self.estimate_dpi(image)
        if est_dpi <= 0:
            return image
        scale = float(self.min_dpi) / float(est_dpi)
        if scale <= 1.0:
            return image
        # Cap to 2x for safety with FSRCNN_x2 model
        scale = min(scale, 2.0)

        # If we have the DNN SR available, use it (it expects BGR color images)
        is_gray = (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[2] == 1)
        if is_gray:
            if len(image.shape) == 3:
                gray_img = image[:, :, 0]
            else:
                gray_img = image
            color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        else:
            color_img = image

        if self.use_dnn_sr and self.sr is not None:
            try:
                upscaled = self.sr.upsample(color_img)
            except Exception:
                logger.exception("DNN SR failed during upsample; falling back to cv2.resize")
                new_w = int(round(color_img.shape[1] * scale))
                new_h = int(round(color_img.shape[0] * scale))
                upscaled = cv2.resize(color_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            # Fallback: use cv2.resize with cubic interpolation
            new_w = int(round(color_img.shape[1] * scale))
            new_h = int(round(color_img.shape[0] * scale))
            upscaled = cv2.resize(color_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        if is_gray:
            upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

        return upscaled
