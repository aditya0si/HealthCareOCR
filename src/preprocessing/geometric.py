import cv2
import numpy as np

class GeometricCorrector:
    def __init__(self, border_color: int = 255):
        """
        Initializes the geometric corrector.
        border_color: grayscale value for fill color (default: 255/white)
        """
        self.border_color = border_color

    def correct_skew(self, image: np.ndarray, skew_angle: float) -> np.ndarray:
        """
        Rotates the input image by the specified skew_angle to straighten it.
        Borders are filled with white color.
        """
        if abs(skew_angle) < 0.1:
            return image.copy()

        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix (negative angle rotates clockwise, but deskew angle is normally negative
        # when text is tilted clockwise, so we rotate by negative angle)
        # Note: deskew library returns angle where negative means tilted clockwise, so we need to
        # rotate by negative of the returned skew angle.
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        
        # Calculate new bounding dimensions of the rotated image to avoid clipping
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to take translation into account
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Determine border value based on BGR or Grayscale
        if len(image.shape) == 3:
            border_value = (self.border_color, self.border_color, self.border_color)
        else:
            border_value = self.border_color

        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                 flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=border_value)
        return rotated
