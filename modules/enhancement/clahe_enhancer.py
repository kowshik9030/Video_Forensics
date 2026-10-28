"""CLAHE low-quality footage enhancement.

Surveillance footage is frequently under-exposed, low-contrast or shot at
night. This module applies CLAHE (Contrast Limited Adaptive Histogram
Equalization) on the L channel of the LAB color space so that YOLO detection
receives frames with recovered local contrast instead of the raw degraded
input.
"""
import cv2
import numpy as np


class CLAHEEnhancer:
    """Enhances low-quality surveillance frames with CLAHE."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), enabled=True):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.enabled = enabled
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )

    def enhance(self, frame):
        """Return a CLAHE-enhanced copy of the frame.

        Works on the LAB lightness channel so colors are preserved while
        local contrast in dark/washed-out regions is recovered.
        """
        if not self.enabled or frame is None:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self._clahe.apply(l_channel)
        lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def enhance_gray(self, frame):
        """CLAHE on a grayscale conversion - used by detectors that only
        need single-channel input."""
        if not self.enabled or frame is None:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._clahe.apply(gray)
