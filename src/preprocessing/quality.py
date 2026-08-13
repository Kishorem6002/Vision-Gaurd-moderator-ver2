import cv2
import numpy as np


class ImageQualityAnalyzer:
    """
    Analyzes the quality of a camera frame before
    sending it to the NSFW detection pipeline.

    Currently checks:
        1. Blur
        2. Brightness
        3. Resolution

    IMPORTANT:
    This class does NOT modify the image.
    It only measures its quality.
    """

    def __init__(
        self,
        blur_threshold=100.0,
        brightness_low=50.0,
        brightness_high=200.0,
        min_width=224,
        min_height=224,
    ):
        self.blur_threshold = blur_threshold
        self.brightness_low = brightness_low
        self.brightness_high = brightness_high

        self.min_width = min_width
        self.min_height = min_height

    # ==========================================================
    # BLUR DETECTION
    # ==========================================================

    def calculate_blur_score(self, image):
        """
        Calculate blur using variance of Laplacian.

        Higher score  -> sharper image
        Lower score   -> blurrier image

        Returns:
            float
        """

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        laplacian = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        blur_score = laplacian.var()

        return float(blur_score)

    # ==========================================================
    # BRIGHTNESS
    # ==========================================================

    def calculate_brightness(self, image):
        """
        Calculate average image brightness.

        Returns:
            float between approximately 0 and 255
        """

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        brightness = np.mean(gray)

        return float(brightness)

    # ==========================================================
    # RESOLUTION
    # ==========================================================

    def check_resolution(self, image):
        """
        Check whether image resolution is sufficient.

        Returns:
            True  -> resolution is acceptable
            False -> resolution is too small
        """

        height, width = image.shape[:2]

        return (
            width >= self.min_width
            and
            height >= self.min_height
        )

    # ==========================================================
    # COMPLETE QUALITY ANALYSIS
    # ==========================================================

    def analyze(self, image):
        """
        Analyze the complete image quality.

        Returns a dictionary containing:

            blur_score
            is_blurry
            brightness
            is_dark
            is_overexposed
            width
            height
            resolution_ok
            quality_status
        """

        # ------------------------------------------------------
        # Validate image
        # ------------------------------------------------------

        if image is None:
            raise ValueError("Image is None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Image must be a NumPy array."
            )

        # ------------------------------------------------------
        # Resolution
        # ------------------------------------------------------

        height, width = image.shape[:2]

        resolution_ok = self.check_resolution(image)

        # ------------------------------------------------------
        # Blur
        # ------------------------------------------------------

        blur_score = self.calculate_blur_score(image)

        is_blurry = (
            blur_score < self.blur_threshold
        )

        # ------------------------------------------------------
        # Brightness
        # ------------------------------------------------------

        brightness = self.calculate_brightness(image)

        is_dark = (
            brightness < self.brightness_low
        )

        is_overexposed = (
            brightness > self.brightness_high
        )

        # ------------------------------------------------------
        # Overall quality status
        # ------------------------------------------------------

        if not resolution_ok:
            quality_status = "LOW_RESOLUTION"

        elif is_blurry:
            quality_status = "BLURRY"

        elif is_dark:
            quality_status = "LOW_LIGHT"

        elif is_overexposed:
            quality_status = "OVEREXPOSED"

        else:
            quality_status = "GOOD"

        # ------------------------------------------------------
        # Return results
        # ------------------------------------------------------

        return {
            "blur_score": round(blur_score, 2),
            "is_blurry": is_blurry,

            "brightness": round(brightness, 2),
            "is_dark": is_dark,
            "is_overexposed": is_overexposed,

            "width": width,
            "height": height,
            "resolution_ok": resolution_ok,

            "quality_status": quality_status,
        }