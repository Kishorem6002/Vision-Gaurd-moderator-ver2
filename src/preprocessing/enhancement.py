import cv2
import numpy as np


class ImageEnhancer:
    """
    Applies conditional image enhancement based on
    the quality analysis.

    Supported operations:
        - Mild sharpening for blurry images
        - CLAHE for low-light images
        - Mild denoising

    The original image is never modified directly.
    A new processed image is returned.
    """

    def __init__(
        self,
        enable_sharpen=True,
        enable_clahe=True,
        enable_denoise=True,
    ):
        self.enable_sharpen = enable_sharpen
        self.enable_clahe = enable_clahe
        self.enable_denoise = enable_denoise

    # ==========================================================
    # SHARPENING
    # ==========================================================

    def sharpen(self, image):
        """
        Apply mild unsharp masking.

        This is intended for mild blur.
        It cannot recover information that was never captured.
        """

        gaussian = cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=1.2
        )

        sharpened = cv2.addWeighted(
            image,
            1.5,
            gaussian,
            -0.5,
            0
        )

        return sharpened

    # ==========================================================
    # CLAHE
    # ==========================================================

    def enhance_low_light(self, image):
        """
        Improve local contrast using CLAHE.

        CLAHE is applied to the L channel of LAB color space
        so that color information is not unnecessarily distorted.
        """

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced_l = clahe.apply(l_channel)

        enhanced_lab = cv2.merge(
            (
                enhanced_l,
                a_channel,
                b_channel
            )
        )

        enhanced = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR
        )

        return enhanced

    # ==========================================================
    # DENOISING
    # ==========================================================

    def denoise(self, image):
        """
        Apply mild color image denoising.
        """

        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=3,
            hColor=3,
            templateWindowSize=7,
            searchWindowSize=21
        )

    # ==========================================================
    # VIDEO FRAME ENHANCEMENT
    # ==========================================================

    def enhance_video_frame(self, frame):
        """
        Lightweight enhancement for video frames.

        No quality analysis required.

        Pipeline:
            CLAHE  →  unsharp mask

        Input : BGR numpy array
        Output: BGR numpy array, same shape and dtype
        """

        if frame is None:
            raise ValueError("Frame is None.")

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "Frame must be a NumPy array."
            )

        processed = frame.copy()

        # ------------------------------------------------------
        # CLAHE — always applied, improves local contrast
        # ------------------------------------------------------

        processed = self.enhance_low_light(
            processed
        )

        # ------------------------------------------------------
        # Unsharp mask — always applied, recovers mild blur
        # ------------------------------------------------------

        processed = self.sharpen(
            processed
        )

        return processed

    # ==========================================================
    # COMPLETE ENHANCEMENT
    # ==========================================================

    def enhance(self, image, quality_result):
        """
        Apply enhancement based on quality_result.

        Expected quality_result format:

        {
            "is_blurry": bool,
            "is_dark": bool,
            "quality_status": str
        }

        Returns:
            enhanced image
        """

        if image is None:
            raise ValueError("Image is None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Image must be a NumPy array."
            )

        # ------------------------------------------------------
        # NEVER modify the original frame
        # ------------------------------------------------------

        processed = image.copy()

        # ------------------------------------------------------
        # Denoise first
        # ------------------------------------------------------

        if self.enable_denoise:

            processed = self.denoise(processed)

        # ------------------------------------------------------
        # Low-light enhancement
        # ------------------------------------------------------

        if (
            self.enable_clahe
            and quality_result.get("is_dark", False)
        ):

            processed = self.enhance_low_light(
                processed
            )

        # ------------------------------------------------------
        # Sharpen if blurry
        # ------------------------------------------------------

        if (
            self.enable_sharpen
            and quality_result.get("is_blurry", False)
        ):

            processed = self.sharpen(
                processed
            )

        return processed