from nudenet import NudeDetector
from PIL import Image
import numpy as np
import cv2


class NudeNetDetector:
    """
    Pretrained NudeNet nudity detector.

    This detector performs object-level nudity detection.

    It returns:
        - detected class
        - confidence score
        - bounding box
    """

    def __init__(self):

        print("[NudeNet] Loading model...")

        # --------------------------------------------------
        # Load pretrained NudeNet model
        # --------------------------------------------------

        self.detector = NudeDetector()

        print("[NudeNet] Model loaded successfully.")

    # ======================================================
    # DETECT
    # ======================================================

    def detect(self, image):
        """
        Run NudeNet detection.

        Accepted input:
            - PIL Image
            - OpenCV NumPy image

        Returns:
            List of detections.
        """

        # --------------------------------------------------
        # Convert PIL -> OpenCV
        # --------------------------------------------------

        if isinstance(image, Image.Image):

            image = image.convert("RGB")

            image = np.array(image)

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

        # --------------------------------------------------
        # Validate image
        # --------------------------------------------------

        if not isinstance(image, np.ndarray):

            raise TypeError(
                "Image must be PIL Image or NumPy array."
            )

        # --------------------------------------------------
        # Run NudeNet
        # --------------------------------------------------

        detections = self.detector.detect(
            image
        )

        # --------------------------------------------------
        # Return detections
        # --------------------------------------------------

        return detections