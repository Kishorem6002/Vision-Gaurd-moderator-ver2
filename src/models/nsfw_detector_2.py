import torch
from PIL import Image

from nsfw_image_detector import NSFWDetector


class FreepikNsfwDetector:
    """
    Pretrained Freepik NSFW detector.

    The model provides cumulative NSFW probabilities
    for four levels:

        neutral
        low
        medium
        high

    Interpretation:

        low    = P(low + medium + high)
        medium = P(medium + high)
        high   = P(high)

    Therefore, these scores must NOT be added together.
    """

    def __init__(self):

        # ==================================================
        # DEVICE
        # ==================================================

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[FreepikNSFW] Device: "
            f"{self.device}"
        )

        if torch.cuda.is_available():

            print(
                f"[FreepikNSFW] GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

        # ==================================================
        # DATA TYPE
        # ==================================================

        if (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):

            self.dtype = torch.bfloat16

        else:

            self.dtype = torch.float32

        print(
            f"[FreepikNSFW] "
            f"Dtype: {self.dtype}"
        )

        # ==================================================
        # LOAD MODEL
        # ==================================================

        print(
            "[FreepikNSFW] Loading model..."
        )

        self.detector = NSFWDetector(
            dtype=self.dtype,
            device=self.device
        )

        print(
            "[FreepikNSFW] "
            "Model loaded successfully."
        )

    # ======================================================
    # DETECT
    # ======================================================

    def detect(
        self,
        image: Image.Image
    ):
        """
        Run Freepik NSFW classification.

        Returns:

            {
                "neutral_score": float,
                "low_score": float,
                "medium_score": float,
                "high_score": float,
                "level": str
            }

        The returned scores are cumulative thresholds:

            low    = low + medium + high
            medium = medium + high
            high   = high
        """

        # --------------------------------------------------
        # Validate image
        # --------------------------------------------------

        if not isinstance(
            image,
            Image.Image
        ):

            raise TypeError(
                "Input must be a PIL Image."
            )

        image = image.convert("RGB")

        # --------------------------------------------------
        # Run prediction
        # --------------------------------------------------

        probabilities = (
            self.detector.predict_proba(
                image
            )
        )

        # --------------------------------------------------
        # Normalize output
        # --------------------------------------------------

        scores = {}

        for item in probabilities:

            for key, value in item.items():

                label = (
                    key.value
                    if hasattr(key, "value")
                    else str(key)
                )

                scores[
                    label.lower()
                ] = float(value)

        # --------------------------------------------------
        # Extract cumulative scores
        # --------------------------------------------------

        neutral_score = scores.get(
            "neutral",
            0.0
        )

        low_score = scores.get(
            "low",
            0.0
        )

        medium_score = scores.get(
            "medium",
            0.0
        )

        high_score = scores.get(
            "high",
            0.0
        )

        # ==================================================
        # DETERMINE RISK LEVEL
        # ==================================================

        # IMPORTANT:
        # We do NOT sum low + medium + high.
        #
        # The scores are cumulative thresholds.
        #
        # Example:
        #
        # low    = 1.00
        # medium = 1.00
        # high   = 0.45
        #
        # This means:
        #
        # P(NSFW >= low)    = 1.00
        # P(NSFW >= medium) = 1.00
        # P(NSFW >= high)   = 0.45

        if high_score >= 0.50:

            level = "high"

        elif medium_score >= 0.50:

            level = "medium"

        elif low_score >= 0.50:

            level = "low"

        else:

            level = "neutral"

        # --------------------------------------------------
        # Return
        # --------------------------------------------------

        return {
            "neutral_score": neutral_score,
            "low_score": low_score,
            "medium_score": medium_score,
            "high_score": high_score,
            "level": level
        }