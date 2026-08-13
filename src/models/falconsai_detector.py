import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)


class FalconsAINsfwDetector:
    """
    Pretrained FalconsAI NSFW image detector.

    This class:
    1. Loads the pretrained FalconsAI model.
    2. Uses GPU when CUDA is available.
    3. Accepts a PIL image.
    4. Returns SAFE/NSFW probabilities.
    """

    MODEL_NAME = "Falconsai/nsfw_image_detection"

    def __init__(self):
        # --------------------------------------------------
        # Select device
        # --------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"[FalconsAI] Device: {self.device}")

        if torch.cuda.is_available():
            print(
                f"[FalconsAI] GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

        # --------------------------------------------------
        # Load image processor
        # --------------------------------------------------

        print("[FalconsAI] Loading processor...")

        self.processor = AutoImageProcessor.from_pretrained(
            self.MODEL_NAME
        )

        # --------------------------------------------------
        # Load pretrained model
        # --------------------------------------------------

        print("[FalconsAI] Loading model...")

        self.model = AutoModelForImageClassification.from_pretrained(
            self.MODEL_NAME
        )

        # Move model to GPU/CPU
        self.model = self.model.to(self.device)

        # Evaluation mode
        self.model.eval()

        print("[FalconsAI] Model loaded successfully.")

        # --------------------------------------------------
        # Display model labels
        # --------------------------------------------------

        print(
            f"[FalconsAI] Labels: "
            f"{self.model.config.id2label}"
        )

    # ======================================================
    # DETECT
    # ======================================================

    def detect(self, image: Image.Image):
        """
        Run NSFW detection on one PIL image.

        Returns:
            {
                "nsfw_score": float,
                "safe_score": float,
                "predicted_class": str
            }
        """

        # --------------------------------------------------
        # Make sure image is RGB
        # --------------------------------------------------

        image = image.convert("RGB")

        # --------------------------------------------------
        # Preprocess
        # --------------------------------------------------

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        # Move tensors to GPU/CPU
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # --------------------------------------------------
        # Inference
        # --------------------------------------------------

        with torch.no_grad():

            outputs = self.model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

        # --------------------------------------------------
        # Extract probabilities using model labels
        # --------------------------------------------------

        scores = {}

        for class_id, probability in enumerate(probabilities):

            label = self.model.config.id2label[class_id]

            scores[label.lower()] = probability.item()

        # --------------------------------------------------
        # Get SAFE and NSFW scores
        # --------------------------------------------------

        nsfw_score = scores.get("nsfw", 0.0)
        safe_score = scores.get("normal", 0.0)

        # --------------------------------------------------
        # Determine predicted class
        # --------------------------------------------------

        predicted_class_id = torch.argmax(
            probabilities
        ).item()

        predicted_class = self.model.config.id2label[
            predicted_class_id
        ]

        # --------------------------------------------------
        # Return clean result
        # --------------------------------------------------

        return {
            "nsfw_score": nsfw_score,
            "safe_score": safe_score,
            "predicted_class": predicted_class,
        }