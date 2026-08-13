import os
import sys
from PIL import Image


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


# ==========================================================
# IMPORT FALCONSAI DETECTOR
# ==========================================================

from src.models.falconsai_detector import (
    FalconsAINsfwDetector
)


# ==========================================================
# TEST IMAGE DIRECTORY
# ==========================================================

TEST_IMAGE_FOLDER = os.path.join(
    PROJECT_ROOT,
    "test_images"
)


VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
)


# ==========================================================
# FIND TEST IMAGES
# ==========================================================

def get_test_images(folder):

    image_files = []

    for root, dirs, files in os.walk(folder):

        for filename in files:

            if filename.lower().endswith(
                VALID_EXTENSIONS
            ):

                image_files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    return sorted(image_files)


# ==========================================================
# MAIN TEST
# ==========================================================

def main():

    print("=" * 70)
    print("VISIONGUARD - FALCONSAI NSFW DETECTOR TEST")
    print("=" * 70)

    # ------------------------------------------------------
    # Load detector
    # ------------------------------------------------------

    detector = FalconsAINsfwDetector()

    # ------------------------------------------------------
    # Find images
    # ------------------------------------------------------

    image_files = get_test_images(
        TEST_IMAGE_FOLDER
    )

    if not image_files:

        raise FileNotFoundError(
            f"No test images found in:\n"
            f"{TEST_IMAGE_FOLDER}"
        )

    print("\n")
    print("=" * 70)
    print(f"TEST IMAGES FOUND: {len(image_files)}")
    print("=" * 70)

    # ------------------------------------------------------
    # Run detection
    # ------------------------------------------------------

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        print("\n" + "-" * 70)

        relative_path = os.path.relpath(
            image_path,
            PROJECT_ROOT
        )

        print(
            f"[{index}/{len(image_files)}] "
            f"{relative_path}"
        )

        try:

            # Load image
            image = Image.open(
                image_path
            ).convert("RGB")

            # Run FalconS-AI
            result = detector.detect(
                image
            )

            # --------------------------------------------------
            # Display result
            # --------------------------------------------------

            print(
                f"NSFW Score : "
                f"{result['nsfw_score']:.4f}"
            )

            print(
                f"Safe Score : "
                f"{result['safe_score']:.4f}"
            )

            print(
                f"Prediction : "
                f"{result['predicted_class']}"
            )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

    print("\n")
    print("=" * 70)
    print("FALCONSAI TEST COMPLETED")
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()