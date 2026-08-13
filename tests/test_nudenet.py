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
# IMPORT NUDENET DETECTOR
# ==========================================================

from src.models.nudenet_detector import (
    NudeNetDetector
)


# ==========================================================
# PERSON CROPS DIRECTORY
# ==========================================================

PERSON_FOLDER = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "person_detection"
)


VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
)


# ==========================================================
# FIND PERSON CROPS
# ==========================================================

def get_person_images(folder):

    image_files = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            VALID_EXTENSIONS
        ):

            image_files.append(
                os.path.join(
                    folder,
                    filename
                )
            )

    return sorted(image_files)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("VISIONGUARD - NUDENET PERSON-CROP TEST")
    print("=" * 70)

    # ------------------------------------------------------
    # Check folder
    # ------------------------------------------------------

    if not os.path.exists(
        PERSON_FOLDER
    ):

        raise FileNotFoundError(
            f"Person detection folder not found:\n"
            f"{PERSON_FOLDER}\n\n"
            f"Run test_person_detector.py first."
        )

    # ------------------------------------------------------
    # Load NudeNet
    # ------------------------------------------------------

    detector = NudeNetDetector()

    # ------------------------------------------------------
    # Find person crops
    # ------------------------------------------------------

    person_images = get_person_images(
        PERSON_FOLDER
    )

    if not person_images:

        raise FileNotFoundError(
            "No person crop images found."
        )

    print(
        f"\nPerson crops found: "
        f"{len(person_images)}"
    )

    # ------------------------------------------------------
    # Run NudeNet on every person crop
    # ------------------------------------------------------

    for index, image_path in enumerate(
        person_images,
        start=1
    ):

        print("\n" + "=" * 70)

        filename = os.path.basename(
            image_path
        )

        print(
            f"[{index}/{len(person_images)}]"
        )

        print(
            f"Person crop: {filename}"
        )

        try:

            # --------------------------------------------------
            # Load crop
            # --------------------------------------------------

            image = Image.open(
                image_path
            ).convert("RGB")

            print(
                f"Crop size: {image.size}"
            )

            # --------------------------------------------------
            # NudeNet
            # --------------------------------------------------

            detections = detector.detect(
                image
            )

            # --------------------------------------------------
            # No detection
            # --------------------------------------------------

            if not detections:

                print(
                    "Result: NO DETECTIONS"
                )

                continue

            # --------------------------------------------------
            # Display detections
            # --------------------------------------------------

            print(
                f"Detections: "
                f"{len(detections)}"
            )

            for detection in detections:

                label = detection.get(
                    "class",
                    "UNKNOWN"
                )

                score = detection.get(
                    "score",
                    0.0
                )

                box = detection.get(
                    "box",
                    []
                )

                print(
                    f"  Class : {label}"
                )

                print(
                    f"  Score : {score:.4f}"
                )

                print(
                    f"  Box   : {box}"
                )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

    print("\n")
    print("=" * 70)
    print("PERSON-CROP NUDENET TEST COMPLETED")
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()