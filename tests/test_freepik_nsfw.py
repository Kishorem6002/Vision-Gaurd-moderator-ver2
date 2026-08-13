import os
import sys
import tkinter as tk

from tkinter import filedialog

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
# IMPORT MODEL
# ==========================================================

from src.models.nsfw_detector_2 import (
    FreepikNsfwDetector
)


# ==========================================================
# FILE PICKER
# ==========================================================

def select_image():

    root = tk.Tk()

    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Image for Freepik NSFW Test",
        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.png *.webp *.bmp"
            ),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()

    return file_path


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("VISIONGUARD - FREEPIK NSFW TEST")
    print("=" * 70)

    # ------------------------------------------------------
    # Select image
    # ------------------------------------------------------

    image_path = select_image()

    if not image_path:

        print(
            "\nNo image selected."
        )

        return

    print(
        f"\nImage:"
    )

    print(
        image_path
    )

    # ------------------------------------------------------
    # Load image
    # ------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    print(
        f"\nImage size: "
        f"{image.size}"
    )

    # ------------------------------------------------------
    # Load detector
    # ------------------------------------------------------

    detector = FreepikNsfwDetector()

    # ------------------------------------------------------
    # Detect
    # ------------------------------------------------------

    result = detector.detect(
        image
    )

    # ------------------------------------------------------
    # Display results
    # ------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FREEPIK NSFW RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"Neutral : "
        f"{result['neutral_score']:.4f}"
    )

    print(
        f"Low     : "
        f"{result['low_score']:.4f}"
    )

    print(
        f"Medium  : "
        f"{result['medium_score']:.4f}"
    )

    print(
        f"High    : "
        f"{result['high_score']:.4f}"
    )

    print(
        f"Level   : "
        f"{result['level']}"
    )

    print(
        "=" * 70
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()