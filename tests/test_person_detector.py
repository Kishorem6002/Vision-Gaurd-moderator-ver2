import os
import sys
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageDraw


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

from src.models.person_detector import (
    PersonDetector
)


# ==========================================================
# OUTPUT DIRECTORIES
# ==========================================================

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "person_detection"
)

CROPS_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "crops"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

os.makedirs(
    CROPS_FOLDER,
    exist_ok=True
)


# ==========================================================
# SELECT IMAGE
# ==========================================================

def select_image():
    """
    Open Windows File Explorer and allow the user
    to select an image.
    """

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Image for Person Detection",
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
    print("VISIONGUARD - ADAPTIVE PERSON DETECTOR TEST")
    print("=" * 70)

    # ------------------------------------------------------
    # Select image
    # ------------------------------------------------------

    print(
        "\n[Input] Select an image..."
    )

    image_path = select_image()

    if not image_path:

        print(
            "\nNo image selected."
        )

        return

    print(
        f"\nSelected image:"
    )

    print(
        image_path
    )

    # ------------------------------------------------------
    # Load detector
    # ------------------------------------------------------

    detector = PersonDetector(
        confidence_threshold=0.25,
        image_size=640
    )

    # ------------------------------------------------------
    # Load image
    # ------------------------------------------------------

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as error:

        print(
            f"\nFailed to load image:"
        )

        print(error)

        return

    print(
        f"\nImage size: {image.size}"
    )

    # ------------------------------------------------------
    # Create annotated image
    # ------------------------------------------------------

    annotated_image = image.copy()

    draw = ImageDraw.Draw(
        annotated_image
    )

    # ------------------------------------------------------
    # Detect people + adaptive crops
    # ------------------------------------------------------

    people = detector.detect_and_crop(
        image
    )

    print(
        f"\nPeople detected: "
        f"{len(people)}"
    )

    # ======================================================
    # SAVE ORIGINAL IMAGE
    # ======================================================

    original_path = os.path.join(
        OUTPUT_FOLDER,
        "original_input.jpg"
    )

    image.save(
        original_path
    )

    print(
        f"\nOriginal image saved:"
    )

    print(
        original_path
    )

    # ======================================================
    # PROCESS PERSONS
    # ======================================================

    for person in people:

        person_id = person[
            "person_id"
        ]

        confidence = person[
            "confidence"
        ]

        original_box = person[
            "box"
        ]

        crop_box = person[
            "crop_box"
        ]

        padding = person[
            "padding"
        ]

        crop = person[
            "crop"
        ]

        print(
            "\n" + "-" * 70
        )

        print(
            f"Person ID       : {person_id}"
        )

        print(
            f"Confidence      : "
            f"{confidence:.4f}"
        )

        print(
            f"Original box    : "
            f"{original_box}"
        )

        print(
            f"Adaptive box    : "
            f"{crop_box}"
        )

        print(
            f"Horizontal pad  : "
            f"{padding['horizontal'] * 100:.2f}%"
        )

        print(
            f"Vertical pad    : "
            f"{padding['vertical'] * 100:.2f}%"
        )

        print(
            f"Crop size       : "
            f"{crop.size}"
        )

        # ==================================================
        # DRAW ORIGINAL YOLO BOX
        # ==================================================

        x1, y1, x2, y2 = original_box

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=4
        )

        draw.text(
            (
                x1,
                max(0, y1 - 20)
            ),
            f"P{person_id} YOLO",
            fill="red"
        )

        # ==================================================
        # DRAW ADAPTIVE CROP BOX
        # ==================================================

        cx1, cy1, cx2, cy2 = crop_box

        draw.rectangle(
            [cx1, cy1, cx2, cy2],
            outline="green",
            width=4
        )

        draw.text(
            (
                cx1,
                min(
                    image.height - 20,
                    cy1 + 5
                )
            ),
            f"P{person_id} CROP",
            fill="green"
        )

        # ==================================================
        # SAVE PERSON CROP
        # ==================================================

        crop_path = os.path.join(
            CROPS_FOLDER,
            f"person_{person_id}.jpg"
        )

        crop.save(
            crop_path
        )

        print(
            f"Crop saved:"
        )

        print(
            crop_path
        )

    # ======================================================
    # SAVE ANNOTATED IMAGE
    # ======================================================

    annotated_path = os.path.join(
        OUTPUT_FOLDER,
        "persons_adaptive_crops.jpg"
    )

    annotated_image.save(
        annotated_path
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "Annotated image saved:"
    )

    print(
        annotated_path
    )

    # ======================================================
    # OUTPUT SUMMARY
    # ======================================================

    print(
        "\nOUTPUT STRUCTURE:"
    )

    print(
        f"{OUTPUT_FOLDER}/"
    )

    print(
        "├── original_input.jpg"
    )

    print(
        "├── persons_adaptive_crops.jpg"
    )

    print(
        "└── crops/"
    )

    for person in people:

        print(
            f"    └── "
            f"person_{person['person_id']}.jpg"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "ADAPTIVE PERSON DETECTION TEST COMPLETED"
    )

    print(
        "=" * 70
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()