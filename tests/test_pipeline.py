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
# IMPORT ORCHESTRATOR
# ==========================================================

from src.pipeline.moderation_orchestrator import (
    ModerationOrchestrator
)


# ==========================================================
# FILE PICKER
# ==========================================================

def select_image():

    root = tk.Tk()

    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Image for VisionGuard Pipeline Test",
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
    print("VISIONGUARD - FULL PIPELINE TEST")
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
    # Load orchestrator
    # ------------------------------------------------------

    orchestrator = ModerationOrchestrator()

    # ------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------

    result = orchestrator.moderate(
        image
    )

    # ------------------------------------------------------
    # Final decision
    # ------------------------------------------------------

    decision = result.get(
        "decision",
        "UNKNOWN"
    )

    reason = result.get(
        "reason",
        "NO_REASON"
    )

    print("\n")
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Decision : {decision}"
    )

    print(
        f"Reason   : {reason}"
    )

    print("=" * 70)

    # ------------------------------------------------------
    # Per-person summary
    # ------------------------------------------------------

    persons = result.get(
        "persons",
        []
    )

    if not persons:

        print(
            "\n[Summary] No people detected."
        )

        return

    print(
        f"\n[Summary] "
        f"{len(persons)} person(s) detected."
    )

    for person in persons:

        person_id = person.get(
            "person_id",
            -1
        )

        print("\n" + "-" * 70)

        print(
            f"PERSON {person_id}"
        )

        print("-" * 70)

        # --------------------------------------------------
        # FalconS-AI
        # --------------------------------------------------

        falconsai = person.get(
            "falconsai",
            {}
        )

        print(
            f"\n[FalconS-AI]"
        )

        print(
            f"  NSFW score : "
            f"{falconsai.get('nsfw_score', 0.0):.4f}"
        )

        print(
            f"  Safe score : "
            f"{falconsai.get('safe_score', 0.0):.4f}"
        )

        print(
            f"  Prediction : "
            f"{falconsai.get('predicted_class', 'N/A')}"
        )

        # --------------------------------------------------
        # Freepik NSFW
        # --------------------------------------------------

        freepik = person.get(
            "freepik",
            {}
        )

        print(
            f"\n[Freepik NSFW]"
        )

        print(
            f"  Neutral : "
            f"{freepik.get('neutral_score', 0.0):.4f}"
        )

        print(
            f"  Low     : "
            f"{freepik.get('low_score', 0.0):.4f}"
        )

        print(
            f"  Medium  : "
            f"{freepik.get('medium_score', 0.0):.4f}"
        )

        print(
            f"  High    : "
            f"{freepik.get('high_score', 0.0):.4f}"
        )

        print(
            f"  Level   : "
            f"{freepik.get('level', 'N/A')}"
        )

        # --------------------------------------------------
        # NudeNet
        # --------------------------------------------------

        nudenet = person.get(
            "nudeNet",
            []
        )

        print(
            f"\n[NudeNet]"
        )

        if not nudenet:

            print(
                "  No detections."
            )

        else:

            print(
                f"  {len(nudenet)} detection(s)."
            )

            for detection in nudenet:

                print(
                    f"    Class : "
                    f"{detection.get('class', 'UNKNOWN')}"
                )

                print(
                    f"    Score : "
                    f"{detection.get('score', 0.0):.4f}"
                )

                print(
                    f"    Box   : "
                    f"{detection.get('box', [])}"
                )

    print("\n" + "=" * 70)
    print("PIPELINE TEST COMPLETED")
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
