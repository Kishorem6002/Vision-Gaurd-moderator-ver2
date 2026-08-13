import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image

from src.pipeline.moderation_orchestrator import (
    ModerationOrchestrator
)


def select_file():
    """
    Open Windows File Explorer and allow
    the user to select an image or video.
    """

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Media for VisionGuard",
        filetypes=[
            (
                "Media Files",
                "*.jpg *.jpeg *.png *.bmp *.webp "
                "*.mp4 *.avi *.mov *.mkv"
            ),
            ("All Files", "*.*")
        ]
    )

    root.destroy()

    return file_path


def main():

    print("=" * 70)
    print("VISIONGUARD")
    print("Privacy-Sensitive Image Moderation")
    print("=" * 70)

    # ------------------------------------------------------
    # STEP 1 - SELECT RAW IMAGE
    # ------------------------------------------------------

    print("\n[INPUT] Select a media file...")

    file_path = select_file()

    if not file_path:

        print("\nNo file selected.")

        return

    print(
        f"\n[INPUT] Selected file:"
    )

    print(
        f"        {file_path}"
    )

    file_extension = (
        Path(file_path)
        .suffix
        .lower()
    )

    # ------------------------------------------------------
    # CREATE ORCHESTRATOR
    # ------------------------------------------------------

    print(
        "\n[System] Initializing VisionGuard..."
    )

    orchestrator = ModerationOrchestrator()

    # ------------------------------------------------------
    # START COMPLETE PIPELINE
    # ------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("VISIONGUARD PIPELINE STARTED")
    print("=" * 70)

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    video_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    }

    try:

        if file_extension in image_extensions:

            image = Image.open(
                file_path
            ).convert("RGB")

            result = orchestrator.moderate(
                image=image,
                image_path=file_path
            )

        elif file_extension in video_extensions:

            result = (
                orchestrator.moderate_video(
                    file_path
                )
            )

        else:

            print(
                "Unsupported file type."
            )

            return

    except Exception as error:

        print("\n[ERROR] Pipeline failed:")

        print(error)

        return

    # ------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("VISIONGUARD FINAL RESULT")
    print("=" * 70)

    print(
        f"File     : {file_path}"
    )

    print(
        f"Decision : {result['decision']}"
    )

    print(
        f"Reason   : {result['reason']}"
    )

    if result.get("saved_path"):

        print(
            f"Stored At : "
            f"{result['saved_path']}"
        )

    print("=" * 70)

    # ------------------------------------------------------
    # GUI RESULT
    # ------------------------------------------------------

    root = tk.Tk()
    root.withdraw()

    if result["decision"] == "BLOCK":

        messagebox.showwarning(
            "VisionGuard - BLOCKED",
            "Sensitive content detected.\n\n"
            "The image must NOT be stored."
        )

    else:

        messagebox.showinfo(
            "VisionGuard - ALLOWED",
            "No explicit sensitive content detected.\n\n"
            "Image can proceed to storage."
        )

    root.destroy()


if __name__ == "__main__":
    main()