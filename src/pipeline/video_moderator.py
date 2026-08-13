import cv2
import importlib.util
from pathlib import Path

from PIL import Image

from src.video.frame_sampler import (
    FrameSampler
)

from src.video.batch_processor import (
    BatchProcessor
)


# ============================================================
# LOAD CropExtractor
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

CROP_EXTRACTOR_PATH = (
    PROJECT_ROOT
    / "video"
    / "crop.extractor.py"
)

if not CROP_EXTRACTOR_PATH.exists():

    raise FileNotFoundError(
        "\n[VideoModerator] CropExtractor file not found.\n"
        f"Expected path:\n"
        f"{CROP_EXTRACTOR_PATH}\n"
    )


_spec = importlib.util.spec_from_file_location(
    "crop_extractor",
    CROP_EXTRACTOR_PATH
)

if _spec is None or _spec.loader is None:

    raise ImportError(
        f"Could not load CropExtractor from:\n"
        f"{CROP_EXTRACTOR_PATH}"
    )


_mod = importlib.util.module_from_spec(
    _spec
)

_spec.loader.exec_module(
    _mod
)

CropExtractor = _mod.CropExtractor


# ============================================================
# VIDEO MODERATOR
# ============================================================

class VideoModerator:

    def __init__(
        self,
        orchestrator,
        sample_every=6
    ):
        """
        orchestrator:
            Existing ModerationOrchestrator.

        sample_every:
            Kept for compatibility with the existing API.

        NOTE:
            FrameSampler currently uses fps_to_analyze=2
            inside moderate_video().
        """

        self.orchestrator = orchestrator
        self.sample_every = sample_every


    # ========================================================
    # VIDEO MODERATION
    # ========================================================

    def moderate_video(
        self,
        video_path
    ):
        """
        Moderate video frame-by-frame.

        Pipeline:

            Video
              ↓
            FrameSampler
              ↓
            BatchProcessor
              ↓
            YOLO batch person detection
              ↓
            CropExtractor
              ↓
            Person crop
              ↓
            NudeNet
              ↓
            Explicit NudeNet?
              ↓
            BLOCK immediately
              ↓
            Save exact violating frame
              ↓
            Stop entire video

        If NudeNet does not find explicit content:

            FalconS-AI
              ↓
            Freepik
              ↓
            ALLOW / UNCERTAIN
        """

        print("\n")
        print("=" * 70)
        print("VISIONGUARD VIDEO MODERATION")
        print("=" * 70)

        print(
            f"[VIDEO] Input: {video_path}"
        )

        # ====================================================
        # FRAME SAMPLER
        # ====================================================

        sampler = FrameSampler(
            fps_to_analyze=2
        )

        # ====================================================
        # BATCH PROCESSOR
        # ====================================================

        processor = BatchProcessor(
            batch_size=16
        )

        # ====================================================
        # CROP EXTRACTOR
        # ====================================================

        extractor = CropExtractor()

        # ====================================================
        # FRAME GENERATOR
        # ====================================================

        frame_generator = sampler.sample(
            video_path
        )

        batches = processor.create_batches(
            frame_generator
        )

        analyzed_frames = 0

        # ====================================================
        # BATCH LOOP
        # ====================================================

        for batch in batches:

            print(
                f"\n[VIDEO] Processing batch "
                f"of {len(batch)} frames"
            )

            # =================================================
            # FRAME ENHANCEMENT + YOLO INPUT
            # =================================================
            #
            # Enhancement runs first so YOLO, CropExtractor,
            # and all models receive the enhanced frame.
            #
            # OpenCV frame = BGR
            # PIL image    = RGB
            #
            # YOLO receives PIL images, so BGR→RGB conversion
            # must follow enhancement.
            #
            # CropExtractor separately handles its own
            # BGR→RGB conversion for moderation crops.
            #
            # =================================================

            pil_images = []

            for item in batch:

                enhanced = (
                    self.orchestrator
                    .enhancer
                    .enhance_video_frame(
                        item["frame"]
                    )
                )

                item["frame"] = enhanced

                frame_rgb = cv2.cvtColor(
                    enhanced,
                    cv2.COLOR_BGR2RGB
                )

                pil_images.append(
                    Image.fromarray(
                        frame_rgb
                    )
                )

            # =================================================
            # YOLO BATCH DETECTION
            # =================================================

            batch_detections = (
                self.orchestrator
                .person_detector
                .detect_batch(
                    pil_images
                )
            )

            # =================================================
            # FRAME LOOP
            # =================================================

            for frame_index, detections in enumerate(
                batch_detections
            ):

                frame_item = batch[
                    frame_index
                ]

                frame_id = frame_item[
                    "frame_id"
                ]

                frame = frame_item[
                    "frame"
                ]

                analyzed_frames += 1

                print(
                    f"\n[Frame {frame_id}] "
                    f"{len(detections)} persons detected"
                )

                # =============================================
                # NO PERSONS
                # =============================================

                if not detections:

                    continue

                # =============================================
                # EXTRACT PERSON CROPS
                # =============================================

                crops = extractor.extract(
                    frame,
                    detections
                )

                # =============================================
                # SAFETY CHECK
                # =============================================

                if len(crops) != len(detections):

                    raise RuntimeError(
                        f"[Frame {frame_id}] "
                        f"Crop/detection mismatch: "
                        f"{len(detections)} detections, "
                        f"{len(crops)} crops."
                    )

                # =============================================
                # PERSON LOOP
                # =============================================

                for person_index, detection in enumerate(
                    detections
                ):

                    person_id = detection[
                        "person_id"
                    ]

                    crop = crops[
                        person_index
                    ]

                    print(
                        f"[Frame {frame_id}] "
                        f"[Person {person_id}] "
                        f"Moderating..."
                    )

                    # =========================================
                    # MODERATE PERSON
                    # =========================================

                    result = (
                        self.orchestrator
                        .moderate_person_crop(
                            crop=crop,
                            person_id=person_id,
                            frame_id=frame_id,
                            frame=frame
                        )
                    )

                    # =========================================
                    # IMMEDIATE BLOCK
                    # =========================================

                    if result.get(
                        "decision"
                    ) == "BLOCK":

                        print("\n")
                        print("=" * 70)
                        print("VIDEO VIOLATION DETECTED")
                        print("=" * 70)

                        print(
                            f"[VIDEO] BLOCK"
                        )

                        print(
                            f"[VIDEO] Frame  : "
                            f"{frame_id}"
                        )

                        print(
                            f"[VIDEO] Person : "
                            f"{person_id}"
                        )

                        print(
                            f"[VIDEO] Reason : "
                            f"{result.get('reason')}"
                        )

                        if result.get(
                            "flagged_frame_path"
                        ):

                            print(
                                f"[VIDEO] Flagged frame: "
                                f"{result['flagged_frame_path']}"
                            )

                        if result.get(
                            "flagged_crop_path"
                        ):

                            print(
                                f"[VIDEO] Flagged crop: "
                                f"{result['flagged_crop_path']}"
                            )

                        # =====================================
                        # STOP EVERYTHING
                        # =====================================
                        #
                        # This return exits:
                        #
                        # person loop
                        # frame loop
                        # batch loop
                        # video moderation
                        #
                        # =====================================

                        _video_block_result = {
                            "decision": "BLOCK",

                            "reason": result.get(
                                "reason",
                                "NUDENET_EXPLICIT_CONTENT"
                            ),

                            "frame_id": frame_id,

                            "person_id": person_id,

                            "flagged_frame_path": result.get(
                                "flagged_frame_path"
                            ),

                            "flagged_crop_path": result.get(
                                "flagged_crop_path"
                            ),

                            "evidence": result.get(
                                "evidence",
                                []
                            ),

                            "analyzed_frames": (
                                analyzed_frames
                            )
                        }

                        print("\n")
                        print("=" * 56)
                        print("VIDEO MODERATOR RETURNING DECISION:")
                        print(_video_block_result)
                        print("=" * 56)

                        return _video_block_result

        # ====================================================
        # ALL FRAMES PASSED
        # ====================================================

        print("\n")
        print("=" * 70)
        print("VIDEO MODERATION COMPLETE")
        print("=" * 70)

        print(
            f"[VIDEO] ALLOW — "
            f"{analyzed_frames} frames analyzed, "
            f"no violations found."
        )

        _video_allow_result = {
            "decision": "ALLOW",

            "reason": (
                "NO_SENSITIVE_CONTENT_DETECTED"
            ),

            "analyzed_frames": (
                analyzed_frames
            ),

            "flagged_frame_path": None,

            "flagged_crop_path": None
        }

        print("\n")
        print("=" * 56)
        print("VIDEO MODERATOR RETURNING DECISION:")
        print(_video_allow_result)
        print("=" * 56)

        return _video_allow_result