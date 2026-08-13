import cv2
import numpy as np

from PIL import Image

from pathlib import Path

from src.preprocessing.quality import (
    ImageQualityAnalyzer
)

from src.preprocessing.enhancement import (
    ImageEnhancer
)

from src.models.person_detector import (
    PersonDetector
)

from src.models.falconsai_detector import (
    FalconsAINsfwDetector
)

from src.models.nsfw_detector_2 import (
    FreepikNsfwDetector
)

from src.models.nudenet_detector import (
    NudeNetDetector
)

from src.pipeline.video_moderator import (
    VideoModerator
)


# ============================================================
# MODERATION ORCHESTRATOR
# ============================================================

class ModerationOrchestrator:

    def __init__(
        self,
        nsfw_threshold=0.50,
        nudenet_threshold=0.50,
        person_confidence=0.25
    ):

        print("\n")
        print("=" * 70)
        print("VISIONGUARD - MODERATION ORCHESTRATOR")
        print("=" * 70)

        # ==================================================
        # CONFIGURATION
        # ==================================================

        self.nsfw_threshold = (
            nsfw_threshold
        )

        self.nudenet_threshold = (
            nudenet_threshold
        )

        self.person_confidence = (
            person_confidence
        )

        print(
            f"[Orchestrator] "
            f"FalconS-AI threshold: "
            f"{self.nsfw_threshold}"
        )

        print(
            f"[Orchestrator] "
            f"NudeNet threshold: "
            f"{self.nudenet_threshold}"
        )

        print(
            f"[Orchestrator] "
            f"Person confidence: "
            f"{self.person_confidence}"
        )

        # ==================================================
        # INITIALIZE COMPONENTS
        # ==================================================

        print(
            "\n[Orchestrator] "
            "Initializing components..."
        )

        self.quality_analyzer = (
            ImageQualityAnalyzer()
        )

        self.enhancer = (
            ImageEnhancer()
        )

        self.person_detector = (
            PersonDetector(
                confidence_threshold=(
                    self.person_confidence
                )
            )
        )

        self.falconsai = (
            FalconsAINsfwDetector()
        )

        self.freepik = (
            FreepikNsfwDetector()
        )

        self.nudenet = (
            NudeNetDetector()
        )

        print(
            "\n[Orchestrator] "
            "All components ready."
        )


    # ========================================================
    # VIDEO MODERATION
    # ========================================================

    def moderate_video(
        self,
        video_path
    ):

        moderator = VideoModerator(
            orchestrator=self,
            sample_every=6
        )

        return moderator.moderate_video(
            video_path
        )


    # ========================================================
    # PER-PERSON CROP MODERATION
    # ========================================================

    def moderate_person_crop(
        self,
        crop,
        person_id,
        frame_id,
        frame=None
    ):
        """
        Moderate one person crop from a video.

        Decision priority:

        1. NudeNet explicit detection
           -> BLOCK immediately

        2. Falcon / Freepik suspicious
           -> UNCERTAIN

        3. No evidence
           -> ALLOW

        NudeNet is the ONLY model allowed to BLOCK.

        When NudeNet blocks:

            - exact original frame is saved
            - violating crop is saved
            - evidence is returned
            - processing stops immediately
        """

        # ==================================================
        # OUTPUT DIRECTORIES
        # ==================================================

        PROJECT_ROOT = (
            Path(__file__)
            .resolve()
            .parent
            .parent
            .parent
        )

        outputs_dir = (
            PROJECT_ROOT
            / "outputs"
        )

        debug_dir = (
            outputs_dir
            / "debug"
        )

        flagged_frame_dir = (
            outputs_dir
            / "flagged_frames"
        )

        flagged_crop_dir = (
            outputs_dir
            / "flagged_crops"
        )

        debug_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        flagged_frame_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        flagged_crop_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ==================================================
        # EXPLICIT NUDENET CLASSES
        # ==================================================

        EXPLICIT_NUDENET_CLASSES = {
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "BUTTOCKS_EXPOSED",
            "ANUS_EXPOSED"
        }

        # ==================================================
        # DEBUG HEADER
        # ==================================================

        print("\n")
        print("=" * 56)
        print(
            f"FRAME {frame_id} | PERSON {person_id}"
        )
        print("=" * 56)

        # ==================================================
        # FALCONSAI — DEBUG
        # ==================================================

        falconsai_debug = (
            self.falconsai.detect(
                crop
            )
        )

        print("\nFALCONSAI RAW RESULT:")
        print(falconsai_debug)

        _falcon_nsfw = float(
            falconsai_debug.get(
                "nsfw_score",
                0.0
            )
        )

        _falcon_safe = float(
            falconsai_debug.get(
                "safe_score",
                0.0
            )
        )

        _falcon_class = falconsai_debug.get(
            "predicted_class",
            ""
        )

        print("\nExtracted:")
        print(f"nsfw_score      = {_falcon_nsfw:.4f}")
        print(f"safe_score      = {_falcon_safe:.4f}")
        print(f"predicted_class = {_falcon_class}")

        print("\n" + "-" * 56)

        # ==================================================
        # FREEPIK — DEBUG
        # ==================================================

        freepik_debug = (
            self.freepik.detect(
                crop
            )
        )

        print("\nFREEPIK RAW RESULT:")
        print(freepik_debug)

        _fp_neutral = float(
            freepik_debug.get(
                "neutral_score",
                0.0
            )
        )

        _fp_low = float(
            freepik_debug.get(
                "low_score",
                0.0
            )
        )

        _fp_medium = float(
            freepik_debug.get(
                "medium_score",
                0.0
            )
        )

        _fp_high = float(
            freepik_debug.get(
                "high_score",
                0.0
            )
        )

        _fp_level = freepik_debug.get(
            "level",
            "neutral"
        )

        print("\nExtracted:")
        print(f"neutral = {_fp_neutral:.4f}")
        print(f"low     = {_fp_low:.4f}")
        print(f"medium  = {_fp_medium:.4f}")
        print(f"high    = {_fp_high:.4f}")
        print(f"level   = {_fp_level}")

        print("\n" + "-" * 56)

        # ==================================================
        # NUDENET — RUN FIRST
        # ==================================================

        print(
            f"[Frame {frame_id}] "
            f"[Person {person_id}] "
            f"Running NudeNet..."
        )

        nudenet_results = (
            self.nudenet.detect(
                crop
            )
        )

        print(
            f"[Frame {frame_id}] "
            f"[Person {person_id}] "
            f"NudeNet: "
            f"{len(nudenet_results)} detections"
        )

        # --------------------------------------------------
        # NUDENET RAW DEBUG
        # --------------------------------------------------

        print("\nNUDENET RAW RESULT:")
        print("[")

        for _i, _det in enumerate(nudenet_results):

            print(f"    {_det}")

        print("]")

        # ==================================================
        # INSPECT EVERY NUDENET RESULT
        # ==================================================

        explicit_evidence = []

        for detection in nudenet_results:

            # ----------------------------------------------
            # Normalize class name
            # ----------------------------------------------

            raw_class = (
                detection.get(
                    "class",
                    detection.get(
                        "label",
                        detection.get(
                            "category",
                            ""
                        )
                    )
                )
            )

            class_name = str(
                raw_class
            ).strip().upper()

            # ----------------------------------------------
            # Score
            # ----------------------------------------------

            try:

                score = float(
                    detection.get(
                        "score",
                        detection.get(
                            "confidence",
                            0.0
                        )
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                score = 0.0

            # ----------------------------------------------
            # Bounding box
            # ----------------------------------------------

            box = detection.get(
                "box",
                detection.get(
                    "bbox",
                    []
                )
            )

            # ----------------------------------------------
            # Print EVERYTHING
            # ----------------------------------------------

            print(
                f"    class = "
                f"{class_name}"
            )

            print(
                f"    score = "
                f"{score:.4f}"
            )

            print(
                f"    box   = "
                f"{box}"
            )

            # ----------------------------------------------
            # NUDENET FOCUS DEBUG
            # ----------------------------------------------

            print("\nRaw class received:")
            print(f"    {class_name}")
            print("\nThreshold:")
            print(f"    {self.nudenet_threshold:.2f}")
            print("\nDetection score:")
            print(f"    {score:.4f}")
            print("\nClass in dangerous_classes?")
            print(
                f"    "
                f"{'True' if class_name in EXPLICIT_NUDENET_CLASSES else 'False'}"
            )
            print("\nWill trigger BLOCK?")
            print(
                f"    "
                f"{'True' if class_name in EXPLICIT_NUDENET_CLASSES and score >= self.nudenet_threshold else 'False'}"
            )

            # ----------------------------------------------
            # EXPLICIT CONTENT
            # ----------------------------------------------

            if (
                class_name
                in EXPLICIT_NUDENET_CLASSES
                and
                score >= self.nudenet_threshold
            ):

                explicit_evidence.append(
                    {
                        "class": class_name,
                        "score": score,
                        "box": box
                    }
                )

        # ==================================================
        # DECISION ANALYSIS DEBUG
        # ==================================================

        _falcon_suspicious = (
            _falcon_nsfw >= self.nsfw_threshold
        )

        _freepik_suspicious = (
            _fp_level in {"high", "medium"}
        )

        print("\n" + "-" * 56)
        print("\nDECISION ANALYSIS:")
        print("")
        print(
            f"Falcon suspicious?  "
            f"{_falcon_suspicious}"
        )
        print(
            f"Freepik suspicious? "
            f"{_freepik_suspicious}"
        )
        print("\nChecking NudeNet detections...")

        for _det in nudenet_results:

            _rc = (
                _det.get(
                    "class",
                    _det.get(
                        "label",
                        _det.get(
                            "category",
                            ""
                        )
                    )
                )
            )

            _cn = str(_rc).strip().upper()

            try:
                _sc = float(
                    _det.get(
                        "score",
                        _det.get(
                            "confidence",
                            0.0
                        )
                    )
                )
            except (TypeError, ValueError):
                _sc = 0.0

            print(f"\nClass: {_cn}")
            print(f"Score: {_sc:.4f}")
            print(
                f"Dangerous class? "
                f"{'True' if _cn in EXPLICIT_NUDENET_CLASSES else 'False'}"
            )
            print(
                f"Above threshold? "
                f"{'True' if _sc >= self.nudenet_threshold else 'False'}"
            )

        # ==================================================
        # NUDENET BLOCK
        # ==================================================

        if explicit_evidence:

            print("\n")
            print("=" * 70)
            print("NUDENET EXPLICIT CONTENT DETECTED")
            print("=" * 70)

            for evidence in explicit_evidence:

                print(
                    f"[NUDENET] "
                    f"{evidence['class']} "
                    f"score={evidence['score']:.4f} "
                    f"box={evidence['box']}"
                )

            print(
                "[NUDENET] >>> BLOCK"
            )

            # ==================================================
            # SAVE EXACT FLAGGED CROP
            # ==================================================

            flagged_crop_path = (
                flagged_crop_dir
                / (
                    f"frame_{frame_id}"
                    f"_person_{person_id}"
                    "_NUDENET_FLAGGED.jpg"
                )
            )

            try:

                crop.save(
                    flagged_crop_path
                )

                print(
                    f"[NUDENET] "
                    f"Flagged crop saved: "
                    f"{flagged_crop_path}"
                )

            except Exception as exc:

                print(
                    f"[WARNING] "
                    f"Could not save flagged crop: "
                    f"{exc}"
                )

                flagged_crop_path = None

            # ==================================================
            # SAVE EXACT ORIGINAL VIDEO FRAME
            # ==================================================

            flagged_frame_path = None

            if frame is not None:

                flagged_frame_path = (
                    flagged_frame_dir
                    / (
                        f"frame_{frame_id}"
                        f"_person_{person_id}"
                        "_NUDENET_FLAGGED.jpg"
                    )
                )

                try:

                    # ------------------------------------------
                    # `frame` comes from OpenCV / VideoSampler
                    # and is therefore BGR.
                    #
                    # cv2.imwrite expects BGR.
                    #
                    # Therefore DO NOT convert here.
                    # ------------------------------------------

                    success = cv2.imwrite(
                        str(
                            flagged_frame_path
                        ),
                        frame
                    )

                    if success:

                        print(
                            f"[NUDENET] "
                            f"Exact flagged frame saved: "
                            f"{flagged_frame_path}"
                        )

                    else:

                        print(
                            "[WARNING] "
                            "cv2.imwrite failed."
                        )

                        flagged_frame_path = None

                except Exception as exc:

                    print(
                        f"[WARNING] "
                        f"Could not save flagged frame: "
                        f"{exc}"
                    )

                    flagged_frame_path = None

            # ==================================================
            # RETURN IMMEDIATELY
            # ==================================================

            print(
                "[NUDENET] "
                "Stopping moderation immediately."
            )

            _block_result = {
                "decision": "BLOCK",

                "reason": (
                    "NUDENET_EXPLICIT_CONTENT: "
                    +
                    ", ".join(
                        [
                            (
                                f"{e['class']} "
                                f"({e['score']:.4f})"
                            )
                            for e
                            in explicit_evidence
                        ]
                    )
                ),

                "frame_id": frame_id,

                "person_id": person_id,

                "flagged_frame_path": (
                    str(
                        flagged_frame_path
                    )
                    if flagged_frame_path
                    else None
                ),

                "flagged_crop_path": (
                    str(
                        flagged_crop_path
                    )
                    if flagged_crop_path
                    else None
                ),

                "evidence": (
                    explicit_evidence
                )
            }

            print("\nFINAL PERSON DECISION:")
            print("BLOCK")
            print("\nReason:")
            print("NudeNet explicit content detected")
            print("\nRETURNING DECISION:")
            print(_block_result)
            print("=" * 56)

            return _block_result

        # ==================================================
        # NO EXPLICIT NUDENET
        # ==================================================

        print(
            f"[Frame {frame_id}] "
            f"[Person {person_id}] "
            f"NudeNet: "
            f"No explicit content."
        )

        # ==================================================
        # FALCON S-AI
        # ==================================================

        suspicious_signals = []

        # Re-use the debug run already executed above
        falconsai_result = falconsai_debug

        nsfw_score = _falcon_nsfw

        print(
            f"[Frame {frame_id}] "
            f"[Person {person_id}] "
            f"FalconS-AI: "
            f"{nsfw_score:.4f}"
        )

        if (
            nsfw_score
            >= self.nsfw_threshold
        ):

            suspicious_signals.append(
                f"FalconS-AI "
                f"{nsfw_score:.4f}"
            )

            debug_filename = (
                f"frame_{frame_id}"
                f"_person_{person_id}"
                "_falcon_suspicious.jpg"
            )

            debug_path = (
                debug_dir
                / debug_filename
            )

            try:

                crop.save(
                    debug_path
                )

                print(
                    f"[DEBUG] "
                    f"Saved suspicious crop: "
                    f"{debug_path}"
                )

            except Exception as exc:

                print(
                    f"[WARNING] "
                    f"Could not save Falcon crop: "
                    f"{exc}"
                )

        # ==================================================
        # FREEPIK
        # ==================================================

        # Re-use the debug run already executed above
        freepik_result = freepik_debug

        freepik_level = _fp_level

        print(
            f"[Frame {frame_id}] "
            f"[Person {person_id}] "
            f"Freepik: "
            f"{freepik_level}"
        )

        if (
            freepik_level
            in {
                "high",
                "medium"
            }
        ):

            suspicious_signals.append(
                f"Freepik "
                f"{freepik_level}"
            )

            debug_filename = (
                f"frame_{frame_id}"
                f"_person_{person_id}"
                f"_freepik_{freepik_level}.jpg"
            )

            debug_path = (
                debug_dir
                / debug_filename
            )

            try:

                crop.save(
                    debug_path
                )

                print(
                    f"[DEBUG] "
                    f"Saved suspicious crop: "
                    f"{debug_path}"
                )

            except Exception as exc:

                print(
                    f"[WARNING] "
                    f"Could not save Freepik crop: "
                    f"{exc}"
                )

        # ==================================================
        # ENSEMBLE DECISION
        # ==================================================

        if suspicious_signals:

            _uncertain_result = {
                "decision": "UNCERTAIN",

                "reason": (
                    "Suspicious signals: "
                    +
                    ", ".join(
                        suspicious_signals
                    )
                ),

                "frame_id": frame_id,

                "person_id": person_id,

                "evidence": []
            }

            print("\nFINAL PERSON DECISION:")
            print("UNCERTAIN")
            print("\nReason:")
            print(_uncertain_result["reason"])
            print("\nRETURNING DECISION:")
            print(_uncertain_result)
            print("=" * 56)

            return _uncertain_result

        # ==================================================
        # ALLOW
        # ==================================================

        _allow_result = {
            "decision": "ALLOW",

            "reason": (
                "NO_SENSITIVE_CONTENT_DETECTED"
            ),

            "frame_id": frame_id,

            "person_id": person_id,

            "evidence": []
        }

        print("\nFINAL PERSON DECISION:")
        print("ALLOW")
        print("\nReason:")
        print("No sensitive content detected")
        print("\nRETURNING DECISION:")
        print(_allow_result)
        print("=" * 56)

        return _allow_result


    # ========================================================
    # MAIN IMAGE MODERATION
    # ========================================================

    def moderate(
        self,
        image: Image.Image,
        image_path=None
    ):
        """
        Existing image moderation pipeline.

        This section remains separate from the video
        pipeline.
        """

        if not isinstance(
            image,
            Image.Image
        ):

            raise TypeError(
                "Input must be a PIL Image."
            )

        image = image.convert(
            "RGB"
        )

        print("\n")
        print("=" * 70)
        print("STARTING MODERATION")
        print("=" * 70)

        # ==================================================
        # PIL -> OpenCV
        # ==================================================

        image_rgb = np.array(
            image
        )

        image_cv = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR
        )

        # ==================================================
        # QUALITY
        # ==================================================

        print(
            "\n[1/6] "
            "Image quality analysis..."
        )

        quality_result = (
            self.quality_analyzer.analyze(
                image_cv
            )
        )

        print(
            f"[Quality] "
            f"{quality_result}"
        )

        # ==================================================
        # ENHANCEMENT
        # ==================================================

        print(
            "\n[2/6] "
            "Image enhancement..."
        )

        quality_status = (
            quality_result.get(
                "quality_status",
                "GOOD"
            )
        )

        if quality_status == "GOOD":

            enhanced_image = image

            print(
                "[Enhancement] "
                "Image quality is GOOD. "
                "Enhancement skipped."
            )

        else:

            enhanced_cv = (
                self.enhancer.enhance(
                    image_cv,
                    quality_result
                )
            )

            enhanced_rgb = cv2.cvtColor(
                enhanced_cv,
                cv2.COLOR_BGR2RGB
            )

            enhanced_image = (
                Image.fromarray(
                    enhanced_rgb
                )
            )

        # ==================================================
        # PERSON DETECTION
        # ==================================================

        print(
            "\n[3/6] "
            "Detecting people..."
        )

        person_results = (
            self.person_detector
            .detect_and_crop(
                enhanced_image
            )
        )

        if not person_results:

            return {
                "decision": "ALLOW",
                "reason": "NO_PERSON_DETECTED",
                "quality": quality_result,
                "persons": [],
                "saved_path": None
            }

        # ==================================================
        # IMAGE PIPELINE
        #
        # Keep your existing image pipeline here if you
        # already have additional reporting requirements.
        # ==================================================

        final_results = []

        for person in person_results:

            person_id = person[
                "person_id"
            ]

            crop = person[
                "crop"
            ]

            print(
                f"\n[Person {person_id}] "
                f"Running NudeNet..."
            )

            nudenet_results = (
                self.nudenet.detect(
                    crop
                )
            )

            final_results.append(
                {
                    **{
                        k: v
                        for k, v
                        in person.items()
                        if k != "crop"
                    },
                    "nudeNet": (
                        nudenet_results
                    )
                }
            )

        decision = (
            self._make_final_decision(
                final_results,
                quality_result
            )
        )

        return {
            "decision": decision[
                "decision"
            ],

            "reason": decision[
                "reason"
            ],

            "quality": quality_result,

            "persons": final_results,

            "saved_path": None
        }


    # ========================================================
    # FINAL IMAGE DECISION
    # ========================================================

    def _make_final_decision(
        self,
        person_results,
        quality_result
    ):

        dangerous_classes = {
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "BUTTOCKS_EXPOSED",
            "ANUS_EXPOSED"
        }

        # ==================================================
        # NUDENET FIRST
        # ==================================================

        for person in person_results:

            person_id = person.get(
                "person_id",
                -1
            )

            for detection in person.get(
                "nudeNet",
                []
            ):

                raw_class = (
                    detection.get(
                        "class",
                        detection.get(
                            "label",
                            detection.get(
                                "category",
                                ""
                            )
                        )
                    )
                )

                class_name = str(
                    raw_class
                ).strip().upper()

                try:

                    score = float(
                        detection.get(
                            "score",
                            0.0
                        )
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    score = 0.0

                if (
                    class_name
                    in dangerous_classes
                    and
                    score
                    >= self.nudenet_threshold
                ):

                    return {
                        "decision": "BLOCK",

                        "reason": (
                            f"NudeNet detected "
                            f"{class_name} "
                            f"on person "
                            f"{person_id} "
                            f"with confidence "
                            f"{score:.4f}"
                        )
                    }

        # ==================================================
        # NO EXPLICIT NUDENET
        # ==================================================

        return {
            "decision": "ALLOW",

            "reason": (
                "NO_SENSITIVE_CONTENT_DETECTED"
            )
        }