import torch
from PIL import Image
from ultralytics import YOLO


class PersonDetector:
    """
    Pretrained YOLO person detector.

    Responsibilities:
        1. Load pretrained YOLO model.
        2. Detect people in an image.
        3. Return bounding boxes and confidence scores.
        4. Create adaptive crops around detected people.
        5. Keep crop coordinates inside the image boundaries.

    This class does NOT perform NSFW or nudity detection.
    """

    MODEL_NAME = "yolov8m.pt"

    # COCO class ID for person
    PERSON_CLASS_ID = 0

    # ------------------------------------------------------
    # Adaptive crop configuration
    # ------------------------------------------------------

    MIN_PADDING = 0.12
    MAX_PADDING = 0.25

    # Extra vertical context because important body regions
    # can be close to the bottom/top of a person bounding box.
    VERTICAL_PADDING_MULTIPLIER = 1.10

    # Very small crops should receive more context.
    SMALL_PERSON_RATIO = 0.20

    def __init__(
        self,
        confidence_threshold=0.25,
        image_size=640
    ):

        # ==================================================
        # DEVICE
        # ==================================================

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[PersonDetector] Device: "
            f"{self.device}"
        )

        if torch.cuda.is_available():
            print(
                f"[PersonDetector] GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

        # ==================================================
        # CONFIGURATION
        # ==================================================

        self.confidence_threshold = (
            confidence_threshold
        )

        self.image_size = image_size

        print(
            f"[PersonDetector] "
            f"Confidence threshold: "
            f"{self.confidence_threshold}"
        )

        print(
            f"[PersonDetector] "
            f"Adaptive padding range: "
            f"{self.MIN_PADDING * 100:.0f}% - "
            f"{self.MAX_PADDING * 100:.0f}%"
        )

        # ==================================================
        # LOAD YOLO
        # ==================================================

        print(
            f"[PersonDetector] Loading model: "
            f"{self.MODEL_NAME}"
        )

        self.model = YOLO(
            self.MODEL_NAME
        )

        print(
            "[PersonDetector] "
            "Model loaded successfully."
        )

    # ======================================================
    # DETECT PEOPLE
    # ======================================================

    def detect(
        self,
        image: Image.Image
    ):
        """
        Detect people in a PIL image.

        Returns:

            [
                {
                    "person_id": 1,
                    "confidence": 0.91,
                    "box": [x1, y1, x2, y2]
                }
            ]
        """

        if not isinstance(
            image,
            Image.Image
        ):
            raise TypeError(
                "Input must be a PIL Image."
            )

        image = image.convert("RGB")

        # --------------------------------------------------
        # YOLO inference
        # --------------------------------------------------

        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=0.50,
            imgsz=self.image_size,
            classes=[self.PERSON_CLASS_ID],
            device=self.device,
            max_det=30,
            verbose=False
        )

        detections = []

        result = results[0]

        if result.boxes is None:
            return detections

        for index, box in enumerate(
            result.boxes
        ):

            coordinates = (
                box.xyxy[0]
                .detach()
                .cpu()
                .tolist()
            )

            x1, y1, x2, y2 = coordinates

            confidence = (
                box.conf[0]
                .detach()
                .cpu()
                .item()
            )

            detections.append(
                {
                    "person_id": index + 1,
                    "confidence": confidence,
                    "box": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ]
                }
            )

        return detections

    # ======================================================
    # BATCH PERSON DETECTION
    # ======================================================

    def detect_batch(
        self,
        images
    ):
        """
        Batch person detection.

        images:
            List[PIL.Image]

        Returns:
            [
                detections_for_image_1,
                detections_for_image_2,
                ...
            ]
        """

        results = self.model.predict(
            source=images,
            conf=self.confidence_threshold,
            iou=0.50,
            imgsz=self.image_size,
            classes=[self.PERSON_CLASS_ID],
            device=self.device,
            max_det=30,
            verbose=False
        )

        all_detections = []

        for result in results:

            detections = []

            if result.boxes is not None:

                for idx, box in enumerate(
                    result.boxes
                ):

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .tolist()
                    )

                    confidence = (
                        box.conf[0]
                        .cpu()
                        .item()
                    )

                    detections.append(
                        {
                            "person_id":
                                idx + 1,

                            "confidence":
                                confidence,

                            "box": [
                                int(x1),
                                int(y1),
                                int(x2),
                                int(y2)
                            ]
                        }
                    )

            all_detections.append(
                detections
            )

        return all_detections

    # ======================================================
    # CALCULATE ADAPTIVE PADDING
    # ======================================================

    def calculate_adaptive_padding(
        self,
        image: Image.Image,
        box
    ):
        """
        Calculate adaptive padding based on the detected
        person's relative size in the image.

        Small people receive more padding.
        Large people receive less padding.

        Returns:
            {
                "horizontal": float,
                "vertical": float
            }
        """

        image_width, image_height = image.size

        x1, y1, x2, y2 = box

        box_width = max(
            1,
            x2 - x1
        )

        box_height = max(
            1,
            y2 - y1
        )

        # --------------------------------------------------
        # Person size relative to image
        # --------------------------------------------------

        width_ratio = (
            box_width / image_width
        )

        height_ratio = (
            box_height / image_height
        )

        # Use the larger relative dimension as the
        # main scale indicator.
        person_ratio = max(
            width_ratio,
            height_ratio
        )

        # --------------------------------------------------
        # Adaptive base padding
        # --------------------------------------------------
        #
        # Small person -> more padding
        # Large person -> less padding
        #
        # Example:
        #   0.05 person ratio -> close to MAX_PADDING
        #   0.50 person ratio -> close to MIN_PADDING
        # --------------------------------------------------

        if person_ratio <= self.SMALL_PERSON_RATIO:

            base_padding = self.MAX_PADDING

        else:

            # Normalize ratio between the small-person
            # reference and a large-person reference.
            normalized = min(
                1.0,
                (person_ratio - self.SMALL_PERSON_RATIO)
                / 0.50
            )

            base_padding = (
                self.MAX_PADDING
                -
                (
                    normalized
                    *
                    (
                        self.MAX_PADDING
                        -
                        self.MIN_PADDING
                    )
                )
            )

        # --------------------------------------------------
        # Slightly more vertical context
        # --------------------------------------------------

        horizontal_padding = base_padding

        vertical_padding = min(
            self.MAX_PADDING,
            base_padding *
            self.VERTICAL_PADDING_MULTIPLIER
        )

        return {
            "horizontal": horizontal_padding,
            "vertical": vertical_padding
        }

    # ======================================================
    # CROP ONE PERSON
    # ======================================================

    def crop_person(
        self,
        image: Image.Image,
        box
    ):
        """
        Create an adaptive crop around one detected person.

        The crop:
            - adapts to person size
            - provides more vertical context
            - stays inside image boundaries
            - preserves the original aspect ratio

        Returns:
            {
                "crop": PIL.Image,
                "crop_box": [x1, y1, x2, y2],
                "padding": {
                    "horizontal": ...,
                    "vertical": ...
                }
            }
        """

        image = image.convert("RGB")

        image_width, image_height = image.size

        x1, y1, x2, y2 = box

        # --------------------------------------------------
        # Calculate adaptive padding
        # --------------------------------------------------

        padding = (
            self.calculate_adaptive_padding(
                image,
                box
            )
        )

        horizontal_padding = (
            padding["horizontal"]
        )

        vertical_padding = (
            padding["vertical"]
        )

        # --------------------------------------------------
        # Calculate person dimensions
        # --------------------------------------------------

        box_width = max(
            1,
            x2 - x1
        )

        box_height = max(
            1,
            y2 - y1
        )

        # --------------------------------------------------
        # Padding in pixels
        # --------------------------------------------------

        pad_x = int(
            box_width *
            horizontal_padding
        )

        pad_y = int(
            box_height *
            vertical_padding
        )

        # --------------------------------------------------
        # Expanded crop coordinates
        # --------------------------------------------------

        crop_x1 = max(
            0,
            x1 - pad_x
        )

        crop_y1 = max(
            0,
            y1 - pad_y
        )

        crop_x2 = min(
            image_width,
            x2 + pad_x
        )

        crop_y2 = min(
            image_height,
            y2 + pad_y
        )

        # --------------------------------------------------
        # Safety check
        # --------------------------------------------------

        if crop_x2 <= crop_x1:
            crop_x2 = min(
                image_width,
                crop_x1 + box_width
            )

        if crop_y2 <= crop_y1:
            crop_y2 = min(
                image_height,
                crop_y1 + box_height
            )

        # --------------------------------------------------
        # Create crop
        # --------------------------------------------------

        crop = image.crop(
            (
                crop_x1,
                crop_y1,
                crop_x2,
                crop_y2
            )
        )

        return {
            "crop": crop,
            "crop_box": [
                int(crop_x1),
                int(crop_y1),
                int(crop_x2),
                int(crop_y2)
            ],
            "padding": padding
        }

    # ======================================================
    # DETECT + ADAPTIVE CROP
    # ======================================================

    def detect_and_crop(
        self,
        image: Image.Image
    ):
        """
        Detect all people and return adaptive crops.

        Returns:

            [
                {
                    "person_id": 1,
                    "confidence": 0.91,

                    "box": [
                        x1, y1, x2, y2
                    ],

                    "crop_box": [
                        x1, y1, x2, y2
                    ],

                    "padding": {
                        "horizontal": 0.20,
                        "vertical": 0.24
                    },

                    "crop": PIL.Image
                }
            ]
        """

        detections = self.detect(
            image
        )

        results = []

        # --------------------------------------------------
        # Process every person
        # --------------------------------------------------

        for detection in detections:

            crop_result = (
                self.crop_person(
                    image=image,
                    box=detection["box"]
                )
            )

            results.append(
                {
                    **detection,

                    "crop_box": (
                        crop_result[
                            "crop_box"
                        ]
                    ),

                    "padding": (
                        crop_result[
                            "padding"
                        ]
                    ),

                    "crop": (
                        crop_result[
                            "crop"
                        ]
                    )
                }
            )

        return results