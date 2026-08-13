import cv2
from PIL import Image


class CropExtractor:

    def extract(
        self,
        frame,
        detections
    ):
        """
        Extract person crops from a video frame.

        Parameters
        ----------
        frame : np.ndarray
            OpenCV frame (BGR format)

        detections : list
            List of YOLO detections containing:
            {
                "box": [x1, y1, x2, y2]
            }

        Returns
        -------
        list[PIL.Image]
            Person crops in RGB format
        """

        crops = []

        # OpenCV -> RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            frame_rgb
        )

        width, height = image.size

        for det in detections:

            x1, y1, x2, y2 = det["box"]

            # Ensure integer coordinates
            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(width, x2))
            y2 = int(min(height, y2))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop(
                (x1, y1, x2, y2)
            )

            crops.append(crop)

        return crops