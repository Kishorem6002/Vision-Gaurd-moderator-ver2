import cv2


class FrameSampler:

    def __init__(self, fps_to_analyze=2):
        self.fps_to_analyze = fps_to_analyze

    def sample(self, video_path):

        cap = cv2.VideoCapture(video_path)

        video_fps = cap.get(cv2.CAP_PROP_FPS)

        interval = max(
            1,
            int(video_fps / self.fps_to_analyze)
        )

        frame_id = 0

        try:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_id += 1

                if frame_id % interval == 0:

                    yield {
                        "frame_id": frame_id,
                        "frame": frame
                    }

        finally:

            cap.release()
