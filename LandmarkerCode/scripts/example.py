import cv2
import numpy as np
from os.path import join
from core.models import DenseLandmarks, convert_model
import time
from neurovc.util.IO_util import draw_landmarks


class GenericCamera:
    def __init__(self):
        pass

    def start_acquisition(self):
        pass

    def get_image(self):
        pass


class WebcamCamera(GenericCamera):
    def __init__(self, cam_id=0):
        self.cam = cv2.VideoCapture(cam_id)
        self.__image = None
        self.frame_counter = 0

    def get_image(self):
        try:
            ret, frame = self.cam.read()
            ts = 1000000 * time.time()
            n_frame = self.frame_counter
            self.frame_counter += 1
        except Exception as e:
            ret = False
            print(e)
        if not ret:
            return (None, None, None)
        return float(n_frame), float(ts), frame


if __name__ == "__main__":
    landmarker = DenseLandmarks(
        model_path="joint_converted_70.pt",
        n_landmarks=70, stride=100)
    cam = WebcamCamera(0)
    while True:
        n_frame, ts, frame = cam.get_image()
        if frame is None:
            continue
        frame = frame[:250]
        lm, _ = landmarker.process(frame, sliding_window=False)
        frame = draw_landmarks(frame, lm)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
