"""Face detector: OpenCV YuNet (opencv_zoo) with Haar-cascade fallback. Detection runs on a copy resized to 640 px width."""
import cv2
import numpy as np
from m3.utils.common import ROOT

YUNET = ROOT / "models_ext/face_detection_yunet_2023mar.onnx"


class Detector:
    def __init__(self, det_width=640, score=0.6):
        self.w = det_width; self.kind = "haar"
        if YUNET.exists():
            try: self.det = cv2.FaceDetectorYN.create(str(YUNET), "", (det_width, det_width), score_threshold=score); self.kind = "yunet"
            except Exception: pass
        if self.kind == "haar": self.det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, frame):
        """Returns int array [N,4] of (x,y,w,h) in full-resolution coordinates."""
        h, w = frame.shape[:2]; s = self.w / w; small = cv2.resize(frame, (self.w, int(round(h * s))))
        if self.kind == "yunet":
            self.det.setInputSize((small.shape[1], small.shape[0])); _, f = self.det.detect(small)
            return np.zeros((0, 4), int) if f is None else (f[:, :4] / s).round().astype(int)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY); b = self.det.detectMultiScale(g, 1.1, 4, minSize=(12, 12))
        return np.zeros((0, 4), int) if len(b) == 0 else (b / s).round().astype(int)
