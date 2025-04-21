import cv2
import numpy as np

class ObjectTracker:
    def __init__(self, frame, box):
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, box)

        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)

        self.miss_count = 0
        self.max_miss = 10
        self.box = box

    def update(self, frame):
        predicted = self.kalman.predict()
        success, box = self.tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            cx, cy = x + w / 2, y + h / 2
            self.kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
            self.miss_count = 0
            self.box = (x, y, w, h)
            return True, self.box
        else:
            self.miss_count += 1
            return False, self.box

    @property
    def is_lost(self):
        return self.miss_count > self.max_miss
