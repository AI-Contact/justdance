"""
포즈 추출기 클래스
"""

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("pip install mediapipe opencv-python numpy")

import cv2
import numpy as np


class PoseExtractor:
    def __init__(self, static_image_mode=False, model_complexity=1, enable_seg=False):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_seg,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def infer(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        arr = np.zeros((33,4), dtype=np.float32)
        for i, p in enumerate(lm):
            arr[i,0] = p.x; arr[i,1] = p.y; arr[i,2] = p.z; arr[i,3] = p.visibility
        return arr

