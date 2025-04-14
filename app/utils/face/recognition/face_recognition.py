from mtcnn import MTCNN
import cv2
import numpy as np
from typing import Union, List

def convert_np(obj):
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(item) for item in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

class MTCNNRecognition:
    def __init__(self,
                 stages :str = "face_and_landmarks_detection",
                 device :str = "CPU:0"):
        self._detector = MTCNN(stages = stages,
                               device = device)

    def detect_faces(self,
                     image :Union[str, np.ndarray],
                     limit :int = 1):
        # Image path case
        if isinstance(image, str): image = cv2.imread(image)
        # Convert chanel from BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Detect
        results = self._detector.detect_faces(rgb_image)

        # When detect no faces
        if len(results) == 0: return []

        # When limit too high
        if len(results) < limit:
            raise Exception(f"Limit: {limit} must be lower than len of input: {len(results)}")
        # Normalize input
        return [convert_np(result) for result in results[:limit]]

    def batch_detect_faces(self,
                           images :List[np.ndarray],
                           limit :int = 1):
        # Detect
        results = self._detector.detect_faces(images)

        # When detect no faces
        if len(results) == 0: return []

        # When limit too high
        if len(results) < limit:
            raise Exception(f"Limit: {limit} must be lower than len of input: {len(results)}")

        # Apply convert
        for i in range(len(results)):
            results[i] = [convert_np(result) for result in results[i][:limit]]
        return results