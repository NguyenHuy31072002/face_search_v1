import numpy as np
import cv2

class ImagePreprocess:
    @staticmethod
    def bytes_to_numpy(img_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Use cv2.IMREAD_GRAYSCALE for grayscale
        return img