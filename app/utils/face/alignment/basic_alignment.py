# Image library
import numpy as np
import cv2
# Dependencies
from pydantic import Field, BaseModel

# Define landmark
class ThreePointsLandmark(BaseModel):
    left_eye :list[int] = Field(min_length=2, max_length=2)
    right_eye :list[int] = Field(min_length=2, max_length=2)
    nose :list[int] = Field(min_length=2, max_length=2)

class FivePointsLandmark(BaseModel):
    left_eye :list[int] = Field(min_length=2, max_length=2)
    right_eye :list[int] = Field(min_length=2, max_length=2)
    nose :list[int] = Field(min_length=2, max_length=2)
    mouth_left: list[int] = Field(min_length=2, max_length=2)
    mouth_right: list[int] = Field(min_length=2, max_length=2)

class BasicAlignment:
    @staticmethod
    def align_face_3points(image: np.ndarray,
                           landmarks: dict,
                           output_size: tuple = (112, 112)) -> np.ndarray:
        """
        Align a face in the image using 3 facial landmarks: left eye, right eye, nose tip.

        Args:
            image (np.ndarray): Input image (BGR).
            landmarks (dict): Dictionary with keys 'left_eye', 'right_eye', 'nose' and tuple values (x, y).
            output_size (tuple): Output image size (width, height). Default: (256, 256)

        Returns:
            np.ndarray: Aligned face image.
        """
        # Convert landmark to defined landmark
        desired_landmark = ThreePointsLandmark(**landmarks)

        # Target (aligned) landmark positions in the output image
        target_landmarks = {
            "left_eye": (0.3 * output_size[0], 0.35 * output_size[1]),
            "right_eye": (0.7 * output_size[0], 0.35 * output_size[1]),
            "nose": (0.5 * output_size[0], 0.6 * output_size[1])
        }

        # Prepare source and destination points
        src_points = np.float32([
            desired_landmark.left_eye,
            desired_landmark.right_eye,
            desired_landmark.nose
        ])
        dst_points = np.float32([
            target_landmarks["left_eye"],
            target_landmarks["right_eye"],
            target_landmarks["nose"]
        ])

        # Compute affine transformation matrix
        M = cv2.getAffineTransform(src_points, dst_points)

        # Apply transformation
        return cv2.warpAffine(image, M, output_size)

    @staticmethod
    def align_face_5points(image: np.ndarray,
                        landmarks: dict,
                        output_size: tuple = (112, 112)) -> np.ndarray:
        """
        Align face using 5-point landmarks (left_eye, right_eye, nose, left_mouth, right_mouth).

        Args:
            image (np.ndarray): Original BGR image
            landmarks (dict): Dictionary with keys 'left_eye', 'right_eye', 'nose','mouth_left','mouth_right' and tuple values (x, y).
            output_size (tuple): Output size of aligned image (w, h)

        Returns:
            np.ndarray: Aligned face image
        """
        # Standard reference points used by ArcFace/InsightFace for 112x112
        ref_5pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype = np.float32)

        # Convert landmark to defined landmark
        desired_landmark = FivePointsLandmark(**landmarks)

        # Scale to desired output size
        scale = output_size[0] / 112.0
        ref_5pts *= scale

        # Ensure landmarks are float32
        src_5pts = np.array([desired_landmark.left_eye,
                             desired_landmark.right_eye,
                             desired_landmark.nose,
                             desired_landmark.mouth_left,
                             desired_landmark.mouth_right], dtype = np.float32)

        # Compute similarity transform (full affine with scaling and rotation)
        M, _ = cv2.estimateAffinePartial2D(src_5pts, ref_5pts, method = cv2.LMEDS)

        # Warp image
        return cv2.warpAffine(image, M, output_size, borderValue=0)