# FastAPI Component
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse
# Image component
from app.utils.image import ImagePreprocess
from app.utils.face.alignment import BasicAlignment
# Other components
from io import BytesIO
# Startup
from app.startup import get_face_recognition_model
# Image model
from app.utils.face.recognition import MTCNNRecognition
# Components
import cv2
# Define route
preview_router = APIRouter()

@preview_router.post("/face_align")
async def face_align(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Uploaded files must be under image format!")
    # Get model
    mtcnn: MTCNNRecognition = get_face_recognition_model()

    # Load file
    file_content = await file.read()
    images_numpy = ImagePreprocess.bytes_to_numpy(file_content)
    # Get face
    detections = mtcnn.detect_faces(images_numpy)
    # When face existed
    if len(detections)  == 0:
        raise HTTPException(status_code = status.HTTP_403_FORBIDDEN,
                            detail = "Face not found!")
    # Get detection
    detections = detections[0]
    landmarks = detections.get("keypoints")
    # Align image
    aligned_image = BasicAlignment.align_face_5points(image = images_numpy,
                                                      landmarks = landmarks)
    # Encode image as JPEG
    _, encoded_image = cv2.imencode('.jpg', aligned_image)
    return StreamingResponse(BytesIO(encoded_image.tobytes()), media_type="image/jpeg")