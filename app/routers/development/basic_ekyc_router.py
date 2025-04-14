# FastAPI Component
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from typing import List
# Image component
from app.utils.image import ImagePreprocess
from app.utils.face.alignment import BasicAlignment
from app.utils.face.embedding import  calculate_similarity
# Startup
from app.startup import get_face_recognition_model, get_face_embedding_model
# Other components
from datetime import datetime
# Config
from app.core.config.constant import DEFAULT_MATCHING_THRESHOLD
# Image model
from app.utils.face.embedding import AdaFaceEmbedding
from app.utils.face.recognition import MTCNNRecognition

# ekyc router
basic_ekyc_router = APIRouter()

@basic_ekyc_router.post("/face_compare")
async def face_compare(files: List[UploadFile] = File(...,
                                                      description = "Upload images for comparison. First image is source, the rest is reference",
                                                      media_type = "image/png")):
    # Get model
    mtcnn : MTCNNRecognition = get_face_recognition_model()
    ada_face : AdaFaceEmbedding = get_face_embedding_model()
    # Raise exception if not enough file
    if len(files) < 2: raise HTTPException(status_code = 400,
                                           detail = "Please provide at least 2 files")
    # Check input file, requires images at all
    files_content = [True if file.content_type.startswith("image/") else False for file in files]
    if not all(files_content):
        raise HTTPException(status_code = status.HTTP_403_FORBIDDEN,
                            detail = "Uploaded files must be under image format!")

    # Define start time
    begin_time = datetime.now().strftime("%Y/%d/%m %H:%M:%S")
    # Read as bytes
    images_byte = [await file.read() for file in files]
    # Convert as numpy
    images_numpy = [ImagePreprocess.bytes_to_numpy(image) for image in images_byte]

    # Detecting face
    face_detections = mtcnn.batch_detect_faces(images_numpy)

    try:
        # Get landmarks
        faces_landmark = [detection[0].get("keypoints") for detection in face_detections]
        # Get faces aligned
        faces_aligned = [BasicAlignment.align_face_5points(image = image,
                                                           landmarks = landmark)for (landmark, image) in zip(faces_landmark,images_numpy)]
        # Embedding
        faces_embeddings = ada_face.embed(faces_aligned)
        # Source embedding
        source_embedding, reference_embeddings = faces_embeddings[0], faces_embeddings[1:]
        # Calculate similarity
        similarities = calculate_similarity(source_embedding, reference_embeddings)
        return {"created_at": begin_time,
                "similarities": similarities.tolist(),
                "embedding_model": ada_face.model_name}

    except Exception as e:
        raise HTTPException(status_code = status.HTTP_409_CONFLICT,
                            detail = e)

@basic_ekyc_router.post("/face_matching")
async def face_matching(files: List[UploadFile] = File(...,
                                                       description = "Upload images for comparison. First image is source, the rest is reference",
                                                       media_type = "image/png"),
                        threshold :float = DEFAULT_MATCHING_THRESHOLD):
    # Get model
    mtcnn: MTCNNRecognition = get_face_recognition_model()
    ada_face: AdaFaceEmbedding = get_face_embedding_model()

    # Raise exception if not enough file
    if len(files) != 2: raise HTTPException(status_code = 400,
                                            detail = "Please provide only 2 image files")
    # Check input file, requires images at all
    files_content = [True if file.content_type.startswith("image/") else False for file in files]
    if not all(files_content):
        raise HTTPException(status_code = status.HTTP_403_FORBIDDEN,
                            detail = "Uploaded files must be under image format!")

    # Define start time
    begin_time = datetime.now().strftime("%Y/%d/%m %H:%M:%S")
    # Read as bytes
    images_byte = [await file.read() for file in files]
    # Convert as numpy
    images_numpy = [ImagePreprocess.bytes_to_numpy(image) for image in images_byte]

    # Detecting face
    face_detections = mtcnn.batch_detect_faces(images_numpy)

    try:
        # Get landmarks
        faces_landmark = [detection[0].get("keypoints") for detection in face_detections]
        # Get faces aligned
        faces_aligned = [BasicAlignment.align_face_5points(image = image,
                                                           landmarks = landmark)for (landmark, image) in zip(faces_landmark,images_numpy)]
        # Embedding
        faces_embeddings = ada_face.embed(faces_aligned)
        # Source embedding
        source_embedding, reference_embeddings = faces_embeddings[0], faces_embeddings[1:]
        # Calculate similarity
        similarities = calculate_similarity(source_embedding, reference_embeddings)
        return {"created_at": begin_time,
                "matching": True if similarities.tolist()[0] > threshold else False,
                "embedding_model": ada_face.model_name}

    except Exception as e:
        raise HTTPException(status_code = status.HTTP_409_CONFLICT,
                            detail = e)
