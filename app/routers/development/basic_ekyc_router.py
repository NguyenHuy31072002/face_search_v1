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
from fastapi import Body
from app.utils.elasticsearch_client import get_es_client
from fastapi import Body, UploadFile, File
import numpy as np
from datetime import datetime
from fastapi import Path

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
    





@basic_ekyc_router.post("/insert_face_image")
async def insert_face_image(
    file: UploadFile = File(...),
    user_name: str = Body(...),
):
    # Kiểm tra ảnh
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    es = get_es_client()

    # 1. Đọc bytes -> numpy
    image_bytes = await file.read()
    image_np = ImagePreprocess.bytes_to_numpy(image_bytes)

    # 2. Detect face và landmark
    mtcnn: MTCNNRecognition = get_face_recognition_model()
    detections = mtcnn.batch_detect_faces([image_np])

    if not detections or not detections[0]:
        raise HTTPException(status_code=404, detail="No face detected.")

    keypoints = detections[0][0]["keypoints"]

    # 3. Align face
    aligned_face = BasicAlignment.align_face_5points(image=image_np, landmarks=keypoints)

    # 4. Embedding
    ada_face: AdaFaceEmbedding = get_face_embedding_model()
    embedding_vector = ada_face.embed([aligned_face])[0].tolist()

    # 5. Tăng ID tự động
    counter_id = "face_id"
    try:
        if es.exists(index="counters", id=counter_id):
            counter_doc = es.get(index="counters", id=counter_id)["_source"]
            next_id = counter_doc["value"] + 1
        else:
            next_id = 1
        es.index(index="counters", id=counter_id, document={"value": next_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Counter error: {e}")

    # 6. Lưu vào Elasticsearch với kiểu dữ liệu dense_vector
    doc = {
        "user_id": next_id,
        "user_name": user_name,
        "embedding": embedding_vector,  # Lưu vector embedding như một dense_vector
        "created_at": datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    }

    try:
        es.index(index="face-embeddings", id=str(next_id), document=doc)
        return {"status": "success", "user_id": next_id, "user_name": user_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert error: {e}")


@basic_ekyc_router.delete("/delete_face_image/{user_id}")
async def delete_face_image(user_id: int = Path(..., description="ID người dùng cần xóa")):
    es = get_es_client()

    # Kiểm tra xem tài liệu có tồn tại không
    if not es.exists(index="face-embeddings", id=str(user_id)):
        raise HTTPException(status_code=404, detail=f"User ID {user_id} không tồn tại.")

    try:
        # Xóa tài liệu
        es.delete(index="face-embeddings", id=str(user_id))
        return {"status": "success", "message": f"Đã xóa dữ liệu user_id {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa: {e}")


@basic_ekyc_router.post("/face_search")
async def face_search(
    file: UploadFile = File(..., description="Upload a face image to search for similar faces in the database"),
    limit: int = Body(10, description="Number of most similar faces to return"),
    threshold: float = Body(DEFAULT_MATCHING_THRESHOLD, description="Minimum similarity threshold")
):
    # Kiểm tra ảnh
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    es = get_es_client()

    # 1. Đọc bytes -> numpy
    image_bytes = await file.read()
    image_np = ImagePreprocess.bytes_to_numpy(image_bytes)

    # 2. Detect face và landmark
    mtcnn: MTCNNRecognition = get_face_recognition_model()
    detections = mtcnn.batch_detect_faces([image_np])

    if not detections or not detections[0]:
        raise HTTPException(status_code=404, detail="No face detected.")

    keypoints = detections[0][0]["keypoints"]

    # 3. Align face
    aligned_face = BasicAlignment.align_face_5points(image=image_np, landmarks=keypoints)

    # 4. Embedding
    ada_face: AdaFaceEmbedding = get_face_embedding_model()
    query_embedding = ada_face.embed([aligned_face])[0]  # Đây có thể đã là tensor

    # 5. Tìm kiếm các khuôn mặt tương tự trong Elasticsearch
    try:
        # Lấy tất cả các face embedding từ Elasticsearch
        search_query = {
            "query": {
                "match_all": {}
            },
            "size": 1000  # Giới hạn số lượng kết quả, có thể điều chỉnh nếu cần
        }
        
        search_results = es.search(
            index="face-embeddings",
            body=search_query
        )
        
        # Chuẩn bị danh sách embedding và thông tin người dùng
        face_embeddings = []
        user_info = []
        
        for hit in search_results["hits"]["hits"]:
            source = hit["_source"]
            # Thêm embedding vào danh sách để tính toán độ tương đồng
            face_embeddings.append(source["embedding"])
            # Lưu thông tin người dùng tương ứng
            user_info.append({
                "user_id": source["user_id"],
                "user_name": source["user_name"],
                "created_at": source["created_at"]
            })
        
        if not face_embeddings:
            return {
                "total_matches": 0,
                "threshold": threshold,
                "embedding_model": ada_face.model_name,
                "matches": []
            }
        
        # Import torch
        import torch
        
        # Đảm bảo query_embedding là tensor và đúng shape cho hàm cosine_similarity
        if not isinstance(query_embedding, torch.Tensor):
            query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
        
        # Reshape query_embedding để phù hợp với yêu cầu của F.cosine_similarity
        # F.cosine_similarity cần x1 có shape [N, D] và x2 có shape [M, D] khi dim=1
        query_embedding = query_embedding.unsqueeze(0)  # [D] -> [1, D]
        
        # Chuyển đổi danh sách embedding thành PyTorch tensor
        db_embeddings = torch.tensor(face_embeddings, dtype=torch.float32)  # [M, D]
        
        # Tính toán độ tương đồng
        similarities = calculate_similarity(query_embedding, db_embeddings)
        
        # Chuyển similarities về dạng Python list
        similarities = similarities.tolist()
        
        # Tạo danh sách kết quả với độ tương đồng và thông tin người dùng
        results = []
        for i, similarity in enumerate(similarities):
            if similarity > threshold:
                results.append({
                    **user_info[i],
                    "similarity_score": float(similarity)
                })
        
        # Sắp xếp kết quả theo độ tương đồng giảm dần
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Giới hạn số lượng kết quả
        results = results[:limit]
        
        return {
            "total_matches": len(results),
            "threshold": threshold,
            "embedding_model": ada_face.model_name,
            "matches": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")