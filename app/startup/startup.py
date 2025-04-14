# init model
from app.utils.face.embedding import AdaFaceEmbedding
from app.utils.face.recognition import MTCNNRecognition
# Import config
from app.core.config.constant import EMBEDDING_MODEL
from app.core.config import HF_TOKEN

# Variable
mtcnn = None
ada_face = None

def init_models():
    """Start Postgres Connection"""
    global mtcnn
    global ada_face
    # Init connection
    mtcnn = MTCNNRecognition()
    ada_face = AdaFaceEmbedding(model_name = EMBEDDING_MODEL,
                                HF_TOKEN = HF_TOKEN)
    return mtcnn, ada_face

def get_face_embedding_model():
    return ada_face

def get_face_recognition_model():
    return mtcnn