from transformers import AutoModel
from huggingface_hub import hf_hub_download
import shutil, os , sys, cv2, torch
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
# Numpy
import numpy as np
# Typing
from typing import Union, List

class AdaFaceEmbedding:
    def __init__(self,
                 HF_TOKEN: str,
                 model_name :str = "minchul/cvlface_DFA_mobilenet",
                 cached_dir :str = "~/.cvlface_cache",
                 force_download :bool = False):
        self._HF_TOKEN = HF_TOKEN
        self._model_name = model_name
        self._cached_dir = os.path.expanduser(f"{cached_dir}/{self._model_name}")
        self._force_download = force_download
        # Load model
        self._model = self._load_model_by_repo_id()

    @property
    def model_name(self):
        return self._model_name

    def _download_repo(self):
        os.makedirs(self._cached_dir, exist_ok=True)
        files_path = os.path.join(self._cached_dir, 'files.txt')
        if not os.path.exists(files_path):
            hf_hub_download(self._model_name, 'files.txt', token = self._HF_TOKEN,
                            local_dir = self._cached_dir,
                            local_dir_use_symlinks = False)

        with open(os.path.join(self._cached_dir, 'files.txt'), 'r') as f:
            files = f.read().split('\n')
        for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
            full_path = os.path.join(self._cached_dir, file)
            if not os.path.exists(full_path):
                hf_hub_download(self._model_name, file,
                                token = self._HF_TOKEN,
                                local_dir = self._cached_dir,
                                local_dir_use_symlinks = False)

    # helpfer function to download huggingface repo and use model
    def _load_model_from_local_path(self):
        cwd = os.getcwd()
        os.chdir(self._cached_dir)
        sys.path.insert(0, self._cached_dir)
        model = AutoModel.from_pretrained(self._cached_dir, trust_remote_code = True, token = self._HF_TOKEN)
        os.chdir(cwd)
        sys.path.pop(0)
        return model

    # Helpfer function to download huggingface repo and use model
    def _load_model_by_repo_id(self):
        # When force download
        if self._force_download:
            if os.path.exists(self._cached_dir):
                shutil.rmtree(self._cached_dir)
        # Check if model existed first, if not download
        if not os.path.exists(self._cached_dir): self._download_repo()
        # Load model
        return self._load_model_from_local_path()

    def embed(self,
              images :Union[str, List[str],np.ndarray, List[np.ndarray]]) -> torch.Tensor:
        # Convert string or ndarray to List[string]
        if isinstance(images, str) or isinstance(images, np.ndarray): images = [images]
        # Check condition
        if len(images) == 0:
            raise Exception("Images cannot be empty!")

        # Define transformation
        trans = Compose([ToTensor(),
                         Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        # Embed
        if isinstance(images[0], str):
            # Path case
            # embeddings = [read_image(image) for image in images]
            embeddings = [Image.open(image) for image in images]
            # Apply transformation
            embeddings = [trans(embedding) for embedding in embeddings]
        else:
            # Numpy case
            embeddings = [trans(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]

        # Stack as batch
        embeddings = torch.stack(embeddings)
        # Return value
        return self._model(embeddings)
