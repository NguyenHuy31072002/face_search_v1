import torch
import torch.nn.functional as F
from typing import Literal

def calculate_similarity(source_vector :torch.Tensor,
                         ref_vector :torch.Tensor,
                         metrics :Literal["cosine"] = "cosine") -> torch.Tensor:
    return F.cosine_similarity(source_vector, ref_vector, dim=1)