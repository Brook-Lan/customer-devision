import time
import os

import torch
from torch import Tensor
from typing import List, Tuple, Any, Optional


def create_model_path(model_dir, model_name):
    model_name = time.strftime(model_name + '_%m%d_%H%M%S.pth')
    model_path = os.path.join(model_dir, model_name)
    return model_path
   

def normalize_transpose(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    # tensor.sub_(mean).div_(std)
    tensor.mul_(std).add_(mean)
    return tensor
