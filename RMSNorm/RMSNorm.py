import torch
import torch.nn as nn
from einops import rearrange
from packaging import version

from typing import Dict, Type

import math
import warnings
from typing import Optional

def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor

def rms_norm(x, weight = None, eps = 1e-5):
    # mean을 0으로 가정하고 variance를 구함 
    out = x * torch.rsqrt(x.pow(2).mean(dim = -1, keepdim = True) + eps)
    if weight is not None:
        return output * weight
    return output

class RMSNorm(nn.Module):
    def __init__(
            self,
            normalized_shape,
            eps = 1e-5,
            weight = True,
            dtype = None,
            device = None,
        ):
            super().__init__()
            self.eps = eps
            if weight:
                self.weight = torch.nn.Parameter(
                            torch.ones(normalized_shape, dtype = dtype, device = device)
                )
            else:
                self.register_parameter('weight', None)

