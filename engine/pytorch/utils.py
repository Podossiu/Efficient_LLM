import math
from typing import Any, Callable, Optional, Tuple
import torch

def cast_if_needed(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast tensor to dtype"""
    """ torch.enable_grad(): no_grad나 set_grad_enabled를 통해 비활성화된 경우 기울기 계산을 활성화"""
    with torch.enable_grad():
        return tensor if tensor is None or tensor.dtype == dtype else tensor.to(dtype)


