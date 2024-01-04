import os 
from contextlib import contextmanager
from collections import deque
from typing import Callable, List, Optional, Dict, Any, Tuple, Union

import torch

__all__ = ["fp8_autocast", "fp8_model_init"]

def check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (9, 0): # hopper and above
        return True, ""
    if get_device_compute_capability() < (8, 9): # pre-ada
        return False, "Device compute capability 8.9 or higher required for FP8 execution."
    if tex.get_cublasLt_version() < 120103:
        return False, "CublasLt version 12.1.3.x or higher required for FP8 execution on Ada."
    if float(torch.version.cuda) < 12.1:
        return False, "Cuda version 12.1 or higher required for FP8 execution on Ada."
    return True, ""

def get_default_fp8_recipe() -> DelayedScaling:
    """FP8 recipe if not provided by user
    Margin = 0, interval = 1, E4M3
    """
    return DelayedScaling()

def get_fp8_te_dtype(
    fp8_recipe: DelayedScaling, fprop_tensor: bool = True
) -> tex.DType:
    """Get fp8 data type according to recipe and tensor"""
    if fp8_recipe.fp8_format == Format.E4M3 or (
        fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor
    ):
        return tex.DType.kFloat8E4M3
    return tex.DType.kFloat8E5M2


