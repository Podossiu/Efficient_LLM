import torch
import torch.nn as nn
from einops import rearrange
from packaging import version

from typing import Dict, Type

import math
import warnings
from typing import Optional
import time
import numpy as np

def func(module, batch_size, seq_len, dim, training = False):
    if training:
        x = torch.randn(batch_size, seq_len, dim).cuda()
        out = module(x)
        out.sum().backward()
    else:
        with torch.no_grad():
            x = torch.randn(batch_size, seq_len, dim).cuda()
            out = module(x)

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
    output = x * torch.rsqrt(x.pow(2).mean(dim = -1, keepdim = True) + eps)
    if weight is not None:
        return output * weight
    return output

# https://github.com/NVIDIA/TransformerEngine/blob/fad3044bde1547eae9543a6a3f80401e59bb629e/transformer_engine/pytorch/module/rmsnorm.py#L84
class _RMSNorm(torch.autograd.Function):
    """ functional RMSNorm """
    @staticmethod
    def forward(
            ctx,
            inp : torch.Tensor,
            rmsnorm_weight: torch.Tensor,
            eps: float,
            fwd_rmsnorm_sm_margin: int,
            bwd_rmsnorm_sm_margin: int,
            zero_centered_gamma: bool,
            is_grad_enabled: bool,
            activation_dtype: torch.dtype,
    ) -> torch.Tensor:
        # Make sure input dimensions are compatible
        in_features = rmsnorm_weight.numel()
        assert inp.is_cuda, "Only GPU Supports (CUDA)."
        assert inp.shape[-1] == in_features, "RMSNorm not possible"
        inputmat = inp.view(-1, in_features)
        
        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        rmsnorm_weight = cast_if_needed(rmsnorm_weight, activation_dtype)

        if is_grad_enabled:
            rmsnorm_out, rsigma = tex.rmsnorm_fwd(inputmat, rmsnorm_weight,
                                                  eps, fwd_rmsnorm_sm_margin,
                                                  zero_centered_gamma)
            ctx.save_for_backward(inputmat, rmsnorm_weight, rsigma)
            ctx.inp_shape = inp.shape
            ctx.bwd_rmsnorm_sm_margin = bwd_rmsnorm_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
        else:
            rmsnorm_out = tex.rmsnorm_fwd_inf(inputmat, rmsnorm_weight,
                                              eps, 
                                              zero_centered_gamma)
        return rmsnorm_out.view_as(inp)

    
    @staticmethod
    def backward(
            ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        inputmat, rmsnorm_weight, rsigma = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        d_rmsnorm_out = grad_output.view(inputmat.shape)
        dxmat, dgamma = tex.rmsnorm_bwd(
                d_rmsnorm_out, inputmat, rsigma, rmsnorm_weight,
                ctx.bwd_rmsnorm_sm_margin, ctx.zero_centered_gamma
        )
        return (
            dxmat.view(ctx.inp_shape),
            dgamma,
            None,
            None,
            None,
            None,
            None,
            None,
        )

class RMSNorm(nn.Module):
    r"""
    Applies Root Mean Square Layer Normalization over a mini-batch of inputs as described in
    the paper `Root Mean Square Layer Normalization <https://arxiv.org/abs/1910.07467>`__

    .. math::
        y = \frac{x}{RMS_\varepsilon(x)} * \gamma

    where

    .. math::
        RMS_\varepsilon(x) = \sqrt{\frac{1}{n}\sum_{i=0}^nx_i^2 + \varepsilon}

    :math:`\gamma` is a learnable affine transform parameter of size :attr:`hidden_size`

    Parameters
    ----------
    hidden_size : int
                size of each input sample.
    eps : float, default = 1e-5
        a value added to the denominator of layer normalization for numerical stability.
    sequence_parallel : bool, default = `False`
                        if set to `True`, uses sequence parallelism.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                    it controls the type used to allocate the initial parameters. Useful when
                    the model is trained with lower precision and the original FP32 parameters
                    would not fit in GPU memory.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in RMSNorm is initialized to 0 and
                         the RMSNorm formula changes to

                         .. math::
                            y = \frac{x}{RMS(x) + \varepsilon} * (1 + \gamma)
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.
    """


    def __init__(
            self,
            hidden_size: int,
            eps: float = 1e-5,
            sequence_parallel: bool = False,
            params_dtype: Optional[torch.dtype] = None,
            zero_centered_gamma: bool = False,
            device: Union[torch.device, str] = "cuda",
        ) -> None:
            super().__init__()

            params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
            self.eps = eps
            self.zero_centered_gamma = zero_centered_gamma
            self.weight = nn.Parameter(
                    torch.empty(
                        hidden_size,
                        device = device,
                        dtype = params_dtype,
                    )
            )
            setattr(self.weight, "sequence_parallel", sequence_parallel)
            self.reset_rmsnorm_parameters()

            self.fwd_rmsnorm_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
            self.bwd_rmsnorm_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))

    
    def reset_rmsnorm_parameters(self) -> None:
        """ Init RMSNorm Params"""
        if not self.zero_centered_gamma:
            init.ones_(self.weight)
        else:
            init.zeros_(self.weight)
    
    @no_torch_dynamo()
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """ RMSNorm FWD"""

        TransformerEngineBaseModule.set_activation_dtype(self, inp)

        if torch.is_grad_enabled():
            fwd_fn = _RMSNorm.apply
            args = []
        else:
            fwd_fn = _RMSNorm.forward
            args = [None]

        args += (
            inp,
            self.weight,
            self.eps,
            self.fwd_rmsnorm_sm_margin,
            self.bwd_rmsnorm_sm_margin,
            self.zero_centered_gamma,
            torch.is_grad_enabled(),
            self.activation_dtype,
        )
        return fwd_fn(*args)



if __name__ == "__main__":

    num_iter = 10
    batch_size = 256
    dim = 4096
    seq_len = 1024
    batch_size = 128
    results = {}
    '''
    try:
        torch.cuda.reset_peak_memory_stats()
        module = nn.LayerNorm(dim).cuda()
        while(True):
            func(module, batch_size, seq_len, dim)
            batch_size = batch_size * 2
    except Exception as e:
        if not str(e).startswith("CUDA out of memory"):
            print(e)
    finally:
        del module
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(2):
        if batch_size > 1:
            batch_size = batch_size // 2
    if batch_size == 0:
        batch_size = 1
    '''
    print(f"seq_len={seq_len}, batch_size={batch_size}, dim={dim}, ", end = "")

    torch.cuda.reset_peak_memory_stats()

    module = nn.LayerNorm(dim).cuda()
    func(module, batch_size, seq_len, dim)
    
    time_list = []
    for _ in range(num_iter):
        torch.cuda.synchronize()
        t0 = time.time()
        func(module, batch_size, seq_len, dim)
        torch.cuda.synchronize()
        t1 = time.time()
        time_list.append((t1 - t0) / batch_size)

    per_inst_time_avg = np.mean(time_list) * 1000
    per_inst_time_std = np.std(time_list) * 1000
    memory_per_inst = torch.cuda.max_memory_allocated() / batch_size / 1024 / 1024


    results[batch_size] = {
        "batch_size":batch_size,
        "per_inst_time_avg (ms)":round(per_inst_time_avg, 3),
        "per_inst_time_std (ms)":round(per_inst_time_std, 3),
        "memory_per_inst (MB)":round(memory_per_inst, 3),
    }

    print("LayerNorm :", results[batch_size])

    del module

    torch.cuda.empty_cache()

    batch_size = 256
    dim = 4096
    seq_len = 1024
    batch_size = 128
    results = {}
    '''
    try:
        torch.cuda.reset_peak_memory_stats()
        module = RMSNorm(dim).cuda()
        while(True):
            func(module, batch_size, seq_len, dim)
            batch_size = batch_size * 2
    except Exception as e:
        if not str(e).startswith("CUDA out of memory"):
            print(e)
    finally:
        del module
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    for _ in range(2):
        if batch_size > 1:
            batch_size = batch_size // 2
    if batch_size == 0:
        batch_size = 1
    '''
    print(f"seq_len={seq_len}, batch_size={batch_size}, dim={dim}, ", end = "")

    torch.cuda.reset_peak_memory_stats()

    module = RMSNorm(dim).cuda()
    func(module, batch_size, seq_len, dim)
    
    time_list = []
    for _ in range(num_iter):
        torch.cuda.synchronize()
        t0 = time.time()
        func(module, batch_size, seq_len, dim)
        torch.cuda.synchronize()
        t1 = time.time()
        time_list.append((t1 - t0) / batch_size)

    per_inst_time_avg = np.mean(time_list) * 1000
    per_inst_time_std = np.std(time_list) * 1000
    memory_per_inst = torch.cuda.max_memory_allocated() / batch_size / 1024 / 1024


    results[batch_size] = {
        "batch_size":batch_size,
        "per_inst_time_avg (ms)":round(per_inst_time_avg, 3),
        "per_inst_time_std (ms)":round(per_inst_time_std, 3),
        "memory_per_inst (MB)":round(memory_per_inst, 3),
    }

    print("RMSNorm : ", results[batch_size])

    del module

    torch.cuda.empty_cache()


