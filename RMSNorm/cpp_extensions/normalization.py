from typing import Optional, Tuple, Union
import torch

__all__ = ['layernorm_fwd_fp8',
           'layernorm_fwd_fp8_inf',
           'layernorm_fwd_inf',
           'rmsnorm_fwd_fp8',
           'rmsnorm_fwd_fp8_inf',
           'rmsnorm_fwd_inf']

def rmsnorm_fwd_inf(
        inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        zero_centered_gamma: bool,
    ) -> torch.Tensor:
        return torch.ops.text_ts.rmsnorm_fwd_inf_ts(
                inp,
                weight,
                eps,
                zero_centered_gamma,
            )
