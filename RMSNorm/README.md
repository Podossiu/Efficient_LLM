# [Root Mean Square Layer Normalization[NeurIPS'19]](https://arxiv.org/pdf/1910.07467.pdf)

## LayerNorm
![image](https://github.com/Podossiu/Efficient_LLM/assets/86233304/39fc0a29-1aca-499d-ae07-34835c6882a6)
![image](https://github.com/Podossiu/Efficient_LLM/assets/86233304/992f5ead-1720-44b4-8918-b3e407466c1f)

LayerNorm의 성공: re-centering & re-scaling invariance property

re-centering: input과 weight 모두에 shift-noise에 대해 insensitive하게 해줌 (mean)
re-scaling: input과 weight가 무작위로 스케일 될 때, 출력 표현을 그대로 유지해줌 (variance)

## RMSNorm
RMSNorm에서는 LayerNorm의 성공이 re-centering이 아닌 re-scaling에 있다고 가정하며 시작
따라서 re-centering을 제외하고 re-scaling invariance에 초점을 맞춰 단순히 Root mean square (RMS)만을 사용 (단순히 mean을 제거) 
![image](https://github.com/Podossiu/Efficient_LLM/assets/86233304/c21177a1-81bc-4537-b1f1-b82c3f70119a)


## Implementation
```python
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

      def forward(self, x):
         return rms_norm(x.float(), self.weight, self.eps).to(dtype = x.dtype)
```
