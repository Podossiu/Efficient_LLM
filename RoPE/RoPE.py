import torch
import torch.nn as nn

from einops import rearrange, repeat

# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("WangZeJun/roformer-sim-base-chinese")

inputs = torch.arange(128, dtype = torch.long).unsqueeze(0)
out= model(inputs)
