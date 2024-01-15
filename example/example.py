import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import time
# Set dimensions.
in_features = 4096
out_features = 3072
hidden_size = 4096

# Initialize model and inputs.
#model = te.LayerNorm(in_features)
model = te.LayerNorm(in_features)
inp = torch.randn(128, 1024, 4096, device="cuda")

# Create an FP8 recipe. Note: All input args are optional.

# Enable autocasting for the forward pass

total = 0
# warmup
for i in range(10):
    out = model(inp)

for i in range(50):
    start = time.time()
    out = model(inp)
    total += time.time() - start

print("MEAN : " + str(total/ 50 * 1000) + " milisecond")
