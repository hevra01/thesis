import os
import torch
import sys

from flextok.model.layers.attention import FlexAttention  # or wherever it's defined

# # Dummy input
# B, N, H * D = 2, 64, 32
# q = torch.randn(B, N, H * D)
# k = torch.randn(B, N, H * D)
# v = torch.randn(B, N, H * D)


# # Instantiate attention
# attn = FlexAttention(dim=8)  # or whatever the class is
# compiled_attn = torch.compile(attn)

# # Forward pass
# out = compiled_attn(q, k, v)
# print(out.shape)

attn = FlexAttention(dim=8, num_heads=4)
q = torch.randn(2, 64, 32)
out = torch.compile(attn)(q, q, q)