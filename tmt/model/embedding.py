from __future__ import annotations

import torch
from torch import nn


class ByteEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = nn.Embedding(256, dim)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(byte_ids.long())
