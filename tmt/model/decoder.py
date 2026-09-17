from __future__ import annotations

import torch
from torch import nn


class ByteDecoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.decode = nn.Linear(dim, 256)
        self.stop = nn.Linear(dim, 1)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.decode(latent)
        stop_prob = torch.sigmoid(self.stop(latent))
        return logits, stop_prob
