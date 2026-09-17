from __future__ import annotations

import torch
from torch import nn


class RecurrentTraceUnit(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.decay_logits = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        self.weights = nn.Linear(dim, dim, bias=False)
        self.silu = nn.SiLU()

    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_logits.float())

    def forward(
        self,
        x: torch.Tensor,
        previous_state: torch.Tensor,
        dummy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decay = self.decay()
        state = decay * previous_state.float() + x.float()
        if dummy is not None:
            state = state + dummy.float()
        normalized = self.norm(state.to(dtype=x.dtype))
        update = self.silu(self.weights(normalized))
        return x + update, state, decay
