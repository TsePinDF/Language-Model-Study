from __future__ import annotations

import torch

from tmt.model import TMTModel

from .retention import retention_probe


def memory_horizon_probe(
    model: TMTModel,
    *,
    distances: tuple[int, ...] = (1024, 10_240, 102_400, 1_048_576),
    device: torch.device | str | None = None,
) -> list[dict[str, float]]:
    return retention_probe(model, distances=distances, device=device)
