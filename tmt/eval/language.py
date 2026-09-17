from __future__ import annotations

import math

import torch

from tmt.bytes import bytes_to_tensor
from tmt.model import TMTModel
from tmt.training import TMTTrainer


def evaluate_language_bytes(
    model: TMTModel,
    data: bytes | str | torch.Tensor,
    *,
    max_bytes: int | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    device = device or next(model.parameters()).device
    byte_tensor = bytes_to_tensor(data, device=device)
    if max_bytes is not None:
        byte_tensor = byte_tensor[:max_bytes]
    if byte_tensor.numel() < 2:
        return {"byte_cross_entropy": float("nan"), "bits_per_byte": float("nan"), "bytes": float(byte_tensor.numel())}

    trainer = TMTTrainer(model)
    state = model.initial_state(batch_size=1, device=device)
    total_ce = 0.0
    count = 0
    for index in range(byte_tensor.numel() - 1):
        result = trainer.step(
            byte_tensor[index],
            state,
            next_byte=byte_tensor[index + 1],
            end=torch.tensor(index == byte_tensor.numel() - 2, device=device),
            update_weights=False,
            traces_enabled=True,
        )
        state = result.state
        total_ce += result.metrics["loss_byte"]
        count += 1
    ce = total_ce / max(1, count)
    return {"byte_cross_entropy": ce, "bits_per_byte": ce / math.log(2.0), "bytes": float(count + 1)}
