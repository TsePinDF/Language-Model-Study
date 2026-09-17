from __future__ import annotations

import torch

from tmt.bytes import bytes_to_tensor
from tmt.model import TMTModel
from tmt.training import TMTTrainer


def retention_probe(
    model: TMTModel,
    *,
    fact: str = "The access code is ZQ-174.\n",
    query: str = "Access code:",
    answer: str = " ZQ-174",
    distances: tuple[int, ...] = (1024, 10_240, 102_400),
    filler_byte: int = 32,
    device: torch.device | str | None = None,
) -> list[dict[str, float]]:
    results = []
    for distance in distances:
        prefix = fact.encode("utf-8") + bytes([filler_byte]) * distance + query.encode("utf-8")
        ce = _score_answer(model, prefix, answer.encode("utf-8"), device=device)
        results.append({"distance_bytes": float(distance), "answer_cross_entropy": ce})
    return results


def _score_answer(
    model: TMTModel,
    prefix: bytes,
    answer: bytes,
    *,
    device: torch.device | str | None = None,
) -> float:
    device = device or next(model.parameters()).device
    trainer = TMTTrainer(model)
    state = model.initial_state(batch_size=1, device=device)
    prefix_tensor = bytes_to_tensor(prefix, device=device)
    for byte in prefix_tensor:
        state = trainer.step(byte, state, update_weights=False).state

    answer_tensor = bytes_to_tensor(answer, device=device)
    if answer_tensor.numel() < 1:
        return float("nan")
    prev = prefix_tensor[-1] if prefix_tensor.numel() else torch.tensor(10, device=device)
    total = 0.0
    for index, target in enumerate(answer_tensor):
        result = trainer.step(prev, state, next_byte=target, update_weights=False)
        state = result.state
        total += result.metrics["loss_byte"]
        prev = target
    return total / float(answer_tensor.numel())
