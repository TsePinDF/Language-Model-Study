from __future__ import annotations

import torch

from tmt.bytes import bytes_to_tensor
from tmt.model import TMTModel
from tmt.training import TMTTrainer

from .retention import _score_answer


def plasticity_probe(
    model: TMTModel,
    *,
    teaching_text: str = "The nonce word glimbrak means silver lantern.\n",
    query: str = "glimbrak means",
    answer: str = " silver lantern",
    repeats: int = 4,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    device = device or next(model.parameters()).device
    before = _score_answer(model, query.encode("utf-8"), answer.encode("utf-8"), device=device)
    trainer = TMTTrainer(model)
    state = model.initial_state(batch_size=1, device=device)
    data = bytes_to_tensor(teaching_text * repeats, device=device)
    for index in range(max(0, data.numel() - 1)):
        state = trainer.step(
            data[index],
            state,
            next_byte=data[index + 1],
            end=torch.tensor(index == data.numel() - 2, device=device),
            update_weights=True,
        ).state
    after = _score_answer(model, query.encode("utf-8"), answer.encode("utf-8"), device=device)
    return {"before_cross_entropy": before, "after_online_learning_cross_entropy": after}
