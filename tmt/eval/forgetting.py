from __future__ import annotations

import torch

from tmt.model import TMTModel

from .retention import _score_answer


def forgetting_probe(
    model: TMTModel,
    *,
    original_fact: str = "The blue key opens vault 17.\n",
    interference: str = "The red key opens vault 42.\n" * 16,
    query: str = "The blue key opens",
    answer: str = " vault 17",
    device: torch.device | str | None = None,
) -> dict[str, float]:
    before = _score_answer(model, original_fact.encode("utf-8") + query.encode("utf-8"), answer.encode("utf-8"), device=device)
    after_prefix = (original_fact + interference + query).encode("utf-8")
    after = _score_answer(model, after_prefix, answer.encode("utf-8"), device=device)
    return {"before_cross_entropy": before, "after_interference_cross_entropy": after}
