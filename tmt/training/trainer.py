from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from tmt.config import TMTConfig
from tmt.memory import TMTState, decay_gradient, embedding_trace_gradient
from tmt.model import TMTModel

from .losses import LossBreakdown, compute_tmt_losses


@dataclass
class StepResult:
    state: TMTState
    losses: LossBreakdown
    metrics: dict[str, float]
    sampled_byte: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    stop_prob: torch.Tensor | None = None


class TMTTrainer:
    def __init__(
        self,
        model: TMTModel,
        config: TMTConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        self.model = model
        self.config = config or model.config
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )
        self.bytes_processed = 0

    def step(
        self,
        current_byte: torch.Tensor,
        state: TMTState,
        *,
        next_byte: torch.Tensor | None = None,
        end: torch.Tensor | None = None,
        update_weights: bool = True,
        update_state: bool = True,
        traces_enabled: bool = True,
        sample: bool = False,
    ) -> StepResult:
        device = next(self.model.parameters()).device
        current_byte = _as_batch(current_byte, device=device, dtype=torch.long)
        next_byte = None if next_byte is None else _as_batch(next_byte, device=device, dtype=torch.long)
        end = None if end is None else _as_batch(end, device=device, dtype=torch.float32)
        state = state.to(device=device, dtype=torch.float32).detach()

        if update_weights:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            dummies = _make_dummies(state) if traces_enabled else None
            with self._autocast_context():
                output = self.model(
                    current_byte,
                    state,
                    dummies=dummies,
                    update_state=update_state,
                    traces_enabled=traces_enabled,
                )
                losses = compute_tmt_losses(self.model, output, next_byte, end, self.config.losses)
            losses.total.backward()
            if traces_enabled and dummies is not None:
                self._apply_trace_gradients(state, output_state=output.state, decays=output.decays, dummies=dummies)
            self.optimizer.step()
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model(
                    current_byte,
                    state,
                    dummies=None,
                    update_state=update_state,
                    traces_enabled=traces_enabled,
                )
                losses = compute_tmt_losses(self.model, output, next_byte, end, self.config.losses)

        sampled = self.model.sample(output.logits) if sample else None
        new_state = output.state.detach()
        self.bytes_processed += int(current_byte.numel())
        metrics = losses.scalars()
        metrics["bytes_processed"] = float(self.bytes_processed)
        return StepResult(
            state=new_state,
            losses=losses,
            metrics=metrics,
            sampled_byte=sampled,
            logits=output.logits.detach(),
            stop_prob=output.stop_prob.detach(),
        )

    def _apply_trace_gradients(
        self,
        previous_state: TMTState,
        *,
        output_state: TMTState,
        decays: tuple[torch.Tensor, ...],
        dummies: list[torch.Tensor],
    ) -> None:
        first_dummy_grad = dummies[0].grad
        if first_dummy_grad is not None:
            grad = embedding_trace_gradient(first_dummy_grad, previous_state.embedding_trace, decays[0])
            embedding_weight = self.model.encoder.embed.weight
            if embedding_weight.grad is None:
                embedding_weight.grad = torch.zeros_like(embedding_weight)
            embedding_weight.grad.add_(grad.to(device=embedding_weight.device, dtype=embedding_weight.dtype))

        for index, layer in enumerate(self.model.layers):
            dummy_grad = dummies[index].grad
            if dummy_grad is None:
                continue
            grad = decay_gradient(dummy_grad, output_state.decay_traces[index])
            layer.decay_logits.grad = grad.to(device=layer.decay_logits.device, dtype=layer.decay_logits.dtype)

    def _autocast_context(self) -> Any:
        device = next(self.model.parameters()).device
        enabled = (
            self.config.precision.autocast
            and self.config.precision.dtype.lower() in {"bf16", "bfloat16"}
            and device.type == "cuda"
        )
        if not enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _make_dummies(state: TMTState) -> list[torch.Tensor]:
    return [torch.zeros_like(layer_state, requires_grad=True) for layer_state in state.layer_states]


def _as_batch(value: torch.Tensor, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.ndim == 0:
        value = value.unsqueeze(0)
    return value
