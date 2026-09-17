from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from tmt.config import TMTConfig, parameter_count
from tmt.memory import TMTState, update_decay_trace, update_embedding_trace

from .decoder import ByteDecoder
from .embedding import ByteEncoder
from .rtu import RecurrentTraceUnit


@dataclass
class TMTForwardOutput:
    logits: torch.Tensor
    stop_prob: torch.Tensor
    latent: torch.Tensor
    state: TMTState
    decays: tuple[torch.Tensor, ...]
    layer_states: tuple[torch.Tensor, ...]


class TMTModel(nn.Module):
    def __init__(
        self,
        config: TMTConfig | None = None,
        *,
        dim: int | None = None,
        num_layers: int | None = None,
        temperature: float | None = None,
    ):
        super().__init__()
        if config is None:
            config = TMTConfig(
                dim=dim or 512,
                num_layers=num_layers or 16,
                temperature=temperature if temperature is not None else 0.75,
            )
        self.config = config
        self.dim = config.dim
        self.num_layers = config.num_layers
        self.temperature = config.temperature

        self.encoder = ByteEncoder(self.dim)
        self.layers = nn.ModuleList(RecurrentTraceUnit(self.dim) for _ in range(self.num_layers))
        self.decoder = ByteDecoder(self.dim)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> TMTState:
        if device is None:
            device = next(self.parameters()).device
        return TMTState.zeros(self.num_layers, batch_size, self.dim, device=device, dtype=dtype)

    def forward(
        self,
        byte_input: torch.Tensor,
        state: TMTState,
        dummies: Sequence[torch.Tensor] | None = None,
        *,
        update_state: bool = True,
        traces_enabled: bool = True,
    ) -> TMTForwardOutput:
        byte_input = byte_input.to(device=next(self.parameters()).device, dtype=torch.long)
        if byte_input.ndim == 0:
            byte_input = byte_input.unsqueeze(0)
        batch_size = int(byte_input.shape[0])
        state.validate(self.num_layers, self.dim, batch_size=batch_size)

        x = self.encoder(byte_input)
        new_layer_states: list[torch.Tensor] = []
        new_decay_traces: list[torch.Tensor] = []
        decays: list[torch.Tensor] = []

        for index, layer in enumerate(self.layers):
            dummy = None if dummies is None else dummies[index]
            x, new_state, decay = layer(x, state.layer_states[index], dummy)
            new_layer_states.append(new_state)
            decays.append(decay)
            if traces_enabled:
                new_decay_traces.append(update_decay_trace(state.decay_traces[index], state.layer_states[index], decay))
            else:
                new_decay_traces.append(state.decay_traces[index])

        logits, stop_prob = self.decoder(x)

        if update_state:
            if traces_enabled:
                embedding_trace = update_embedding_trace(state.embedding_trace, byte_input, decays[0])
            else:
                embedding_trace = state.embedding_trace
            next_state = TMTState(
                layer_states=tuple(new_layer_states),
                decay_traces=tuple(new_decay_traces),
                embedding_trace=embedding_trace,
            )
        else:
            next_state = state

        return TMTForwardOutput(
            logits=logits,
            stop_prob=stop_prob,
            latent=x,
            state=next_state,
            decays=tuple(decays),
            layer_states=tuple(new_layer_states),
        )

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1) / torch.log(torch.tensor(256.0, device=logits.device))
        temp = torch.clamp(self.temperature * (1.0 - self.temperature * entropy), min=0.1)
        sample_probs = torch.softmax(logits.float() / temp[:, None], dim=-1)
        return torch.multinomial(sample_probs, num_samples=1, generator=generator).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def expected_parameter_count(self) -> int:
        return parameter_count(self.dim, self.num_layers)
