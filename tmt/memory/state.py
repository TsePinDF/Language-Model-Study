from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class TMTState:
    layer_states: tuple[torch.Tensor, ...]
    decay_traces: tuple[torch.Tensor, ...]
    embedding_trace: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_layers: int,
        batch_size: int,
        dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "TMTState":
        layer_states = tuple(torch.zeros(batch_size, dim, device=device, dtype=dtype) for _ in range(num_layers))
        decay_traces = tuple(torch.zeros(batch_size, dim, device=device, dtype=dtype) for _ in range(num_layers))
        embedding_trace = torch.zeros(batch_size, 256, dim, device=device, dtype=dtype)
        return cls(layer_states=layer_states, decay_traces=decay_traces, embedding_trace=embedding_trace)

    @property
    def batch_size(self) -> int:
        return int(self.embedding_trace.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embedding_trace.shape[-1])

    @property
    def num_layers(self) -> int:
        return len(self.layer_states)

    @property
    def device(self) -> torch.device:
        return self.embedding_trace.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embedding_trace.dtype

    def detach(self) -> "TMTState":
        return TMTState(
            layer_states=tuple(t.detach() for t in self.layer_states),
            decay_traces=tuple(t.detach() for t in self.decay_traces),
            embedding_trace=self.embedding_trace.detach(),
        )

    def clone(self) -> "TMTState":
        return TMTState(
            layer_states=tuple(t.clone() for t in self.layer_states),
            decay_traces=tuple(t.clone() for t in self.decay_traces),
            embedding_trace=self.embedding_trace.clone(),
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "TMTState":
        return TMTState(
            layer_states=tuple(t.to(device=device, dtype=dtype or t.dtype) for t in self.layer_states),
            decay_traces=tuple(t.to(device=device, dtype=dtype or t.dtype) for t in self.decay_traces),
            embedding_trace=self.embedding_trace.to(device=device, dtype=dtype or self.embedding_trace.dtype),
        )

    def reset(self, mask: torch.Tensor | Iterable[bool] | None = None) -> "TMTState":
        with torch.no_grad():
            if mask is None:
                for tensor in self.layer_states:
                    tensor.zero_()
                for tensor in self.decay_traces:
                    tensor.zero_()
                self.embedding_trace.zero_()
                return self

            mask_tensor = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
            if mask_tensor.ndim == 0:
                mask_tensor = mask_tensor[None]
            for tensor in self.layer_states:
                tensor[mask_tensor] = 0
            for tensor in self.decay_traces:
                tensor[mask_tensor] = 0
            self.embedding_trace[mask_tensor] = 0
        return self

    def validate(self, num_layers: int, dim: int, batch_size: int | None = None) -> None:
        if self.num_layers != num_layers:
            raise ValueError(f"expected {num_layers} layers in state, got {self.num_layers}")
        if self.dim != dim:
            raise ValueError(f"expected state dim {dim}, got {self.dim}")
        if batch_size is not None and self.batch_size != batch_size:
            raise ValueError(f"expected batch size {batch_size}, got {self.batch_size}")

    def as_dict(self) -> dict[str, object]:
        return {
            "layer_states": [tensor.detach().cpu() for tensor in self.layer_states],
            "decay_traces": [tensor.detach().cpu() for tensor in self.decay_traces],
            "embedding_trace": self.embedding_trace.detach().cpu(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object], device: torch.device | str | None = None) -> "TMTState":
        layer_states = tuple(t.to(device=device) for t in data["layer_states"])  # type: ignore[index, union-attr]
        decay_traces = tuple(t.to(device=device) for t in data["decay_traces"])  # type: ignore[index, union-attr]
        embedding_trace = data["embedding_trace"].to(device=device)  # type: ignore[union-attr]
        return cls(layer_states=layer_states, decay_traces=decay_traces, embedding_trace=embedding_trace)
