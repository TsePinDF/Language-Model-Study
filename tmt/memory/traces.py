from __future__ import annotations

import torch
import torch.nn.functional as F


def _decay_for_batch(decay: torch.Tensor, batch_size: int) -> torch.Tensor:
    if decay.ndim == 1:
        return decay.unsqueeze(0).expand(batch_size, -1)
    return decay


def update_embedding_trace(
    previous_trace: torch.Tensor,
    current_byte: torch.Tensor,
    first_layer_decay: torch.Tensor,
) -> torch.Tensor:
    batch_size, _, dim = previous_trace.shape
    current_byte = current_byte.reshape(batch_size).long()
    decay = _decay_for_batch(first_layer_decay.detach().to(previous_trace.dtype), batch_size)
    one_hot = F.one_hot(current_byte, num_classes=256).to(previous_trace.dtype).unsqueeze(-1)
    return previous_trace * decay[:, None, :] + one_hot.expand(-1, -1, dim)


def update_decay_trace(
    previous_trace: torch.Tensor,
    previous_state: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    batch_size = previous_trace.shape[0]
    decay_batched = _decay_for_batch(decay.detach().to(previous_trace.dtype), batch_size)
    return decay_batched * previous_trace + decay_batched * (1.0 - decay_batched) * previous_state


def embedding_trace_gradient(
    dloss_dstate: torch.Tensor,
    previous_embedding_trace: torch.Tensor,
    first_layer_decay: torch.Tensor,
) -> torch.Tensor:
    batch_size = previous_embedding_trace.shape[0]
    decay = _decay_for_batch(first_layer_decay.detach().to(previous_embedding_trace.dtype), batch_size)
    contribution = dloss_dstate.detach().to(previous_embedding_trace.dtype)[:, None, :]
    return (contribution * (previous_embedding_trace * decay[:, None, :])).sum(dim=0)


def decay_gradient(dloss_dstate: torch.Tensor, new_decay_trace: torch.Tensor) -> torch.Tensor:
    return (dloss_dstate.detach().to(new_decay_trace.dtype) * new_decay_trace).sum(dim=0)
