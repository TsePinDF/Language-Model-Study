from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tmt.config import LossWeights
from tmt.model import TMTForwardOutput, TMTModel


@dataclass
class LossBreakdown:
    total: torch.Tensor
    byte: torch.Tensor
    latent: torch.Tensor
    stop: torch.Tensor
    variance: torch.Tensor

    def scalars(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_byte": float(self.byte.detach().cpu()),
            "loss_latent": float(self.latent.detach().cpu()),
            "loss_stop": float(self.stop.detach().cpu()),
            "loss_variance": float(self.variance.detach().cpu()),
        }


def compute_tmt_losses(
    model: TMTModel,
    output: TMTForwardOutput,
    next_byte: torch.Tensor | None,
    end: torch.Tensor | None,
    weights: LossWeights,
) -> LossBreakdown:
    var = output.latent.float().var(dim=-1, unbiased=False)
    variance = torch.relu(1.0 - torch.sqrt(var + 1e-4)).mean()

    zero = output.latent.new_zeros(())
    byte = zero
    latent = zero
    stop = zero

    if next_byte is not None:
        next_byte = next_byte.to(device=output.logits.device, dtype=torch.long)
        if next_byte.ndim == 0:
            next_byte = next_byte.unsqueeze(0)
        target_latent = model.encoder(next_byte).detach()
        latent = F.mse_loss(output.latent.float(), target_latent.float())
        byte = F.cross_entropy(output.logits.float(), next_byte, reduction="mean")
        if end is None:
            end = torch.zeros_like(next_byte, dtype=torch.float32)
        end = end.to(device=output.stop_prob.device, dtype=output.stop_prob.dtype)
        if end.ndim == 0:
            end = end.unsqueeze(0)
        stop = F.mse_loss(output.stop_prob.squeeze(-1), end)

    total = (
        weights.byte * byte
        + weights.latent * latent
        + weights.stop * stop
        + weights.variance * variance
    )
    return LossBreakdown(total=total, byte=byte, latent=latent, stop=stop, variance=variance)
