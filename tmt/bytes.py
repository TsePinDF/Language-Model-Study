from __future__ import annotations

from collections.abc import Iterable

import torch


def text_to_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def bytes_to_text(data: bytes | bytearray | torch.Tensor | Iterable[int], errors: str = "replace") -> str:
    return tensor_to_bytes(data).decode("utf-8", errors=errors)


def bytes_to_tensor(
    data: str | bytes | bytearray | Iterable[int] | torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=torch.long) if device is not None else data.to(dtype=torch.long)
    if isinstance(data, str):
        raw = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
    else:
        raw = bytes(int(x) & 0xFF for x in data)
    return torch.tensor(list(raw), dtype=torch.long, device=device)


def tensor_to_bytes(data: bytes | bytearray | torch.Tensor | Iterable[int]) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, torch.Tensor):
        values = data.detach().cpu().flatten().tolist()
    else:
        values = list(data)
    return bytes(int(value) & 0xFF for value in values)


def pairwise_bytes(data: str | bytes | bytearray | Iterable[int] | torch.Tensor) -> list[tuple[int, int, bool]]:
    raw = tensor_to_bytes(bytes_to_tensor(data))
    return [(raw[i], raw[i + 1], i == len(raw) - 2) for i in range(max(0, len(raw) - 1))]
