from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tmt.config import TMTConfig
from tmt.memory import TMTState
from tmt.model import TMTModel


CHECKPOINT_FORMAT = "tmt-safetensors"
CHECKPOINT_VERSION = 1
_TREE_KEY = "tmt.tree"


def save_model_checkpoint(
    path: str | Path,
    model: TMTModel,
    config: TMTConfig,
    *,
    run_id: str | None = None,
    step: int | None = None,
) -> None:
    payload: dict[str, Any] = {
        "kind": "model",
        "model": model.state_dict(),
        "config": config.to_dict(),
    }
    if run_id is not None:
        payload["run_id"] = run_id
    if step is not None:
        payload["step"] = step
    save_checkpoint(path, payload)


def save_training_checkpoint(
    path: str | Path,
    model: TMTModel,
    optimizer: torch.optim.Optimizer,
    state: TMTState,
    config: TMTConfig,
    *,
    step: int,
    run_id: str | None = None,
    progress: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    rng_state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    save_checkpoint(
        path,
        {
            "kind": "training",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "runtime_state": state.as_dict(),
            "config": config.to_dict(),
            "step": step,
            "run_id": run_id,
            "progress": progress or {},
            "rng_state": rng_state,
            "metadata": metadata or {},
        },
    )


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a nested checkpoint payload in one pickle-free safetensors file."""
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("checkpoint paths must end in .safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    tree = _encode_tree(payload, tensors)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "version": str(CHECKPOINT_VERSION),
        "kind": str(payload.get("kind", "unknown")),
        "step": str(payload.get("step", "")),
        "run_id": str(payload.get("run_id") or ""),
        _TREE_KEY: json.dumps(tree, ensure_ascii=True, separators=(",", ":"), allow_nan=False),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.safetensors")
    try:
        save_file(tensors, temporary, metadata=metadata)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device | None = "cpu",
) -> dict[str, Any]:
    path = Path(path)
    device = str(map_location or "cpu")
    with safe_open(path, framework="pt", device=device) as handle:
        metadata = handle.metadata()
        _validate_metadata(path, metadata)
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    payload = _decode_tree(json.loads(metadata[_TREE_KEY]), tensors)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint root must be a dictionary: {path}")
    return payload


def checkpoint_metadata(path: str | Path) -> dict[str, str]:
    path = Path(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    _validate_metadata(path, metadata)
    return {key: value for key, value in metadata.items() if key != _TREE_KEY}


def restore_model(model: TMTModel, checkpoint: dict[str, Any], strict: bool = True) -> None:
    model.load_state_dict(checkpoint["model"], strict=strict)


def restore_training_state(
    model: TMTModel,
    optimizer: torch.optim.Optimizer,
    checkpoint: dict[str, Any],
    device: torch.device | str | None = None,
    *,
    restore_rng: bool = True,
) -> tuple[TMTState | None, int]:
    restore_model(model, checkpoint)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    state = None
    if "runtime_state" in checkpoint:
        state = TMTState.from_dict(checkpoint["runtime_state"], device=device)
    if restore_rng and "rng_state" in checkpoint:
        _restore_rng_state(checkpoint["rng_state"])
    return state, int(checkpoint.get("step", 0))


def _restore_rng_state(rng_state: dict[str, Any]) -> None:
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in rng_state:
        cuda_states = rng_state["cuda"]
        for index, state in enumerate(cuda_states[: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(state.cpu(), device=index)


def _validate_metadata(path: Path, metadata: dict[str, str] | None) -> None:
    if not metadata or metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"not a TMT safetensors checkpoint: {path}")
    if int(metadata.get("version", "0")) != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported TMT checkpoint version in {path}")
    if _TREE_KEY not in metadata:
        raise ValueError(f"checkpoint structure metadata is missing: {path}")


def _encode_tree(value: Any, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        key = f"tensor_{len(tensors):08d}"
        tensors[key] = value.detach().cpu().contiguous()
        return {"type": "tensor", "key": key}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": [[_encode_tree(key, tensors), _encode_tree(item, tensors)] for key, item in value.items()],
        }
    if isinstance(value, list):
        return {"type": "list", "items": [_encode_tree(item, tensors) for item in value]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode_tree(item, tensors) for item in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": "scalar", "value": value}
    raise TypeError(f"unsupported checkpoint value: {type(value).__name__}")


def _decode_tree(node: dict[str, Any], tensors: dict[str, torch.Tensor]) -> Any:
    node_type = node["type"]
    if node_type == "tensor":
        return tensors[node["key"]]
    if node_type == "dict":
        return {_decode_tree(key, tensors): _decode_tree(value, tensors) for key, value in node["items"]}
    if node_type == "list":
        return [_decode_tree(item, tensors) for item in node["items"]]
    if node_type == "tuple":
        return tuple(_decode_tree(item, tensors) for item in node["items"])
    if node_type == "scalar":
        return node["value"]
    raise ValueError(f"unknown checkpoint node type: {node_type}")
