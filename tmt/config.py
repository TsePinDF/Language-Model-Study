from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class OptimizerConfig:
    learning_rate: float = 5e-4
    weight_decay: float = 0.0


@dataclass
class LossWeights:
    byte: float = 1.0
    latent: float = 1.0
    stop: float = 1.0
    variance: float = 1.0


@dataclass
class PrecisionConfig:
    autocast: bool = True
    dtype: str = "bf16"
    state_dtype: str = "float32"


@dataclass
class StreamConfig:
    data_glob: str = "wikipedia_clean/**/wiki_*"
    batch_size: int = 1
    sequence_length: int = 256
    reset_policy: str = "document"
    reset_interval_bytes: int = 0
    repeat: bool = True


@dataclass
class CheckpointConfig:
    directory: str = "checkpoints"
    interval: int = 500
    resume: str | None = None


@dataclass
class EvaluationConfig:
    interval: int = 1000
    max_bytes: int = 100_000


@dataclass
class LoggingConfig:
    directory: str = "runs"
    console_interval: int = 50
    jsonl: bool = True


@dataclass
class TMTConfig:
    name: str = "tmt_5m"
    dim: int = 512
    num_layers: int = 16
    temperature: float = 0.75
    seed: int = 1337
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    losses: LossWeights = field(default_factory=LossWeights)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def parameter_count_formula(self) -> int:
        return parameter_count(self.dim, self.num_layers)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TMTConfig":
        data = dict(data)
        if "layers" in data and "num_layers" not in data:
            data["num_layers"] = data.pop("layers")
        if "temp" in data and "temperature" not in data:
            data["temperature"] = data.pop("temp")
        if "lr" in data:
            data.setdefault("optimizer", {})
            data["optimizer"]["learning_rate"] = data.pop("lr")
        return _from_dict(cls, data)


T = TypeVar("T")


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    field_map = {field.name: field for field in fields(cls)}
    for key, value in data.items():
        if key not in field_map:
            continue
        field = field_map[key]
        field_type = field.type
        default_value = getattr(cls(), key) if callable(cls) else None
        if is_dataclass(default_value) and isinstance(value, dict):
            kwargs[key] = _from_dict(type(default_value), value)
        elif _is_dataclass_type(field_type) and isinstance(value, dict):
            kwargs[key] = _from_dict(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _is_dataclass_type(value: Any) -> bool:
    try:
        return is_dataclass(value)
    except TypeError:
        return False


def load_config(path: str | Path) -> TMTConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return TMTConfig.from_dict(data)


def save_config(config: TMTConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)


def parameter_count(dim: int, num_layers: int) -> int:
    return num_layers * dim * dim + (3 * num_layers + 513) * dim + 257
