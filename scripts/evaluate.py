from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmt.config import TMTConfig, load_config
from tmt.eval import evaluate_language_bytes, forgetting_probe, memory_horizon_probe, retention_probe
from tmt.model import TMTModel
from tmt.training import load_checkpoint
from tmt.training.checkpoint import restore_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run basic TMT byte-level evaluations.")
    parser.add_argument("--config", default="configs/tmt_5m.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", default=None, help="Optional file for byte-level language evaluation.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-bytes", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = None
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, map_location=args.device)
        if "config" in checkpoint:
            config = TMTConfig.from_dict(checkpoint["config"])

    device = torch.device(args.device)
    model = TMTModel(config).to(device)
    if checkpoint is not None:
        restore_model(model, checkpoint)

    results: dict[str, object] = {
        "config": config.name,
        "parameter_count": model.parameter_count(),
    }
    if args.data:
        data = Path(args.data).read_bytes()
        results["language"] = evaluate_language_bytes(
            model,
            data,
            max_bytes=args.max_bytes or config.evaluation.max_bytes,
            device=device,
        )
    results["retention"] = retention_probe(model, device=device)
    results["memory_horizon"] = memory_horizon_probe(model, device=device)
    results["forgetting"] = forgetting_probe(model, device=device)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
