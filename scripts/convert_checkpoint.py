from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmt.training import save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a trusted legacy TMT .pt file to safetensors.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--step", type=int, default=None, help="Set a step when the source does not contain one.")
    args = parser.parse_args()

    if args.input.suffix != ".pt":
        raise SystemExit("legacy input must be a .pt file")
    if args.output.suffix != ".safetensors":
        raise SystemExit("output must end in .safetensors")

    try:
        payload = torch.load(args.input, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SystemExit(f"Could not read legacy checkpoint: {exc}") from exc
    if not isinstance(payload, dict) or "model" not in payload:
        raise SystemExit("legacy file is not a recognized TMT checkpoint")
    if args.run_id:
        payload["run_id"] = args.run_id
    if args.step is not None:
        payload["step"] = args.step

    save_checkpoint(args.output, payload)
    print(f"converted: {args.input}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
