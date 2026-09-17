from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmt.bytes import bytes_to_text, bytes_to_tensor, tensor_to_bytes
from tmt.config import TMTConfig, load_config
from tmt.model import TMTModel
from tmt.training import TMTTrainer, load_checkpoint
from tmt.training.checkpoint import restore_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bytes with a PyTorch TMT model.")
    parser.add_argument("--config", default="configs/tmt_5m.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-bytes", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--online", action="store_true", help="Allow weight updates during prompt/generation.")
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
    trainer = TMTTrainer(model, config)
    state = model.initial_state(batch_size=1, device=device)

    prompt = bytes_to_tensor(args.prompt, device=device)
    current = torch.tensor(10, device=device, dtype=torch.long)
    if prompt.numel() > 0:
        for byte in prompt:
            state = trainer.step(byte, state, update_weights=args.online).state
            current = byte

    output_bytes: list[int] = []
    for _ in range(args.max_new_bytes):
        result = trainer.step(current, state, update_weights=args.online, sample=True)
        state = result.state
        current = result.sampled_byte.reshape(())  # type: ignore[union-attr]
        output_bytes.append(int(current.detach().cpu()))
        stop = float(result.stop_prob.reshape(-1)[0].detach().cpu()) if result.stop_prob is not None else 0.0
        if stop > args.threshold:
            break

    _write_text(bytes_to_text(tensor_to_bytes(output_bytes), errors="replace"))


def _write_text(text: str) -> None:
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
