from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmt.config import TMTConfig, load_config, save_config
from tmt.model import TMTModel
from tmt.training import load_checkpoint, save_model_checkpoint, save_training_checkpoint
from tmt.training.checkpoint import restore_training_state
from tmt.training.logging import JSONLLogger
from tmt.training.streaming import IndependentByteBatcher, find_input_files
from tmt.training.trainer import TMTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch TMT model on byte streams.")
    parser.add_argument("--config", default="configs/tmt_5m.yaml")
    parser.add_argument("--data", default=None, help="Glob for input files. Defaults to config stream.data_glob.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=1000, help="Absolute target optimizer step.")
    parser.add_argument("--resume", default=None, help="Training .safetensors checkpoint to resume.")
    parser.add_argument("--run-id", default=None, help="Checkpoint/log directory name for this run.")
    args = parser.parse_args()

    config = load_config(args.config)
    resume_path = args.resume or config.checkpoint.resume
    checkpoint = load_checkpoint(resume_path, map_location="cpu") if resume_path else None
    if checkpoint is not None:
        if checkpoint.get("kind") != "training":
            raise SystemExit("--resume requires a training checkpoint, not a model-only checkpoint")
        if "config" in checkpoint:
            config = TMTConfig.from_dict(checkpoint["config"])

    if args.data:
        config.stream.data_glob = args.data
    run_id = _resolve_run_id(args.run_id, checkpoint, config)
    device = torch.device(args.device)
    torch.manual_seed(config.seed)

    model = TMTModel(config).to(device)
    trainer = TMTTrainer(model, config)
    state = model.initial_state(config.stream.batch_size, device=device)

    files = find_input_files(config.stream.data_glob)
    if not files:
        raise SystemExit(f"No input files matched {config.stream.data_glob!r}")
    batcher = IndependentByteBatcher(files, config.stream.batch_size, repeat=config.stream.repeat)
    bytes_since_reset = torch.zeros(config.stream.batch_size, dtype=torch.long)
    start_step = 0

    if checkpoint is not None:
        restored_state, start_step = restore_training_state(
            model,
            trainer.optimizer,
            checkpoint,
            device=device,
        )
        if restored_state is not None:
            state = restored_state
        progress = checkpoint.get("progress", {})
        if isinstance(progress, dict):
            trainer.bytes_processed = int(progress.get("bytes_processed", 0))
            if "bytes_since_reset" in progress:
                bytes_since_reset = torch.as_tensor(progress["bytes_since_reset"], dtype=torch.long).cpu()
            if "batcher" in progress:
                batcher.load_state_dict(progress["batcher"])

    run_dir = Path(config.logging.directory) / run_id
    checkpoint_dir = Path(config.checkpoint.directory) / run_id
    save_config(config, run_dir / "config.yaml")
    logger = JSONLLogger(run_dir / "train.jsonl") if config.logging.jsonl else None

    print(f"run id: {run_id}")
    print(f"config: {config.name}")
    print(f"device: {device}")
    print(f"trainable parameters: {model.parameter_count():,}")
    print(f"expected parameters: {model.expected_parameter_count():,}")
    print(f"git commit: {_git_commit()}")
    print(f"input files: {len(files)}")
    print(f"checkpoint directory: {checkpoint_dir}")
    if start_step:
        print(f"resumed at step: {start_step}")

    started = time.time()
    starting_bytes = trainer.bytes_processed
    last_completed_step = start_step
    last_saved_step: int | None = None
    interrupted = False

    def save_snapshot(step: int) -> None:
        nonlocal last_saved_step
        progress: dict[str, Any] = {
            "bytes_processed": trainer.bytes_processed,
            "bytes_since_reset": bytes_since_reset.clone(),
            "batcher": batcher.state_dict(),
        }
        metadata = {"git_commit": _git_commit()}
        save_training_checkpoint(
            checkpoint_dir / f"step_{step:09d}.safetensors",
            model,
            trainer.optimizer,
            state,
            config,
            step=step,
            run_id=run_id,
            progress=progress,
            metadata=metadata,
        )
        save_model_checkpoint(
            checkpoint_dir / "model_latest.safetensors",
            model,
            config,
            run_id=run_id,
            step=step,
        )
        last_saved_step = step
        print(f"checkpoint saved: step {step}")

    try:
        for step in range(start_step + 1, args.max_steps + 1):
            batch = next(batcher)
            if config.stream.reset_policy == "document":
                state.reset(batch.new_document)

            result = trainer.step(
                batch.current,
                state,
                next_byte=batch.next,
                end=batch.end,
                update_weights=True,
                update_state=True,
                traces_enabled=True,
            )
            state = result.state
            last_completed_step = step

            bytes_since_reset += 1
            if config.stream.reset_policy == "fixed_byte_interval" and config.stream.reset_interval_bytes > 0:
                mask = bytes_since_reset >= config.stream.reset_interval_bytes
                state.reset(mask)
                bytes_since_reset[mask] = 0

            elapsed = max(1e-9, time.time() - started)
            record = {
                "run_id": run_id,
                "config": config.name,
                "step": step,
                "bytes_processed": trainer.bytes_processed,
                "bytes_per_second": (trainer.bytes_processed - starting_bytes) / elapsed,
                "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                "parameter_count": model.parameter_count(),
                "gpu_memory_allocated": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                **result.metrics,
            }
            if logger is not None:
                logger.write(record)
            if step == start_step + 1 or step % config.logging.console_interval == 0:
                print(
                    f"step {step} "
                    f"loss={record['loss_total']:.4f} "
                    f"byte={record['loss_byte']:.4f} "
                    f"latent={record['loss_latent']:.4f} "
                    f"stop={record['loss_stop']:.4f} "
                    f"bytes/s={record['bytes_per_second']:.1f}"
                )

            if config.checkpoint.interval > 0 and step % config.checkpoint.interval == 0:
                save_snapshot(step)
    except KeyboardInterrupt:
        interrupted = True
        print("stop requested; saving the latest completed step")
    finally:
        if last_completed_step > start_step and last_saved_step != last_completed_step:
            save_snapshot(last_completed_step)

    if interrupted:
        print(f"training stopped at step {last_completed_step}")


def _resolve_run_id(requested: str | None, checkpoint: dict[str, Any] | None, config: TMTConfig) -> str:
    if requested:
        run_id = requested
    elif checkpoint and checkpoint.get("run_id"):
        run_id = str(checkpoint["run_id"])
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = re.sub(r"[^A-Za-z0-9._-]+", "-", config.name).strip("-.") or "tmt"
        run_id = f"{base}-{timestamp}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) or run_id in {".", ".."}:
        raise SystemExit("run IDs must use 1-128 letters, numbers, dots, underscores, or hyphens")
    return run_id


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()
