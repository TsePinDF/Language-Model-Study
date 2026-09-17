from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse


HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}


def parse_dataset_id(value: str) -> str:
    """Return owner/name from either a dataset ID or a Hugging Face URL."""
    value = value.strip().rstrip("/")
    if not value:
        raise argparse.ArgumentTypeError("dataset must not be empty")

    if "://" in value:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in HF_HOSTS:
            raise argparse.ArgumentTypeError("expected a huggingface.co dataset URL")
        parts = [part for part in parsed.path.split("/") if part]
        if parts and parts[0] == "datasets":
            parts = parts[1:]
    else:
        parts = [part for part in value.split("/") if part]
        if parts and parts[0] == "datasets":
            parts = parts[1:]

    if len(parts) < 2:
        raise argparse.ArgumentTypeError("expected a dataset ID in owner/name form")
    return f"{parts[0]}/{parts[1]}"


def default_output_dir(dataset_id: str) -> Path:
    return Path("data") / "huggingface" / dataset_id.replace("/", "--")


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset repository to a local directory."
    )
    parser.add_argument(
        "dataset",
        type=parse_dataset_id,
        help="Dataset ID or URL, for example Open-Orca/OpenOrca.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination directory. Defaults to data/huggingface/OWNER--NAME.",
    )
    parser.add_argument("--revision", default="main", help="Branch, tag, or full commit hash.")
    parser.add_argument(
        "--include",
        action="append",
        metavar="GLOB",
        help='Only download matching files; repeat as needed (for example "*.parquet").',
    )
    parser.add_argument(
        "--exclude",
        action="append",
        metavar="GLOB",
        help="Skip matching files; repeat as needed.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent file downloads (default: 8).")
    parser.add_argument("--force", action="store_true", help="Redownload files even if cached.")
    parser.add_argument("--dry-run", action="store_true", help="Show files and size without downloading.")
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing a token for private/gated datasets.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install it with `python -m pip install huggingface_hub`."
        ) from exc

    output = (args.output or default_output_dir(args.dataset)).resolve()
    token = os.environ.get(args.token_env) if args.token_env else None
    request = {
        "repo_id": args.dataset,
        "repo_type": "dataset",
        "revision": args.revision,
        "local_dir": output,
        "allow_patterns": args.include,
        "ignore_patterns": args.exclude,
        "max_workers": args.workers,
        "force_download": args.force,
        "token": token,
        "dry_run": args.dry_run,
    }

    print(f"dataset: {args.dataset}")
    print(f"revision: {args.revision}")
    print(f"destination: {output}")
    if args.include:
        print(f"include: {', '.join(args.include)}")
    if args.exclude:
        print(f"exclude: {', '.join(args.exclude)}")

    try:
        result = snapshot_download(**request)
    except (HfHubHTTPError, OSError, ValueError) as exc:
        raise SystemExit(f"Download failed: {exc}") from exc

    if args.dry_run:
        files = list(result)
        pending = [file for file in files if file.will_download]
        total_size = sum(file.file_size for file in files)
        pending_size = sum(file.file_size for file in pending)
        print(f"selected files: {len(files)} ({format_bytes(total_size)})")
        print(f"to download: {len(pending)} ({format_bytes(pending_size)})")
        for file in files:
            status = "download" if file.will_download else "cached"
            print(f"  {status:8} {format_bytes(file.file_size):>10}  {file.filename}")
    else:
        print(f"download complete: {result}")


if __name__ == "__main__":
    main()
