from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
JOB_DIR = ROOT / "runs" / "jobs"
FINAL_STATUSES = {"completed", "failed", "stopped"}


def start_job(
    kind: str,
    command: Sequence[str],
    parameters: dict[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = f"{kind}-{stamp}-{uuid.uuid4().hex[:6]}"
    job_path = JOB_DIR / f"{job_id}.json"
    log_path = JOB_DIR / f"{job_id}.log"
    job = {
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "command": [str(part) for part in command],
        "parameters": parameters,
        "environment_keys": sorted(environment or {}),
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "return_code": None,
        "error": None,
        "stop_requested": False,
        "launcher_pid": None,
        "child_pid": None,
        "log_path": str(log_path),
    }
    write_job(job_path, job)

    runner = [sys.executable, str(ROOT / "scripts" / "run_job.py"), "--job", str(job_path)]
    kwargs: dict[str, Any] = {
        "cwd": ROOT,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if environment:
        process_environment = os.environ.copy()
        process_environment.update({str(key): str(value) for key, value in environment.items()})
        kwargs["env"] = process_environment
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(runner, **kwargs)
    return job


def list_jobs() -> list[dict[str, Any]]:
    if not JOB_DIR.exists():
        return []
    jobs = []
    for path in JOB_DIR.glob("*.json"):
        try:
            jobs.append(read_job(path))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(jobs, key=lambda item: item.get("created_at") or "", reverse=True)


def request_stop(job_id: str) -> dict[str, Any]:
    path = JOB_DIR / f"{job_id}.json"
    job = read_job(path)
    if job["status"] not in FINAL_STATUSES:
        job["status"] = "stopping"
        job["stop_requested"] = True
        write_job(path, job)
    return job


def tail_log(job: dict[str, Any], max_chars: int = 20_000) -> str:
    path = Path(job["log_path"])
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_chars))
        return handle.read().decode("utf-8", errors="replace")


def read_job(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_job(path: str | Path, job: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(job, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def update_job(path: str | Path, **changes: Any) -> dict[str, Any]:
    job = read_job(path)
    job.update(changes)
    write_job(path, job)
    return job


def utc_now() -> str:
    return _now()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
