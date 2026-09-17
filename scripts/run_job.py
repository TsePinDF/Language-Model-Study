from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmt.jobs import read_job, update_job, utc_now


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one persisted TMT console job.")
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()

    job = read_job(args.job)
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    update_job(args.job, status="running", started_at=utc_now(), launcher_pid=os.getpid())

    try:
        with log_path.open("a", encoding="utf-8", errors="replace", buffering=1) as log:
            log.write(f"$ {' '.join(job['command'])}\n\n")
            kwargs = {
                "cwd": ROOT,
                "stdin": subprocess.DEVNULL,
                "stdout": log,
                "stderr": subprocess.STDOUT,
                "text": True,
            }
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            child = subprocess.Popen(job["command"], **kwargs)
            update_job(args.job, child_pid=child.pid)

            stop_requested = False
            while child.poll() is None:
                time.sleep(0.5)
                current = read_job(args.job)
                if current.get("stop_requested"):
                    stop_requested = True
                    _stop_child(child)
                    break
            return_code = child.wait()

        current = read_job(args.job)
        stop_requested = stop_requested or bool(current.get("stop_requested"))
        if stop_requested:
            status = "stopped"
        else:
            status = "completed" if return_code == 0 else "failed"
        update_job(
            args.job,
            status=status,
            finished_at=utc_now(),
            return_code=return_code,
            child_pid=None,
        )
    except Exception as exc:
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            traceback.print_exc(file=log)
        update_job(
            args.job,
            status="failed",
            finished_at=utc_now(),
            error=str(exc),
            child_pid=None,
        )
        raise


def _stop_child(child: subprocess.Popen[str]) -> None:
    try:
        child.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        child.wait(timeout=10)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()


if __name__ == "__main__":
    main()
