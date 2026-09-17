from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JSONLLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

    def write(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("wall_clock_seconds", time.time() - self.start_time)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
