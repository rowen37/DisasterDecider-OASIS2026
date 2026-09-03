from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Minimal trajectory logger for later DORA-style ablation experiments."""

    def __init__(self, path: str = "runs/trajectories.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, run_id: str, stage: str, event: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": time.time(),
            "run_id": run_id,
            "stage": stage,
            "event": event,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
