"""Append-only JSONL security-event log."""
from __future__ import annotations
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List

_LOCK = threading.Lock()


class EventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def log(self, event_type: str, **fields) -> dict:
        rec = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "event": event_type, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def read(self, limit: int = 500) -> List[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue          # tolerate a truncated final line
        return out
