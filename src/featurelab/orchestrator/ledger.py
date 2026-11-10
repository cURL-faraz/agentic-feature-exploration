"""The hypothesis ledger - the system's long-term memory (SQLite, one file).

Every hypothesis ever tried is recorded, accepted or not.  This one table does
four jobs:

* **Memory for the inventor** - it is told what was tried and why it failed.
* **Deduplication** - identical code (by hash) is never evaluated twice.
* **Statistical accounting** - how many Stage-B tests and lockbox looks have
  been spent under this protocol (multiple-testing budget).
* **Audit trail** - the report and teammates read results from here, never
  from an LLM's memory of them.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created REAL, run_id TEXT, protocol TEXT, source TEXT, name TEXT,
    code_hash TEXT, proposal TEXT, models TEXT,
    status TEXT,          -- proposed | rejected_by_critic | gate_failed | screen_failed |
                          -- incremental_failed | confirm_failed | accepted | error
    stage_reached TEXT, reason TEXT,
    runner TEXT, screen TEXT, incremental TEXT, confirm TEXT
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, run_id TEXT, role TEXT, model TEXT,
    ok INTEGER, cached INTEGER, seconds REAL, prompt_tokens INTEGER, completion_tokens INTEGER, error TEXT
);
"""

JSON_FIELDS = ("proposal", "models", "runner", "screen", "incremental", "confirm")


def code_hash(code: str) -> str:
    normalised = "\n".join(line.rstrip() for line in code.strip().splitlines())
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


class Ledger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def add(self, **fields) -> int:
        fields.setdefault("created", time.time())
        for key in JSON_FIELDS:
            if key in fields and not isinstance(fields[key], str):
                fields[key] = json.dumps(fields[key], default=float)
        cols = ", ".join(fields)
        cur = self.db.execute(f"INSERT INTO hypotheses ({cols}) VALUES ({', '.join('?' * len(fields))})",
                              list(fields.values()))
        self.db.commit()
        return int(cur.lastrowid)

    def update(self, hid: int, **fields) -> None:
        for key in JSON_FIELDS:
            if key in fields and not isinstance(fields[key], str):
                fields[key] = json.dumps(fields[key], default=float)
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE hypotheses SET {sets} WHERE id = ?", [*fields.values(), hid])
        self.db.commit()

    def log_call(self, run_id: str, record: dict) -> None:
        self.db.execute(
            "INSERT INTO llm_calls (ts, run_id, role, model, ok, cached, seconds, prompt_tokens, "
            "completion_tokens, error) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (time.time(), run_id, record.get("role"), record.get("model"), int(bool(record.get("ok"))),
             int(bool(record.get("cached"))), record.get("seconds"), record.get("prompt_tokens"),
             record.get("completion_tokens"), record.get("error")))
        self.db.commit()

    def rows(self, where: str = "1=1", params: tuple = ()) -> list[dict]:
        out = []
        for row in self.db.execute(f"SELECT * FROM hypotheses WHERE {where} ORDER BY id", params):
            item = dict(row)
            for key in JSON_FIELDS:
                if item.get(key):
                    try:
                        item[key] = json.loads(item[key])
                    except Exception:  # noqa: BLE001
                        pass
            out.append(item)
        return out

    def seen(self, protocol: str, digest: str) -> dict | None:
        rows = self.rows("protocol = ? AND code_hash = ? AND status != 'proposed'", (protocol, digest))
        return rows[-1] if rows else None

    def count_stage_b(self, protocol: str) -> int:
        return len(self.rows("protocol = ? AND incremental IS NOT NULL", (protocol,)))

    def count_lockbox_looks(self, protocol: str) -> int:
        return len(self.rows("protocol = ? AND confirm IS NOT NULL", (protocol,)))

    def history_for_inventor(self, protocol: str, limit: int = 15) -> list[dict]:
        """Coarse feedback only - see docs: rich feedback lets the loop overfit the validation data."""
        out = []
        for row in self.rows("protocol = ?", (protocol,))[-limit:]:
            inc = row.get("incremental") if isinstance(row.get("incremental"), dict) else None
            outcome = {
                "accepted": "ACCEPTED - confirmed on unseen subjects",
                "rejected_by_critic": "rejected by reviewer: " + (row.get("reason") or "")[:120],
                "gate_failed": "code failed the contract/runtime checks",
                "screen_failed": "rejected at screening: " + (row.get("reason") or "")[:140],
                "incremental_failed": "no reliable gain over the baseline"
                + (f" (mean gain {inc['mean_delta']:+.3f})" if inc else ""),
                "confirm_failed": "gain did not replicate on unseen subjects",
            }.get(row.get("status"), row.get("status") or "")
            out.append({"name": row.get("name"), "outcome": outcome})
        return out
