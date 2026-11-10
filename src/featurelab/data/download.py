"""Download Sleep-EDF Expanded recordings from PhysioNet's open S3 mirror.

Stdlib only, so it can run before the scientific environment exists.
Every file is verified against PhysioNet's published SHA256SUMS, and a file that
is already present with the right checksum is skipped (safe to re-run).

Cohorts (see docs/featurelab/02_data_protocol.md for why these three):

* ``sc20``      - SC subjects 0-19, the "Sleep-EDF-20" benchmark used by most
                  single-channel papers (DeepSleepNet, AttnSleep, ...).  Discovery set.
* ``sc_lockbox``- SC subjects 20-39.  Same population, never seen during discovery.
* ``st``        - Sleep Telemetry (22 subjects, placebo + temazepam nights).
                  Different study and population: the external test.
* ``sc78``      - all 78 SC subjects (the "Sleep-EDF-78" benchmark), optional.
"""

from __future__ import annotations

import argparse
import hashlib
import socket
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0"

socket_timeout = 120  # seconds; without it a stalled connection hangs forever


def data_root() -> Path:
    return Path(os.environ.get("FEATURELAB_DATA", "data/featurelab")).expanduser().resolve()


def _sums() -> dict[str, str]:
    with urllib.request.urlopen(f"{BASE}/SHA256SUMS.txt", timeout=60) as response:
        text = response.read().decode()
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def _subject_of(name: str) -> tuple[str, int]:
    """'sleep-cassette/SC4012E0-PSG.edf' -> ('SC', 1); ST7011J0 -> ('ST', 1)."""
    stem = Path(name).name
    if stem.startswith("SC4"):
        return "SC", int(stem[3:5])
    if stem.startswith("ST7"):
        return "ST", int(stem[3:5])
    return "?", -1


def select_files(cohort: str, sums: dict[str, str]) -> list[str]:
    files = [n for n in sums if n.endswith(".edf")]
    picked = []
    for name in files:
        kind, subject = _subject_of(name)
        if cohort == "sc20" and kind == "SC" and subject < 20:
            picked.append(name)
        elif cohort == "sc_lockbox" and kind == "SC" and 20 <= subject < 40:
            picked.append(name)
        elif cohort == "sc78" and kind == "SC":
            picked.append(name)
        elif cohort == "st" and kind == "ST":
            picked.append(name)
    return sorted(picked)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, expected: str, root: Path) -> str:
    target = root / "raw" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and _sha256(target) == expected:
        return f"skip {name}"
    tmp = target.with_suffix(target.suffix + ".part")
    for attempt in range(6):
        if attempt:
            time.sleep(min(120, 5 * 2**attempt))  # exponential backoff: the server is throttling us
        try:
            urllib.request.urlretrieve(f"{BASE}/{name}", tmp)
            if _sha256(tmp) != expected:
                raise IOError("checksum mismatch")
            tmp.replace(target)
            return f"ok   {name}"
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            last = exc
    return f"FAIL {name}: {last}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cohorts", nargs="+", choices=["sc20", "sc_lockbox", "st", "sc78"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    socket.setdefaulttimeout(socket_timeout)
    root = data_root()
    sums = _sums()
    names: list[str] = []
    for cohort in args.cohorts:
        names += [n for n in select_files(cohort, sums) if n not in names]
    print(f"{len(names)} files -> {root / 'raw'}", flush=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, name, sums[name], root) for name in names]
        for i, future in enumerate(as_completed(futures), 1):
            message = future.result()
            failures += message.startswith("FAIL")
            print(f"[{i}/{len(names)}] {message}", flush=True)
    print("done" if not failures else f"done with {failures} failures", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
