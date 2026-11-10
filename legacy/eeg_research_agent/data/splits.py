"""Subject-grouped data splitting -- the project's leakage firewall.

Epochs from one participant are highly correlated, so splitting *epochs* at
random leaks subject identity into the test set and inflates every metric.
Splits here are therefore made at the **subject** level, are deterministic given
a seed, are written to a manifest, and are re-verified by the verification agent.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..logging_utils import get_logger
from ..eeg.segmentation import EpochSet

__all__ = [
    "DatasetSplits",
    "make_subject_splits",
    "build_epoch_index",
    "verify_splits",
]

logger = get_logger(__name__)

SPLIT_ORDER = ("train", "validation", "test")


@dataclass
class DatasetSplits:
    """Subject-level train/validation/test assignment."""

    train: list[str] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    seed: int = 0

    def split_for(self, subject_id: str) -> str | None:
        if subject_id in self.train:
            return "train"
        if subject_id in self.validation:
            return "validation"
        if subject_id in self.test:
            return "test"
        return None

    def all_subjects(self) -> list[str]:
        return [*self.train, *self.validation, *self.test]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "train_subjects": list(self.train),
            "validation_subjects": list(self.validation),
            "test_subjects": list(self.test),
            "n_subjects": len(self.all_subjects()),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DatasetSplits":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            train=list(payload.get("train_subjects", [])),
            validation=list(payload.get("validation_subjects", [])),
            test=list(payload.get("test_subjects", [])),
            seed=int(payload.get("seed", 0)),
        )


def make_subject_splits(
    subjects: Sequence[str],
    test_fraction: float = 0.25,
    validation_fraction: float = 0.25,
    seed: int = 0,
    min_subjects_per_split: int = 1,
) -> DatasetSplits:
    """Allocate subjects to train/validation/test deterministically.

    ``validation_fraction`` is taken from the non-test remainder.  When too few
    subjects are available the split is trimmed (and a warning is logged) rather
    than producing empty partitions.
    """
    unique = sorted(set(subjects))
    if not unique:
        raise ValueError("cannot split an empty subject list")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")

    rng = np.random.default_rng(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = int(round(n_total * test_fraction))
    remainder = n_total - n_test
    n_val = int(round(remainder * validation_fraction))

    # guarantee at least one subject in train/validation/test when possible
    if n_total >= 3:
        n_test = max(min_subjects_per_split, min(n_test, n_total - 2 * min_subjects_per_split))
        n_val = max(min_subjects_per_split, min(n_val, n_total - n_test - min_subjects_per_split))
    elif n_total == 2:
        n_test, n_val = 1, 0
        logger.warning("only 2 subjects available: validation split is empty")
    else:
        n_test, n_val = 0, 0
        logger.warning("only 1 subject available: all data used for training (no evaluation possible)")

    test = shuffled[:n_test]
    validation = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return DatasetSplits(train=sorted(train), validation=sorted(validation), test=sorted(test), seed=seed)


def build_epoch_index(epoch_sets: Iterable[EpochSet]) -> list[dict[str, Any]]:
    """Flatten epoch sets into an index table (one row per epoch)."""
    rows: list[dict[str, Any]] = []
    for epoch_set in epoch_sets:
        for index, stage in enumerate(epoch_set.stages):
            rows.append(
                {
                    "subject_id": epoch_set.subject_id,
                    "recording_id": epoch_set.recording_id,
                    "epoch_id": index,
                    "onset_s": float(epoch_set.onsets[index]),
                    "stage": stage,
                    "ch_name": epoch_set.ch_name,
                    "sfreq": float(epoch_set.sfreq),
                    "data_origin": epoch_set.data_origin,
                    "epoch_seconds": float(epoch_set.epoch_seconds),
                }
            )
    return rows


def verify_splits(rows: Sequence[dict[str, Any]], splits: DatasetSplits) -> dict[str, Any]:
    """Check that the split bookkeeping is internally consistent.

    Returns a report containing ``ok`` and the individual checks, so that it can
    be embedded in the validation report and asserted in tests.
    """
    by_subject: dict[str, set[str]] = {}
    for row in rows:
        by_subject.setdefault(str(row["subject_id"]), set()).add(str(row["split"]))

    overlaps = {subject: sorted(splits_) for subject, splits_ in by_subject.items() if len(splits_) > 1}
    described = set(splits.all_subjects())
    observed = set(by_subject)
    stage_distribution: dict[str, dict[str, int]] = {}
    for split in SPLIT_ORDER:
        counter = Counter(row["stage"] for row in rows if row["split"] == split)
        stage_distribution[split] = dict(counter)

    checks = {
        "no_subject_in_multiple_splits": not overlaps,
        "subjects_are_partitioned": described == observed,
        "every_split_non_empty": all(
            len(getattr(splits, name)) > 0 for name in SPLIT_ORDER if len(described) >= 3
        ),
        "epochs_assigned": len(rows) > 0,
    }
    report = {
        "ok": all(checks.values()),
        "checks": checks,
        "overlaps": overlaps,
        "n_epochs": len(rows),
        "n_subjects": len(observed),
        "epochs_per_split": {
            split: sum(1 for row in rows if row["split"] == split) for split in SPLIT_ORDER
        },
        "subjects_per_split": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "stage_distribution": stage_distribution,
        "seed": splits.seed,
    }
    return report


def assign_splits(rows: list[dict[str, Any]], splits: DatasetSplits) -> list[dict[str, Any]]:
    """Attach a ``split`` field to each epoch row (in place, returns the rows)."""
    for row in rows:
        split = splits.split_for(str(row["subject_id"]))
        if split is None:
            raise KeyError(f"subject {row['subject_id']} is not part of any split")
        row["split"] = split
    return rows
