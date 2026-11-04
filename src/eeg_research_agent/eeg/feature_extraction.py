"""Deterministic feature-extraction engine.

Responsibilities
----------------
* run every approved feature implementation over every epoch,
* record **every** failure and every NaN instead of dropping epochs silently,
* write a feature table plus a machine-readable extraction report
  (configuration, software versions, failures, quality statistics, timings),
* optionally add explicitly-labelled *non-causal* sequence features.

Caching
-------
The cache key is a hash of (data fingerprint, feature ids, relevant configuration
subtree, package versions).  A cache hit is only used when the fingerprint matches
exactly, so a stale cache can never silently substitute for fresh computation.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import ProjectConfig
from ..logging_utils import get_logger
from ..data.splits import EpochSet
from .feature_registry import FeatureRegistry, SignalContext, load_default_registry

__all__ = [
    "ExtractionReport",
    "FeatureTable",
    "add_sequence_features",
    "extract_features",
    "save_feature_table",
    "load_feature_table",
    "cache_key",
]

logger = get_logger(__name__)

METADATA_COLUMNS = (
    "subject_id",
    "recording_id",
    "epoch_id",
    "onset_s",
    "stage",
    "ch_name",
    "data_origin",
    # Bookkeeping columns that are added around extraction.  They must be listed
    # here because :func:`add_sequence_features` derives its feature set from the
    # frame's own columns: a text column such as ``split`` would otherwise be
    # coerced to float and abort the sequence-feature stage.
    "split",
    "sfreq",
    "epoch_seconds",
    "timestamp",
)

#: Sequence features need neighbouring epochs, so they can never be causal.
NON_CAUSAL_FEATURES = ("feature_delta_norm", "feature_delta_norm_sigma")


@dataclass
class ExtractionReport:
    """Bookkeeping for one extraction pass."""

    n_epochs: int = 0
    n_features: int = 0
    feature_ids: list[str] = field(default_factory=list)
    failures: dict[str, dict[str, int]] = field(default_factory=dict)
    nan_counts: dict[str, int] = field(default_factory=dict)
    durations_s: list[float] = field(default_factory=list)
    excluded_epochs: int = 0
    exclusion_reasons: dict[str, int] = field(default_factory=dict)
    software: dict[str, Any] = field(default_factory=dict)
    extraction_seconds: float = 0.0
    data_origin: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_epochs": self.n_epochs,
            "n_features": self.n_features,
            "feature_ids": list(self.feature_ids),
            "failures": {k: dict(v) for k, v in self.failures.items()},
            "nan_counts": {k: v for k, v in self.nan_counts.items() if v},
            "non_causal_features": list(NON_CAUSAL_FEATURES),
            "excluded_epochs": self.excluded_epochs,
            "exclusion_reasons": dict(self.exclusion_reasons),
            "software": self.software,
            "extraction_seconds": round(self.extraction_seconds, 3),
            "per_epoch_seconds_mean": (
                round(float(np.mean(self.durations_s)), 6) if self.durations_s else None
            ),
            "data_origin": self.data_origin,
        }


@dataclass
class FeatureTable:
    """Feature matrix plus provenance."""

    table: pd.DataFrame
    feature_ids: list[str]
    report: ExtractionReport
    feature_definitions: list[dict[str, Any]]

    @property
    def numeric_features(self) -> list[str]:
        return [c for c in self.feature_ids if c in self.table.columns]


def _software_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module_name in ("numpy", "scipy", "sklearn", "pandas", "mne", "matplotlib", "pydantic", "langgraph"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:  # pragma: no cover - optional
            versions[module_name] = None
    return versions


def _config_fingerprint(config: ProjectConfig) -> str:
    payload = json.dumps(
        {
            "features": config.features.model_dump(mode="json"),
            "dataset": {
                "epoch_seconds": config.dataset.signal.epoch_seconds,
                "sfreq": config.dataset.signal.target_sampling_frequency,
            },
            "seed": config.seed,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_key(epoch_sets: Sequence[EpochSet], config: ProjectConfig, feature_ids: Sequence[str]) -> str:
    """Deterministic cache key for a feature-extraction result."""
    data_payload = "|".join(
        f"{es.subject_id}:{es.recording_id}:{es.n_epochs}:{es.data.shape[1]}:{round(float(np.sum(es.data)), 8)}"
        for es in epoch_sets
    )
    payload = f"{data_payload}|{','.join(sorted(feature_ids))}|{_config_fingerprint(config)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def extract_features(
    epoch_sets: Sequence[EpochSet],
    config: ProjectConfig,
    feature_ids: Sequence[str] | None = None,
    registry: FeatureRegistry | None = None,
) -> FeatureTable:
    """Extract registered features for every epoch of every epoch set."""
    registry = registry or load_default_registry()
    selected = list(feature_ids) if feature_ids is not None else registry.ids()
    report = ExtractionReport(feature_ids=list(selected), software=_software_versions())
    started = time.time()

    rows: list[dict[str, Any]] = []
    for epoch_set in epoch_sets:
        report.data_origin = epoch_set.data_origin
        report.excluded_epochs += len(epoch_set.exclusions)
        for exclusion in epoch_set.exclusions:
            reason = str(exclusion.get("reason", "unknown"))
            report.exclusion_reasons[reason] = report.exclusion_reasons.get(reason, 0) + 1
        for index, (stage, onset) in enumerate(zip(epoch_set.stages, epoch_set.onsets)):
            context = SignalContext(epoch_set.data[index], epoch_set.sfreq, config, stage=stage)
            epoch_started = time.time()
            values, errors = registry.compute(context, selected)
            report.durations_s.append(time.time() - epoch_started)
            for feature_id, message in errors.items():
                bucket = report.failures.setdefault(feature_id, {})
                key = message.split(":")[0]
                bucket[key] = bucket.get(key, 0) + 1
            row = {
                "subject_id": epoch_set.subject_id,
                "recording_id": epoch_set.recording_id,
                "epoch_id": index,
                "onset_s": float(onset),
                "stage": stage,
                "ch_name": epoch_set.ch_name,
                "data_origin": epoch_set.data_origin,
            }
            row.update(values)
            rows.append(row)

    table = pd.DataFrame(rows)
    for feature_id in selected:
        if feature_id not in table.columns:
            table[feature_id] = np.nan
        column = table[feature_id].astype(float)
        report.nan_counts[feature_id] = int(np.sum(~np.isfinite(column)))
    report.n_epochs = len(table)
    report.n_features = len([f for f in selected if f in table.columns])
    report.extraction_seconds = time.time() - started
    definitions = [registry.get(feature_id).describe() for feature_id in selected if feature_id in registry]
    logger.info(
        "extracted %d features for %d epochs in %.2f s (%d feature computations failed)",
        report.n_features,
        report.n_epochs,
        report.extraction_seconds,
        sum(sum(v.values()) for v in report.failures.values()),
    )
    return FeatureTable(table=table, feature_ids=list(selected), report=report, feature_definitions=definitions)


def _standardise(
    frame: pd.DataFrame, columns: Sequence[str], stats: dict[str, tuple[float, float]] | None
) -> pd.DataFrame:
    """Robust z-scores; uses supplied train statistics when available.

    Guarantees
    ----------
    * Columns that are entirely non-finite are dropped instead of poisoning the
      whole distance computation - one dead feature must not void an epoch.
    * Non-finite entries of otherwise usable columns are imputed with ``0.0``
      ("at the central value"), the least-informative choice, applied identically
      regardless of split.
    * The result is assembled with a single ``pd.concat`` to avoid the
      DataFrame-fragmentation performance warning.
    """
    loaded: list[pd.Series] = []
    for column in columns:
        values = frame[column].astype(float)
        finite = np.isfinite(values.to_numpy(dtype=float))
        if not finite.any():
            continue
        values = values.where(finite)
        if stats and column in stats:
            center, scale = stats[column]
            if not (np.isfinite(center) and np.isfinite(scale) and scale):
                center, scale = 0.0, 1.0
        else:
            center = float(np.nanmedian(values))
            scale = float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))
            if not np.isfinite(scale) or scale == 0.0:
                scale = float(np.nanstd(values)) or 1.0
        loaded.append(((values - center) / (scale if np.isfinite(scale) and scale else 1.0)).rename(column))
    if not loaded:
        return pd.DataFrame(index=frame.index)
    return pd.concat(loaded, axis=1).fillna(0.0)


def add_sequence_features(
    table: pd.DataFrame,
    columns: Sequence[str] | None = None,
    lookback: int = 1,
    stats: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Add *non-causal* epoch-sequence features.

    Two features are produced per epoch (never for the first epoch of a recording):

    ``feature_delta_norm``
        Euclidean norm of the change in the standardised feature vector relative to
        the previous epoch - the "feature-space transition" hypothesis of the
        research brief.

    ``feature_delta_norm_sigma``
        The same quantity restricted to the sigma-band family, a targeted
        spindle-transition probe.

    Both are marked non-causal because they require the preceding epoch, which is
    available in offline scoring but *not* at the moment a single epoch must be
    classified in isolation.  They are therefore excluded from the causal
    ablation comparison by the evaluation module.
    """
    frame = table.copy().reset_index(drop=True)
    feature_columns = list(columns or [c for c in frame.columns if c not in METADATA_COLUMNS])
    sigma_columns = [c for c in feature_columns if "sigma" in c or c.startswith("spindle")]
    z_scores = _standardise(frame, feature_columns, stats)
    z_sigma = _standardise(frame, sigma_columns, stats) if sigma_columns else None

    delta_norm = np.full(len(frame), np.nan)
    delta_sigma = np.full(len(frame), np.nan)
    matrix = z_scores.to_numpy(dtype=float) if len(z_scores.columns) else None
    sigma_matrix = z_sigma.to_numpy(dtype=float) if z_sigma is not None and len(z_sigma.columns) else None
    for _, indices in frame.groupby(["subject_id", "recording_id"], sort=False).indices.items():
        ordered = list(indices)
        for position in range(lookback, len(ordered)):
            current, previous = ordered[position], ordered[position - lookback]
            if matrix is not None:
                delta_norm[current] = float(np.linalg.norm(matrix[current] - matrix[previous]))
            if sigma_matrix is not None:
                delta_sigma[current] = float(np.linalg.norm(sigma_matrix[current] - sigma_matrix[previous]))
    frame.index = table.index
    frame["feature_delta_norm"] = delta_norm
    frame["feature_delta_norm_sigma"] = delta_sigma
    frame.attrs["non_causal_features"] = list(NON_CAUSAL_FEATURES)
    return frame


def save_feature_table(
    feature_table: FeatureTable,
    config: ProjectConfig,
    name: str = "features",
    write_parquet: bool = True,
) -> dict[str, str]:
    """Persist the feature table plus CSV/JSON companions; returns written paths."""
    features_dir = config.path("features")
    features_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    csv_path = features_dir / f"{name}.csv.gz"
    feature_table.table.to_csv(csv_path, index=False, compression="gzip")
    paths["csv"] = str(csv_path)

    if write_parquet:
        try:
            parquet_path = features_dir / f"{name}.parquet"
            feature_table.table.to_parquet(parquet_path, index=False)
            paths["parquet"] = str(parquet_path)
        except Exception as exc:  # pragma: no cover - pyarrow optional
            logger.info("parquet output unavailable (%s); CSV used instead", type(exc).__name__)
            paths["parquet_error"] = str(exc)

    report_path = features_dir / f"{name}_extraction_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(feature_table.report.to_dict(), handle, indent=2, default=str)
    paths["report"] = str(report_path)

    definitions_path = config.path("metadata") / f"{name}_definitions.json"
    definitions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(definitions_path, "w", encoding="utf-8") as handle:
        json.dump(feature_table.feature_definitions, handle, indent=2, default=str)
    paths["definitions"] = str(definitions_path)
    return paths


def load_feature_table(config: ProjectConfig, name: str = "features") -> pd.DataFrame:
    """Load a previously saved feature table (CSV/Parquet)."""
    features_dir = config.path("features")
    parquet_path = features_dir / f"{name}.parquet"
    csv_path = features_dir / f"{name}.csv.gz"
    if parquet_path.is_file():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:  # pragma: no cover - fall back to CSV
            pass
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"no cached feature table found under {features_dir}")


def feature_columns(table: pd.DataFrame, exclude_metadata: bool = True) -> list[str]:
    """Feature columns of a table (metadata and bookkeeping removed)."""
    skip = set(METADATA_COLUMNS) | {"split", "label", "fold"}
    columns = [c for c in table.columns if c not in skip]
    if exclude_metadata:
        columns = [c for c in columns if np.issubdtype(table[c].dtype, np.number)]
    return columns


def subset_table(table: pd.DataFrame, feature_ids: Iterable[str]) -> pd.DataFrame:
    """Keep metadata columns plus the requested features."""
    keep = [c for c in METADATA_COLUMNS if c in table.columns]
    keep += [c for c in feature_ids if c in table.columns]
    return table.loc[:, list(dict.fromkeys(keep))].copy()
