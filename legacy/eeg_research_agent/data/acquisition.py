"""Dataset acquisition with an explicit, recorded provenance trail.

Two origins are supported:

``sleep-edf``
    Sleep-EDF Expanded (PhysioNet, ``sleep-cassette`` collection).  Files are
    fetched through MNE's dataset helper when available, otherwise directly over
    HTTPS.  Downloads are cached in ``data/raw``; existing files are reused after
    a structural integrity check (readable, non-empty, EDF/header parseable).

``synthetic``
    Deterministic surrogate EEG (:mod:`eeg_research_agent.data.synthetic`) used
    when no network access is available.  The manifest always records why the
    fallback happened so no report can accidentally attribute synthetic results
    to the clinical dataset.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ProjectConfig
from ..logging_utils import get_logger
from .annotations import StageMapper
from .inspection import pick_channel
from .recording import Recording, StageAnnotation
from .synthetic import generate_corpus

__all__ = [
    "AcquisitionResult",
    "acquire_dataset",
    "network_available",
    "download_file",
    "sha256_file",
]

logger = get_logger(__name__)

#: Hosts probed by :func:`network_available` (cheap, HEAD-like requests).
_PROBE_HOSTS = ("https://physionet.org/robots.txt", "https://pypi.org/simple/")
_PROBE_TIMEOUT = 4.0


@dataclass
class AcquisitionResult:
    """Recordings plus the provenance manifest describing how they were obtained."""

    recordings: list[Recording]
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def origin(self) -> str:
        return str(self.manifest.get("data_origin", "unknown"))


def network_available(timeout: float = _PROBE_TIMEOUT) -> bool:
    """Return ``True`` when at least one external host is reachable."""
    from urllib import request

    for host in _PROBE_HOSTS:
        try:
            with request.urlopen(host, timeout=timeout):  # noqa: S310 - fixed https hosts
                return True
        except Exception:  # noqa: BLE001 - any failure means "not reachable"
            continue
    return False


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path, timeout: float = 60.0, retries: int = 3) -> dict[str, Any]:
    """Stream ``url`` to ``destination`` (atomic replace), with retries."""
    import httpx

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
                response.raise_for_status()
                with open(tmp, "wb") as handle:
                    for chunk in response.iter_bytes(1 << 20):
                        handle.write(chunk)
            tmp.replace(destination)
            return {
                "url": url,
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "attempts": attempt,
            }
        except Exception as exc:  # noqa: BLE001 - network layer failures are expected
            last_error = exc
            time.sleep(min(2.0 * attempt, 10.0))
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last_error}")


# --------------------------------------------------------------------------- #
# Sleep-EDF Expanded
# --------------------------------------------------------------------------- #
def _sleep_edf_paths(config: ProjectConfig, raw_dir: Path) -> list[tuple[str, str]]:
    """Return ``(psg, hypnogram)`` path pairs, downloading when necessary."""
    src = config.dataset.source
    subset = config.dataset.subset
    pairs: list[tuple[str, str]] = []
    for subject in subset.subjects:
        for night in subset.recordings:
            psg_name = src.file_pattern.format(subj=subject, night=night, recording=night)
            hyp_name = src.annotation_pattern.format(subj=subject, night=night, recording=night)
            pairs.append((str(raw_dir / psg_name), str(raw_dir / hyp_name)))
    return pairs


def _ensure_sleep_edf_files(config: ProjectConfig, raw_dir: Path) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """Download missing Sleep-EDF files; return pairs and a per-file audit trail."""
    attempts: list[dict[str, Any]] = []
    pairs = _sleep_edf_paths(config, raw_dir)
    base = config.dataset.source.file_base_url.rstrip("/")
    collection = config.dataset.source.collection

    for psg_path, hyp_path in pairs:
        for path in (Path(psg_path), Path(hyp_path)):
            url = f"{base}/{collection}/{path.name}"
            if path.is_file() and path.stat().st_size > 0:
                attempts.append(
                    {"file": path.name, "status": "cached", "bytes": path.stat().st_size,
                     "sha256": sha256_file(path)}
                )
                continue
            info = download_file(url, path)
            info["status"] = "downloaded"
            info["file"] = path.name
            attempts.append(info)
    return pairs, attempts


def read_sleep_edf_recording(
    psg_path: str,
    hypnogram_path: str,
    config: ProjectConfig,
    subject_id: str,
    recording_id: str,
) -> Recording:
    """Read one Sleep-EDF recording, map its annotations to canonical stages."""
    import mne

    raw = mne.io.read_raw_edf(psg_path, preload=True, stim_channel=None, verbose="ERROR")
    channel = pick_channel(raw.ch_names, list(config.dataset.signal.channel_preference))
    if channel is None:
        raise RuntimeError(f"no EEG channel found in {psg_path}; channels={raw.ch_names}")
    raw.pick([channel])

    annotations = mne.read_annotations(hypnogram_path)
    mapper = StageMapper(
        config.dataset.stages.annotation_map,
        supported=list(config.dataset.stages.supported) + ["UNKNOWN"],
    )
    raw_annotations = [
        (float(onset), float(duration), str(description))
        for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description)
    ]
    mapped, report = mapper.map_annotations(raw_annotations)

    max_seconds = config.dataset.subset.max_recording_seconds
    if max_seconds:
        mapped = [ann for ann in mapped if ann.onset < float(max_seconds)]

    signal = raw.get_data(picks=[0])[0]  # volts
    sfreq = float(raw.info["sfreq"])
    if max_seconds:
        signal = signal[: int(float(max_seconds) * sfreq)]

    return Recording(
        signal=signal,
        sfreq=sfreq,
        ch_name=channel,
        annotations=mapped,
        subject_id=subject_id,
        recording_id=recording_id,
        data_origin="sleep-edf",
        source_file=str(psg_path),
        channel_candidates=list(raw.ch_names) + [channel],
        metadata={
            "annotation_report": report.to_dict(),
            "hypnogram_file": str(hypnogram_path),
            "recording_start": str(raw.info.get("meas_date")),
            "device": str(raw.info.get("device_info", "")),
            "truncated_to_seconds": max_seconds,
        },
    )


def _acquire_sleep_edf(config: ProjectConfig) -> tuple[list[Recording], dict[str, Any]]:
    raw_dir = config.path("raw")
    pairs, attempts = _ensure_sleep_edf_files(config, raw_dir)
    recordings: list[Recording] = []
    errors: list[dict[str, Any]] = []
    for psg_path, hyp_path in pairs:
        subject_token = Path(psg_path).stem[:4]
        recording_id = Path(psg_path).stem
        try:
            rec = read_sleep_edf_recording(psg_path, hyp_path, config, subject_token, recording_id)
        except Exception as exc:  # noqa: BLE001 - a broken file must not kill the run
            logger.warning("failed to read %s: %s", psg_path, exc)
            errors.append({"file": psg_path, "error": f"{type(exc).__name__}: {exc}"})
            continue
        recordings.append(rec)
    if not recordings:
        raise RuntimeError("Sleep-EDF acquisition produced no usable recordings")
    manifest = {
        "data_origin": "sleep-edf",
        "dataset": config.dataset.dataset_name,
        "doi": config.dataset.source.doi,
        "landing_page": config.dataset.source.landing_page,
        "collection": config.dataset.source.collection,
        "files": attempts,
        "errors": errors,
    }
    return recordings, manifest


# --------------------------------------------------------------------------- #
# synthetic fallback
# --------------------------------------------------------------------------- #
def _acquire_synthetic(config: ProjectConfig, reason: str) -> tuple[list[Recording], dict[str, Any]]:
    fallback = config.dataset.fallback
    max_subjects = config.settings.execution.max_subjects
    n_subjects = min(int(fallback.subjects), int(max_subjects)) if max_subjects else int(fallback.subjects)
    epochs = int(fallback.epochs_per_subject)
    if config.settings.execution.mode == "smoke":
        n_subjects = min(n_subjects, 3)
        epochs = min(epochs, 40)
    recordings = generate_corpus(
        n_subjects=n_subjects,
        epochs_per_subject=epochs,
        sfreq=float(config.dataset.signal.target_sampling_frequency),
        epoch_seconds=float(config.dataset.signal.epoch_seconds),
        seed=int(fallback.seed),
        effect_strength=float(fallback.effect_strength),
    )
    manifest = {
        "data_origin": "synthetic",
        "dataset": "synthetic_psg_v1",
        "requested_dataset": config.dataset.dataset_name,
        "fallback_reason": reason,
        "generator": {
            "mode": fallback.mode,
            "subjects": n_subjects,
            "epochs_per_subject": epochs,
            "seed": int(fallback.seed),
            "effect_strength": float(fallback.effect_strength),
            "sampling_frequency": float(config.dataset.signal.target_sampling_frequency),
            "epoch_seconds": float(config.dataset.signal.epoch_seconds),
        },
        "caveat": (
            "Stage-specific spectral/event structure is imposed by construction; these data "
            "exercise the pipeline and cannot support claims about human sleep EEG."
        ),
        "files": [],
        "errors": [],
    }
    return recordings, manifest


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def acquire_dataset(
    config: ProjectConfig,
    origin: str | None = None,
    force_offline: bool = False,
) -> AcquisitionResult:
    """Acquire recordings, honouring ``origin`` (``auto``/``sleep-edf``/``synthetic``)."""
    requested = (origin or config.dataset.data_origin or "auto").lower()
    allow_network = config.settings.offline_mode != "always" and not force_offline

    if requested == "synthetic":
        recordings, manifest = _acquire_synthetic(config, "explicitly requested origin=synthetic")
        return AcquisitionResult(recordings, manifest)

    if allow_network and network_available():
        try:
            recordings, manifest = _acquire_sleep_edf(config)
            manifest["requested_origin"] = requested
            return AcquisitionResult(recordings, manifest)
        except Exception as exc:  # noqa: BLE001 - fall back, but record everything
            reason = f"Sleep-EDF acquisition failed: {type(exc).__name__}: {exc}"
            logger.warning("%s", reason)
            recordings, manifest = _acquire_synthetic(config, reason)
            return AcquisitionResult(recordings, manifest)

    if requested == "sleep-edf":
        raise RuntimeError(
            "Sleep-EDF was requested explicitly but no network access is available. "
            "Re-run with network access or use origin=synthetic."
        )
    reason = "no network access detected (offline sandbox); used the deterministic synthetic generator"
    logger.warning("%s", reason)
    recordings, manifest = _acquire_synthetic(config, reason)
    return AcquisitionResult(recordings, manifest)


def write_acquisition_manifest(result: AcquisitionResult, config: ProjectConfig) -> Path:
    """Persist the provenance manifest as JSON next to the metadata artifacts."""
    path = config.path("metadata") / "acquisition_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "origin": result.origin,
        "recording_count": len(result.recordings),
        "subjects": sorted({rec.subject_id for rec in result.recordings}),
        **result.manifest,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def recording_to_npz(recording: Recording, destination: Path) -> Path:
    """Cache a conditioned recording as ``.npz`` (never commits clinical data)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        signal=recording.signal,
        sfreq=recording.sfreq,
        onsets=np.array([ann.onset for ann in recording.annotations]),
        durations=np.array([ann.duration for ann in recording.annotations]),
        stages=np.array([ann.stage for ann in recording.annotations]),
    )
    return destination


def load_npz_recording(path: Path, subject_id: str, recording_id: str, ch_name: str) -> Recording:
    """Inverse of :func:`recording_to_npz`."""
    data = np.load(path, allow_pickle=False)
    annotations = [
        StageAnnotation(float(onset), float(duration), str(stage), "cached")
        for onset, duration, stage in zip(data["onsets"], data["durations"], data["stages"])
    ]
    return Recording(
        signal=data["signal"],
        sfreq=float(data["sfreq"]),
        ch_name=ch_name,
        annotations=annotations,
        subject_id=subject_id,
        recording_id=recording_id,
        data_origin="cached",
        source_file=str(path),
    )
