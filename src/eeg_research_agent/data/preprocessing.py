"""Signal conditioning and quality control.

Order of operations (applied to the *continuous* signal, before epoching, so that
filter transients never contaminate epoch boundaries):

1. optional resampling to the configured target sampling frequency,
2. zero-phase Butterworth band-pass (default 0.3-45 Hz),
3. optional IIR notch at the mains frequency (50 or 60 Hz),
4. optional constant detrending.

Quality metrics are computed per epoch on the conditioned signal and stored with
the epoch set; epochs are never dropped silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from scipy import signal as sps

from ..config import ProjectConfig
from .recording import Recording

__all__ = ["PreprocessingReport", "preprocess_recording", "epoch_quality_metrics"]


@dataclass
class PreprocessingReport:
    """Record of exactly what preprocessing was applied."""

    sfreq_in: float
    sfreq_out: float
    bandpass: tuple[float, float]
    notch_hz: float | None
    detrend: bool
    resampled: bool
    n_samples: int
    amplitude_scale_uv: float
    clipping_fraction: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sfreq_in": self.sfreq_in,
            "sfreq_out": self.sfreq_out,
            "bandpass_hz": list(self.bandpass),
            "notch_hz": self.notch_hz,
            "detrend": self.detrend,
            "resampled": self.resampled,
            "n_samples": self.n_samples,
            "signal_scale_uv": round(self.amplitude_scale_uv, 4),
            "clipping_fraction": round(self.clipping_fraction, 8),
            "notes": list(self.notes),
        }


def _resample(signal: np.ndarray, sfreq: float, target: float) -> tuple[np.ndarray, float]:
    if abs(sfreq - target) < 1e-9:
        return signal, sfreq
    ratio = Fraction(target / sfreq).limit_denominator(1000)
    resampled = sps.resample_poly(signal, ratio.numerator, ratio.denominator)
    return resampled, sfreq * ratio.numerator / ratio.denominator


def preprocess_recording(recording: Recording, config: ProjectConfig) -> tuple[Recording, PreprocessingReport]:
    """Condition a recording and attach the preprocessing report."""
    cfg = config.dataset.preprocessing
    signal = np.asarray(recording.signal, dtype=np.float64)
    sfreq_in = float(recording.sfreq)
    notes: list[str] = []

    low, high = float(cfg.bandpass[0]), float(cfg.bandpass[1])
    nyquist = sfreq_in / 2.0
    if high >= nyquist:
        high = nyquist * 0.98
        notes.append(f"high cut clamped to {high:.2f} Hz to stay below Nyquist")

    signal, sfreq = _resample(signal, sfreq_in, float(config.dataset.signal.target_sampling_frequency))
    resampled = abs(sfreq - sfreq_in) > 1e-9
    if resampled:
        nyquist = sfreq / 2.0
        high = min(high, nyquist * 0.98)
        notes.append(f"resampled {sfreq_in:.1f} -> {sfreq:.1f} Hz (polyphase)")

    if cfg.detrend:
        signal = sps.detrend(signal, type="constant")

    sos = sps.butter(4, [low / (sfreq / 2.0), high / (sfreq / 2.0)], btype="bandpass", output="sos")
    signal = sps.sosfiltfilt(sos, signal, padlen=min(3 * (sos.shape[0] * 2), max(0, signal.size - 1)))
    notes.append(f"zero-phase Butterworth band-pass {low}-{high:.1f} Hz (order 4)")

    notch_hz: float | None = None
    if cfg.apply_notch and low < cfg.notch_hz < high:
        notch_hz = float(cfg.notch_hz)
        b_notch, a_notch = sps.iirnotch(notch_hz, cfg.notch_quality, fs=sfreq)
        signal = sps.filtfilt(b_notch, a_notch, signal, padlen=min(3 * max(len(a_notch), len(b_notch)), max(0, signal.size - 1)))
        notes.append(f"IIR notch at {notch_hz:.1f} Hz (Q={cfg.notch_quality:g})")
    elif cfg.apply_notch:
        notes.append(
            f"notch at {cfg.notch_hz:.1f} Hz skipped: outside the {low}-{high:.1f} Hz pass-band"
        )

    scale = float(np.std(signal))
    quantisation = float(np.abs(np.diff(np.unique(signal))).min()) if signal.size > 1 else 0.0
    clipping_fraction = float(quantisation == 0.0)

    processed = Recording(
        signal=signal,
        sfreq=sfreq,
        ch_name=recording.ch_name,
        annotations=list(recording.annotations),
        subject_id=recording.subject_id,
        recording_id=recording.recording_id,
        data_origin=recording.data_origin,
        source_file=recording.source_file,
        channel_candidates=list(recording.channel_candidates),
        metadata=dict(recording.metadata),
        quality=dict(recording.quality),
    )
    report = PreprocessingReport(
        sfreq_in=sfreq_in,
        sfreq_out=sfreq,
        bandpass=(low, high),
        notch_hz=notch_hz,
        detrend=bool(cfg.detrend),
        resampled=resampled,
        n_samples=int(signal.size),
        amplitude_scale_uv=scale * 1e6,
        clipping_fraction=clipping_fraction,
        notes=notes,
    )
    processed.quality["preprocessing"] = report.to_dict()
    return processed, report


def epoch_quality_metrics(epoch: np.ndarray, sfreq: float, config: ProjectConfig) -> dict[str, Any]:
    """Per-epoch quality metrics used for exclusion decisions."""
    cfg = config.dataset.preprocessing
    epoch = np.asarray(epoch, dtype=np.float64)
    if epoch.size == 0:
        return {"ok": False, "reason": "empty_epoch"}

    peak_to_peak_uv = float(np.ptp(epoch) * 1e6)
    amplitude_uv = float(np.std(epoch) * 1e6)
    median = float(np.median(epoch))
    mad = float(np.median(np.abs(epoch - median)))
    robust_scale_uv = max(mad * 1.4826 * 1e6, 1e-9)
    flat_fraction = float(np.mean(np.abs(epoch) < 1e-9))
    derivative = np.abs(np.diff(epoch))
    max_step_uv = float(derivative.max() * 1e6) if derivative.size else 0.0

    reasons: list[str] = []
    if amplitude_uv < cfg.min_amplitude_uv:
        reasons.append("amplitude_below_min")
    if amplitude_uv > cfg.max_amplitude_uv:
        reasons.append("amplitude_above_max")
    if peak_to_peak_uv > cfg.clip_sigma * robust_scale_uv * 10:
        reasons.append("peak_to_peak_outlier")
    if flat_fraction > 0.25:
        reasons.append("flat_signal")

    return {
        "ok": not reasons,
        "reason": ",".join(reasons) if reasons else None,
        "amplitude_uv": amplitude_uv,
        "peak_to_peak_uv": peak_to_peak_uv,
        "robust_scale_uv": robust_scale_uv,
        "flat_fraction": flat_fraction,
        "max_step_uv": max_step_uv,
        "n_samples": int(epoch.size),
    }
