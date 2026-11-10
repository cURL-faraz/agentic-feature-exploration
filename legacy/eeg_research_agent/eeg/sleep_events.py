"""Event-based features: sleep spindles, K-complexes and slow waves.

Detection is rule-based and fully documented; each detector is a *pipeline* of
independently testable steps rather than a black box:

spindles
    1. band-pass 11-16 Hz (zero-phase Butterworth, order 4),
    2. analytic envelope via the Hilbert transform,
    3. envelope smoothing with a moving average (0.1 s),
    4. adaptive threshold = ``factor`` x moving RMS over a 2 s window,
    5. contiguous supra-threshold segments shorter than 0.5 s or longer than 3 s
       are rejected,
    6. the intra-event zero-crossing rate must lie within 10-18 Hz (rejects
       broadband transients that survive the band-pass),
    7. peak envelope amplitude must exceed the 75th percentile of the epoch's own
       envelope (epoch-relative amplitude criterion),
    8. events closer than the 0.5 s refractory period are merged.

K-complexes
    1. low-pass 4 Hz,
    2. negative peaks below ``-2.5 x`` robust scale (1.4826 x MAD) *and* below an
       absolute floor (default 30 uV; R&K requires >= 75 uV peak-to-peak),
    3. the surrounding zero-crossing interval must last 0.5-2.0 s,
    4. an optional positive rebound within 1 s after the negative peak is
       required by default,
    5. 1 s refractory period.

slow waves (0.5-2 Hz)
    1. band-pass 0.5-2 Hz, half-waves delimited by zero crossings,
    2. negative half-wave amplitude below ``-1.5 x`` robust scale *and* below an
       absolute floor (default 20 uV, motivated by the -40 uV criterion of
       Riedner et al. 2007),
    3. half-wave duration 0.3-1.5 s,
    4. slope measured on the steepest 0.2 s window of the half-wave.

Important: sigma-band power and spindle density are **not** the same quantity.
Sigma power measures ongoing band energy, spindle density counts discrete
transient events; both are reported so that their complementary information can
be quantified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .feature_registry import SignalContext, register_feature

__all__ = [
    "EventCollection",
    "detect_spindles",
    "detect_k_complexes",
    "detect_slow_waves",
]

UV = 1e6


@dataclass
class EventCollection:
    """Detected events with per-event measurements."""

    onsets: np.ndarray
    durations: np.ndarray
    amplitudes: np.ndarray          # volts
    frequencies: np.ndarray         # Hz
    slopes: np.ndarray              # volts/s (slow waves only)
    epoch_seconds: float

    def __post_init__(self) -> None:
        self.onsets = np.asarray(self.onsets, dtype=float)
        self.durations = np.asarray(self.durations, dtype=float)
        self.amplitudes = np.asarray(self.amplitudes, dtype=float)
        self.frequencies = np.asarray(self.frequencies, dtype=float)
        self.slopes = np.asarray(self.slopes, dtype=float)

    @property
    def count(self) -> int:
        return int(self.onsets.size)

    @property
    def density_per_min(self) -> float:
        return float(self.count / (self.epoch_seconds / 60.0)) if self.epoch_seconds > 0 else 0.0

    @property
    def mean_duration(self) -> float:
        return float(np.mean(self.durations)) if self.count else 0.0

    @property
    def mean_amplitude(self) -> float:
        return float(np.mean(self.amplitudes)) if self.count else 0.0

    @property
    def mean_frequency(self) -> float:
        return float(np.mean(self.frequencies)) if self.count else 0.0

    @property
    def mean_slope(self) -> float:
        return float(np.mean(np.abs(self.slopes))) if self.count else 0.0

    @property
    def occupancy(self) -> float:
        return float(np.sum(self.durations) / self.epoch_seconds) if self.epoch_seconds > 0 else 0.0

    @property
    def max_amplitude(self) -> float:
        return float(np.max(self.amplitudes)) if self.count else 0.0


def _robust_scale(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) or float(np.std(x)) or 1e-12


def _moving_rms(x: np.ndarray, window: int) -> np.ndarray:
    window = max(2, int(window))
    kernel = np.ones(window) / window
    power = np.convolve(x**2, kernel, mode="same")
    return np.sqrt(np.maximum(power, 0.0))


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as ``(start, stop)`` index pairs."""
    if not np.any(mask):
        return []
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def detect_spindles(context: SignalContext) -> EventCollection:
    """Detect spindle-like bursts; see module docstring step list."""
    cfg = context.config.features.events.spindle
    sfreq = context.sfreq
    banded = context.band_limited(cfg.band_hz[0], cfg.band_hz[1], order=4)
    envelope = np.abs(sps.hilbert(banded))
    smoothing = max(1, int(round(cfg.envelope_smoothing_seconds * sfreq)))
    if smoothing > 1:
        envelope = np.convolve(envelope, np.ones(smoothing) / smoothing, mode="same")

    reference = _moving_rms(banded, max(4, int(round(cfg.threshold_window_seconds * sfreq))))
    mask = envelope > cfg.threshold_factor * reference
    # Two amplitude criteria must both be met: an epoch-relative one (the peak must
    # stand out from this epoch's own sigma envelope) and an absolute one (in
    # microvolts).  The relative criterion alone cannot reject broadband noise,
    # because the top quartile of any envelope passes it by construction.
    amplitude_floor = max(
        float(np.percentile(envelope, cfg.amplitude_percentile)),
        float(cfg.min_amplitude_uv) / UV,
    )

    onsets: list[float] = []
    durations: list[float] = []
    amplitudes: list[float] = []
    frequencies: list[float] = []
    last_stop = -np.inf
    for start, stop in _segments(mask):
        duration = (stop - start) / sfreq
        if duration < cfg.min_duration_s or duration > cfg.max_duration_s:
            continue
        onset = start / sfreq
        if onset - last_stop < cfg.refractory_s:
            continue
        segment = banded[start:stop]
        if segment.size < 4:
            continue
        crossings = np.count_nonzero(np.diff(np.signbit(segment)))
        frequency = crossings / (2.0 * duration)
        if not (cfg.min_zero_cross_rate_hz <= frequency <= cfg.max_zero_cross_rate_hz):
            continue
        peak = float(np.max(envelope[start:stop]))
        if peak < amplitude_floor:
            continue
        onsets.append(onset)
        durations.append(duration)
        amplitudes.append(peak)
        frequencies.append(frequency)
        last_stop = stop / sfreq

    return EventCollection(
        onsets=np.asarray(onsets),
        durations=np.asarray(durations),
        amplitudes=np.asarray(amplitudes),
        frequencies=np.asarray(frequencies),
        slopes=np.zeros(len(onsets)),
        epoch_seconds=context.n_samples / sfreq,
    )


def detect_k_complexes(context: SignalContext) -> EventCollection:
    """Detect K-complex-like biphasic slow deflections."""
    cfg = context.config.features.events.k_complex
    sfreq = context.sfreq
    slow = context.lowpass(cfg.lowpass_hz, order=4)
    scale = _robust_scale(slow)
    threshold = -cfg.neg_threshold_factor * scale

    amplitude_floor = cfg.min_amplitude_uv / UV
    # required negative-peak magnitude in volts (positive number): the stricter of
    # the epoch-relative criterion and the absolute amplitude floor
    required = max(-threshold, amplitude_floor)
    minima, _ = sps.find_peaks(-slow, height=required)
    crossings = np.flatnonzero(np.diff(np.signbit(slow)))
    onsets: list[float] = []
    durations: list[float] = []
    amplitudes: list[float] = []
    last_onset = -np.inf
    for index in minima:
        onset_index = crossings[crossings < index]
        stop_index = crossings[crossings > index]
        if onset_index.size == 0 or stop_index.size == 0:
            continue
        start, stop = int(onset_index[-1]), int(stop_index[0])
        duration = (stop - start) / sfreq
        if duration < cfg.min_duration_s or duration > cfg.max_duration_s:
            continue
        onset = start / sfreq
        if onset - last_onset < cfg.refractory_s:
            continue
        if cfg.rebound_required:
            rebound_window = slow[index : min(slow.size, index + int(1.0 * sfreq))]
            if rebound_window.size == 0 or float(np.max(rebound_window)) < 0.3 * abs(float(slow[index])):
                continue
        onsets.append(onset)
        durations.append(duration)
        amplitudes.append(abs(float(slow[index])))
        last_onset = onset

    return EventCollection(
        onsets=np.asarray(onsets),
        durations=np.asarray(durations),
        amplitudes=np.asarray(amplitudes),
        frequencies=np.asarray([1.0 / d if d > 0 else 0.0 for d in durations]),
        slopes=np.zeros(len(onsets)),
        epoch_seconds=context.n_samples / sfreq,
    )


def detect_slow_waves(context: SignalContext) -> EventCollection:
    """Detect 0.5-2 Hz negative half-waves with a steep descending slope."""
    cfg = context.config.features.events.slow_wave
    sfreq = context.sfreq
    banded = context.band_limited(cfg.band_hz[0], cfg.band_hz[1], order=4)
    scale = _robust_scale(banded)
    threshold = -cfg.neg_threshold_factor * scale

    zero_crossings = np.flatnonzero(np.diff(np.signbit(banded)))
    onsets: list[float] = []
    durations: list[float] = []
    amplitudes: list[float] = []
    slopes: list[float] = []
    last_stop = -np.inf
    slope_window = max(2, int(round(cfg.slope_window_s * sfreq)))

    for start, stop in zip(zero_crossings[:-1], zero_crossings[1:]):
        segment = banded[start:stop]
        if segment.size < 4:
            continue
        minimum = float(np.min(segment))
        if minimum >= 0 or abs(minimum) < max(abs(threshold), cfg.min_amplitude_uv / UV):
            continue
        duration = (stop - start) / sfreq
        if duration < cfg.min_duration_s or duration > cfg.max_duration_s:
            continue
        onset = start / sfreq
        if onset - last_stop < cfg.refractory_s:
            continue
        derivative = np.gradient(segment, 1.0 / sfreq)
        if derivative.size >= slope_window:
            kernel = np.ones(slope_window) / slope_window
            smoothed = np.convolve(derivative, kernel, mode="valid")
            slope = float(np.min(smoothed)) if smoothed.size else float(np.min(derivative))
        else:
            slope = float(np.min(derivative))
        onsets.append(onset)
        durations.append(duration)
        amplitudes.append(abs(minimum))
        slopes.append(slope)
        last_stop = stop / sfreq

    return EventCollection(
        onsets=np.asarray(onsets),
        durations=np.asarray(durations),
        amplitudes=np.asarray(amplitudes),
        frequencies=np.asarray([1.0 / d if d > 0 else 0.0 for d in durations]),
        slopes=np.asarray(slopes),
        epoch_seconds=context.n_samples / sfreq,
    )


# --------------------------------------------------------------------------- #
# registry entries
# --------------------------------------------------------------------------- #
def _memo_events(context: SignalContext, key: str, detector) -> EventCollection:
    cache_key = f"events:{key}"
    if cache_key not in context.memo:
        context.memo[cache_key] = detector(context)
    return context.memo[cache_key]


def _spindle(context: SignalContext) -> EventCollection:
    return _memo_events(context, "spindle", detect_spindles)


def _kcomplex(context: SignalContext) -> EventCollection:
    return _memo_events(context, "kcomplex", detect_k_complexes)


def _slow_wave(context: SignalContext) -> EventCollection:
    return _memo_events(context, "slow_wave", detect_slow_waves)


_SPINDLE_META = dict(
    required_sampling_frequency=40.0,
    min_signal_samples=200,
    computational_cost="medium",
    literature_references=(
        "De Gennaro & Ferrara (2003) sleep spindles: an overview",
        "Wamsley et al. (2012) spindle detection methodology",
    ),
    validation_status="synthetic-validated",
)

for _suffix, _fn, _definition, _motivation, _unit in [
    (
        "count",
        lambda ctx: float(_spindle(ctx).count),
        r"N_{sp}",
        "Number of discrete spindle events; the density that distinguishes N2 from N1/REM.",
        "count",
    ),
    (
        "density",
        lambda ctx: _spindle(ctx).density_per_min,
        r"\rho_{sp}=N_{sp}/T_{epoch}\ \text{[min}^{-1}\text{]}",
        "Spindle density is the standard N2 marker (typically 1-8 per minute).",
        "1/min",
    ),
    (
        "duration_mean",
        lambda ctx: _spindle(ctx).mean_duration,
        r"\overline{d}_{sp}",
        "Mean spindle duration; ~0.5-2 s in physiological recordings.",
        "s",
    ),
    (
        "amplitude_mean",
        lambda ctx: _spindle(ctx).mean_amplitude * UV,
        r"\overline{A}_{sp}",
        "Mean peak envelope amplitude of detected spindles.",
        "uV",
    ),
    (
        "frequency_mean",
        lambda ctx: _spindle(ctx).mean_frequency,
        r"\overline{f}_{sp}",
        "Mean intra-spindle oscillation frequency (should stay inside 11-16 Hz).",
        "Hz",
    ),
    (
        "occupancy",
        lambda ctx: _spindle(ctx).occupancy,
        r"o_{sp}=\sum d / T",
        "Fraction of the epoch occupied by spindles; complements density when events merge.",
        "fraction",
    ),
]:
    register_feature(
        feature_id=f"spindle_{_suffix}",
        name=f"Spindle {_suffix.replace('_', ' ')}",
        category="events",
        family="events",
        function=_fn,
        mathematical_definition=_definition,
        biological_motivation=_motivation,
        output_units=_unit,
        novelty_status="established",
        **_SPINDLE_META,
    )

_KC_META = dict(
    required_sampling_frequency=20.0,
    min_signal_samples=200,
    computational_cost="medium",
    literature_references=("Cash et al. (2009) K-complex detection", "Colrain et al. (2011) K-complex review"),
    validation_status="synthetic-validated",
)

for _suffix, _fn, _definition, _motivation, _unit in [
    ("count", lambda ctx: float(_kcomplex(ctx).count), r"N_{KC}", "K-complex count; a defining N2 feature.", "count"),
    (
        "density",
        lambda ctx: _kcomplex(ctx).density_per_min,
        r"\rho_{KC}=N_{KC}/T",
        "K-complex density; expected to peak in N2 and be rare in REM.",
        "1/min",
    ),
    (
        "duration_mean",
        lambda ctx: _kcomplex(ctx).mean_duration,
        r"\overline{d}_{KC}",
        "Mean K-complex duration; typically 0.5-2 s including the rebound.",
        "s",
    ),
    (
        "amplitude_mean",
        lambda ctx: _kcomplex(ctx).mean_amplitude * UV,
        r"\overline{A}_{KC}",
        "Mean negative peak amplitude of detected K-complexes.",
        "uV",
    ),
    (
        "occupancy",
        lambda ctx: _kcomplex(ctx).occupancy,
        r"o_{KC}=\sum d / T",
        "Fraction of the epoch occupied by K-complexes.",
        "fraction",
    ),
]:
    register_feature(
        feature_id=f"kcomplex_{_suffix}",
        name=f"K-complex {_suffix.replace('_', ' ')}",
        category="events",
        family="events",
        function=_fn,
        mathematical_definition=_definition,
        biological_motivation=_motivation,
        output_units=_unit,
        novelty_status="established",
        **_KC_META,
    )

_SW_META = dict(
    required_sampling_frequency=20.0,
    min_signal_samples=200,
    computational_cost="medium",
    literature_references=(
        "Massimini et al. (2004) sleep slow waves",
        "Riedner et al. (2007) slow-wave detection and analysis",
    ),
    validation_status="synthetic-validated",
)

for _suffix, _fn, _definition, _motivation, _unit in [
    ("count", lambda ctx: float(_slow_wave(ctx).count), r"N_{SW}", "Slow-wave count; the N3 hallmark.", "count"),
    (
        "density",
        lambda ctx: _slow_wave(ctx).density_per_min,
        r"\rho_{SW}=N_{SW}/T",
        "Slow-wave density; high in N3, near-zero in REM.",
        "1/min",
    ),
    (
        "duration_mean",
        lambda ctx: _slow_wave(ctx).mean_duration,
        r"\overline{d}_{SW}",
        "Mean slow-wave half-wave duration.",
        "s",
    ),
    (
        "amplitude_mean",
        lambda ctx: _slow_wave(ctx).mean_amplitude * UV,
        r"\overline{A}_{SW}",
        "Mean slow-wave peak amplitude (large negative deflection).",
        "uV",
    ),
    (
        "slope_mean",
        lambda ctx: _slow_wave(ctx).mean_slope * UV,
        r"\overline{s}_{SW}",
        "Mean absolute descending slope of slow waves, a marker of sleep intensity.",
        "uV/s",
    ),
    (
        "occupancy",
        lambda ctx: _slow_wave(ctx).occupancy,
        r"o_{SW}=\sum d / T",
        "Fraction of the epoch occupied by detected slow waves.",
        "fraction",
    ),
]:
    register_feature(
        feature_id=f"slow_wave_{_suffix}",
        name=f"Slow wave {_suffix.replace('_', ' ')}",
        category="events",
        family="events",
        function=_fn,
        mathematical_definition=_definition,
        biological_motivation=_motivation,
        output_units=_unit,
        novelty_status="established",
        **_SW_META,
    )
