"""Synthetic signals with *known* properties, and the checks built on them.

Every feature in the registry is ultimately validated against signals whose
expected behaviour can be derived on paper:

* a pure sinusoid has all of its power at one frequency, so spectral entropy and
  spectral flatness must be near their minima and the peak frequency must match;
* white noise has a flat spectrum, so spectral flatness must be near 1;
* scaling the amplitude by ``c`` multiplies absolute power by ``c^2`` and leaves
  *relative* power (and every normalised ratio) unchanged;
* a signal with a delta-band component added must have more delta power than the
  same signal without it;
* an injected spindle must be detected by the spindle detector, and white noise
  must produce (almost) no detections.

None of these is a claim about human sleep.  They are unit tests for the
mathematics of the feature implementations.

Units
-----
Amplitudes are *given* in microvolts because that is how the literature quotes
EEG, but every generator **returns volts**, which is the project's internal unit
(:class:`~eeg_research_agent.data.recording.Recording` also stores volts, matching
MNE).  Feature functions therefore see the same units during verification as
during a real extraction run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "sine",
    "multi_tone",
    "white_noise",
    "pink_noise",
    "amplitude_modulated",
    "frequency_modulated",
    "spindle_burst",
    "slow_wave",
    "kcomplex_like",
    "slow_wave_transient",
    "eeg_like_background",
    "with_artifact",
    "SignalCheck",
    "run_synthetic_signal_checks",
]

#: Volts per microvolt.
UV = 1e-6


def _time(n_samples: int, sfreq: float) -> np.ndarray:
    return np.arange(n_samples, dtype=float) / float(sfreq)


def sine(freq: float, n_samples: int, sfreq: float, amplitude: float = 20.0, phase: float = 0.0) -> np.ndarray:
    """Pure sinusoid; ``amplitude`` is in microvolts and the result is in volts."""
    return UV * amplitude * np.sin(2.0 * np.pi * freq * _time(n_samples, sfreq) + phase)


def multi_tone(
    components: Sequence[tuple[float, float]], n_samples: int, sfreq: float
) -> np.ndarray:
    """Sum of sinusoids, each ``(frequency_hz, amplitude_uv)``."""
    signal = np.zeros(n_samples, dtype=float)
    for freq, amplitude in components:
        signal += sine(freq, n_samples, sfreq, amplitude)
    return signal


def white_noise(n_samples: int, sfreq: float, amplitude: float = 10.0, seed: int = 20240517) -> np.ndarray:
    """Gaussian white noise; SD ``amplitude`` in microvolts, returned in volts."""
    rng = np.random.default_rng(seed)
    return UV * rng.normal(0.0, amplitude, n_samples)


def pink_noise(
    n_samples: int, sfreq: float, amplitude: float = 10.0, exponent: float = 1.0, seed: int = 20240517
) -> np.ndarray:
    """1/f^exponent noise normalised to the requested RMS in microvolts (returned in volts)."""
    rng = np.random.default_rng(seed)
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    spectrum = rng.normal(0.0, 1.0, frequencies.size) + 1j * rng.normal(0.0, 1.0, frequencies.size)
    scaling = np.ones_like(frequencies)
    scaling[1:] = frequencies[1:] ** (-exponent / 2.0)
    shaped = spectrum * scaling
    signal = np.fft.irfft(shaped, n=n_samples)
    current = float(np.std(signal))
    scaled = signal * (amplitude / current) if current > 0 else signal
    return UV * scaled


def amplitude_modulated(
    carrier_hz: float, mod_hz: float, n_samples: int, sfreq: float, depth: float = 0.5, amplitude: float = 20.0
) -> np.ndarray:
    """Carrier with a modulated envelope; ``amplitude`` in microvolts (returned in volts)."""
    time = _time(n_samples, sfreq)
    envelope = 1.0 + depth * np.sin(2.0 * np.pi * mod_hz * time)
    return UV * amplitude * envelope * np.sin(2.0 * np.pi * carrier_hz * time)


def frequency_modulated(
    carrier_hz: float, deviation_hz: float, mod_hz: float, n_samples: int, sfreq: float, amplitude: float = 20.0
) -> np.ndarray:
    """Frequency-modulated carrier; ``amplitude`` in microvolts (returned in volts)."""
    time = _time(n_samples, sfreq)
    phase = 2.0 * np.pi * carrier_hz * time + (deviation_hz / mod_hz) * np.sin(2.0 * np.pi * mod_hz * time)
    return UV * amplitude * np.sin(phase)


def spindle_burst(
    n_samples: int,
    sfreq: float,
    frequency: float = 13.0,
    duration_s: float = 1.0,
    amplitude: float = 20.0,
    start_s: float = 0.5,
) -> np.ndarray:
    """A Gaussian-windowed sigma-band burst (synthetic sleep spindle), returned in volts."""
    signal = np.zeros(n_samples, dtype=float)
    start = int(start_s * sfreq)
    length = int(duration_s * sfreq)
    if start >= n_samples or length <= 0:
        return signal
    stop = min(n_samples, start + length)
    local = np.arange(stop - start, dtype=float) / sfreq
    window = np.exp(-0.5 * ((local - local.mean()) / (0.25 * local.max() if local.max() else 1.0)) ** 2)
    signal[start:stop] = UV * amplitude * window * np.sin(2.0 * np.pi * frequency * local)
    return signal


def slow_wave(
    n_samples: int, sfreq: float, frequency: float = 1.0, amplitude: float = 60.0, start_s: float = 0.5
) -> np.ndarray:
    """A single large slow oscillation (synthetic slow wave / K-complex-like), returned in volts."""
    signal = np.zeros(n_samples, dtype=float)
    start = int(start_s * sfreq)
    if start >= n_samples:
        return signal
    local = np.arange(n_samples - start, dtype=float) / sfreq
    signal[start:] = UV * amplitude * np.sin(2.0 * np.pi * frequency * local)
    return signal


def with_artifact(
    signal: np.ndarray, sfreq: float, start_s: float, duration_s: float, amplitude: float = 500.0
) -> np.ndarray:
    """Add a large step-like movement artefact; ``amplitude`` in microvolts."""
    out = np.array(signal, dtype=float, copy=True)
    start = int(start_s * sfreq)
    stop = min(out.size, start + int(duration_s * sfreq))
    if start < out.size:
        out[start:stop] += UV * amplitude
    return out


def eeg_like_background(
    n_samples: int, sfreq: float, amplitude: float = 12.0, alpha_uv: float = 4.0, seed: int = 20240517
) -> np.ndarray:
    """A plausible resting-EEG background: 1/f activity plus a weak alpha rhythm.

    detectors are calibrated against a *background*, not against silence or
    against a signal that consists entirely of the event they are looking for.
    """
    return pink_noise(n_samples, sfreq, amplitude=amplitude, seed=seed) + sine(
        10.0, n_samples, sfreq, amplitude=alpha_uv
    )


def kcomplex_like(
    n_samples: int, sfreq: float, amplitude: float = 150.0, start_s: float = 10.0
) -> np.ndarray:
    """A localised biphasic K-complex: sharp negative deflection, positive rebound.

    The shape follows the AASM description (a well-delineated negative sharp wave
    immediately followed by a positive component) rather than being a pure sine,
    because the detector's duration and rebound criteria are defined on that
    morphology.  Returned in volts; ``amplitude`` in microvolts.
    """
    signal = np.zeros(n_samples, dtype=float)
    start = int(start_s * sfreq)
    if start >= n_samples:
        return signal
    time = np.arange(n_samples - start, dtype=float) / sfreq
    negative = -np.exp(-0.5 * ((time - 0.35) / 0.13) ** 2)
    rebound = 0.55 * np.exp(-0.5 * ((time - 0.85) / 0.20) ** 2)
    window = np.zeros_like(time)
    window[time <= 2.0] = 1.0
    signal[start:] = UV * amplitude * (negative + rebound) * window
    return signal


def slow_wave_transient(
    n_samples: int, sfreq: float, amplitude: float = 90.0, start_s: float = 10.0, duration_s: float = 0.7
) -> np.ndarray:
    """A single negative half-wave with a steep descending slope (a slow wave)."""
    signal = np.zeros(n_samples, dtype=float)
    start = int(start_s * sfreq)
    length = int(duration_s * sfreq)
    stop = min(n_samples, start + length)
    if start >= n_samples or stop <= start:
        return signal
    local = np.linspace(0.0, np.pi, stop - start, endpoint=False)
    signal[start:stop] = UV * amplitude * np.sin(local)  # positive, then negate below
    signal[start:stop] *= -1.0
    return signal


@dataclass
class SignalCheck:
    """One synthetic-signal assertion."""

    name: str
    passed: bool
    detail: str
    observed: Any = None
    expected: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "detail": self.detail,
            "observed": _plain(self.observed),
            "expected": _plain(self.expected),
        }


def _plain(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run_synthetic_signal_checks(registry: Any, config: Any, sfreq: float = 100.0) -> list[SignalCheck]:
    """Property-based checks of the registered features on known signals."""
    from ..eeg.feature_registry import SignalContext

    checks: list[SignalCheck] = []
    n_samples = int(30 * sfreq)

    def context(signal: np.ndarray) -> SignalContext:
        return SignalContext(epoch=signal, sfreq=sfreq, config=config)

    def value(signal: np.ndarray, feature_id: str) -> float:
        return float(registry.compute(context(signal), [feature_id])[0].get(feature_id, float("nan")))

    tone = sine(10.0, n_samples, sfreq)
    noise = white_noise(n_samples, sfreq)
    pink = pink_noise(n_samples, sfreq)

    # -- 1. a pure tone has a concentrated spectrum --------------------------- #
    checks.append(
        _check(
            "pure_tone_peak_frequency",
            value(tone, "peak_frequency"),
            lambda observed: abs(observed - 10.0) <= 1.0,
            expected="10 +/- 1 Hz",
            fmt="{:.3f} Hz",
        )
    )
    flatness_tone = value(tone, "spectral_flatness")
    flatness_noise = value(noise, "spectral_flatness")
    checks.append(
        SignalCheck(
            "pure_tone_less_flat_than_noise",
            bool(flatness_tone < flatness_noise),
            "spectral flatness of a tone must be below that of white noise",
            observed={"tone": flatness_tone, "noise": flatness_noise},
            expected="tone < noise",
        )
    )
    checks.append(
        _check(
            "white_noise_spectral_flatness_high",
            flatness_noise,
            lambda observed: observed > 0.3,
            expected="> 0.3 (flat spectrum)",
            fmt="{:.4f}",
        )
    )
    # -- 2. spectral entropy ordering ---------------------------------------- #
    checks.append(
        SignalCheck(
            "spectral_entropy_tone_below_noise",
            bool(value(tone, "spectral_entropy") < value(noise, "spectral_entropy")),
            "spectral entropy must be lower for a pure tone than for white noise",
            observed={"tone": value(tone, "spectral_entropy"), "noise": value(noise, "spectral_entropy")},
            expected="tone < noise",
        )
    )
    # -- 3. amplitude scaling invariants ------------------------------------- #
    for scale in (2.0, 3.0):
        base_power = value(tone, "abs_power_alpha")
        scaled_power = value(tone * scale, "abs_power_alpha")
        ratio = scaled_power / base_power if base_power else float("nan")
        checks.append(
            SignalCheck(
                f"absolute_power_scales_quadratically_x{scale:g}",
                bool(np.isfinite(ratio) and abs(ratio - scale**2) / scale**2 < 1e-6),
                f"absolute band power must scale with amplitude^2 (x{scale:g} amplitude -> x{scale**2:g} power)",
                observed=ratio,
                expected=scale**2,
            )
        )
    relative_base = value(tone, "rel_power_alpha")
    relative_scaled = value(tone * 3.0, "rel_power_alpha")
    checks.append(
        SignalCheck(
            "relative_power_invariant_to_amplitude",
            bool(abs(relative_base - relative_scaled) < 1e-9),
            "relative band power must be invariant to a uniform amplitude change",
            observed={"base": relative_base, "scaled": relative_scaled},
            expected="equal",
        )
    )
    checks.append(
        SignalCheck(
            "relative_band_powers_sum_to_one",
            bool(abs(sum(value(tone, f"rel_power_{b}") for b in config.bands) - 1.0) < 0.05),
            "relative band powers should approximately sum to 1 over the covered range",
            observed=sum(value(tone, f"rel_power_{b}") for b in config.bands),
            expected=1.0,
        )
    )
    # -- 4. delta power responds to a delta component ------------------------ #
    delta_added = tone + sine(2.0, n_samples, sfreq, amplitude=40.0)
    checks.append(
        SignalCheck(
            "delta_power_increases_with_delta_component",
            bool(value(delta_added, "abs_power_delta") > 10.0 * max(value(tone, "abs_power_delta"), 1e-12)),
            "adding a 2 Hz component must increase delta-band power",
            observed={
                "with_delta": value(delta_added, "abs_power_delta"),
                "without": value(tone, "abs_power_delta"),
            },
            expected="with_delta >> without",
        )
    )
    # -- 5. sigma power responds to a sigma component ------------------------ #
    sigma_added = tone + sine(13.0, n_samples, sfreq, amplitude=15.0)
    checks.append(
        SignalCheck(
            "sigma_power_increases_with_sigma_component",
            bool(value(sigma_added, "abs_power_sigma") > value(tone, "abs_power_sigma")),
            "adding a 13 Hz component must increase sigma-band power",
            observed={
                "with_sigma": value(sigma_added, "abs_power_sigma"),
                "without": value(tone, "abs_power_sigma"),
            },
        )
    )
    # -- 6. time-domain sanity ----------------------------------------------- #
    checks.append(
        _check(
            "zero_crossing_rate_matches_tone_frequency",
            value(tone, "zero_crossing_rate_hz"),
            lambda observed: abs(observed - 10.0) <= 0.5,
            expected="~10 Hz",
            fmt="{:.3f} Hz",
        )
    )
    checks.append(
        _check(
            "hjorth_complexity_near_one_for_sinusoid",
            value(tone, "hjorth_complexity"),
            lambda observed: abs(observed - 1.0) <= 0.05,
            expected="~1.0",
            fmt="{:.4f}",
        )
    )
    checks.append(
        _check(
            "rms_matches_sine_amplitude_over_sqrt2",
            value(tone, "rms_amplitude"),
            lambda observed: abs(observed - 20.0 / np.sqrt(2.0)) < 0.5,
            expected=20.0 / np.sqrt(2.0),
            fmt="{:.4f} uV",
        )
    )
    checks.append(
        SignalCheck(
            "line_length_increases_with_frequency",
            bool(value(sine(20.0, n_samples, sfreq), "line_length") > value(sine(5.0, n_samples, sfreq), "line_length")),
            "line length (total variation) must grow with frequency for equal amplitude",
            observed={
                "5 Hz": value(sine(5.0, n_samples, sfreq), "line_length"),
                "20 Hz": value(sine(20.0, n_samples, sfreq), "line_length"),
            },
        )
    )
    # -- 7. pink noise is more low-frequency weighted than white noise ------- #
    checks.append(
        SignalCheck(
            "pink_noise_has_more_delta_than_white_noise",
            bool(value(pink, "rel_power_delta") > value(noise, "rel_power_delta")),
            "1/f noise must place relatively more power in the delta band than white noise",
            observed={
                "pink": value(pink, "rel_power_delta"),
                "white": value(noise, "rel_power_delta"),
            },
        )
    )
    # -- 8. wavelet energy behaves as expected ------------------------------- #
    low = value(sine(1.0, n_samples, sfreq), "wavelet_low_high_ratio")
    high = value(sine(40.0, n_samples, sfreq), "wavelet_low_high_ratio")
    checks.append(
        SignalCheck(
            "wavelet_low_high_ratio_orders_by_frequency",
            bool(np.isfinite(low) and np.isfinite(high) and low > high),
            "a 1 Hz tone must place more energy in the approximation/low detail than a 40 Hz tone",
            observed={"1Hz": low, "40Hz": high},
        )
    )
    # -- 9. entropy features respond to complexity --------------------------- #
    checks.append(
        SignalCheck(
            "sample_entropy_higher_for_noise_than_tone",
            bool(value(noise, "sample_entropy") > value(tone, "sample_entropy")),
            "sample entropy of white noise must exceed that of a pure tone",
            observed={
                "noise": value(noise, "sample_entropy"),
                "tone": value(tone, "sample_entropy"),
            },
        )
    )
    checks.append(
        SignalCheck(
            "permutation_entropy_higher_for_noise_than_tone",
            bool(value(noise, "permutation_entropy") > value(tone, "permutation_entropy")),
            "permutation entropy of white noise must exceed that of a pure tone",
            observed={
                "noise": value(noise, "permutation_entropy"),
                "tone": value(tone, "permutation_entropy"),
            },
        )
    )
    checks.append(
        SignalCheck(
            "higuchi_fd_higher_for_noise_than_tone",
            bool(
                value(noise, "higuchi_fractal_dimension")
                > value(sine(7.0, n_samples, sfreq), "higuchi_fractal_dimension")
            ),
            "the fractal dimension of noise must exceed that of a smooth sinusoid "
            "(a 7 Hz tone is used because an exactly periodic 10 Hz tone at 100 Hz is a "
            "documented degeneracy of the Higuchi estimator, not a property of the signal)",
            observed={
                "noise": value(noise, "higuchi_fractal_dimension"),
                "tone_7Hz": value(sine(7.0, n_samples, sfreq), "higuchi_fractal_dimension"),
            },
        )
    )
    # -- 10. detrended fluctuation analysis on 1/f noise --------------------- #
    alpha1 = value(pink, "dfa_alpha1")
    checks.append(
        SignalCheck(
            "dfa_alpha1_in_plausible_range_for_pink_noise",
            bool(np.isfinite(alpha1) and 0.2 < alpha1 < 2.2),
            "DFA alpha1 for 1/f noise should be near 1 (reported for information)",
            observed=alpha1,
            expected="~1",
        )
    )
    # -- 11. event detectors -------------------------------------------------- #
    # Event detectors are defined on *transients within a background*, so the
    # controlled comparison is background vs background+event.  Comparing against
    # silence or against a signal made only of the target waveform would test the
    # wrong property: a stationary 1 Hz sine is not a slow-wave transient, and its
    # own robust scale defeats any relative threshold.
    background = eeg_like_background(n_samples, sfreq, seed=5)
    event_cases = {
        "spindle": (
            background + spindle_burst(n_samples, sfreq, frequency=13.0, duration_s=1.0, amplitude=25.0, start_s=10.0),
            background,
            "spindle_count",
        ),
        "k_complex": (
            background + kcomplex_like(n_samples, sfreq, amplitude=120.0, start_s=10.0),
            background,
            "kcomplex_count",
        ),
        "slow_wave": (
            background + slow_wave_transient(n_samples, sfreq, amplitude=90.0, start_s=10.0),
            background,
            "slow_wave_count",
        ),
    }
    for name, (with_event, without_event, feature) in event_cases.items():
        detected = value(with_event, feature)
        spurious = value(without_event, feature)
        checks.append(
            SignalCheck(
                f"{name}_detected_when_injected",
                bool(detected >= 1),
                f"an injected {name.replace('_', ' ')} must be detected on an EEG-like background",
                observed=detected,
                expected=">= 1",
            )
        )
        checks.append(
            SignalCheck(
                f"{name}_specific_to_the_injected_event",
                bool(detected > spurious),
                f"the {name.replace('_', ' ')} detector must fire more often with the event than without it",
                observed={"with_event": detected, "background_only": spurious},
                expected="with_event > background_only",
            )
        )
    checks.append(
        SignalCheck(
            "spindle_detector_low_false_positive_rate_on_noise",
            bool(
                value(white_noise(n_samples, sfreq, amplitude=8.0, seed=11), "spindle_density") <= 1.0
            ),
            "pure Gaussian noise (not an EEG background) must not produce more than 1 spindle/min",
            observed=value(white_noise(n_samples, sfreq, amplitude=8.0, seed=11), "spindle_density"),
            expected="<= 1 per minute",
        )
    )
    # -- 12. amplitude-modulated signals have a transient sigma envelope ----- #
    am = amplitude_modulated(13.0, 0.4, n_samples, sfreq, depth=0.9)
    checks.append(
        SignalCheck(
            "transient_index_higher_for_modulated_than_stationary",
            bool(
                value(am, "sigma_envelope_transient_index")
                > value(sine(13.0, n_samples, sfreq), "sigma_envelope_transient_index")
            ),
            "a modulated sigma envelope must look more transient than a constant-amplitude tone",
            observed={
                "modulated": value(am, "sigma_envelope_transient_index"),
                "stationary": value(sine(13.0, n_samples, sfreq), "sigma_envelope_transient_index"),
            },
        )
    )
    # -- 13. artefacts -------------------------------------------------------- #
    artefact = with_artifact(tone, sfreq, start_s=10.0, duration_s=0.5, amplitude=500.0)
    checks.append(
        SignalCheck(
            "peak_to_peak_detects_artefact",
            bool(value(artefact, "peak_to_peak") > 10.0 * value(tone, "peak_to_peak")),
            "a 500 uV step must dominate the peak-to-peak amplitude",
            observed={"with": value(artefact, "peak_to_peak"), "without": value(tone, "peak_to_peak")},
        )
    )
    checks.append(
        SignalCheck(
            "ar_histogram_entropy_finite_under_artefact",
            bool(np.isfinite(value(artefact, "amplitude_histogram_entropy"))),
            "the amplitude histogram entropy must remain finite for an artefact-contaminated epoch",
            observed=value(artefact, "amplitude_histogram_entropy"),
        )
    )
    # -- 14. numerically stable paths ---------------------------------------- #
    zeros = np.zeros(n_samples)
    silent_ok = True
    detail = ""
    for feature_id in registry.ids():
        try:
            computed = value(zeros, feature_id)
            if not np.isfinite(computed):
                silent_ok = False
                detail = f"{feature_id} returned {computed} for a zero signal"
                break
        except Exception as exc:  # noqa: BLE001
            silent_ok = False
            detail = f"{feature_id} raised {type(exc).__name__}: {exc}"
            break
    checks.append(
        SignalCheck(
            "all_features_finite_on_zero_signal",
            silent_ok,
            "every registered feature must return a finite value for an all-zero epoch (no silent NaN propagation)",
            observed=detail or "all finite",
        )
    )
    # -- 15. constant signal -------------------------------------------------- #
    constant = np.full(n_samples, 5.0)
    constant_ok = True
    detail = ""
    for feature_id in registry.ids():
        try:
            computed = value(constant, feature_id)
            if not np.isfinite(computed):
                constant_ok = False
                detail = f"{feature_id} returned {computed} for a constant signal"
                break
        except Exception as exc:  # noqa: BLE001
            constant_ok = False
            detail = f"{feature_id} raised {type(exc).__name__}: {exc}"
            break
    checks.append(
        SignalCheck(
            "all_features_finite_on_constant_signal",
            constant_ok,
            "every registered feature must handle a constant epoch without dividing by zero",
            observed=detail or "all finite",
        )
    )
    # -- 16. very short epoch ------------------------------------------------- #
    short = sine(10.0, 40, sfreq)
    short_failures: list[str] = []
    for feature_id in registry.ids():
        definition = registry.get(feature_id)
        if context(short).n_samples < definition.min_signal_samples:
            continue
        try:
            computed = value(short, feature_id)
            if not np.isfinite(computed):
                short_failures.append(feature_id)
        except Exception:  # noqa: BLE001
            short_failures.append(feature_id)
    checks.append(
        SignalCheck(
            "features_declaring_support_handle_short_epoch",
            not short_failures,
            "a feature that declares a minimum sample count must actually work at that length",
            observed=short_failures[:10],
            expected=[],
        )
    )
    return checks


def _check(name: str, observed: float, predicate: Callable[[float], bool], expected: Any, fmt: str = "{:.4f}") -> SignalCheck:
    try:
        passed = bool(predicate(observed))
    except Exception:  # pragma: no cover - defensive
        passed = False
    return SignalCheck(
        name=name,
        passed=passed,
        detail=f"observed {fmt.format(observed) if np.isfinite(observed) else observed}",
        observed=observed,
        expected=expected,
    )
