"""Time-domain amplitude, morphology and Hjorth features.

All amplitude quantities are reported in microvolts (the signal itself is stored
in volts), so that the feature table stays human readable while remaining
dimensionally consistent with the spectral family (uV^2 for power).

Hjorth parameters (Hjorth 1970) are the classical time-domain triplet:

* activity  -- variance of the signal, = power,
* mobility  -- square root of the variance ratio of the first derivative to the
  signal, i.e. an estimate of mean frequency,
* complexity -- mobility ratio between derivative and signal, an estimate of
  bandwidth (1.0 for a pure sinusoid).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sstats

from .feature_registry import SignalContext, register_feature

__all__ = ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"]

UV = 1e6  # volts -> microvolts


def _diff(signal: np.ndarray) -> np.ndarray:
    return np.diff(signal) if signal.size > 1 else np.zeros_like(signal)


def hjorth_activity(context: SignalContext) -> float:
    """Variance of the epoch (uV^2)."""
    return float(np.var(context.epoch) * UV**2)


def hjorth_mobility(context: SignalContext) -> float:
    """sqrt(var(x')/var(x)); dimensionless, ~2*pi*f for a sinusoid at f Hz."""
    variance = float(np.var(context.epoch))
    derivative_variance = float(np.var(_diff(context.epoch)))
    return float(np.sqrt(derivative_variance / variance)) if variance > 0 else 0.0


def hjorth_complexity(context: SignalContext) -> float:
    """Mobility of x' divided by mobility of x; 1.0 for a pure sinusoid."""
    mobility = hjorth_mobility(context)
    if mobility <= 0:
        return 0.0
    derivative = _diff(context.epoch)
    derivative_variance = float(np.var(derivative))
    second_variance = float(np.var(_diff(derivative)))
    derivative_mobility = float(np.sqrt(second_variance / derivative_variance)) if derivative_variance > 0 else 0.0
    return derivative_mobility / mobility


# --------------------------------------------------------------------------- #
# amplitude / morphology
# --------------------------------------------------------------------------- #
register_feature(
    feature_id="mean_amplitude",
    name="Mean amplitude",
    category="temporal",
    family="temporal",
    function=lambda context: float(np.mean(context.epoch) * UV),
    mathematical_definition=r"\bar{x}=\frac{1}{N}\sum_i x_i",
    biological_motivation="DC offset after band-pass filtering; near zero unless residual drift remains.",
    expected_range=(-100.0, 100.0),
    output_units="uV",
    computational_cost="low",
    validation_status="unit-tested",
)

register_feature(
    feature_id="std_amplitude",
    name="Standard deviation of amplitude",
    category="temporal",
    family="temporal",
    function=lambda context: float(np.std(context.epoch) * UV),
    mathematical_definition=r"\sigma_x=\sqrt{\frac{1}{N}\sum_i (x_i-\bar{x})^2}",
    biological_motivation="Broadband amplitude; largest in N3, smallest in REM/light sleep.",
    expected_range=(0.0, 500.0),
    output_units="uV",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="var_amplitude",
    name="Variance of amplitude",
    category="temporal",
    family="temporal",
    function=lambda context: float(np.var(context.epoch) * UV**2),
    mathematical_definition=r"\sigma_x^2",
    biological_motivation="Signal power in the time domain; equals total spectral power (Parseval).",
    expected_range=(0.0, None),
    output_units="uV^2",
    computational_cost="low",
    validation_status="unit-tested",
)

register_feature(
    feature_id="rms_amplitude",
    name="Root mean square amplitude",
    category="temporal",
    family="temporal",
    function=lambda context: float(np.sqrt(np.mean(context.epoch**2)) * UV),
    mathematical_definition=r"x_{rms}=\sqrt{\frac{1}{N}\sum_i x_i^2}",
    biological_motivation="Amplitude including any DC component; robust summary of epoch energy.",
    expected_range=(0.0, 500.0),
    output_units="uV",
    computational_cost="low",
    validation_status="unit-tested",
)

register_feature(
    feature_id="peak_to_peak",
    name="Peak-to-peak amplitude",
    category="temporal",
    family="temporal",
    function=lambda context: float(np.ptp(context.epoch) * UV),
    mathematical_definition=r"A_{pp}=\max_i x_i-\min_i x_i",
    biological_motivation="Sensitive to K-complexes and slow waves; also the primary artefact indicator.",
    expected_range=(0.0, None),
    output_units="uV",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="iqr_amplitude",
    name="Interquartile range of amplitude",
    category="temporal",
    family="temporal",
    function=lambda context: float(
        (np.percentile(context.epoch, 75) - np.percentile(context.epoch, 25)) * UV
    ),
    mathematical_definition=r"\mathrm{IQR}=Q_{0.75}-Q_{0.25}",
    biological_motivation="Robust spread estimator, insensitive to isolated artefacts.",
    expected_range=(0.0, None),
    output_units="uV",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="mad_amplitude",
    name="Median absolute deviation of amplitude",
    category="temporal",
    family="temporal",
    function=lambda context: float(
        np.median(np.abs(context.epoch - np.median(context.epoch))) * UV
    ),
    mathematical_definition=r"\mathrm{MAD}=\mathrm{median}_i|x_i-\mathrm{median}(x)|",
    biological_motivation="Robust scale estimate used for artefact thresholds and effect sizes.",
    expected_range=(0.0, None),
    output_units="uV",
    computational_cost="low",
    validation_status="unit-tested",
)

register_feature(
    feature_id="skewness",
    name="Amplitude skewness",
    category="temporal",
    family="temporal",
    function=lambda context: float(sstats.skew(context.epoch, bias=False)) if np.std(context.epoch) > 0 else 0.0,
    mathematical_definition=r"\gamma_1=\mathrm{E}\left[(x-\mu)^3\right]/\sigma^3",
    biological_motivation="Slow waves and K-complexes skew the amplitude distribution negatively.",
    expected_range=(-5.0, 5.0),
    output_units="dimensionless",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="kurtosis",
    name="Amplitude excess kurtosis",
    category="temporal",
    family="temporal",
    function=lambda context: float(sstats.kurtosis(context.epoch, fisher=True, bias=False))
    if np.std(context.epoch) > 0
    else 0.0,
    mathematical_definition=r"\gamma_2=\mathrm{E}\left[(x-\mu)^4\right]/\sigma^4-3",
    biological_motivation="Transient high-amplitude events (spindles, K-complexes) raise kurtosis.",
    expected_range=(-3.0, 50.0),
    output_units="dimensionless",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="zero_crossing_rate",
    name="Zero-crossing rate",
    category="temporal",
    family="temporal",
    function=lambda context: float(
        np.count_nonzero(np.diff(np.signbit(context.epoch))) / context.n_samples
    ),
    mathematical_definition=r"Z=\frac{1}{N-1}\#\{i:\mathrm{sgn}(x_i)\ne\mathrm{sgn}(x_{i+1})\}",
    biological_motivation="Dominant-frequency proxy in the time domain; low in N3, higher in wake.",
    expected_range=(0.0, 0.5),
    output_units="crossings/sample",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="zero_crossing_rate_hz",
    name="Zero-crossing rate (Hz equivalent)",
    category="temporal",
    family="temporal",
    function=lambda context: float(
        0.5 * np.count_nonzero(np.diff(np.signbit(context.epoch))) / (context.n_samples / context.sfreq)
    ),
    mathematical_definition=r"f_{zc}=\frac{1}{2T}\#\{\text{sign changes}\}",
    biological_motivation="Time-domain frequency proxy, comparable across sampling frequencies.",
    expected_range=(0.0, 50.0),
    output_units="Hz",
    computational_cost="low",
    validation_status="unit-tested",
)

register_feature(
    feature_id="line_length",
    name="Line length",
    category="temporal",
    family="temporal",
    function=lambda context: float(np.sum(np.abs(_diff(context.epoch))) * UV),
    mathematical_definition=r"L=\sum_{i=1}^{N-1}|x_{i+1}-x_i|",
    biological_motivation="Curve length combines amplitude and frequency; a classic seizure/arousal detector.",
    literature_references=("Esteller et al. (2001) line length feature",),
    expected_range=(0.0, None),
    output_units="uV",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="line_length_ratio",
    name="Normalised line length",
    category="temporal",
    family="temporal",
    function=lambda context: float(
        np.sum(np.abs(_diff(context.epoch))) / max(context.n_samples * np.std(context.epoch), 1e-18)
    ),
    mathematical_definition=r"L_{norm}=L\,/\,\left(N\sigma_x\right)",
    biological_motivation="Amplitude-normalised curve length: a pure frequency/complexity measure.",
    expected_range=(0.0, 1.0),
    output_units="dimensionless",
    computational_cost="low",
    validation_status="unit-tested",
)

register_feature(
    feature_id="mean_abs_derivative",
    name="Mean absolute first derivative",
    category="temporal",
    family="temporal",
    function=lambda context: float(np.mean(np.abs(_diff(context.epoch))) * UV),
    mathematical_definition=r"\overline{|\dot x|}=\frac{1}{N-1}\sum_i|x_{i+1}-x_i|",
    biological_motivation="Slope magnitude; steep components dominate in wake and REM, flat in N3.",
    expected_range=(0.0, None),
    output_units="uV/sample",
    computational_cost="low",
    validation_status="unit-tested",
)

register_feature(
    feature_id="hjorth_activity",
    name="Hjorth activity",
    category="temporal",
    family="temporal",
    function=hjorth_activity,
    mathematical_definition=r"A=\mathrm{Var}(x)",
    biological_motivation="Time-domain power; equivalent to the variance of the epoch.",
    literature_references=("Hjorth (1970) EEG description by moments",),
    expected_range=(0.0, None),
    output_units="uV^2",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="hjorth_mobility",
    name="Hjorth mobility",
    category="temporal",
    family="temporal",
    function=hjorth_mobility,
    mathematical_definition=r"M=\sqrt{\mathrm{Var}(\dot x)/\mathrm{Var}(x)}",
    biological_motivation="Proportional to the mean frequency of the epoch.",
    literature_references=("Hjorth (1970) EEG description by moments",),
    expected_range=(0.0, 2.0),
    output_units="dimensionless",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="hjorth_complexity",
    name="Hjorth complexity",
    category="temporal",
    family="temporal",
    function=hjorth_complexity,
    mathematical_definition=r"C=M(\dot x)/M(x)",
    biological_motivation="Bandwidth proxy; equals 1 for a pure sinusoid and rises with added components.",
    literature_references=("Hjorth (1970) EEG description by moments",),
    expected_range=(1.0, 6.0),
    output_units="dimensionless",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="amplitude_histogram_entropy",
    name="Amplitude distribution entropy",
    category="temporal",
    family="temporal",
    function=lambda context: float(
        sstats.entropy(
            np.histogram(context.epoch, bins=16, density=False)[0] + 1e-9
        )
        / np.log(16)
    ),
    mathematical_definition=r"H_a=-\frac{1}{\ln B}\sum_b \hat p_b\ln\hat p_b,\quad \hat p_b=\text{histogram}",
    biological_motivation=(
        "Shape of the amplitude distribution: Gaussian-like for broadband activity, more "
        "concentrated when a single rhythm or slow waves dominate."
    ),
    expected_range=(0.0, 1.0),
    output_units="normalised",
    computational_cost="low",
    novelty_status="literature-derived",
    validation_status="unit-tested",
)

register_feature(
    feature_id="crest_factor",
    name="Crest factor",
    category="temporal",
    family="temporal",
    function=lambda context: float(
        np.max(np.abs(context.epoch)) / max(np.sqrt(np.mean(context.epoch**2)), 1e-18)
    ),
    mathematical_definition=r"CF=\max|x|/x_{rms}",
    biological_motivation="Peakiness; elevated when isolated high-amplitude transients are present.",
    expected_range=(1.0, 20.0),
    output_units="dimensionless",
    computational_cost="low",
    validation_status="unit-tested",
)
