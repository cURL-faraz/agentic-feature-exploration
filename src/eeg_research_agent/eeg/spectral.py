"""Frequency-domain features.

Conventions
-----------
* Welch PSD with ``scaling="density"`` -> units V^2/Hz; band powers integrate the
  PSD over the band -> V^2, reported in microvolts squared (uV^2).
* Relative powers use the *total* power over the union of the configured bands
  (default 0.5-45 Hz) as denominator, so they sum to <= 1 by construction.
* Log powers are natural logarithms of (power + eps) with eps = 1e-12, which
  keeps them finite for numerically silent bands.
* Spectral shape statistics treat the normalised PSD as a density over
  frequency, making centroid/spread/skewness/kurtosis comparable across
  recordings and amplifiers.
* Band edges are configuration (``config/features.yaml``), never hidden
  constants; the non-parametric flag ``max_required_sampling_frequency`` records
  the Nyquist constraint implied by each band.

References
----------
Rechtschaffen & Kales (1968); Iber et al. (AASM, 2007); Inouye et al. (1991)
for spectral entropy; Rampil & Laster (1992) for spectral edge frequency.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .feature_registry import SignalContext, register_feature

__all__ = ["EPS", "BAND_REFERENCES"]

EPS = 1e-12

BAND_REFERENCES = {
    "delta": "slow-wave band 0.5-4 Hz (R&K/AASM)",
    "theta": "theta activity 4-8 Hz, characteristic of N1",
    "alpha": "8-12 Hz alpha rhythm, prominent in relaxed wakefulness",
    "sigma": "12-16 Hz sigma band, contains sleep spindles",
    "beta": "16-30 Hz fast activity, elevated in wake and REM",
    "gamma": "30-45 Hz high-frequency activity (artefact-sensitive)",
}

#: Highest frequency each band needs; a Nyquist below this invalidates the feature.
BAND_NYQUIST = {"delta": 4.0, "theta": 8.0, "alpha": 12.0, "sigma": 16.0, "beta": 30.0, "gamma": 45.0}


# --------------------------------------------------------------------------- #
# shared helpers (memoised in the SignalContext)
# --------------------------------------------------------------------------- #
def normalised_psd(context: SignalContext) -> tuple[np.ndarray, np.ndarray]:
    """PSD normalised to unit area over the configured band union."""
    key = "normalised_psd"
    if key not in context.memo:
        freqs, psd = context.psd()
        low = min(band[0] for band in context.bands.values())
        high = max(band[1] for band in context.bands.values())
        mask = (freqs >= low) & (freqs < high)
        band_freqs, band_psd = freqs[mask], psd[mask]
        total = float(np.trapezoid(band_psd, band_freqs)) if band_psd.size else 0.0
        if total <= 0:
            context.memo[key] = (band_freqs, np.zeros_like(band_psd))
        else:
            context.memo[key] = (band_freqs, band_psd / total)
    return context.memo[key]


def _centroid(context: SignalContext) -> float:
    freqs, psd = normalised_psd(context)
    return float(np.trapezoid(freqs * psd, freqs)) if freqs.size else 0.0


def _spread(context: SignalContext) -> float:
    freqs, psd = normalised_psd(context)
    if not freqs.size:
        return 0.0
    return float(np.sqrt(max(np.trapezoid((freqs - _centroid(context)) ** 2 * psd, freqs), 0.0)))


def _spectral_moment(context: SignalContext, order: int) -> float:
    freqs, psd = normalised_psd(context)
    if not freqs.size:
        return 0.0
    centroid = _centroid(context)
    sigma = max(_spread(context), np.sqrt(EPS))
    return float(np.trapezoid(((freqs - centroid) / sigma) ** order * psd, freqs))


def _rolloff(context: SignalContext, fraction: float) -> float:
    freqs, psd = normalised_psd(context)
    if not freqs.size:
        return 0.0
    cumulative = np.cumsum(psd)
    if cumulative[-1] <= 0:
        return 0.0
    cumulative = cumulative / cumulative[-1]
    index = int(np.searchsorted(cumulative, fraction))
    return float(freqs[min(index, freqs.size - 1)])


# --------------------------------------------------------------------------- #
# band powers
# --------------------------------------------------------------------------- #
def _make_absolute(band: str):
    def feature(context: SignalContext) -> float:
        """Absolute band power in uV^2."""
        return context.band_power(band) * 1e12

    return feature


def _make_relative(band: str):
    def feature(context: SignalContext) -> float:
        """Band power divided by total power over the configured bands."""
        total = context.total_power()
        return float(context.band_power(band) / total) if total > 0 else 0.0

    return feature


def _make_log_absolute(band: str):
    def feature(context: SignalContext) -> float:
        """Natural log of the absolute band power."""
        return float(np.log(context.band_power(band) + EPS))

    return feature


for _band, _reference in BAND_REFERENCES.items():
    _nyquist = BAND_NYQUIST[_band]
    register_feature(
        feature_id=f"abs_power_{_band}",
        name=f"Absolute {_band} power",
        category="spectral",
        family="spectral",
        function=_make_absolute(_band),
        mathematical_definition=(
            rf"P_{{{_band}}}=\int_{{f_1}}^{{f_2}}\mathrm{{PSD}}(f)\,\mathrm{{d}}f\quad[\mu V^2]"
        ),
        biological_motivation=f"Absolute power in the {_band} band ({_reference}).",
        literature_references=(
            "Rechtschaffen & Kales (1968) manual of standardized terminology",
            "Iber et al. (2007) AASM scoring manual",
        ),
        required_sampling_frequency=2 * _nyquist,
        expected_range=(0.0, None),
        output_units="uV^2",
        computational_cost="low",
        validation_status="reproduced-from-literature",
        notes="Welch PSD integrated over the configured band boundaries.",
    )
    register_feature(
        feature_id=f"rel_power_{_band}",
        name=f"Relative {_band} power",
        category="spectral",
        family="spectral",
        function=_make_relative(_band),
        mathematical_definition=rf"\rho_{{{_band}}}=P_{{{_band}}}\big/\textstyle\sum_b P_b",
        biological_motivation=(
            f"{_band.capitalize()} power normalised by total power; removes constant gain and "
            "electrode-impedance scaling, at the price of inducing negative correlations "
            "between bands."
        ),
        literature_references=("Iber et al. (2007) AASM scoring manual",),
        required_sampling_frequency=2 * _nyquist,
        expected_range=(0.0, 1.0),
        output_units="fraction",
        computational_cost="low",
        validation_status="reproduced-from-literature",
        notes="Denominator is the union of the configured bands; relative powers are compositional.",
    )
    register_feature(
        feature_id=f"log_power_{_band}",
        name=f"Log absolute {_band} power",
        category="spectral",
        family="spectral",
        function=_make_log_absolute(_band),
        mathematical_definition=rf"\ln\!\left(P_{{{_band}}}+\epsilon\right),\ \epsilon=10^{{-12}}",
        biological_motivation=(
            "Epoch-level band power is approximately log-normal, so the log transform stabilises "
            "variance for parametric tests and additive models."
        ),
        literature_references=("Standard qEEG practice (log-transformed band power)",),
        required_sampling_frequency=2 * _nyquist,
        expected_range=(None, None),
        output_units="ln(uV^2)",
        computational_cost="low",
        validation_status="unit-tested",
    )


def total_power(context: SignalContext) -> float:
    """Total power across the configured bands, in uV^2."""
    return context.total_power() * 1e12


def spectral_coverage_fraction(context: SignalContext) -> float:
    """Fraction of full-band Welch power captured by the configured bands."""
    freqs, psd = context.psd()
    full = float(np.trapezoid(psd, freqs)) if freqs.size else 0.0
    return float(context.total_power() / full) if full > 0 else 0.0


def mean_frequency(context: SignalContext) -> float:
    """Amplitude-weighted mean frequency over 0.5-45 Hz."""
    freqs, psd = context.psd()
    mask = (freqs >= 0.5) & (freqs < 45.0)
    denominator = float(np.sum(psd[mask]))
    return float(np.sum(freqs[mask] * psd[mask]) / denominator) if denominator > 0 else 0.0


def spectral_entropy(context: SignalContext) -> float:
    """Normalised Shannon entropy of the discretised PSD.

    The density is re-normalised so that the PSD bins form a probability vector
    (``sum_k p_k = 1``).  Without that step the entropy would depend on the
    frequency resolution rather than on the spectrum's shape.
    """
    _, psd = normalised_psd(context)
    if psd.size == 0:
        return 0.0
    total = float(np.sum(psd))
    if total <= 0:
        return 0.0
    discrete = psd / total
    entropy = float(-np.sum(discrete * np.log(discrete + EPS)))
    return entropy / max(np.log(discrete.size), EPS)


def spectral_flatness(context: SignalContext) -> float:
    """Wiener entropy (geometric mean / arithmetic mean) of the normalised PSD."""
    _, psd = normalised_psd(context)
    if psd.size == 0:
        return 0.0
    arithmetic = float(np.mean(psd)) + EPS
    geometric = float(np.exp(np.mean(np.log(psd + EPS))))
    return geometric / arithmetic


def peak_frequency(context: SignalContext) -> float:
    """Frequency of the largest PSD bin within the configured bands."""
    freqs, psd = normalised_psd(context)
    if not freqs.size:
        return 0.0
    return float(freqs[int(np.argmax(psd))])


def _welch_segment(segment: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = int(min(segment.size, max(8, round(segment.size / 2))))
    return sps.welch(segment, fs=sfreq, window="hann", nperseg=nperseg, detrend="constant", scaling="density")


def spectral_flux(context: SignalContext) -> float:
    """Mean RMS difference between normalised sub-window spectra (non-stationarity proxy)."""
    window_seconds = float(context.config.features.novel.subwindow_seconds)
    n_per_window = int(round(window_seconds * context.sfreq))
    if context.n_samples < 2 * n_per_window or n_per_window < 16:
        return 0.0
    spectra: list[np.ndarray] = []
    for start in range(0, context.n_samples - n_per_window + 1, n_per_window):
        freqs, psd = _welch_segment(context.epoch[start : start + n_per_window], context.sfreq)
        total = float(np.trapezoid(psd, freqs))
        if total <= 0:
            continue
        spectra.append(psd / total)
    if len(spectra) < 2:
        return 0.0
    grid = min(len(s) for s in spectra)
    stacked = np.vstack([s[:grid] for s in spectra])
    diffs = np.diff(stacked, axis=0)
    return float(np.mean(np.sqrt(np.mean(diffs**2, axis=1))))


register_feature(
    feature_id="total_power",
    name="Total power 0.5-45 Hz",
    category="spectral",
    family="spectral",
    function=total_power,
    mathematical_definition=r"P_{tot}=\int_{0.5}^{45}\mathrm{PSD}(f)\,\mathrm{d}f",
    biological_motivation="Overall EEG amplitude; decreases from N3 towards REM and wakefulness.",
    literature_references=("Iber et al. (2007) AASM scoring manual",),
    required_sampling_frequency=90.0,
    expected_range=(0.0, None),
    output_units="uV^2",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="log_total_power",
    name="Log total power",
    category="spectral",
    family="spectral",
    function=lambda context: float(np.log(total_power(context) * 1e-12 + EPS)),
    mathematical_definition=r"\ln(P_{tot}+\epsilon)",
    biological_motivation="Variance-stabilised overall amplitude; the primary N3 versus N1/REM contrast.",
    required_sampling_frequency=90.0,
    expected_range=(None, None),
    output_units="ln(uV^2)",
    computational_cost="low",
    validation_status="unit-tested",
)

register_feature(
    feature_id="spectral_coverage_fraction",
    name="Fraction of Welch power inside the configured bands",
    category="spectral",
    family="spectral",
    function=spectral_coverage_fraction,
    mathematical_definition=r"\kappa=\int_{0.5}^{45}\mathrm{PSD}\,\mathrm{d}f\big/\int_0^{f_N}\mathrm{PSD}\,\mathrm{d}f",
    biological_motivation="Data-integrity feature: near 1 when the configured bands cover the analysis band.",
    expected_range=(0.0, 1.0),
    output_units="fraction",
    computational_cost="low",
    validation_status="unit-tested",
)

register_feature(
    feature_id="spectral_centroid",
    name="Spectral centroid",
    category="spectral",
    family="spectral",
    function=_centroid,
    mathematical_definition=r"f_c=\int f\,p(f)\,\mathrm{d}f,\quad p=\mathrm{PSD}/\int\mathrm{PSD}",
    biological_motivation="Centre of spectral mass; moves down in N3 and up in wake.",
    literature_references=("Standard qEEG spectral descriptors",),
    expected_range=(0.5, 45.0),
    output_units="Hz",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="spectral_spread",
    name="Spectral bandwidth",
    category="spectral",
    family="spectral",
    function=_spread,
    mathematical_definition=r"\sigma_f=\sqrt{\int (f-f_c)^2 p(f)\,\mathrm{d}f}",
    biological_motivation="Width of the dominant spectral peak; narrow when one rhythm dominates.",
    expected_range=(0.0, 30.0),
    output_units="Hz",
    computational_cost="low",
    validation_status="unit-tested",
)

register_feature(
    feature_id="spectral_skewness",
    name="Spectral skewness",
    category="spectral",
    family="spectral",
    function=lambda context: _spectral_moment(context, 3),
    mathematical_definition=r"\gamma_1=\int\big((f-f_c)/\sigma_f\big)^3 p(f)\,\mathrm{d}f",
    biological_motivation="Asymmetry of the spectral density; sensitive to a low-frequency shoulder in N3.",
    expected_range=(None, None),
    output_units="dimensionless",
    computational_cost="low",
    validation_status="unit-tested",
)

register_feature(
    feature_id="spectral_kurtosis",
    name="Spectral kurtosis",
    category="spectral",
    family="spectral",
    function=lambda context: _spectral_moment(context, 4),
    mathematical_definition=r"\gamma_2=\int\big((f-f_c)/\sigma_f\big)^4 p(f)\,\mathrm{d}f",
    biological_motivation="Peakedness of the spectrum; elevated when one rhythm dominates.",
    expected_range=(None, None),
    output_units="dimensionless",
    computational_cost="low",
    validation_status="unit-tested",
)

register_feature(
    feature_id="spectral_entropy",
    name="Spectral entropy",
    category="spectral",
    family="spectral",
    function=spectral_entropy,
    mathematical_definition=r"H_{spec}=-\frac{1}{\ln N}\sum_k p(f_k)\ln p(f_k)",
    biological_motivation=(
        "Low for rhythm-dominated spectra (slow waves, spindles), high for broadband "
        "desynchronised activity."
    ),
    literature_references=("Inouye et al. (1991) quantification of EEG regularity",),
    expected_range=(0.0, 1.0),
    output_units="normalised",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="spectral_flatness",
    name="Spectral flatness (Wiener entropy)",
    category="spectral",
    family="spectral",
    function=spectral_flatness,
    mathematical_definition=r"F=\exp\!\left(\tfrac1N\sum\ln p(f_k)\right)\big/\left(\tfrac1N\sum p(f_k)\right)",
    biological_motivation="Near 0 for tonal spectra, near 1 for noise-like spectra.",
    literature_references=("Dubnov (2004) generalised spectral flatness",),
    expected_range=(0.0, 1.0),
    output_units="dimensionless",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="spectral_rolloff_85",
    name="Spectral roll-off 85%",
    category="spectral",
    family="spectral",
    function=lambda context: _rolloff(context, 0.85),
    mathematical_definition=r"f_{85}=\min\{f:\int_{0.5}^{f}p(u)\,\mathrm{d}u\ge0.85\}",
    biological_motivation="Robust analogue of the spectral edge; separates slow from fast activity.",
    expected_range=(0.5, 45.0),
    output_units="Hz",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="spectral_edge_95",
    name="Spectral edge frequency 95%",
    category="spectral",
    family="spectral",
    function=lambda context: _rolloff(context, 0.95),
    mathematical_definition=r"f_{95}=\min\{f:\int_{0.5}^{f}p(u)\,\mathrm{d}u\ge0.95\}",
    biological_motivation="Classical EEG arousal/depth index; high in wake and REM.",
    literature_references=("Rampil & Laster (1992) spectral edge frequency",),
    expected_range=(0.5, 45.0),
    output_units="Hz",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="peak_frequency",
    name="Peak frequency",
    category="spectral",
    family="spectral",
    function=peak_frequency,
    mathematical_definition=r"f_{peak}=\arg\max_f p(f)",
    biological_motivation="Dominant rhythm: alpha in wake, slow waves in N3, spindles in N2.",
    expected_range=(0.5, 45.0),
    output_units="Hz",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="mean_frequency",
    name="Mean frequency (amplitude weighted)",
    category="spectral",
    family="spectral",
    function=mean_frequency,
    mathematical_definition=r"f_{mean}=\sum_k f_k\,\mathrm{PSD}(f_k)\big/\sum_k\mathrm{PSD}(f_k)",
    biological_motivation="Amplitude-weighted centre frequency, complementary to the density centroid.",
    expected_range=(0.5, 45.0),
    output_units="Hz",
    computational_cost="low",
    validation_status="unit-tested",
)

register_feature(
    feature_id="spectral_flux",
    name="Within-epoch spectral flux",
    category="spectral",
    family="spectral",
    function=spectral_flux,
    mathematical_definition=(
        r"\Phi=\frac{1}{T-1}\sum_t\sqrt{\frac1N\sum_k\big(\hat p_{t+1}(f_k)-\hat p_t(f_k)\big)^2}"
    ),
    biological_motivation=(
        "Quantifies non-stationarity inside one 30 s epoch; transitional stages are expected to be "
        "less stationary than consolidated N2/N3."
    ),
    required_sampling_frequency=20.0,
    min_signal_samples=400,
    expected_range=(0.0, None),
    output_units="dimensionless",
    computational_cost="medium",
    novelty_status="literature-derived",
    validation_status="unit-tested",
)
