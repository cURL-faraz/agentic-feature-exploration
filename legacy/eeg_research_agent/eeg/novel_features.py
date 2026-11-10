"""Literature-derived and novel hypothesis features.

Two groups live here, and the distinction matters:

**Literature-derived** (``novelty_status="literature-derived"``)
    Ratios, variability and coupling measures whose *form* appears in the sleep-EEG
    literature (band ratios, spectral asymmetry, temporal variability of band
    power, cross-frequency coupling proxies) but whose exact parameterisation is
    this project's choice.  They are subject to the same validation requirements
    as any feature.

**Novel hypotheses** (``novelty_status="novel-hypothesis"``)
    Features proposed as *mathematical hypotheses* (spectral concentration,
    cross-band entropy imbalance, within-epoch band-power instability, multiscale
    energy entropy, sigma-envelope transient index, sub-window RMS instability).
    A small p-value for such a feature is **not** evidence that it is a meaningful
    biomarker: it must pass the synthetic-signal behaviour tests, survive held-out
    predictive evaluation and be interpreted with its confounders in mind.

Every feature below is a deterministic pure function of one epoch, is registered
with its equation, units and expected range, and is covered by tests in
``tests/test_features.py`` and ``tests/test_verification.py``.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .feature_registry import SignalContext, register_feature
from .spectral import normalised_psd

__all__ = [
    "spectral_concentration",
    "cross_band_entropy",
    "band_power_instability",
    "band_power_variability",
    "sigma_envelope_transient_index",
    "subwindow_rms_instability",
    "delta_sigma_coupling",
    "subwindow_band_powers",
]


def subwindow_band_powers(context: SignalContext, band: str) -> np.ndarray:
    """Band power computed on consecutive short sub-windows inside the epoch.

    With a 30 s epoch and 2 s sub-windows this yields 15 values, which is what the
    instability/variability features are built from.  The sub-window PSD uses the
    same Welch settings as the epoch-level PSD where the sub-window is long
    enough, so the numbers stay comparable.
    """
    return context.memo.setdefault(
        ("subwindow_power", band),
        _subwindow_band_powers(context, band),
    )


def _subwindow_band_powers(context: SignalContext, band: str) -> np.ndarray:
    cfg = context.config.features.novel
    window = int(round(cfg.subwindow_seconds * context.sfreq))
    low, high = context.bands[band]
    if window < 16 or context.n_samples < 2 * window:
        return np.asarray([])
    values: list[float] = []
    for start in range(0, context.n_samples - window + 1, window):
        segment = context.epoch[start : start + window]
        nperseg = int(min(segment.size, max(16, segment.size // 2)))
        freqs, psd = sps.welch(segment, fs=context.sfreq, nperseg=nperseg, detrend="constant", scaling="density")
        mask = (freqs >= low) & (freqs < high)
        values.append(float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0)
    return np.asarray(values, dtype=float)


# --------------------------------------------------------------------------- #
# novel hypotheses from the research brief
# --------------------------------------------------------------------------- #
def spectral_concentration(context: SignalContext, alpha: float) -> float:
    r"""Spectral concentration :math:`C_\alpha=\int p(f)^\alpha df`."""
    freqs, psd = normalised_psd(context)
    if psd.size == 0:
        return 0.0
    return float(np.trapezoid(psd**alpha, freqs))


def cross_band_entropy(context: SignalContext, epsilon: float | None = None) -> float:
    r"""Normalised entropy of the relative band-power distribution.

    :math:`H_{bands}=-\sum_b r_b\ln(r_b+\epsilon)/\ln B` with
    :math:`r_b = P_b / \sum_{b'} P_{b'}`.  Unlike spectral entropy (which is
    defined over PSD bins) this quantity operates on the *pre-defined bands*, so
    it measures how evenly the physiological rhythms share the total power.

    ``epsilon`` is a declared parameter so the registry can pass it explicitly;
    when omitted the configured default is used, which keeps the value identical
    for callers that invoke the implementation directly.
    """
    relative = np.asarray([context.band_power(band) for band in context.bands], dtype=float)
    total = relative.sum()
    if total <= 0:
        return 0.0
    relative = relative / total
    epsilon = float(context.config.features.novel.epsilon if epsilon is None else epsilon)
    entropy = -float(np.sum(relative * np.log(relative + epsilon)))
    return entropy / max(np.log(relative.size), epsilon)


def band_power_instability(context: SignalContext, band: str) -> float:
    r"""Mean absolute log-ratio of successive sub-window band powers.

    :math:`I_b=\frac{1}{T-1}\sum_{t=1}^{T-1}\left|\ln(P_{t+1}(b)+\epsilon)-\ln(P_t(b)+\epsilon)\right|`
    """
    powers = subwindow_band_powers(context, band)
    if powers.size < 2:
        return 0.0
    epsilon = context.config.features.novel.epsilon
    logs = np.log(powers + epsilon)
    return float(np.mean(np.abs(np.diff(logs))))


def band_power_variability(context: SignalContext, band: str) -> float:
    """Coefficient of variation of the sub-window band powers (CV = sigma/mu)."""
    powers = subwindow_band_powers(context, band)
    if powers.size < 2:
        return 0.0
    mean = float(np.mean(powers))
    return float(np.std(powers) / mean) if mean > 0 else 0.0


def sigma_envelope_transient_index(context: SignalContext) -> float:
    r"""Crest factor of the sigma-band envelope: :math:`\max_t |e(t)|/\overline{|e(t)|}`.

    A transient-sensitive measure: spindles produce short envelope peaks that raise this
    ratio relative to stationary sigma activity of the same average power.
    """
    sigma = context.band_limited(11.0, 16.0, order=4)
    envelope = np.abs(sps.hilbert(sigma))
    mean = float(np.mean(envelope))
    return float(np.max(envelope) / mean) if mean > 0 else 0.0


def subwindow_rms_instability(context: SignalContext) -> float:
    """Coefficient of variation of sub-window RMS values (non-stationarity proxy)."""
    cfg = context.config.features.novel
    window = int(round(cfg.subwindow_seconds * context.sfreq))
    if window < 16 or context.n_samples < 2 * window:
        return 0.0
    rms = [
        float(np.sqrt(np.mean(context.epoch[start : start + window] ** 2)))
        for start in range(0, context.n_samples - window + 1, window)
    ]
    rms_array = np.asarray(rms)
    mean = float(np.mean(rms_array))
    return float(np.std(rms_array) / mean) if mean > 0 else 0.0


def delta_sigma_coupling(context: SignalContext) -> float:
    r"""Delta-phase / sigma-amplitude coupling proxy (mean vector length).

    :math:`\mathrm{MVL}=\left|\frac{1}{N}\sum_t a_\sigma(t)e^{i\varphi_\delta(t)}\right|/\overline{a_\sigma}`

    where :math:`\varphi_\delta` is the analytic phase of the 0.5-2 Hz component and
    :math:`a_\sigma` the 11-15 Hz envelope.  This is a *phase-amplitude coupling
    proxy*, not a validated coupling estimator (no surrogate normalisation), and
    is labelled literature-derived for that reason.
    """
    slow = context.band_limited(0.5, 2.0, order=4)
    fast = context.band_limited(11.0, 15.0, order=4)
    if slow.size == 0 or fast.size == 0:
        return 0.0
    phase = np.angle(sps.hilbert(slow))
    amplitude = np.abs(sps.hilbert(fast))
    mean_amplitude = float(np.mean(amplitude))
    if mean_amplitude <= 0:
        return 0.0
    return float(np.abs(np.mean(amplitude * np.exp(1j * phase))) / mean_amplitude)


# --------------------------------------------------------------------------- #
# registry entries: novel hypotheses
# --------------------------------------------------------------------------- #
for _alpha in (2.0, 3.0):
    register_feature(
        feature_id=f"spectral_concentration_a{int(_alpha)}",
        name=f"Spectral concentration (alpha={int(_alpha)})",
        category="novel",
        family="novel",
        function=lambda context, alpha=_alpha: spectral_concentration(context, alpha),
        mathematical_definition=rf"C_{{{int(_alpha)}}}=\int p(f)^{{{int(_alpha)}}}\,\mathrm{{d}}f,\quad p=\mathrm{{PSD}}\big/\int\mathrm{{PSD}}",
        biological_motivation=(
            "High-order moments of the normalised spectrum respond quadratically/cubically to "
            "spectral concentration, so they weight the dominant rhythm far more strongly than "
            "band power does."
        ),
        parameters={"alpha": _alpha},
        required_sampling_frequency=90.0,
        expected_range=(0.0, None),
        output_units=f"Hz^{{1-{int(_alpha)}}}",
        computational_cost="low",
        novelty_status="novel-hypothesis",
        validation_status="unit-tested",
        notes="Hypothesis from the project research brief; not a published biomarker.",
    )

register_feature(
    feature_id="cross_band_entropy",
    name="Cross-band entropy imbalance",
    category="novel",
    family="novel",
    function=cross_band_entropy,
    mathematical_definition=r"H_{bands}=-\sum_b r_b\ln(r_b+\epsilon)\big/\ln B",
    biological_motivation=(
        "How evenly power is distributed across the classic bands; low when a single rhythm "
        "dominates (N3 slow waves, W alpha), high for mixed-frequency stages (REM, N1)."
    ),
    parameters={"epsilon": 1e-12},
    required_sampling_frequency=90.0,
    expected_range=(0.0, 1.0),
    output_units="normalised",
    computational_cost="low",
    novelty_status="novel-hypothesis",
    validation_status="unit-tested",
)

for _band in ("delta", "sigma", "beta"):
    register_feature(
        feature_id=f"band_power_instability_{_band}",
        name=f"{_band.capitalize()} power instability",
        category="novel",
        family="novel",
        function=lambda context, band=_band: band_power_instability(context, band),
        mathematical_definition=(
            rf"I_{{{_band}}}=\frac{{1}}{{T-1}}\sum_t\left|\ln(P_{{t+1}}+\epsilon)-\ln(P_t+\epsilon)\right|"
        ),
        biological_motivation=(
            "Within-epoch variability of band power.  Transitional stages (N1, REM) are expected to "
            "change faster than consolidated N2/N3, so variability may carry information that "
            "average power does not."
        ),
        parameters={"band": _band},
        required_sampling_frequency=40.0,
        min_signal_samples=400,
        expected_range=(0.0, None),
        output_units="ln-ratio",
        computational_cost="medium",
        novelty_status="novel-hypothesis",
        validation_status="unit-tested",
    )

for _band in ("delta", "alpha", "sigma"):
    register_feature(
        feature_id=f"band_power_variability_{_band}",
        name=f"{_band.capitalize()} power variability (CV)",
        category="novel",
        family="novel",
        function=lambda context, band=_band: band_power_variability(context, band),
        mathematical_definition=rf"\mathrm{{CV}}_{{{_band}}}=\sigma(P_t)/\bar P_t",
        biological_motivation="Temporal variability of band power; complementary to mean power.",
        literature_references=("Sleep EEG micro-architecture variability literature",),
        parameters={"band": _band},
        required_sampling_frequency=40.0,
        min_signal_samples=400,
        expected_range=(0.0, None),
        output_units="dimensionless",
        computational_cost="medium",
        novelty_status="literature-derived",
        validation_status="unit-tested",
    )

register_feature(
    feature_id="sigma_envelope_transient_index",
    name="Sigma envelope transient index",
    category="novel",
    family="novel",
    function=sigma_envelope_transient_index,
    mathematical_definition=r"\mathrm{TIX}=\max_t|e_\sigma(t)|\big/\overline{|e_\sigma(t)|}",
    biological_motivation=(
        "Differentiates spindles (transient sigma envelopes) from low-amplitude stationary sigma "
        "activity of the same mean power."
    ),
    required_sampling_frequency=40.0,
    min_signal_samples=200,
    expected_range=(1.0, None),
    output_units="dimensionless",
    computational_cost="medium",
    novelty_status="novel-hypothesis",
    validation_status="unit-tested",
)

register_feature(
    feature_id="subwindow_rms_instability",
    name="Sub-window RMS instability",
    category="novel",
    family="novel",
    function=subwindow_rms_instability,
    mathematical_definition=r"\mathrm{NI}=\sigma(x_{rms}^{t})/\overline{x_{rms}^{t}}",
    biological_motivation="Amplitude non-stationarity within the epoch, a proxy for stage transition.",
    required_sampling_frequency=40.0,
    min_signal_samples=400,
    expected_range=(0.0, None),
    output_units="dimensionless",
    computational_cost="medium",
    novelty_status="novel-hypothesis",
    validation_status="unit-tested",
)

register_feature(
    feature_id="delta_sigma_coupling",
    name="Delta-sigma phase-amplitude coupling proxy",
    category="novel",
    family="novel",
    function=delta_sigma_coupling,
    mathematical_definition=r"\mathrm{MVL}=\left|\frac1N\sum_t a_\sigma(t)e^{i\varphi_\delta(t)}\right|\big/\overline{a_\sigma}",
    biological_motivation=(
        "Slow-wave phase is known to modulate spindle occurrence; this measures the strength of "
        "that modulation inside one epoch."
    ),
    literature_references=("Mölle et al. (2002) grouping of spindles by slow oscillations",),
    required_sampling_frequency=40.0,
    min_signal_samples=400,
    expected_range=(0.0, 1.0),
    output_units="dimensionless",
    computational_cost="medium",
    novelty_status="literature-derived",
    validation_status="unit-tested",
    notes="Not surrogate-normalised: a coupling *proxy*, not a significance-tested coupling value.",
)

# --------------------------------------------------------------------------- #
# literature-derived spectral ratios
# --------------------------------------------------------------------------- #
_RATIOS = {
    "delta_beta": ("delta", "beta", r"\mathrm{DAR}=P_\delta/P_\beta", "Classic slow/fast ratio, elevated in N3."),
    "delta_theta": ("delta", "theta", r"P_\delta/P_\theta", "Separates N3 from the theta-rich transition stage N1."),
    "sigma_delta": ("sigma", "delta", r"P_\sigma/P_\delta", "Spindle-permissive versus slow-wave-dominant activity."),
    "theta_alpha": ("theta", "alpha", r"P_\theta/P_\alpha", "Distinguishes alpha-rich wake from theta-rich N1."),
    "sigma_beta": ("sigma", "beta", r"P_\sigma/P_\beta", "Contrasts spindle-band with fast activity."),
    "alpha_delta": ("alpha", "delta", r"P_\alpha/P_\delta", "Arousal versus deep sleep."),
}


def _make_ratio(numerator: str, denominator: str):
    def feature(context: SignalContext) -> float:
        upper = context.band_power(numerator)
        lower = context.band_power(denominator)
        return float(upper / lower) if lower > 0 else 0.0

    return feature


for _name, (_num, _den, _equation, _motivation) in _RATIOS.items():
    register_feature(
        feature_id=f"ratio_{_name}",
        name=f"Spectral ratio {_num}/{_den}",
        category="literature-derived",
        family="literature_derived",
        function=_make_ratio(_num, _den),
        mathematical_definition=_equation,
        biological_motivation=_motivation,
        literature_references=("Standard qEEG band-ratio practice",),
        expected_range=(0.0, None),
        output_units="dimensionless",
        computational_cost="low",
        novelty_status="literature-derived",
        validation_status="reproduced-from-literature",
    )


def _spectral_asymmetry(context: SignalContext) -> float:
    """(slow - fast) / (slow + fast) power over the classic split at 8 Hz."""
    slow = sum(context.band_power(band) for band in ("delta", "theta"))
    fast = sum(context.band_power(band) for band in ("alpha", "sigma", "beta"))
    total = slow + fast
    return float((slow - fast) / total) if total > 0 else 0.0


register_feature(
    feature_id="spectral_asymmetry_index",
    name="Spectral asymmetry index",
    category="literature-derived",
    family="literature_derived",
    function=_spectral_asymmetry,
    mathematical_definition=r"\mathrm{SAI}=\frac{(P_\delta+P_\theta)-(P_\alpha+P_\sigma+P_\beta)}{P_{0.5-30}}",
    biological_motivation="Bounded slow-versus-fast balance; less skewed than a bare ratio.",
    literature_references=("Standard qEEG spectral asymmetry",),
    expected_range=(-1.0, 1.0),
    output_units="dimensionless",
    computational_cost="low",
    novelty_status="literature-derived",
    validation_status="reproduced-from-literature",
)
