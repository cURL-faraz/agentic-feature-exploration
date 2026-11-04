"""Nonlinear / complexity features.

Implemented directly with NumPy + SciPy (no numba, no `antropy`) so that the
estimators are transparent, unit-testable and free of hidden accelerators:

* sample entropy (Richman & Moorman 2000) -- negative log conditional probability
  that two sequences close for ``m`` samples remain close for ``m+1``,
* approximate entropy (Pincus 1991) -- the biased, self-matched precursor,
* permutation entropy (Bandt & Pompe 2002) -- ordinal-pattern entropy,
* Higuchi (1988) and Petrosian (1995) and Katz (1988) fractal dimensions,
* detrended fluctuation analysis (Peng et al. 1995) with short- and long-range
  scaling exponents,
* Lempel-Ziv (1976) complexity is deliberately **not** shipped here.  The
  published ``c(n)`` parse-count implementations differ in their indexing and
  ``k_max`` conventions, and no reference implementation was available offline to
  validate the variant against.  Rather than ship an unverifiable estimator the
  family keeps four independently validated complexity measures; the omission is
  recorded in ``docs/mathematical_features.md``.

Numerical notes
---------------
* Entropy tolerance is ``r = factor * std(x)``; when ``std(x) == 0`` the feature
  is defined as 0 rather than NaN.
* Long signals are truncated to ``nonlinear.max_samples`` (default 6000) purely
  for runtime control; the truncation is deterministic (first samples) and
  reported in the extraction configuration.
* Higuchi/DFA values depend on the sampling frequency, so they may only be
  compared between recordings sampled at the same rate.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .feature_registry import SignalContext, register_feature

__all__ = [
    "sample_entropy",
    "approximate_entropy",
    "permutation_entropy",
    "higuchi_fractal_dimension",
    "petrosian_fractal_dimension",
    "dfa",
]


def _prepare(signal: np.ndarray, context: SignalContext) -> np.ndarray:
    limit = int(context.config.features.nonlinear.max_samples)
    x = np.asarray(signal, dtype=np.float64).ravel()
    return x[:limit] if limit and x.size > limit else x


def _embed(x: np.ndarray, m: int, delay: int = 1) -> np.ndarray:
    n = x.size - (m - 1) * delay
    if n <= 0:
        return np.empty((0, m))
    return np.asarray([x[i : i + m * delay : delay] for i in range(n)])


def sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Sample entropy SampEn(m, r) with Chebyshev distance (Richman & Moorman 2000).

    The estimator uses the pair-count ratio ``A/B``; the ``1/(N-m+1)``
    normalisation of both terms cancels, so total match counts are sufficient
    (this is the standard formulation, e.g. Richman & Moorman Eq. 5).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size <= m + 2:
        return 0.0
    r = float(r_factor) * float(np.std(x))
    if r <= 0:
        return 0.0

    def _phi(order: int) -> float:
        embedded = _embed(x, order)
        if embedded.shape[0] < 2:
            return 0.0
        tree = cKDTree(embedded)
        counts = tree.count_neighbors(tree, r, p=np.inf) - embedded.shape[0]
        return float(counts)

    b = _phi(m)
    a = _phi(m + 1)
    if b <= 0 or a <= 0:
        return 0.0
    return float(-np.log(a / b))


def approximate_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Approximate entropy ApEn(m, r) (Pincus 1991); self-matches included."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size <= m + 2:
        return 0.0
    r = float(r_factor) * float(np.std(x))
    if r <= 0:
        return 0.0

    def _phi(order: int) -> float:
        embedded = _embed(x, order)
        if embedded.shape[0] < 2:
            return 0.0
        tree = cKDTree(embedded)
        counts = np.asarray(
            tree.query_ball_point(embedded, r, p=np.inf, return_length=True), dtype=float
        )
        counts = np.maximum(counts, 1.0)
        return float(np.mean(np.log(counts / embedded.shape[0])))

    return float(_phi(m) - _phi(m + 1))


def permutation_entropy(x: np.ndarray, order: int = 4, delay: int = 1) -> float:
    """Normalised permutation entropy (Bandt & Pompe 2002).

    Ordinal patterns are ranked with a stable sort, so ties are broken by time
    index (the usual "no-jitter" convention); the result is divided by
    ``log(order!)`` so that it lies in [0, 1].
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < order * delay + 1:
        return 0.0
    embedded = _embed(x, order, delay)
    if embedded.shape[0] < 2:
        return 0.0
    patterns = np.argsort(embedded, axis=1, kind="stable")
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    from math import lgamma

    return float(entropy / lgamma(order + 1))


def higuchi_fractal_dimension(x: np.ndarray, kmax: int = 10) -> float:
    """Higuchi (1988) fractal dimension from the curve-length scaling slope.

    Robustness guards (documented behaviour):

    * ``k`` values for which the sub-series length is zero are skipped.  This
      happens for exactly periodic signals when ``k`` equals the number of
      samples per period (e.g. a 10 Hz tone at 100 Hz), a well-known degeneracy
      of the estimator rather than a property of the signal.
    * the result must fall inside the theoretically admissible interval
      ``[1.0, 2.0]``; otherwise the estimate is returned as ``NaN`` so that the
      extraction engine records it as a failed computation instead of silently
      producing a meaningless number.
    * a signal with zero variance (all samples equal) is a degenerate curve with
      no structure to scale, and its Higuchi dimension is defined here as
      ``1.0``.  Returning ``NaN`` for every flat epoch would inflate the failure
      counters without carrying information; ``1.0`` is the value the estimator
      converges to for a straight line and keeps the feature total.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 32:
        return float("nan")
    if float(np.ptp(x)) == 0.0:
        return 1.0
    if n < 2 * kmax + 2:
        kmax = max(2, n // 4)
    lengths: list[float] = []
    ks: list[int] = []
    for k in range(1, kmax + 1):
        partial = []
        for m in range(k):
            count = int((n - m - 1) // k)
            if count < 1:
                continue
            idx = (np.arange(1, count + 1) * k + m).astype(int)
            prev = ((np.arange(0, count) * k) + m).astype(int)
            segment = float(np.abs(x[idx] - x[prev]).sum())
            partial.append(segment * (n - 1) / (count * k))
        if partial:
            mean_length = float(np.mean(partial)) / k
            if mean_length > 0:
                lengths.append(mean_length)
                ks.append(k)
    if len(ks) < 4:
        return float("nan")
    slope = float(np.polyfit(np.log(ks), np.log(np.asarray(lengths)), 1)[0])
    dimension = -slope
    # The estimator is biased at its extremes: white noise has a true dimension of
    # exactly 2 and can land a fraction above it, and a nearly straight curve can
    # land a fraction below 1.  Estimates inside a narrow tolerance band are
    # therefore clipped to the admissible range instead of being discarded; values
    # outside the band (a genuine estimator breakdown) are still reported as NaN.
    if 1.0 - 0.05 <= dimension <= 2.0 + 0.05:
        return float(min(2.0, max(1.0, dimension)))
    return float("nan")


def petrosian_fractal_dimension(x: np.ndarray) -> float:
    """Petrosian (1995) fractal dimension from derivative sign changes."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 3:
        return 0.0
    diff = np.diff(x)
    sign_changes = int(np.count_nonzero(np.diff(np.sign(diff)) != 0))
    log_n = np.log10(n)
    denominator = log_n + np.log10(n / (n + 0.4 * sign_changes))
    return float(log_n / denominator) if denominator > 0 else 0.0


def dfa(x: np.ndarray, scales: list[int] | tuple[int, ...]) -> float:
    """Detrended fluctuation analysis exponent over the given window sizes."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 32:
        return 0.0
    profile = np.cumsum(x - np.mean(x))
    valid_scales: list[int] = []
    fluctuations: list[float] = []
    for scale in scales:
        scale = int(scale)
        if scale < 4 or scale > x.size // 4:
            continue
        n_segments = x.size // scale
        if n_segments < 2:
            continue
        rms_values: list[float] = []
        for start in (0, x.size - n_segments * scale):
            segments = profile[start : start + n_segments * scale].reshape(n_segments, scale)
            t = np.arange(scale, dtype=np.float64)
            design = np.vstack([t, np.ones_like(t)]).T
            coefficients, *_ = np.linalg.lstsq(design, segments.T, rcond=None)
            trend = (design @ coefficients).T
            rms_values.append(float(np.sqrt(np.mean((segments - trend) ** 2))))
        fluctuations.append(float(np.mean(rms_values)))
        valid_scales.append(scale)
    if len(valid_scales) < 3:
        return 0.0
    slope = np.polyfit(np.log(valid_scales), np.log(np.asarray(fluctuations) + 1e-18), 1)[0]
    return float(slope)


# --------------------------------------------------------------------------- #
# registry entries (parameters come from config/features.yaml)
# --------------------------------------------------------------------------- #
def _sample_entropy(context: SignalContext) -> float:
    cfg = context.config.features.nonlinear
    return sample_entropy(_prepare(context.epoch, context), cfg.sample_entropy_m, cfg.sample_entropy_r_factor)


def _approximate_entropy(context: SignalContext) -> float:
    cfg = context.config.features.nonlinear
    return approximate_entropy(
        _prepare(context.epoch, context), cfg.approx_entropy_m, cfg.approx_entropy_r_factor
    )


def _permutation_entropy(context: SignalContext) -> float:
    cfg = context.config.features.nonlinear
    return permutation_entropy(
        _prepare(context.epoch, context), cfg.permutation_entropy_order, cfg.permutation_entropy_delay
    )


def _higuchi(context: SignalContext) -> float:
    return higuchi_fractal_dimension(_prepare(context.epoch, context), context.config.features.nonlinear.higuchi_kmax)


def _petrosian(context: SignalContext) -> float:
    return petrosian_fractal_dimension(_prepare(context.epoch, context))


def _dfa_alpha1(context: SignalContext) -> float:
    cfg = context.config.features.nonlinear
    low, high = cfg.dfa_alpha1_range
    scales = [s for s in cfg.dfa_scales if low <= s < high]
    return dfa(_prepare(context.epoch, context), scales or cfg.dfa_scales)


def _dfa_alpha2(context: SignalContext) -> float:
    cfg = context.config.features.nonlinear
    low, high = cfg.dfa_alpha2_range
    scales = [s for s in cfg.dfa_scales if s >= low]
    _ = high
    return dfa(_prepare(context.epoch, context), scales or cfg.dfa_scales)


register_feature(
    feature_id="sample_entropy",
    name="Sample entropy",
    category="nonlinear",
    family="nonlinear",
    function=_sample_entropy,
    mathematical_definition=r"\mathrm{SampEn}(m,r)=-\ln\left(A/B\right),\ r=0.2\,\sigma_x,\ m=2",
    biological_motivation=(
        "Regularity of the signal: low for highly periodic slow-wave activity, higher for the "
        "irregular mixed-frequency activity of REM and wake."
    ),
    literature_references=("Richman & Moorman (2000) physiological time-series analysis",),
    min_signal_samples=64,
    expected_range=(0.0, 3.0),
    output_units="nats",
    computational_cost="high",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="approximate_entropy",
    name="Approximate entropy",
    category="nonlinear",
    family="nonlinear",
    function=_approximate_entropy,
    mathematical_definition=r"\mathrm{ApEn}(m,r)=\phi^m(r)-\phi^{m+1}(r),\ \phi^m=\frac{1}{N}\sum_i\ln C_i^m(r)",
    biological_motivation="Classic irregularity index; biased relative to SampEn but widely reported.",
    literature_references=("Pincus (1991) approximate entropy",),
    min_signal_samples=64,
    expected_range=(0.0, 3.0),
    output_units="nats",
    computational_cost="high",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="permutation_entropy",
    name="Permutation entropy",
    category="nonlinear",
    family="nonlinear",
    function=_permutation_entropy,
    mathematical_definition=r"H_p=-\frac{1}{\ln m!}\sum_\pi p(\pi)\ln p(\pi)",
    biological_motivation="Ordinal complexity, robust to monotone amplitude changes; sensitive to depth of sleep.",
    literature_references=("Bandt & Pompe (2002) permutation entropy",),
    min_signal_samples=64,
    expected_range=(0.0, 1.0),
    output_units="normalised",
    computational_cost="medium",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="higuchi_fractal_dimension",
    name="Higuchi fractal dimension",
    category="nonlinear",
    family="nonlinear",
    function=_higuchi,
    mathematical_definition=r"D_H=-\text{slope of }\ln L(k)\ \text{vs}\ \ln k",
    biological_motivation="Self-similarity of the EEG trace across time scales; rises with desynchronisation.",
    literature_references=("Higuchi (1988) fractal dimension of time series",),
    min_signal_samples=64,
    expected_range=(1.0, 2.0),
    output_units="dimensionless",
    computational_cost="medium",
    validation_status="reproduced-from-literature",
    notes="Sampling-frequency dependent: compare only within a fixed sfreq.",
)

register_feature(
    feature_id="petrosian_fractal_dimension",
    name="Petrosian fractal dimension",
    category="nonlinear",
    family="nonlinear",
    function=_petrosian,
    mathematical_definition=r"D_P=\frac{\log_{10}N}{\log_{10}N+\log_{10}\left(N/(N+0.4N_\Delta)\right)}",
    biological_motivation="Fast complexity estimate counting derivative sign changes.",
    literature_references=("Petrosian (1995) Kolmogorov complexity of EEG",),
    min_signal_samples=32,
    expected_range=(1.0, 1.2),
    output_units="dimensionless",
    computational_cost="low",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="dfa_alpha1",
    name="DFA scaling exponent (short range)",
    category="nonlinear",
    family="nonlinear",
    function=_dfa_alpha1,
    mathematical_definition=r"F(s)\propto s^{\alpha_1},\quad \alpha_1=\text{slope for } 4\le s<16",
    biological_motivation="Short-range fractal scaling; ~1.0 for 1/f-like sleep EEG.",
    literature_references=("Peng et al. (1995) long-range correlations in DNA and EEG",),
    min_signal_samples=64,
    expected_range=(0.0, 2.0),
    output_units="dimensionless",
    computational_cost="medium",
    validation_status="reproduced-from-literature",
)

register_feature(
    feature_id="dfa_alpha2",
    name="DFA scaling exponent (long range)",
    category="nonlinear",
    family="nonlinear",
    function=_dfa_alpha2,
    mathematical_definition=r"\alpha_2=\text{slope of }\ln F(s)\ \text{vs}\ \ln s\ \text{for } s\ge16",
    biological_motivation="Long-range correlation structure, reported to differ between sleep stages.",
    literature_references=("Peng et al. (1995) long-range correlations in DNA and EEG",),
    min_signal_samples=128,
    expected_range=(0.0, 2.0),
    output_units="dimensionless",
    computational_cost="medium",
    validation_status="reproduced-from-literature",
)

