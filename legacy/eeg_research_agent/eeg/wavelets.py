"""Multiresolution (wavelet) energy features with a self-contained DWT.

Why a hand-written DWT
----------------------
PyWavelets is the usual choice, but it was not installable in the reference
(offline) environment.  The dyadic decomposition implemented here uses the
standard orthonormal Daubechies filter coefficients and periodic boundary
handling, and is validated by an explicit perfect-reconstruction test
(``tests/test_features.py::test_wavelet_perfect_reconstruction``).  If PyWavelets
is available the coefficients below are identical to ``pywt.Wavelet("db4").dec_lo``,
so results are comparable.

Feature interpretation
----------------------
For a sampling frequency ``fs`` the level-``j`` detail band covers
``[fs/2^(j+1), fs/2^j]`` Hz, i.e. with ``fs = 100`` Hz the levels isolate
~25-50, 12.5-25, 6.25-12.5, 3.1-6.25 and 1.6-3.1 Hz.  Level-5 detail therefore
overlaps the delta band and level-2/3 detail overlaps sigma/alpha, which is what
makes the energy distribution across levels informative for sleep staging.
"""

from __future__ import annotations

import numpy as np

from .feature_registry import SignalContext, register_feature

__all__ = ["DB_FILTERS", "dwt", "idwt", "wavelet_energy_distribution", "detail_band_hz"]

#: Orthonormal Daubechies scaling filters (matches PyWavelets' ``dec_lo``).
DB_FILTERS: dict[str, np.ndarray] = {
    "db2": np.array(
        [
            0.4829629131445341,
            0.8365163037378079,
            0.2241438680420134,
            -0.1294095225512604,
        ]
    ),
    "db4": np.array(
        [
            0.2303778133088964,
            0.7148465705529154,
            0.6308807679298587,
            -0.0279837694168599,
            -0.1870348117190931,
            0.0308413818355607,
            0.0328830116668852,
            -0.0105974017850690,
        ]
    ),
}


def _analysis_filters(wavelet: str) -> tuple[np.ndarray, np.ndarray]:
    h = DB_FILTERS[wavelet]
    g = np.array([(-1) ** n * h[len(h) - 1 - n] for n in range(len(h))])
    return h, g


def dwt(signal: np.ndarray, wavelet: str = "db4") -> tuple[np.ndarray, np.ndarray]:
    """One dyadic decomposition step with periodic boundaries.

    ``cA[k] = sum_n h[n] x[(2k+n) mod N]`` and ``cD`` uses the quadrature mirror
    filter ``g``.  Odd-length inputs are extended by one period sample so that
    the transform stays invertible (:func:`idwt` truncates again).
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size < 2:
        return x.copy(), np.zeros_like(x)
    if x.size % 2:  # periodic single-sample extension keeps the transform invertible
        x = np.concatenate([x, x[:1]])
    n = x.size
    h, g = _analysis_filters(wavelet)
    taps = h.size
    half = n // 2
    indices = (2 * np.arange(half)[:, None] + np.arange(taps)[None, :]) % n
    windows = x[indices]
    return windows @ h, windows @ g


def idwt(c_a: np.ndarray, c_d: np.ndarray, wavelet: str = "db4", length: int | None = None) -> np.ndarray:
    """Inverse of :func:`dwt` (orthogonal synthesis, periodic boundaries).

    Synthesis is the adjoint of the correlation-based analysis step, i.e. the
    same taps are scattered back with the same alignment.  Orthonormality of the
    Daubechies filters (``sum_k h[m-2k] h[n-2k] + g[m-2k] g[n-2k] = delta_mn``)
    then guarantees perfect reconstruction; the test suite asserts a maximum
    reconstruction error below 1e-10.
    """
    c_a = np.asarray(c_a, dtype=np.float64).ravel()
    c_d = np.asarray(c_d, dtype=np.float64).ravel()
    h, g = _analysis_filters(wavelet)
    taps = h.size
    padded_size = 2 * c_a.size
    n = padded_size
    out = np.zeros(n, dtype=np.float64)
    indices = (2 * np.arange(c_a.size)[:, None] + np.arange(taps)[None, :]) % n
    for tap in range(taps):
        np.add.at(out, indices[:, tap], c_a * h[tap] + c_d * g[tap])
    if length is not None and length < padded_size:
        out = out[:length]
    return out


def _effective_levels(n_samples: int, levels: int) -> int:
    max_levels = int(np.floor(np.log2(max(n_samples, 2))))
    return max(1, min(levels, max_levels))


def approximation_sizes(n_samples: int, levels: int) -> list[int]:
    """Lengths of the approximation signal before each decomposition step."""
    sizes = [n_samples]
    for _ in range(_effective_levels(n_samples, levels)):
        if sizes[-1] < 2:
            break
        # odd lengths are padded by one periodic sample, so the coefficient
        # count is ceil(n/2) rather than floor(n/2)
        sizes.append(-(-sizes[-1] // 2))
    return sizes


def wavelet_decomposition(signal: np.ndarray, wavelet: str = "db4", levels: int = 5) -> list[np.ndarray]:
    """Return ``[cD_1, ..., cD_L, cA_L]`` (details first, approximation last)."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    details: list[np.ndarray] = []
    approximation = x
    for _ in range(_effective_levels(x.size, levels)):
        if approximation.size < 2:
            break
        approximation, detail = dwt(approximation, wavelet)
        details.append(detail)
    details.append(approximation)
    return details


def wavelet_reconstruct(parts: list[np.ndarray], wavelet: str = "db4", length: int | None = None) -> np.ndarray:
    """Reconstruct a signal from ``[cD_1, ..., cD_L, cA_L]``.

    Used by the validation tests to prove that the decomposition loses no
    information (perfect reconstruction up to floating-point precision).  Odd
    lengths are handled by tracking the approximation size of every level.
    """
    if not parts:
        raise ValueError("no decomposition parts supplied")
    details = list(parts[:-1])
    approximation = np.asarray(parts[-1], dtype=np.float64)
    if length is None:
        length = int(approximation.size * 2 ** len(details))
    sizes = approximation_sizes(length, len(details))
    for level in range(len(details) - 1, -1, -1):
        approximation = idwt(approximation, details[level], wavelet, length=sizes[level])
    return approximation


def wavelet_energy_distribution(
    signal: np.ndarray, wavelet: str = "db4", levels: int = 5
) -> dict[str, float]:
    """Absolute and relative energies of every detail/approximation sub-band."""
    parts = wavelet_decomposition(signal, wavelet, levels)
    energies = [float(np.sum(part**2)) for part in parts]
    total = float(sum(energies)) or 1.0
    result: dict[str, float] = {}
    for index, energy in enumerate(energies[:-1], start=1):
        result[f"detail_energy_{index}"] = energy
        result[f"detail_energy_relative_{index}"] = energy / total
    result["approximation_energy"] = energies[-1]
    result["approximation_energy_relative"] = energies[-1] / total
    result["wavelet_total_energy"] = total
    return result


def detail_band_hz(sfreq: float, level: int) -> tuple[float, float]:
    """Frequency band covered by detail level ``level`` (1-based)."""
    return (sfreq / 2 ** (level + 1), sfreq / 2**level)


def _energy(context: SignalContext) -> dict[str, float]:
    if "wavelet_energy" not in context.memo:
        cfg = context.config.features.wavelet
        context.memo["wavelet_energy"] = wavelet_energy_distribution(
            context.epoch, cfg.wavelet, cfg.levels
        )
    return context.memo["wavelet_energy"]


for _level in range(1, 6):

    def _detail_absolute(context: SignalContext, level: int = _level) -> float:
        return _energy(context).get(f"detail_energy_{level}", 0.0) * 1e12

    def _detail_relative(context: SignalContext, level: int = _level) -> float:
        return _energy(context).get(f"detail_energy_relative_{level}", 0.0)

    register_feature(
        feature_id=f"wavelet_detail_energy_{_level}",
        name=f"Wavelet detail energy level {_level}",
        category="wavelet",
        family="wavelet",
        function=_detail_absolute,
        mathematical_definition=rf"E_{{{_level}}}=\sum_k |c_{{{_level},k}}|^2",
        biological_motivation=(
            f"Energy in the {_level}-th dyadic sub-band "
            f"(~{100/2**(_level+1):.1f}-{100/2**_level:.1f} Hz at 100 Hz); a "
            "time-frequency decomposition that does not assume stationarity within the epoch."
        ),
        literature_references=("Daubechies (1992) ten lectures on wavelets",),
        required_sampling_frequency=2 ** (_level + 1),
        expected_range=(0.0, None),
        output_units="uV^2",
        computational_cost="medium",
        validation_status="unit-tested",
    )
    register_feature(
        feature_id=f"wavelet_detail_ratio_{_level}",
        name=f"Relative wavelet energy level {_level}",
        category="wavelet",
        family="wavelet",
        function=_detail_relative,
        mathematical_definition=rf"q_{{{_level}}}=E_{{{_level}}}\big/\sum_j E_j",
        biological_motivation="Scale-invariant distribution of energy across time scales.",
        required_sampling_frequency=2 ** (_level + 1),
        expected_range=(0.0, 1.0),
        output_units="fraction",
        computational_cost="medium",
        validation_status="unit-tested",
    )


def multiscale_energy_entropy(context: SignalContext) -> float:
    r"""Wavelet energy entropy :math:`M=-\sum_j q_j\ln(q_j+\epsilon)/\ln J`."""
    energy = _energy(context)
    relative = np.array(
        [value for key, value in energy.items() if key.startswith("detail_energy_relative")]
    )
    if relative.size == 0:
        return 0.0
    relative = relative / max(relative.sum(), 1e-12)
    entropy = -float(np.sum(relative * np.log(relative + 1e-12)))
    return entropy / max(np.log(relative.size), 1e-12)


def wavelet_energy_slope(context: SignalContext) -> float:
    """Slope of log relative energy versus decomposition level (1/f-like scaling)."""
    energy = _energy(context)
    relative = [
        energy.get(f"detail_energy_relative_{level}", 0.0) for level in range(1, 6)
    ]
    relative = np.asarray(relative, dtype=float)
    if np.all(relative <= 0):
        return 0.0
    levels = np.arange(1, relative.size + 1)
    x = np.log2(2.0**levels)
    y = np.log(relative + 1e-12)
    return float(np.polyfit(x, y, 1)[0])


def wavelet_low_high_ratio(context: SignalContext) -> float:
    """Energy ratio between coarse (level >= 4) and fine (level <= 2) sub-bands."""
    energy = _energy(context)
    low = sum(energy.get(f"detail_energy_{level}", 0.0) for level in (4, 5))
    high = sum(energy.get(f"detail_energy_{level}", 0.0) for level in (1, 2))
    return float(low / high) if high > 0 else 0.0


register_feature(
    feature_id="wavelet_energy_entropy",
    name="Multiscale wavelet energy entropy",
    category="wavelet",
    family="wavelet",
    function=multiscale_energy_entropy,
    mathematical_definition=r"M=-\sum_j q_j\ln(q_j+\epsilon)\big/\ln J,\quad q_j=E_j/\sum_i E_i",
    biological_motivation=(
        "Distribution of energy across temporal scales.  Low when one scale dominates "
        "(slow-wave sleep), high for broadband activity (wake, REM)."
    ),
    literature_references=("Rosso et al. (2001) wavelet entropy of EEG",),
    required_sampling_frequency=32.0,
    expected_range=(0.0, 1.0),
    output_units="normalised",
    computational_cost="medium",
    novelty_status="literature-derived",
    validation_status="unit-tested",
)

register_feature(
    feature_id="wavelet_energy_slope",
    name="Wavelet energy scaling slope",
    category="wavelet",
    family="wavelet",
    function=wavelet_energy_slope,
    mathematical_definition=r"s=\text{slope of }\ln q_j\ \text{vs}\ \log_2(2^j)",
    biological_motivation="Steep negative slope indicates 1/f-like dominance of coarse scales (N3).",
    required_sampling_frequency=32.0,
    expected_range=(-6.0, 3.0),
    output_units="dimensionless",
    computational_cost="medium",
    novelty_status="literature-derived",
    validation_status="unit-tested",
)

register_feature(
    feature_id="wavelet_low_high_ratio",
    name="Coarse/fine wavelet energy ratio",
    category="wavelet",
    family="wavelet",
    function=wavelet_low_high_ratio,
    mathematical_definition=r"R_{wh}=(E_4+E_5)/(E_1+E_2)",
    biological_motivation="Direct time-frequency analogue of the delta/beta power ratio.",
    required_sampling_frequency=32.0,
    expected_range=(0.0, None),
    output_units="dimensionless",
    computational_cost="medium",
    novelty_status="literature-derived",
    validation_status="unit-tested",
)
