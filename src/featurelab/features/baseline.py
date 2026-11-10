"""The baseline: standard single-channel sleep features a candidate must beat.

A new feature is only interesting if it carries information the *standard* toolkit
does not.  So the baseline has to be honest and reasonably strong: band powers,
their ratios, spectral shape, Hjorth parameters and amplitude statistics - the
features every classical single-channel paper starts from.

Everything is vectorised over all epochs at once (seconds for ~40k epochs), and
every feature is cached per cohort, because the baseline is reused by every
hypothesis test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
    "gamma": (30.0, 45.0),
}
EPS = 1e-12


def _band(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs < hi)
    return psd[:, mask].sum(axis=1) * float(freqs[1] - freqs[0])


def baseline_features(X: np.ndarray, psd: np.ndarray, freqs: np.ndarray, chunk: int = 4096) -> pd.DataFrame:
    psd = np.asarray(psd, dtype=np.float64)
    band = {name: _band(psd, freqs, lo, hi) for name, (lo, hi) in BANDS.items()}
    total = _band(psd, freqs, 0.5, 45.0) + EPS
    out: dict[str, np.ndarray] = {}
    for name, power in band.items():
        out[f"rel_{name}"] = power / total
    out["log_total_power"] = np.log(total)
    out["ratio_theta_delta"] = np.log((band["theta"] + EPS) / (band["delta"] + EPS))
    out["ratio_alpha_theta"] = np.log((band["alpha"] + EPS) / (band["theta"] + EPS))
    out["ratio_sigma_theta"] = np.log((band["sigma"] + EPS) / (band["theta"] + EPS))
    out["ratio_beta_delta"] = np.log((band["beta"] + EPS) / (band["delta"] + EPS))

    in_range = (freqs >= 0.5) & (freqs < 45.0)
    p = psd[:, in_range]
    f = freqs[in_range]
    pn = p / (p.sum(axis=1, keepdims=True) + EPS)
    out["spectral_entropy"] = -(pn * np.log(pn + EPS)).sum(axis=1) / np.log(pn.shape[1])
    cumulative = np.cumsum(pn, axis=1)
    out["sef95"] = f[np.argmax(cumulative >= 0.95, axis=1)]
    out["sef50"] = f[np.argmax(cumulative >= 0.50, axis=1)]
    out["spectral_centroid"] = (pn * f).sum(axis=1)
    peak_mask = (freqs >= 0.5) & (freqs < 30.0)
    out["peak_frequency"] = freqs[peak_mask][np.argmax(psd[:, peak_mask], axis=1)]

    # time-domain features, computed in chunks to keep memory flat
    cols = {k: [] for k in ("std", "hjorth_mobility", "hjorth_complexity", "zero_cross_rate",
                            "line_length", "kurtosis", "skewness")}
    for start in range(0, X.shape[0], chunk):
        x = np.asarray(X[start : start + chunk], dtype=np.float64)
        x = x - x.mean(axis=1, keepdims=True)
        d1 = np.diff(x, axis=1)
        d2 = np.diff(d1, axis=1)
        v0, v1, v2 = x.var(axis=1) + EPS, d1.var(axis=1) + EPS, d2.var(axis=1) + EPS
        mobility = np.sqrt(v1 / v0)
        cols["std"].append(np.sqrt(v0))
        cols["hjorth_mobility"].append(mobility)
        cols["hjorth_complexity"].append(np.sqrt(v2 / v1) / mobility)
        cols["zero_cross_rate"].append((np.diff(np.signbit(x), axis=1) != 0).mean(axis=1))
        cols["line_length"].append(np.abs(d1).mean(axis=1) / np.sqrt(v0))
        cols["kurtosis"].append(stats.kurtosis(x, axis=1))
        cols["skewness"].append(stats.skew(x, axis=1))
    for key, parts in cols.items():
        out[key] = np.concatenate(parts)
    out["log_std"] = np.log(out.pop("std") + EPS)
    return pd.DataFrame(out).astype(np.float32)


def load_or_compute_baseline(cohort) -> pd.DataFrame:
    path = cohort.path / "baseline_features.parquet"
    csv = cohort.path / "baseline_features.csv.gz"
    for candidate in (path, csv):
        if candidate.is_file():
            try:
                return pd.read_parquet(candidate) if candidate.suffix == ".parquet" else pd.read_csv(candidate)
            except Exception:  # noqa: BLE001 - recompute on unreadable cache
                pass
    frame = baseline_features(cohort.X, cohort.psd, cohort.freqs)
    try:
        frame.to_parquet(path)
    except Exception:  # noqa: BLE001 - pyarrow is optional
        frame.to_csv(csv, index=False)
    return frame
