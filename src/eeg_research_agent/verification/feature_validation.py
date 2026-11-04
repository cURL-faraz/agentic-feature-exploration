"""Mathematical verification of the feature implementations.

These checks answer the questions the brief asks directly: is the equation
implemented correctly, are the units sensible, is the transform invertible, are
logarithms numerically stable, and is the code deterministic?
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "verify_wavelet_reconstruction",
    "verify_registry_contract",
    "verify_determinism",
    "verify_sampling_frequency_sensitivity",
    "verify_numerical_stability",
    "verify_expected_ranges",
    "run_feature_validation",
]


def _check(name: str, passed: bool, detail: str, observed: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail, "observed": observed}


def verify_wavelet_reconstruction(config: Any) -> dict[str, Any]:
    """The hand-written DWT/IDWT pair must invert to machine precision."""
    from ..eeg.wavelets import dwt, idwt
    from ..verification.synthetic_signals import pink_noise, sine

    results: dict[str, Any] = {}
    for wavelet in ("db2", "db4"):
        for n_samples in (2000, 2001, 3000, 3001):
            rng = np.random.default_rng(n_samples)
            signal = rng.normal(0, 10, n_samples)
            c_a, c_d = dwt(signal, wavelet)
            reconstructed = idwt(c_a, c_d, wavelet, length=signal.size)
            error = float(np.max(np.abs(reconstructed - signal)))
            results[f"{wavelet}_{n_samples}"] = error
    checks = [
        _check(
            f"wavelet_perfect_reconstruction_{key}",
            value < 1e-9,
            f"max |x - idwt(dwt(x))| = {value:.3e} (tolerance 1e-9)",
            value,
        )
        for key, value in results.items()
    ]
    # energy conservation: ||x||^2 == ||cA||^2 + ||cD||^2
    signal = sine(7.0, 2048, 100.0, amplitude=15.0) + pink_noise(2048, 100.0, amplitude=5.0, seed=3)
    c_a, c_d = dwt(signal, "db4")
    checks.append(
        _check(
            "wavelet_energy_conservation",
            abs(float(np.sum(signal**2)) - float(np.sum(c_a**2) + np.sum(c_d**2))) / float(np.sum(signal**2)) < 1e-9,
            "Parseval: signal energy must equal the sum of coefficient energies for an orthonormal basis",
            {
                "signal_energy": float(np.sum(signal**2)),
                "coefficient_energy": float(np.sum(c_a**2) + np.sum(c_d**2)),
            },
        )
    )
    return {"ok": all(c["passed"] for c in checks), "checks": checks}


def verify_registry_contract(registry: Any) -> dict[str, Any]:
    """Every registered feature must be fully documented and consistently typed."""
    required_text = ("mathematical_definition", "biological_motivation")
    valid_novelty = {"established", "literature-derived", "novel-hypothesis"}
    valid_validation = {
        "unit-tested",
        "synthetic-validated",
        "reproduced-from-literature",
        "unvalidated",
        "rejected",
    }
    problems: dict[str, list[str]] = {}
    for definition in registry:
        issues: list[str] = []
        if not callable(definition.function):
            issues.append("implementation function is not callable")
        for attribute in required_text:
            if not str(getattr(definition, attribute, "") or "").strip():
                issues.append(f"{attribute} is empty")
        if not definition.category:
            issues.append("category is empty")
        if not definition.family:
            issues.append("family is empty")
        if definition.novelty_status not in valid_novelty:
            issues.append(f"invalid novelty_status: {definition.novelty_status}")
        if definition.validation_status not in valid_validation:
            issues.append(f"invalid validation_status: {definition.validation_status}")
        if definition.min_signal_samples < 0:
            issues.append("min_signal_samples is negative")
        if issues:
            problems[definition.feature_id] = issues
    # a novel hypothesis must never claim to be reproduced from literature
    mislabelled = [
        d.feature_id
        for d in registry
        if d.novelty_status == "novel-hypothesis" and d.validation_status == "reproduced-from-literature"
    ]
    checks = [
        _check(
            "registry_metadata_complete",
            not problems,
            "every feature declares an equation, motivation, category, family and valid statuses",
            problems,
        ),
        _check(
            "novelty_status_not_overclaimed",
            not mislabelled,
            "no feature marked 'novel-hypothesis' claims to be reproduced from the literature",
            mislabelled,
        ),
    ]
    return {
        "ok": all(c["passed"] for c in checks),
        "checks": checks,
        "n_features": len(registry),
        "novelty_breakdown": _counts(registry, "novelty_status"),
        "family_breakdown": _counts(registry, "family"),
        "validation_breakdown": _counts(registry, "validation_status"),
    }


def _counts(registry: Any, attribute: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for definition in registry:
        value = str(getattr(definition, attribute))
        counts[value] = counts.get(value, 0) + 1
    return counts


def verify_determinism(registry: Any, config: Any, sfreq: float = 100.0, n_samples: int = 3000) -> dict[str, Any]:
    """Recomputing a feature on the same epoch must give bit-identical results."""
    from ..eeg.feature_registry import SignalContext
    from .synthetic_signals import pink_noise

    signal = pink_noise(n_samples, sfreq, amplitude=12.0, seed=99)
    first, _ = registry.compute(SignalContext(epoch=signal, sfreq=sfreq, config=config))
    second, _ = registry.compute(SignalContext(epoch=signal, sfreq=sfreq, config=config))
    unstable = sorted(
        feature for feature in first if not np.isclose(first[feature], second.get(feature, np.nan), rtol=0, atol=0)
    )
    return {
        "ok": not unstable,
        "checks": [
            _check(
                "feature_determinism",
                not unstable,
                "identical inputs must give bit-identical outputs",
                unstable,
            )
        ],
        "n_features_checked": len(first),
    }


def verify_sampling_frequency_sensitivity(
    registry: Any, config: Any, features: list[str] | None = None
) -> dict[str, Any]:
    """Check how each feature behaves when the sampling rate changes.

    A tone at 10 Hz sampled at 100 Hz and 200 Hz represents the same physical
    signal, so *relative* and *normalised* measures should agree closely.
    Features whose values change markedly are flagged - not as failures (some
    quantities legitimately depend on the sampling rate) but as candidates that
    must never be pooled across recordings with different rates.
    """
    from ..eeg.feature_registry import SignalContext
    from .synthetic_signals import sine

    selected = features or registry.ids()
    sensitive: dict[str, float] = {}
    for feature_id in selected:
        values = []
        for sfreq in (100.0, 200.0):
            n_samples = int(30 * sfreq)
            signal = sine(10.0, n_samples, sfreq, amplitude=20.0) + sine(2.0, n_samples, sfreq, amplitude=10.0)
            context = SignalContext(epoch=signal, sfreq=sfreq, config=config)
            computed = registry.compute(context, [feature_id])[0].get(feature_id)
            values.append(computed)
        if any(value is None or not np.isfinite(value) for value in values):
            continue
        denominator = max(abs(values[0]), abs(values[1]), 1e-12)
        relative_change = abs(values[1] - values[0]) / denominator
        if relative_change > 0.05:
            sensitive[feature_id] = float(relative_change)
    ordered = dict(sorted(sensitive.items(), key=lambda item: -item[1]))
    return {
        "ok": True,
        "checks": [
            _check(
                "sampling_frequency_stability",
                True,
                f"{len(ordered)} of {len(selected)} features change by >5% between 100 Hz and 200 Hz; "
                "these must not be pooled across recordings with different sampling rates",
                ordered,
            )
        ],
        "sensitive_features": ordered,
    }


def verify_numerical_stability(registry: Any, config: Any, sfreq: float = 100.0) -> dict[str, Any]:
    """Degenerate inputs must produce finite values, never NaN/Inf or exceptions."""
    from ..eeg.feature_registry import SignalContext

    cases = {
        "zeros": np.zeros(3000),
        "constant_5uv": np.full(3000, 5.0),
        "tiny_amplitude": np.random.default_rng(1).normal(0, 1e-9, 3000),
        "huge_amplitude": np.random.default_rng(2).normal(0, 1e5, 3000),
        "single_spike": np.concatenate([np.zeros(1500), [500.0], np.zeros(1499)]),
    }
    failures: dict[str, dict[str, str]] = {}
    for case_name, signal in cases.items():
        problems: dict[str, str] = {}
        for feature_id in registry.ids():
            try:
                computed = registry.compute(
                    SignalContext(epoch=signal, sfreq=sfreq, config=config), [feature_id]
                )[0].get(feature_id)
            except Exception as exc:  # noqa: BLE001
                problems[feature_id] = f"{type(exc).__name__}: {exc}"
                continue
            if computed is None or not np.isfinite(computed):
                problems[feature_id] = f"non-finite result: {computed}"
        if problems:
            failures[case_name] = problems
    return {
        "ok": not failures,
        "checks": [
            _check(
                "numerical_stability_degenerate_inputs",
                not failures,
                "all features must return finite values for zero, constant, tiny, huge and spiky inputs",
                failures,
            )
        ],
        "cases": list(cases),
    }


def verify_expected_ranges(registry: Any, config: Any, sfreq: float = 100.0) -> dict[str, Any]:
    """Informational check: do declared expected ranges hold on synthetic signals?"""
    from ..eeg.feature_registry import SignalContext
    from .synthetic_signals import multi_tone, pink_noise, white_noise

    n_samples = int(30 * sfreq)
    signals = {
        "pink": pink_noise(n_samples, sfreq, amplitude=12.0, seed=5),
        "white": white_noise(n_samples, sfreq, amplitude=12.0),
        "mixed": multi_tone([(2.0, 20.0), (13.0, 10.0), (20.0, 6.0)], n_samples, sfreq),
    }
    violations: dict[str, dict[str, Any]] = {}
    for feature_id in registry.ids():
        definition = registry.get(feature_id)
        low, high = definition.expected_range
        if low is None and high is None:
            continue
        for name, signal in signals.items():
            computed = registry.compute(SignalContext(epoch=signal, sfreq=sfreq, config=config), [feature_id])[0].get(feature_id)
            if computed is None or not np.isfinite(computed):
                continue
            if (low is not None and computed < low - 1e-9) or (high is not None and computed > high + 1e-9):
                violations.setdefault(feature_id, {})[name] = computed
    return {
        "ok": True,
        "checks": [
            _check(
                "declared_expected_ranges",
                True,
                "informational: declared expected ranges are heuristic and are reported, not enforced",
                violations,
            )
        ],
        "violations": violations,
        "note": "expected ranges are documentation, not acceptance criteria",
    }


def run_feature_validation(registry: Any, config: Any, sfreq: float = 100.0) -> dict[str, Any]:
    """Run every mathematical/implementation verification."""
    sections = {
        "wavelet_reconstruction": verify_wavelet_reconstruction(config),
        "registry_contract": verify_registry_contract(registry),
        "determinism": verify_determinism(registry, config, sfreq=sfreq),
        "numerical_stability": verify_numerical_stability(registry, config, sfreq=sfreq),
        "sampling_frequency_sensitivity": verify_sampling_frequency_sensitivity(registry, config),
        "expected_ranges": verify_expected_ranges(registry, config, sfreq=sfreq),
    }
    checks = [check for section in sections.values() for check in section.get("checks", [])]
    return {
        "ok": all(section.get("ok", True) for section in sections.values()),
        "checks": checks,
        "sections": sections,
        "n_features": len(registry),
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
    }
