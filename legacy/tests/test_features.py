"""Feature-engine tests.

The most important test in the repository is
:func:`test_every_registered_feature_computes`: a registry entry whose function
signature does not match the parameters the registry passes to it fails silently
for every epoch, and that is exactly how a broken feature once shipped.  It is
cheap to run and it must never be deleted.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeg_research_agent.eeg.feature_extraction import feature_columns
from eeg_research_agent.eeg.feature_registry import SignalContext
from eeg_research_agent.verification.synthetic_signals import multi_tone, sine, white_noise


def _context(signal: np.ndarray, sfreq: float, config) -> SignalContext:
    return SignalContext(epoch=signal, sfreq=sfreq, config=config)


def test_every_registered_feature_computes(registry, base_config):
    """Every registered feature must produce a finite number on a normal epoch."""
    signal = multi_tone([(1.0, 30.0), (6.0, 10.0), (12.0, 8.0)], n_samples=3000, sfreq=100.0)
    values, errors = registry.compute(_context(signal, 100.0, base_config))
    assert not errors, f"feature computations failed: {errors}"
    assert len(values) == len(registry)
    non_finite = {name: value for name, value in values.items() if not np.isfinite(value)}
    assert not non_finite, f"non-finite feature values: {non_finite}"


def test_features_handle_short_and_degenerate_inputs(registry, base_config):
    """A feature may decline an epoch, but must never raise or return a wild value."""
    for signal in (
        np.zeros(600),
        np.ones(600) * 5e-6,
        white_noise(600, 100.0, amplitude=50.0, seed=7),
    ):
        values, errors = registry.compute(_context(signal, 100.0, base_config))
        assert set(values) | set(errors) <= set(registry.ids())
        for name, value in values.items():
            assert np.isfinite(value), f"{name} returned {value} on a degenerate signal"


def test_absolute_power_scales_quadratically_and_relative_power_does_not(
    registry, base_config
):
    """Doubling the waveform quadruples absolute power; relative power is unchanged."""
    fundamental = sine(2.0, n_samples=3000, sfreq=100.0, amplitude=20.0)
    quiet = _context(fundamental, 100.0, base_config)
    loud = _context(fundamental * 2.0, 100.0, base_config)

    assert loud.band_power("delta") / quiet.band_power("delta") == pytest.approx(4.0, rel=1e-6)

    quiet_relative, _ = registry.compute(quiet, ["rel_power_delta"])
    loud_relative, _ = registry.compute(loud, ["rel_power_delta"])
    assert loud_relative["rel_power_delta"] == pytest.approx(
        quiet_relative["rel_power_delta"], rel=1e-6
    )


def test_band_power_tracks_the_injected_band(registry, base_config):
    """Injecting energy in a band must raise that band's power and little else."""
    noise = white_noise(3000, 100.0, amplitude=10.0, seed=3)
    baseline = _context(noise, 100.0, base_config)
    delta_tone = _context(noise + sine(1.5, n_samples=3000, sfreq=100.0, amplitude=40.0), 100.0, base_config)
    assert delta_tone.band_power("delta") > 10 * baseline.band_power("delta")
    assert delta_tone.band_power("beta") == pytest.approx(baseline.band_power("beta"), rel=0.25)


def test_window_length_does_not_change_normalised_features(registry, base_config):
    """A stationary signal scored on 30 s and 60 s windows must agree qualitatively."""
    long_signal = white_noise(6000, 100.0, amplitude=20.0, seed=11)
    half, _ = registry.compute(_context(long_signal[:3000], 100.0, base_config))
    whole, _ = registry.compute(_context(long_signal, 100.0, base_config))
    for name in ("spectral_entropy", "hjorth_mobility", "sample_entropy"):
        assert whole[name] == pytest.approx(half[name], rel=0.5, abs=0.2), name


def test_feature_table_is_finite_and_traceable(feature_table):
    frame, _table, splits = feature_table
    columns = feature_columns(frame)
    assert len(columns) >= 100, "the registered catalog should be fully extracted"
    non_causal = {"feature_delta_norm", "feature_delta_norm_sigma"}
    causal = [name for name in columns if name not in non_causal]
    assert np.isfinite(frame[causal].to_numpy(dtype=float)).all()
    for column in ("subject_id", "recording_id", "epoch_id", "onset_s", "stage", "split"):
        assert column in frame.columns
    assert set(frame["split"]).issubset({"train", "validation", "test"})
    assert frame["subject_id"].nunique() == len(splits.all_subjects())


def test_sequence_features_are_absent_on_the_first_epoch(feature_table):
    frame, _table, _splits = feature_table
    first = frame.groupby(["subject_id", "recording_id"], sort=False).head(1)
    assert first["feature_delta_norm"].isna().all()
    rest = frame.drop(first.index)
    assert rest["feature_delta_norm"].notna().mean() > 0.95
    assert (rest["feature_delta_norm"] >= 0).all()


def test_registry_rejects_unregistered_and_out_of_contract_specifications(registry):
    outcome = registry.validate_specification(
        {
            "features": [
                {"feature_id": "abs_power_delta"},
                {"feature_id": "definitely_not_a_feature"},
                {"feature_id": "abs_power_delta", "parameters": {"made_up": 1}},
            ]
        },
        sfreq=100.0,
    )
    approved = {item["feature_id"] for item in outcome["approved"]}
    rejected = {item["feature_id"] for item in outcome["rejected"]}
    assert "abs_power_delta" in approved
    assert "definitely_not_a_feature" in rejected
    assert any(item["feature_id"] == "abs_power_delta" for item in outcome["rejected"])


def test_sampling_frequency_requirements_are_enforced(registry):
    """A feature that needs a higher sampling rate must be rejected below it."""
    demanding = [definition for definition in registry if definition.required_sampling_frequency > 20.0]
    assert demanding, "the registry should contain features with a sampling-rate requirement"
    feature_id = demanding[0].feature_id
    outcome = registry.validate_specification({"features": [{"feature_id": feature_id}]}, sfreq=8.0)
    assert not outcome["approved"]
    assert "sfreq" in outcome["rejected"][0]["reason"]
