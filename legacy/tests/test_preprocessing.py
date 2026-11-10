"""Preprocessing, segmentation and annotation-mapping tests."""

from __future__ import annotations

import numpy as np

from eeg_research_agent.data.acquisition import acquire_dataset
from eeg_research_agent.data.annotations import StageMapper, summarise_annotations
from eeg_research_agent.data.preprocessing import preprocess_recording
from eeg_research_agent.eeg.segmentation import segment_recordings


def _tone(sfreq: float, seconds: float, freq: float, amplitude_uv: float) -> np.ndarray:
    time = np.arange(int(sfreq * seconds)) / sfreq
    return amplitude_uv * 1e-6 * np.sin(2 * np.pi * freq * time)


def test_synthetic_corpus_is_deterministic_and_labelled(small_corpus, base_config):
    subjects = {recording.subject_id for recording in small_corpus}
    assert subjects, "the synthetic corpus must contain recordings"
    assert all(recording.annotations for recording in small_corpus), "every recording needs annotations"
    again = acquire_dataset(base_config, origin="synthetic").recordings
    assert [r.subject_id for r in again] == [r.subject_id for r in small_corpus]
    assert np.allclose(again[0].signal, small_corpus[0].signal)


def test_preprocessing_is_recorded_and_preserves_length_or_resamples(preprocessed, base_config):
    cleaned, reports = preprocessed
    assert len(cleaned) == len(reports)
    for recording, report in zip(cleaned, reports):
        assert report.sfreq_out == base_config.dataset.signal.target_sampling_frequency
        assert report.bandpass == base_config.dataset.preprocessing.bandpass
        assert report.clipping_fraction < 0.05, "preprocessing must not clip a large fraction of the signal"
        assert np.isfinite(recording.signal).all(), "preprocessed signals must be finite"


def test_bandpass_removes_out_of_band_energy(base_config):
    """The 0.3-45 Hz band-pass removes out-of-band energy and keeps in-band energy.

    The pass-band is measured directly (sampling rate held at 200 Hz so the
    corner at 45 Hz is well below Nyquist), which lets the test assert the two
    properties the preprocessing actually promises: a 60 Hz interferer is
    suppressed, and a 10 Hz component is preserved.
    """
    from eeg_research_agent.data.recording import Recording
    from scipy import signal as sps

    sfreq = 200.0
    signal = (
        _tone(sfreq, 120.0, 10.0, 20.0)  # in band
        + _tone(sfreq, 120.0, 60.0, 20.0)  # out of band (above 45 Hz)
        + _tone(sfreq, 120.0, 0.05, 5.0)  # out of band (slow drift)
    )
    recording = Recording(
        subject_id="T0",
        recording_id="R1",
        ch_name="EEG",
        sfreq=sfreq,
        signal=signal,
        annotations=[],
        data_origin="synthetic",
    )
    config = base_config.model_copy(
        update={
            "dataset": base_config.dataset.model_copy(
                update={
                    "signal": base_config.dataset.signal.model_copy(
                        update={"target_sampling_frequency": sfreq}
                    )
                }
            )
        }
    )
    cleaned, report = preprocess_recording(recording, config)
    assert report.bandpass == (0.3, 45.0)
    assert report.sfreq_out == sfreq, "the sampling rate is held fixed for this test"

    frequencies, raw_psd = sps.welch(recording.signal, fs=sfreq, nperseg=1024)
    clean_freq, clean_psd = sps.welch(cleaned.signal, fs=cleaned.sfreq, nperseg=1024)

    def band_power(omega: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
        mask = (omega >= low) & (omega <= high)
        return float(psd[mask].mean())

    in_band_raw = band_power(frequencies, raw_psd, 9.0, 11.0)
    in_band_clean = band_power(clean_freq, clean_psd, 9.0, 11.0)
    assert in_band_clean > 0.8 * in_band_raw, "the 10 Hz component must survive the band-pass"

    high_raw = band_power(frequencies, raw_psd, 58.0, 62.0)
    high_clean = band_power(clean_freq, clean_psd, 58.0, 62.0)
    assert high_clean < high_raw / 50.0, "the 60 Hz component must be suppressed"


def test_annotations_map_to_canonical_stages_and_preserve_unknowns(base_config):
    config = base_config.dataset.stages
    mapper = StageMapper(config.annotation_map, config.supported)
    raw = [
        (0.0, 30.0, "Sleep stage W"),
        (30.0, 30.0, "Sleep stage 2"),
        (60.0, 30.0, "Movement time"),
        (90.0, 30.0, "Sleep stage R"),
    ]
    annotations, report = mapper.map_annotations(raw)
    stages = [annotation.stage for annotation in annotations]
    assert stages[0] == "W"
    assert stages[1] == "N2"
    assert stages[3] == "REM"
    assert stages[2] == "UNKNOWN", "an unmapped description must not become a sleep stage"
    assert report.unmapped_counts, "unmapped descriptions must be counted and reported"
    assert all(
        annotation.description for annotation in annotations
    ), "the original annotation text must be preserved"
    summary = summarise_annotations(annotations)
    assert summary["counts"]["UNKNOWN"] == 1
    assert "REM" in summary["stages_present"]
    assert "N1" in summary["stages_missing"]


def test_epoch_construction_respects_the_configured_window(preprocessed, base_config):
    cleaned, _ = preprocessed
    epoch_sets, reports = segment_recordings(cleaned, base_config)
    epoch_seconds = base_config.dataset.signal.epoch_seconds
    for epoch_set in epoch_sets:
        assert epoch_set.epoch_seconds == epoch_seconds
        assert epoch_set.sfreq == base_config.dataset.signal.target_sampling_frequency
        if epoch_set.n_epochs:
            assert epoch_set.data.shape[1] == int(round(epoch_seconds * epoch_set.sfreq))
        # every kept epoch has an onset inside the recording and a stage label
        assert all(0 <= onset for onset in epoch_set.onsets)
        assert all(stage for stage in epoch_set.stages)
    # exclusions are recorded with a reason rather than silently dropped
    for report in reports:
        for exclusion in report.exclusions:
            assert exclusion["reason"]
        assert report.n_kept + len(report.exclusions) == report.n_windows
