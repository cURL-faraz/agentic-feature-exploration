# Verification and Validation Report

> **Data provenance.** This run used a *synthetic* polysomnography surrogate (origin: `synthetic`), not human recordings. Every association reported below is a property of that simulation and must not be read as a statement about human sleep. Re-run with network access to use Sleep-EDF.

## Verdict

| Field | Value |
| --- | --- |
| overall | PASS |
| checks executed | 83 |
| checks failed | 0 |
| failed sections | none |
| data origin | synthetic |

A failing check is not hidden: it is recorded here, the affected conclusion is downgraded in the final report, and nothing that failed verification is presented as a validated finding.

## Synthetic-signal checks (signal processing)

| Field | Value |
| --- | --- |
| status | PASS |
| checks | 33 |
| failed | none |
| check | passed | detail |
| --- | --- | --- |
| pure_tone_peak_frequency | yes | observed 10.000 Hz |
| pure_tone_less_flat_than_noise | yes | spectral flatness of a tone must be below that of white noise |
| white_noise_spectral_flatness_high | yes | observed 0.9647 |
| spectral_entropy_tone_below_noise | yes | spectral entropy must be lower for a pure tone than for white noise |
| absolute_power_scales_quadratically_x2 | yes | absolute band power must scale with amplitude^2 (x2 amplitude -> x4 power) |
| absolute_power_scales_quadratically_x3 | yes | absolute band power must scale with amplitude^2 (x3 amplitude -> x9 power) |
| relative_power_invariant_to_amplitude | yes | relative band power must be invariant to a uniform amplitude change |
| relative_band_powers_sum_to_one | yes | relative band powers should approximately sum to 1 over the covered range |
| delta_power_increases_with_delta_component | yes | adding a 2 Hz component must increase delta-band power |
| sigma_power_increases_with_sigma_component | yes | adding a 13 Hz component must increase sigma-band power |
| zero_crossing_rate_matches_tone_frequency | yes | observed 9.983 Hz |
| hjorth_complexity_near_one_for_sinusoid | yes | observed 1.0005 |
| rms_matches_sine_amplitude_over_sqrt2 | yes | observed 14.1421 uV |
| line_length_increases_with_frequency | yes | line length (total variation) must grow with frequency for equal amplitude |
| pink_noise_has_more_delta_than_white_noise | yes | 1/f noise must place relatively more power in the delta band than white noise |
| wavelet_low_high_ratio_orders_by_frequency | yes | a 1 Hz tone must place more energy in the approximation/low detail than a 40 Hz tone |
| sample_entropy_higher_for_noise_than_tone | yes | sample entropy of white noise must exceed that of a pure tone |
| permutation_entropy_higher_for_noise_than_tone | yes | permutation entropy of white noise must exceed that of a pure tone |
| higuchi_fd_higher_for_noise_than_tone | yes | the fractal dimension of noise must exceed that of a smooth sinusoid (a 7 Hz tone is used because an exactly periodic 10 Hz tone at 100 Hz is a documented degeneracy of the Higuchi estimator, not a property of the signal) |
| dfa_alpha1_in_plausible_range_for_pink_noise | yes | DFA alpha1 for 1/f noise should be near 1 (reported for information) |
| spindle_detected_when_injected | yes | an injected spindle must be detected on an EEG-like background |
| spindle_specific_to_the_injected_event | yes | the spindle detector must fire more often with the event than without it |
| k_complex_detected_when_injected | yes | an injected k complex must be detected on an EEG-like background |
| k_complex_specific_to_the_injected_event | yes | the k complex detector must fire more often with the event than without it |
| slow_wave_detected_when_injected | yes | an injected slow wave must be detected on an EEG-like background |
| slow_wave_specific_to_the_injected_event | yes | the slow wave detector must fire more often with the event than without it |
| spindle_detector_low_false_positive_rate_on_noise | yes | pure Gaussian noise (not an EEG background) must not produce more than 1 spindle/min |
| transient_index_higher_for_modulated_than_stationary | yes | a modulated sigma envelope must look more transient than a constant-amplitude tone |
| peak_to_peak_detects_artefact | yes | a 500 uV step must dominate the peak-to-peak amplitude |
| ar_histogram_entropy_finite_under_artefact | yes | the amplitude histogram entropy must remain finite for an artefact-contaminated epoch |
| all_features_finite_on_zero_signal | yes | every registered feature must return a finite value for an all-zero epoch (no silent NaN propagation) |
| all_features_finite_on_constant_signal | yes | every registered feature must handle a constant epoch without dividing by zero |
| features_declaring_support_handle_short_epoch | yes | a feature that declares a minimum sample count must actually work at that length |

## Feature mathematics

| Field | Value |
| --- | --- |
| status | PASS |
| checks | 15 |
| failed | none |
| check | passed | detail |
| --- | --- | --- |
| wavelet_perfect_reconstruction_db2_2000 | yes | max \|x - idwt(dwt(x))\| = 5.329e-15 (tolerance 1e-9) |
| wavelet_perfect_reconstruction_db2_2001 | yes | max \|x - idwt(dwt(x))\| = 7.105e-15 (tolerance 1e-9) |
| wavelet_perfect_reconstruction_db2_3000 | yes | max \|x - idwt(dwt(x))\| = 7.105e-15 (tolerance 1e-9) |
| wavelet_perfect_reconstruction_db2_3001 | yes | max \|x - idwt(dwt(x))\| = 7.105e-15 (tolerance 1e-9) |
| wavelet_perfect_reconstruction_db4_2000 | yes | max \|x - idwt(dwt(x))\| = 2.842e-14 (tolerance 1e-9) |
| wavelet_perfect_reconstruction_db4_2001 | yes | max \|x - idwt(dwt(x))\| = 2.132e-14 (tolerance 1e-9) |
| wavelet_perfect_reconstruction_db4_3000 | yes | max \|x - idwt(dwt(x))\| = 2.132e-14 (tolerance 1e-9) |
| wavelet_perfect_reconstruction_db4_3001 | yes | max \|x - idwt(dwt(x))\| = 2.487e-14 (tolerance 1e-9) |
| wavelet_energy_conservation | yes | Parseval: signal energy must equal the sum of coefficient energies for an orthonormal basis |
| registry_metadata_complete | yes | every feature declares an equation, motivation, category, family and valid statuses |
| novelty_status_not_overclaimed | yes | no feature marked 'novel-hypothesis' claims to be reproduced from the literature |
| feature_determinism | yes | identical inputs must give bit-identical outputs |
| numerical_stability_degenerate_inputs | yes | all features must return finite values for zero, constant, tiny, huge and spiky inputs |
| sampling_frequency_stability | yes | 28 of 107 features change by >5% between 100 Hz and 200 Hz; these must not be pooled across recordings with different sampling rates |
| declared_expected_ranges | yes | informational: declared expected ranges are heuristic and are reported, not enforced |

## Statistical validation

| Field | Value |
| --- | --- |
| status | PASS |
| checks | 23 |
| failed | none |
| check | passed | detail |
| --- | --- | --- |
| type_i_error_rate_near_alpha | yes | empirical type-I error 0.055 vs alpha=0.05 (Monte-Carlo tolerance 0.033 at 3 SE, n=400) |
| p_values_uniform_under_null | yes | Kolmogorov-Smirnov test against U(0,1) gives p=0.5649 |
| cohens_d_recovers_known_effect | yes | Cohen's d=0.8011 for a population effect of 0.8 |
| hedges_g_recovers_known_effect | yes | Hedges' g=0.8010 |
| rank_biserial_matches_analytic_value | yes | rank-biserial=0.4281 vs analytic 2*Phi(d/sqrt2)-1=0.4284 |
| standardised_effect_is_scale_invariant | yes | multiplying both groups by 10 must not change Cohen's d |
| benjamini_hochberg_matches_reference | yes | BH adjustment matches scipy's false_discovery_control(method='bh') |
| bh_monotone | yes | adjusted p-values must be monotone in the raw p-values |
| adjusted_geq_raw | yes | adjusted p-values must never be smaller than the raw p-values |
| holm_geq_bonferroni | yes | Holm is uniformly less conservative than Bonferroni |
| fdr_controlled_under_global_null | yes | proportion of false discoveries 0.00050 under the global null must stay near alpha=0.05 |
| kappa_one_for_perfect_agreement | yes | kappa=1.000000 |
| kappa_near_zero_for_chance | yes | kappa=-0.043750 on random labels |
| kappa_handles_degenerate_matrix | yes | an all-equal confusion matrix (p_o == p_e) must not produce NaN |
| perfect_prediction_gives_full_metrics | yes | a perfect predictor must score 1.0 on balanced accuracy and macro F1 |
| clustered_permutation_type_i_error | yes | empirical type-I error 0.067 over 60 null datasets (tolerance 0.134 including Monte-Carlo noise) |
| normality_accepts_gaussian_data | yes | D'Agostino-Pearson p=0.6713220259994116 on Gaussian data |
| normality_rejects_skewed_data | yes | D'Agostino-Pearson p=7.266758521558479e-48 on exponential data |
| decision_table_routes_non_normal_to_rank_test | yes | non-normal, unequal variance -> mann_whitney_u |
| decision_table_routes_k_sample | yes | non-normal k-sample -> kruskal_wallis |
| end_to_end_route_on_skewed_groups | yes | automatic selection chose kruskal_wallis |
| dunn_z_matches_closed_form | yes | Dunn's z statistics reproduce the analytic expression with tie correction |
| dunn_returns_all_pairs | yes | three groups must produce three pairwise comparisons |

## Machine-learning validation

| Field | Value |
| --- | --- |
| status | PASS |
| checks | 4 |
| failed | none |
| check | passed | detail |
| --- | --- | --- |
| subject_separation | yes | no subject appears in more than one split |
| reproducible_predictions | yes | two identically seeded models must produce identical predictions |
| no_duplicate_epoch_identities | yes | epoch identities are unique and never span splits |
| non_causal_features_excluded_from_causal_sets | yes | sequence features stay out of the causal ablation |

## Data-leakage audit

| Field | Value |
| --- | --- |
| status | PASS |
| checks | 4 |
| failed | none |
| check | passed | detail |
| --- | --- | --- |
| subject_separation | yes | checks={'no_subject_in_multiple_splits': True, 'subjects_are_partitioned': True, 'every_split_non_empty': True, 'epochs_assigned': True}; overlaps={}; n_epochs=720; n_subjects=6; epochs_per_split={'train': 360, 'validation': 120, 'test': 240}; subjects_per_split={'train': 3, 'validation': 1, 'test': 2}; stage_distribution={'train': {'W': 30, 'N1': 37, 'N2': 164, 'REM': 36, 'N3': 93}, 'validation': {'W': 8, 'N1': 21, 'N2': 57, 'REM': 16, 'N3': 18}, 'test': {'W': 11, 'N1': 38, 'N2': 123, 'REM': 22, 'N3': 46}}; seed=20240517; observed_subjects=['SYN00', 'SYN01', 'SYN02', 'SYN03', 'SYN04', 'SYN05']; declared_subjects=['SYN00', 'SYN01', 'SYN02', 'SYN03', 'SYN04', 'SYN05']; undeclared_subjects=[] |
| duplicate_rows | yes | key=['subject_id', 'recording_id', 'epoch_id']; n_duplicate_rows=0; n_identities_in_multiple_splits=0 |
| feature_selection_scope | yes | n_train_rows_used_for_selection=360; n_total_rows=720; selection_used_only_training_rows=True |
| non_causal_features | yes | non_causal_features=['feature_delta_norm', 'feature_delta_norm_sigma']; leaks={} |

## Evidence and citation integrity

| Field | Value |
| --- | --- |
| status | PASS |
| checks | 4 |
| failed | none |

### claims

| Field | Value |
| --- | --- |
| ok | PASS |
| n_claims | 21 |
| n_verified | 21 |

### citations

| Field | Value |
| --- | --- |
| ok | PASS |
| n_papers | 4 |
| n_incomplete | 0 |
| n_incomplete_unverified | 2 |
| n_unverified_offline | 4 |
| note | records whose evidence_level is unverified-offline/metadata-only were built without live retrieval (or from abstracts alone): their metadata gaps are listed under `incomplete_unverified` and they must never be presented as fully verified sources. Only incomplete records that claim a verified evidence level fail this check. |
| index | title | evidence_level | missing |
| --- | --- | --- | --- |
| 0 | AttnSleep: An Efficient Sleep Staging Model Based on Attention Mechanisms | unverified-offline | year |
| 3 | MNE-Python sleep staging tutorial (clinical tutorial 60) | unverified-offline | year |

### novelty_labelling

| Field | Value |
| --- | --- |
| ok | PASS |
| n_offenders | 0 |
| offenders | none |
| note | only claim descriptions that appear verbatim in the report text are checked; this check is conservative and finding nothing is not proof of careful wording |

### report_files

| Field | Value |
| --- | --- |
| ok | PASS |
| n_checked | 5 |

## What these checks do not establish
- Passing synthetic-signal checks proves the implementation behaves as specified, not that the feature is a sleep biomarker.
- Passing the leakage audit proves the recorded split and selection scopes are sound; it cannot prove that unrecorded preprocessing was also leak-free.
- Passing the claim check proves the report matches the saved artifacts, not that the underlying measurements are correct.
- Citation validation checks structure and traceability. When retrieval was offline, no source was verified against a publisher and the records say so.
