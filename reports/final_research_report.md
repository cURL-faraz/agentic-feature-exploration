# Single-Channel EEG Feature Discovery for Sleep Staging

> **Data provenance.** This run used a *synthetic* polysomnography surrogate (origin: `synthetic`), not human recordings. Every association reported below is a property of that simulation and must not be read as a statement about human sleep. Re-run with network access to use Sleep-EDF.

## 1. Executive Summary
This report is the output of an autonomous research pipeline over a single EEG derivation. It (i) assembles a corpus and maps its sleep-stage annotations, (ii) segments it into 30 s epochs, (iii) extracts a registered catalog of spectral, temporal, nonlinear, event-based and novel mathematical features, (iv) compares each feature across stages with repeated-measures-aware statistics and FDR correction, (v) evaluates whether the features predict stage on *unseen subjects*, and (vi) verifies the implementations on synthetic signals with known properties.

| Field | Value |
| --- | --- |
| research question | Which single-channel EEG characteristics are statistically associated with sleep stages, and which of them improve subject-independent stage prediction? |
| data origin | synthetic |
| subjects | 6 |
| epochs | 720 |
| derivation | EEG Fpz-Cz |
| registered features in the catalog | 107 |
| feature columns analysed in this run | 109 |
| significance after FDR | 103 |
| alpha / correction | 0.05 / fdr_bh |
| primary analysis unit | subject x stage means (repeated measures where the design allows) |
| best held-out model | svm on top_k_mutual_information |
| verification | PASS |

The analysed table is the registered catalog plus 2 derived column(s) computed from *adjacent* epochs (`feature_delta_norm`, `feature_delta_norm_sigma`). Those are flagged non-causal and are excluded from every causal ablation; they must not be read as per-epoch markers.

## 2. Dataset

| Field | Value |
| --- | --- |
| dataset | sleep-edf-expanded |
| source | n/a |
| origin | synthetic |
| subjects | 6 |
| recordings | 6 |
| derivation | EEG Fpz-Cz |
| sampling frequency (Hz) | 100.0000 |
| epoch duration (s) | 30.0000 |
| available stages | W, N1, N2, N3, REM |
| class distribution | W=49, N1=96, N2=344, N3=157, REM=74 |

Quality control:

| Field | Value |
| --- | --- |
| recording_count | 6 |
| channel_consistency | yes |
| sampling_frequency_consistent | yes |
| mean_std_uv | 31.3808 |
| max_abs_step_uv | 372.6426 |
| fraction_flat_samples | 7.407e-05 |

Epoch construction keeps a subject/recording/epoch identity triple and the recording timestamp for every epoch, so any epoch can be traced back to its source. Epochs that failed quality gates are recorded with their reason rather than silently dropped.

## 3. Literature Review
4 bibliographic record(s) are available in this run (retrieval mode: `offline`). See `reports/literature_review.md` for the full review, the planned search queries, and an explicit statement of what could not be verified. No numeric claim in this report attributes a value to a published author.

## 4. Feature Catalog

| Field | Value |
| --- | --- |
| registered features | 107 |
| established | 83 |
| literature-derived | 16 |
| novel hypotheses | 8 |
| families | spectral=32, temporal=19, nonlinear=7, novel=12, events=17, literature_derived=7, wavelet=13 |

Formulas, motivations, units and validation status for every feature are in `reports/feature_catalog.md`. Novel features are labelled there as *novel hypothesis - not a validated finding*.

## 5. Feature Distributions
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_all_features_gradient_boosting.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_all_features_logistic_regression.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_all_features_majority.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_all_features_random_forest.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_all_features_svm.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_conventional_only_gradient_boosting.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_conventional_only_logistic_regression.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_conventional_only_majority.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_conventional_only_random_forest.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_conventional_only_svm.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_literature_derived_only_gradient_boosting.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_literature_derived_only_logistic_regression.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_literature_derived_only_majority.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_literature_derived_only_random_forest.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_literature_derived_only_svm.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_no_events_gradient_boosting.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_no_events_logistic_regression.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_no_events_majority.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_no_events_random_forest.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_no_events_svm.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_no_spectral_gradient_boosting.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_no_spectral_logistic_regression.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_no_spectral_majority.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_no_spectral_random_forest.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_no_spectral_svm.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_novel_only_gradient_boosting.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_novel_only_logistic_regression.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_novel_only_majority.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_novel_only_random_forest.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_novel_only_svm.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_top_k_mutual_information_gradient_boosting.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_top_k_mutual_information_logistic_regression.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_top_k_mutual_information_majority.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_top_k_mutual_information_random_forest.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/confusion_top_k_mutual_information_svm.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_abs_power_alpha.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_abs_power_beta.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_abs_power_gamma.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_abs_power_theta.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_approximate_entropy.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_correlation.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_dfa_alpha1.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_distributions.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_higuchi_fractal_dimension.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_hjorth_complexity.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_hjorth_mobility.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_line_length.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_line_length_ratio.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_log_power_alpha.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_log_power_beta.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_log_power_gamma.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_ratio_alpha_delta.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_ratio_delta_beta.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_ratio_sigma_beta.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_rel_power_alpha.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_rel_power_beta.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/feature_rel_power_gamma.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/model_comparison_test.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/stage_distribution.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/subject_level_abs_power_alpha.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/subject_level_abs_power_beta.png`

## 6. Statistical Comparisons
Full tables, assumption checks and post-hoc results are in `reports/statistical_analysis.md`. The headline comparison is repeated-measures (Friedman) on subject x stage means, with paired Wilcoxon post-hoc tests and fdr_bh correction across the 109 tested features.

| feature | primary test | p (raw) | p (FDR) | effect | effect value |
| --- | --- | --- | --- | --- | --- |
| abs_power_alpha | friedman | 7.987e-05 | 2.481e-04 | kendall_w | 1.0000 |
| abs_power_beta | friedman | 7.987e-05 | 2.481e-04 | kendall_w | 1.0000 |
| abs_power_delta | friedman | 1.569e-04 | 2.515e-04 | kendall_w | 0.9389 |
| abs_power_gamma | friedman | 7.987e-05 | 2.481e-04 | kendall_w | 1.0000 |
| abs_power_sigma | friedman | 1.569e-04 | 2.515e-04 | kendall_w | 0.9389 |
| abs_power_theta | friedman | 1.086e-04 | 2.481e-04 | kendall_w | 0.9722 |
| amplitude_histogram_entropy | friedman | 0.0048 | 0.0055 | kendall_w | 0.6222 |
| approximate_entropy | friedman | 7.987e-05 | 2.481e-04 | kendall_w | 1.0000 |
| band_power_instability_beta | friedman | 6.758e-04 | 8.468e-04 | kendall_w | 0.8056 |
| band_power_instability_delta | friedman | 0.0051 | 0.0058 | kendall_w | 0.6167 |
| band_power_instability_sigma | friedman | 0.0628 | 0.0646 | kendall_w | 0.3722 |
| band_power_variability_alpha | friedman | 0.0164 | 0.0177 | kendall_w | 0.5056 |
| band_power_variability_delta | friedman | 0.0065 | 0.0071 | kendall_w | 0.5944 |
| band_power_variability_sigma | friedman | 0.0082 | 0.0089 | kendall_w | 0.5722 |
| crest_factor | friedman | 0.0700 | 0.0713 | kendall_w | 0.3611 |

## 7. Predictive Evaluation
Models are fitted on training subjects, validated on a disjoint validation subject set, and reported once on held-out test subjects. Scaling, imputation and any supervised feature selection are fitted inside the training folds only. Epoch-level random splits are never used.

| feature set | classifier | features | balanced accuracy | accuracy | macro F1 | kappa | F1 N1 | F1 N2 | F1 REM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_features | logistic_regression | 107 | 0.7998 | 0.8417 | 0.7155 | 0.7661 | 0.1000 | 0.9959 | 0.5250 |
| all_features | gradient_boosting | 107 | 0.7818 | 0.8333 | 0.8542 | 0.7534 | n/a | 1.0000 | 0.5000 |
| all_features | svm | 107 | 0.7711 | 0.8292 | 0.8528 | 0.7478 | n/a | 0.9959 | 0.5060 |
| all_features | random_forest | 107 | 0.7689 | 0.8292 | 0.6730 | 0.7473 | 0.0513 | 1.0000 | 0.4675 |
| all_features | majority | 107 | 0.2000 | 0.5125 | 0.6777 | 0.0000 | n/a | 0.6777 | n/a |
| conventional_only | logistic_regression | 83 | 0.7764 | 0.8292 | 0.6816 | 0.7479 | 0.0513 | 0.9959 | 0.4810 |
| conventional_only | svm | 83 | 0.7711 | 0.8292 | 0.8528 | 0.7478 | n/a | 0.9959 | 0.5060 |
| conventional_only | gradient_boosting | 83 | 0.7545 | 0.8208 | 0.8141 | 0.7352 | n/a | 1.0000 | 0.4416 |
| conventional_only | random_forest | 83 | 0.7234 | 0.8083 | 0.6244 | 0.7169 | 0.0513 | 1.0000 | 0.3611 |
| conventional_only | majority | 83 | 0.2000 | 0.5125 | 0.6777 | 0.0000 | n/a | 0.6777 | n/a |
| literature_derived_only | logistic_regression | 16 | 0.8103 | 0.8500 | 0.7363 | 0.7782 | 0.1905 | 0.9959 | 0.5385 |
| literature_derived_only | gradient_boosting | 16 | 0.8062 | 0.8708 | 0.7548 | 0.8081 | 0.5660 | 1.0000 | 0.4746 |
| literature_derived_only | random_forest | 16 | 0.7802 | 0.8292 | 0.8516 | 0.7479 | n/a | 0.9959 | 0.4938 |
| literature_derived_only | svm | 16 | 0.7778 | 0.8375 | 0.7020 | 0.7596 | 0.1463 | 0.9959 | 0.5195 |
| literature_derived_only | majority | 16 | 0.2000 | 0.5125 | 0.6777 | 0.0000 | n/a | 0.6777 | n/a |
| no_events | logistic_regression | 90 | 0.7998 | 0.8417 | 0.7155 | 0.7661 | 0.1000 | 0.9959 | 0.5250 |
| no_events | gradient_boosting | 90 | 0.7802 | 0.8292 | 0.8516 | 0.7479 | n/a | 0.9959 | 0.4938 |
| no_events | svm | 90 | 0.7711 | 0.8292 | 0.8528 | 0.7478 | n/a | 0.9959 | 0.5060 |
| no_events | random_forest | 90 | 0.7598 | 0.8250 | 0.6627 | 0.7412 | 0.0513 | 1.0000 | 0.4474 |
| no_events | majority | 90 | 0.2000 | 0.5125 | 0.6777 | 0.0000 | n/a | 0.6777 | n/a |
| no_spectral | logistic_regression | 75 | 0.8089 | 0.8458 | 0.7205 | 0.7723 | 0.1000 | 0.9959 | 0.5500 |
| no_spectral | gradient_boosting | 75 | 0.7802 | 0.8292 | 0.8516 | 0.7479 | n/a | 0.9959 | 0.4938 |
| no_spectral | svm | 75 | 0.7727 | 0.8333 | 0.8559 | 0.7527 | n/a | 0.9960 | 0.5185 |
| no_spectral | random_forest | 75 | 0.7545 | 0.8208 | 0.8141 | 0.7352 | n/a | 1.0000 | 0.4416 |
| no_spectral | majority | 75 | 0.2000 | 0.5125 | 0.6777 | 0.0000 | n/a | 0.6777 | n/a |
| novel_only | logistic_regression | 8 | 0.5354 | 0.6958 | 0.4869 | 0.5644 | 0.0377 | 0.9270 | 0.2373 |
| novel_only | svm | 8 | 0.5073 | 0.6708 | 0.4829 | 0.5319 | 0.0937 | 0.8860 | 0.2807 |
| novel_only | random_forest | 8 | 0.4835 | 0.6958 | 0.4601 | 0.5529 | 0.0755 | 0.9218 | 0.1569 |
| novel_only | gradient_boosting | 8 | 0.4425 | 0.6667 | 0.5155 | 0.5113 | n/a | 0.9016 | 0.0513 |
| novel_only | majority | 8 | 0.2000 | 0.5125 | 0.6777 | 0.0000 | n/a | 0.6777 | n/a |
| top_k_mutual_information | svm | 40 | 0.8105 | 0.8500 | 0.7300 | 0.7778 | 0.1000 | 1.0000 | 0.5500 |
| top_k_mutual_information | logistic_regression | 40 | 0.8067 | 0.8500 | 0.7283 | 0.7778 | 0.1463 | 1.0000 | 0.5385 |
| top_k_mutual_information | random_forest | 40 | 0.7764 | 0.8292 | 0.6816 | 0.7479 | 0.0513 | 0.9959 | 0.4810 |
| top_k_mutual_information | gradient_boosting | 40 | 0.7711 | 0.8250 | 0.8377 | 0.7419 | n/a | 0.9959 | 0.4750 |
| top_k_mutual_information | majority | 40 | 0.2000 | 0.5125 | 0.6777 | 0.0000 | n/a | 0.6777 | n/a |

### Subject-grouped cross-validation on training subjects

| classifier | features | folds | balanced accuracy | std | macro F1 |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 107 | 3 | n/a | n/a | n/a |
| random_forest | 107 | 3 | n/a | n/a | n/a |

## 8. Feature Interactions

| pair | role | CV MI gain | passes screen |
| --- | --- | --- | --- |
| interaction::rel_power_gamma*spectral_flatness | exploratory | -0.0155 | no |
| interaction::rel_power_alpha*rel_power_gamma | exploratory | -0.0177 | no |
| interaction::rel_power_alpha*spectral_flatness | exploratory | -0.0209 | no |
| interaction::rel_power_alpha*abs_power_gamma | exploratory | -0.0214 | no |
| interaction::rel_power_gamma*abs_power_gamma | exploratory | -0.0217 | no |
| interaction::rel_power_gamma*log_power_gamma | exploratory | -0.0322 | no |
| interaction::rel_power_alpha*log_power_gamma | exploratory | -0.0682 | no |
| interaction::spectral_flatness*abs_power_gamma | exploratory | -0.0873 | no |
| interaction::rel_power_alpha/ratio_sigma_beta | exploratory | -0.0927 | no |
| interaction::rel_power_gamma/ratio_sigma_beta | exploratory | -0.1238 | no |

Interaction search is bounded to a prespecified candidate list and scored with cross-validated mutual-information gain, so it is a controlled search rather than a scan over arbitrary combinations. Surviving candidates remain *exploratory* until they are confirmed on held-out subjects.

## 9. Verification

| Field | Value |
| --- | --- |
| verification verdict | PASS |
| checks | 83 |
| failed checks | 0 |
| failed sections | none |

Synthetic signals with known properties (pure tones, dual tones, white and pink noise, amplitude- and frequency-modulated signals, spindle bursts, slow waves, artefacts) are used to check that features respond as specified. Statistical routines are calibrated on null data. Splits and selection scopes are audited for leakage. Details: `reports/validation_report.md`.

## 10. Discussion
Of 109 features compared across stages, 103 survive FDR correction at alpha=0.05. Surviving features cluster by family as follows: events (17), literature_derived (7), nonlinear (7), novel (10), spectral (32), temporal (16), unknown (1), wavelet (13).

Because the classes are heavily imbalanced, the meaningful comparison is not overall accuracy but balanced accuracy, macro F1 and per-stage F1 - particularly for N1, which is the stage most often merged with its neighbours.

The best held-out model was `svm` on the `top_k_mutual_information` feature set, with balanced accuracy 0.8105 and macro F1 0.7300.

Statistically associated features that do not improve held-out subject-independent prediction are reported as associations only; the ablation table in section 7 is the evidence for predictive usefulness, and no feature is described as predictively useful without it.

## 11. Limitations
- Single derivation only: without EOG and EMG, REM and N1 are expected to be the hardest stages.
- Sleep-stage labels are assumed correct; inter-scorer disagreement is not modelled, and the annotation protocol defines the label semantics.
- Class imbalance is severe; balanced accuracy and per-stage F1 are reported instead of accuracy alone.
- Association, predictive value, biological plausibility and causality are four different questions; this report answers the first two and does not claim the others.
- Multiple comparisons are controlled, so reported significance means `significant after correction across the tested family`, not `significant in isolation`.
- Monte-Carlo p-values cannot go below 1/(n_iterations+1).
- Results are specific to this corpus and segmentation; transfer to another dataset, montage or population is untested.
- This run used synthetic data, so no statement here describes human sleep.

## 12. Future Work
- Compare against learned representations (AttnSleep, BIOT) using the same subject-grouped splits.
- Fuse the extracted feature table with deep-learning embeddings and test whether the features add information beyond the embedding.
- Add domain adaptation / cross-dataset evaluation (Sleep-EDF vs SHHS) to test transfer.
- Improve event detection (spindle, K-complex) with validated detectors and re-test the event family.
- Model label uncertainty (multi-scorer agreement) instead of treating annotations as ground truth.
- Extend the interaction search to a prespecified structural family and confirm on held-out subjects.

## 13. References

| title | year | identifier | type | evidence level |
| --- | --- | --- | --- | --- |
| AttnSleep: An Efficient Sleep Staging Model Based on Attention Mechanisms | n/a | https://github.com/emadeldeen24/AttnSleep | url | unverified-offline |
| BIOT: Biosignal Transformer for Cross-data Learning in the Wild | 2023 | https://arxiv.org/abs/2305.10351 | arxiv | unverified-offline |
| Sleep-EDF Expanded (version 1.0.0) | 2013 | https://physionet.org/content/sleep-edfx/1.0.0/ | url | unverified-offline |
| MNE-Python sleep staging tutorial (clinical tutorial 60) | n/a | https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html | url | unverified-offline |

Records marked `unverified-offline` were **not retrieved** in this run; their metadata was not checked against a publisher and they are listed as pointers for a networked run, not as verified citations.

## Provenance
Every number in this report is registered in `artifacts/tables/claims.json` with the artifact file and locator it came from, and the verification stage re-reads those artifacts to confirm the report matches the computation.
