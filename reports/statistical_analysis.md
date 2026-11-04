# Statistical Analysis of EEG Features Across Sleep Stages

> **Data provenance.** This run used a *synthetic* polysomnography surrogate (origin: `synthetic`), not human recordings. Every association reported below is a property of that simulation and must not be read as a statement about human sleep. Re-run with network access to use Sleep-EDF.

## 1. Design and decision rules

| Field | Value |
| --- | --- |
| epochs | 720 |
| subjects | 6 |
| features tested | 109 |
| unit of analysis (primary) | subject x stage means (repeated measures where the design allows) |
| family-wise correction | fdr_bh |
| alpha | 0.0500 |
| stages | W, N1, N2, N3, REM |
| stage counts | N2=344, N3=157, N1=96, REM=74, W=49 |
| minimum group size | 15 |

Test selection is deterministic, not model-chosen: normality is screened with D'Agostino-Pearson (Shapiro-Wilk for small samples), variance homogeneity with Levene, and the resulting decision table selects the test. With repeated measures across stages within the same subject, the primary analysis is Friedman with paired Wilcoxon post-hoc comparisons; the epoch-level Kruskal-Wallis result is reported only to show the effect of pseudo-replication.

## 2. Multiple-comparison correction

| Field | Value |
| --- | --- |
| method | fdr_bh |
| tests | 109 |
| rejected | 103 |
| alpha | 0.0500 |

Both raw and adjusted p-values are stored in `artifacts/tables/feature_stage_comparisons.csv`. Adjusted values are computed across the family of tested features, so the adjusted p-value of a single feature depends on how many features entered this run.

## 3. Stage-wise descriptive statistics

#### abs_power_alpha

| stage | n | missing | mean | median | std | IQR | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W | 49 | 0.0000 | 221.6503 | 220.9837 | 63.1632 | 99.4204 | 120.3481 | 347.1207 |
| N1 | 96 | 0.0000 | 27.5798 | 29.7369 | 8.0570 | 13.1948 | 13.1791 | 62.4930 |
| N2 | 344 | 0.0000 | 14.3127 | 14.6862 | 4.2892 | 7.9513 | 6.9411 | 33.6171 |
| N3 | 157 | 0.0000 | 7.4135 | 7.5336 | 1.9700 | 3.6067 | 3.4501 | 11.3170 |
| REM | 74 | 0.0000 | 53.3388 | 53.8118 | 15.0963 | 20.9518 | 28.1890 | 100.8797 |

#### abs_power_beta

| stage | n | missing | mean | median | std | IQR | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W | 49 | 0.0000 | 73.0809 | 74.0718 | 25.8723 | 51.8586 | 38.5788 | 108.3647 |
| N1 | 96 | 0.0000 | 17.6626 | 18.3161 | 5.1461 | 9.5439 | 8.6566 | 25.3170 |
| N2 | 344 | 0.0000 | 10.3049 | 10.2028 | 3.3480 | 6.1046 | 5.2209 | 17.4683 |
| N3 | 157 | 0.0000 | 3.3155 | 3.6594 | 1.0044 | 1.7195 | 1.5552 | 5.3881 |
| REM | 74 | 0.0000 | 23.7137 | 25.6023 | 6.4180 | 9.5936 | 12.0485 | 38.4953 |

#### abs_power_delta

| stage | n | missing | mean | median | std | IQR | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W | 49 | 0.0000 | 86.9924 | 81.5137 | 26.8568 | 24.3911 | 45.3606 | 170.6202 |
| N1 | 96 | 0.0000 | 126.7173 | 117.8884 | 45.7255 | 54.0524 | 57.0520 | 294.9928 |
| N2 | 344 | 0.0000 | 343.5481 | 329.7455 | 124.0530 | 168.8237 | 123.6650 | 736.5564 |
| N3 | 157 | 0.0000 | 1501.3196 | 1406.3439 | 588.8983 | 811.0119 | 628.1190 | 2650.7143 |
| REM | 74 | 0.0000 | 124.3866 | 122.9429 | 46.0445 | 77.1838 | 41.7538 | 214.1660 |

#### abs_power_gamma

| stage | n | missing | mean | median | std | IQR | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W | 49 | 0.0000 | 13.8933 | 14.0803 | 2.8490 | 4.5896 | 9.3814 | 20.7685 |
| N1 | 96 | 0.0000 | 3.3158 | 3.3319 | 0.8833 | 1.0010 | 1.9652 | 9.2889 |
| N2 | 344 | 0.0000 | 1.3704 | 1.4056 | 0.3639 | 0.4369 | 0.7801 | 5.9804 |
| N3 | 157 | 0.0000 | 0.5141 | 0.5093 | 0.1179 | 0.1446 | 0.2877 | 0.9355 |
| REM | 74 | 0.0000 | 7.3599 | 7.5129 | 1.5244 | 1.1611 | 4.2764 | 15.5376 |

#### abs_power_sigma

| stage | n | missing | mean | median | std | IQR | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W | 49 | 0.0000 | 19.7149 | 18.0662 | 6.4548 | 9.9030 | 11.0551 | 35.3680 |
| N1 | 96 | 0.0000 | 16.2635 | 14.9558 | 5.3819 | 8.3922 | 8.8891 | 29.4032 |
| N2 | 344 | 0.0000 | 112.0996 | 97.8674 | 35.0203 | 61.8232 | 65.3866 | 188.9028 |
| N3 | 157 | 0.0000 | 7.7606 | 8.0448 | 2.7152 | 4.2775 | 3.6853 | 17.3524 |
| REM | 74 | 0.0000 | 15.5501 | 15.4630 | 4.7336 | 7.7849 | 8.1421 | 35.2927 |

#### abs_power_theta

| stage | n | missing | mean | median | std | IQR | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W | 49 | 0.0000 | 54.5934 | 41.3070 | 23.9757 | 35.8390 | 27.8385 | 100.1625 |
| N1 | 96 | 0.0000 | 191.3516 | 131.2008 | 109.3972 | 173.8859 | 73.6219 | 389.8773 |
| N2 | 344 | 0.0000 | 62.8894 | 47.7998 | 31.8535 | 54.7337 | 25.1112 | 131.5160 |
| N3 | 157 | 0.0000 | 90.7895 | 91.2693 | 40.7368 | 75.1390 | 31.9727 | 202.8549 |
| REM | 74 | 0.0000 | 135.4695 | 151.7069 | 65.3686 | 102.0935 | 45.2609 | 308.6149 |

## 4. Inferential comparisons

| feature | n | primary test | statistic | p (raw) | p (FDR) | significant | effect | effect value | epoch-level p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abs_power_alpha | 30 | friedman | 24.0000 | 7.987e-05 | 2.481e-04 | yes | kendall_w | 1.0000 | 3.705e-119 |
| abs_power_beta | 30 | friedman | 24.0000 | 7.987e-05 | 2.481e-04 | yes | kendall_w | 1.0000 | 3.862e-122 |
| abs_power_gamma | 30 | friedman | 24.0000 | 7.987e-05 | 2.481e-04 | yes | kendall_w | 1.0000 | 9.938e-135 |
| abs_power_theta | 30 | friedman | 23.3333 | 1.086e-04 | 2.481e-04 | yes | kendall_w | 0.9722 | 5.768e-50 |
| approximate_entropy | 30 | friedman | 24.0000 | 7.987e-05 | 2.481e-04 | yes | kendall_w | 1.0000 | 9.883e-128 |
| dfa_alpha1 | 30 | friedman | 23.3333 | 1.086e-04 | 2.481e-04 | yes | kendall_w | 0.9722 | 9.669e-119 |
| higuchi_fractal_dimension | 30 | friedman | 24.0000 | 7.987e-05 | 2.481e-04 | yes | kendall_w | 1.0000 | 8.728e-115 |
| hjorth_complexity | 30 | friedman | 23.3333 | 1.086e-04 | 2.481e-04 | yes | kendall_w | 0.9722 | 1.774e-102 |
| hjorth_mobility | 30 | friedman | 23.3333 | 1.086e-04 | 2.481e-04 | yes | kendall_w | 0.9722 | 4.338e-111 |
| line_length | 30 | friedman | 23.3333 | 1.086e-04 | 2.481e-04 | yes | kendall_w | 0.9722 | 3.878e-91 |
| line_length_ratio | 30 | friedman | 23.3333 | 1.086e-04 | 2.481e-04 | yes | kendall_w | 0.9722 | 5.803e-111 |
| log_power_alpha | 30 | friedman | 24.0000 | 7.987e-05 | 2.481e-04 | yes | kendall_w | 1.0000 | 3.705e-119 |
| log_power_beta | 30 | friedman | 24.0000 | 7.987e-05 | 2.481e-04 | yes | kendall_w | 1.0000 | 3.862e-122 |
| log_power_delta | 30 | friedman | 23.3333 | 1.086e-04 | 2.481e-04 | yes | kendall_w | 0.9722 | 3.799e-127 |
| log_power_gamma | 30 | friedman | 24.0000 | 7.987e-05 | 2.481e-04 | yes | kendall_w | 1.0000 | 9.938e-135 |
| log_power_sigma | 30 | friedman | 23.3333 | 1.086e-04 | 2.481e-04 | yes | kendall_w | 0.9722 | 3.361e-129 |
| log_power_theta | 30 | friedman | 23.3333 | 1.086e-04 | 2.481e-04 | yes | kendall_w | 0.9722 | 5.768e-50 |
| mean_abs_derivative | 30 | friedman | 23.3333 | 1.086e-04 | 2.481e-04 | yes | kendall_w | 0.9722 | 3.878e-91 |
| mean_frequency | 30 | friedman | 24.0000 | 7.987e-05 | 2.481e-04 | yes | kendall_w | 1.0000 | 3.546e-114 |
| permutation_entropy | 30 | friedman | 23.3333 | 1.086e-04 | 2.481e-04 | yes | kendall_w | 0.9722 | 6.433e-134 |
| petrosian_fractal_dimension | 30 | friedman | 23.3333 | 1.086e-04 | 2.481e-04 | yes | kendall_w | 0.9722 | 7.104e-134 |
| ratio_alpha_delta | 30 | friedman | 24.0000 | 7.987e-05 | 2.481e-04 | yes | kendall_w | 1.0000 | 2.272e-134 |
| ratio_delta_beta | 30 | friedman | 24.0000 | 7.987e-05 | 2.481e-04 | yes | kendall_w | 1.0000 | 1.982e-133 |
| ratio_delta_theta | 30 | friedman | 24.0000 | 7.987e-05 | 2.481e-04 | yes | kendall_w | 1.0000 | 5.640e-131 |
| ratio_sigma_beta | 30 | friedman | 24.0000 | 7.987e-05 | 2.481e-04 | yes | kendall_w | 1.0000 | 1.672e-134 |

## 5. Assumption checks

| feature | check | statistic | p | assumption met | note |
| --- | --- | --- | --- | --- | --- |
| abs_power_alpha | normality:W | 4.0867 | 0.1296 | yes | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_alpha | normality:N1 | 10.8975 | 0.0043 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_alpha | normality:N2 | 3.6130 | 0.1642 | yes | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_alpha | normality:N3 | 44.9076 | 1.772e-10 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_alpha | normality:REM | 0.7712 | 0.6800 | yes | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_alpha | variance_homogeneity | 257.2170 | 7.934e-137 | no | Brown-Forsythe Levene; H0 = equal variances |
| abs_power_beta | normality:W | 73.9556 | 8.724e-17 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_beta | normality:N1 | 41.9528 | 7.764e-10 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_beta | normality:N2 | 269.3299 | 3.279e-59 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_beta | normality:N3 | 55.5376 | 8.713e-13 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_beta | normality:REM | 5.9913 | 0.0500 | yes | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_beta | variance_homogeneity | 419.3723 | 7.522e-186 | no | Brown-Forsythe Levene; H0 = equal variances |
| abs_power_delta | normality:W | 14.0694 | 8.808e-04 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_delta | normality:N1 | 19.1257 | 7.029e-05 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_delta | normality:N2 | 14.4037 | 7.452e-04 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_delta | normality:N3 | 28.2194 | 7.451e-07 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_delta | normality:REM | 14.3846 | 7.523e-04 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_delta | variance_homogeneity | 173.8268 | 6.072e-104 | no | Brown-Forsythe Levene; H0 = equal variances |
| abs_power_gamma | normality:W | 3.1222 | 0.2099 | yes | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_gamma | normality:N1 | 97.7990 | 5.797e-22 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_gamma | normality:N2 | 448.5160 | 4.036e-98 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_gamma | normality:N3 | 7.0314 | 0.0297 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_gamma | normality:REM | 47.7440 | 4.291e-11 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_gamma | variance_homogeneity | 154.9695 | 1.958e-95 | no | Brown-Forsythe Levene; H0 = equal variances |
| abs_power_sigma | normality:W | 4.1182 | 0.1276 | yes | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_sigma | normality:N1 | 8.9360 | 0.0115 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_sigma | normality:N2 | 73.3204 | 1.199e-16 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_sigma | normality:N3 | 11.0334 | 0.0040 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_sigma | normality:REM | 18.2313 | 1.099e-04 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_sigma | variance_homogeneity | 103.7604 | 1.134e-69 | no | Brown-Forsythe Levene; H0 = equal variances |
| abs_power_theta | normality:W | 9.4903 | 0.0087 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_theta | normality:N1 | 38.3264 | 4.759e-09 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_theta | normality:N2 | 192.1371 | 1.897e-42 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_theta | normality:N3 | 18.9431 | 7.701e-05 | no | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_theta | normality:REM | 5.8596 | 0.0534 | yes | D'Agostino-Pearson; H0 = normally distributed |
| abs_power_theta | variance_homogeneity | 69.4111 | 1.161e-49 | no | Brown-Forsythe Levene; H0 = equal variances |
| amplitude_histogram_entropy | normality:W | 87.2390 | 1.138e-19 | no | D'Agostino-Pearson; H0 = normally distributed |
| amplitude_histogram_entropy | normality:N1 | 92.3651 | 8.774e-21 | no | D'Agostino-Pearson; H0 = normally distributed |
| amplitude_histogram_entropy | normality:N2 | 229.1146 | 1.772e-50 | no | D'Agostino-Pearson; H0 = normally distributed |
| amplitude_histogram_entropy | normality:N3 | 5.0782 | 0.0789 | yes | D'Agostino-Pearson; H0 = normally distributed |
| amplitude_histogram_entropy | normality:REM | 77.1987 | 1.724e-17 | no | D'Agostino-Pearson; H0 = normally distributed |
| amplitude_histogram_entropy | variance_homogeneity | 1.0607 | 0.3750 | yes | Brown-Forsythe Levene; H0 = equal variances |
| approximate_entropy | normality:W | 17.7624 | 1.390e-04 | no | D'Agostino-Pearson; H0 = normally distributed |
| approximate_entropy | normality:N1 | 26.1934 | 2.052e-06 | no | D'Agostino-Pearson; H0 = normally distributed |
| approximate_entropy | normality:N2 | 6.6364 | 0.0362 | no | D'Agostino-Pearson; H0 = normally distributed |
| approximate_entropy | normality:N3 | 2.9320 | 0.2308 | yes | D'Agostino-Pearson; H0 = normally distributed |
| approximate_entropy | normality:REM | 2.9207 | 0.2322 | yes | D'Agostino-Pearson; H0 = normally distributed |
| approximate_entropy | variance_homogeneity | 33.0496 | 2.594e-25 | no | Brown-Forsythe Levene; H0 = equal variances |

## 6. Post-hoc pairwise comparisons

| feature | stages | test | statistic | p | p (FDR) | effect | effect value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| abs_power_alpha | N1 vs N2 | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | 1.0000 |
| abs_power_alpha | N1 vs N3 | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | 1.0000 |
| abs_power_alpha | N1 vs REM | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_alpha | N1 vs W | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_alpha | N2 vs N3 | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | 1.0000 |
| abs_power_alpha | N2 vs REM | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_alpha | N2 vs W | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_alpha | N3 vs REM | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_alpha | N3 vs W | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_alpha | REM vs W | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_beta | N1 vs N2 | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | 1.0000 |
| abs_power_beta | N1 vs N3 | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | 1.0000 |
| abs_power_beta | N1 vs REM | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_beta | N1 vs W | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_beta | N2 vs N3 | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | 1.0000 |
| abs_power_beta | N2 vs REM | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_beta | N2 vs W | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_beta | N3 vs REM | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_beta | N3 vs W | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_beta | REM vs W | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_delta | N1 vs N2 | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_delta | N1 vs N3 | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |
| abs_power_delta | N1 vs REM | wilcoxon_signed_rank | 2.0000 | 0.0938 | 0.3125 | rank_biserial | 0.8095 |
| abs_power_delta | N1 vs W | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | 1.0000 |
| abs_power_delta | N2 vs N3 | wilcoxon_signed_rank | 0.0000 | 0.0312 | 0.3125 | rank_biserial | -1.0000 |

## 7. Order-free association measures

Multiclass labels are never treated as a continuous variable. Reported here are Kruskal-Wallis epsilon-squared and mutual information (nats); both are invariant to how the stage labels happen to be ordered.

| feature | unit | n subjects | epsilon^2 | MI (nats) |
| --- | --- | --- | --- | --- |
| log_power_gamma | subject_stage_mean | 6 | 0.9611 | 1.4350 |
| abs_power_gamma | subject_stage_mean | 6 | 0.9611 | 1.4284 |
| ratio_sigma_beta | subject_stage_mean | 6 | 0.9559 | 1.4164 |
| rel_power_alpha | subject_stage_mean | 6 | 0.9559 | 1.0247 |
| ratio_alpha_delta | subject_stage_mean | 6 | 0.9510 | 1.0662 |
| sample_entropy | subject_stage_mean | 6 | 0.9455 | 1.1468 |
| approximate_entropy | subject_stage_mean | 6 | 0.9455 | 1.1427 |
| wavelet_detail_ratio_1 | subject_stage_mean | 6 | 0.9421 | 1.0967 |
| spectral_flatness | subject_stage_mean | 6 | 0.9403 | 1.1424 |
| rel_power_beta | subject_stage_mean | 6 | 0.9381 | 1.1203 |
| rel_power_gamma | subject_stage_mean | 6 | 0.9321 | 0.7700 |
| ratio_delta_beta | subject_stage_mean | 6 | 0.9310 | 0.7767 |
| log_power_alpha | subject_stage_mean | 6 | 0.9244 | 1.0054 |
| abs_power_alpha | subject_stage_mean | 6 | 0.9244 | 0.9630 |
| spectral_entropy | subject_stage_mean | 6 | 0.9161 | 1.0548 |
| wavelet_detail_ratio_3 | subject_stage_mean | 6 | 0.9154 | 1.1960 |
| rel_power_delta | subject_stage_mean | 6 | 0.9132 | 1.1139 |
| wavelet_detail_ratio_5 | subject_stage_mean | 6 | 0.9132 | 1.0294 |
| spectral_concentration_a2 | subject_stage_mean | 6 | 0.9041 | 1.0412 |
| permutation_entropy | subject_stage_mean | 6 | 0.9034 | 0.9914 |
| petrosian_fractal_dimension | subject_stage_mean | 6 | 0.9034 | 0.9747 |
| wavelet_detail_ratio_2 | subject_stage_mean | 6 | 0.9028 | 0.9331 |
| spectral_rolloff_85 | subject_stage_mean | 6 | 0.8982 | 1.0313 |
| wavelet_detail_energy_1 | subject_stage_mean | 6 | 0.8954 | 0.9332 |
| spectral_concentration_a3 | subject_stage_mean | 6 | 0.8894 | 0.9389 |

## 8. Feature selection and interactions (training subjects only)

| Field | Value |
| --- | --- |
| training rows | 360 |
| training subjects | 3 |
| candidate features | 109 |
| redundancy clusters | 47 |
| features dropped as redundant | abs_power_alpha, abs_power_beta, abs_power_delta, abs_power_gamma, abs_power_sigma, abs_power_theta, amplitude_histogram_entropy, approximate_entropy, crest_factor, cross_band_entropy, dfa_alpha1, dfa_alpha2, higuchi_fractal_dimension, hjorth_activity, hjorth_complexity, hjorth_mobility, iqr_amplitude, kcomplex_count, kcomplex_occupancy, line_length, line_length_ratio, log_power_alpha, log_power_beta, log_power_delta, log_power_gamma, log_power_sigma, log_power_theta, log_total_power, mad_amplitude, mean_abs_derivative, mean_frequency, peak_to_peak, permutation_entropy, petrosian_fractal_dimension, ratio_alpha_delta, ratio_delta_beta, ratio_delta_theta, ratio_sigma_beta, ratio_sigma_delta, ratio_theta_alpha, rel_power_alpha, rel_power_beta, rel_power_delta, rel_power_gamma, rel_power_sigma, rel_power_theta, rms_amplitude, sample_entropy, slow_wave_amplitude_mean, slow_wave_count, slow_wave_occupancy, slow_wave_slope_mean, spectral_asymmetry_index, spectral_centroid, spectral_concentration_a2, spectral_concentration_a3, spectral_edge_95, spectral_entropy, spectral_flatness, spectral_kurtosis, spectral_rolloff_85, spectral_skewness, spectral_spread, spindle_count, spindle_density, spindle_duration_mean, spindle_frequency_mean, std_amplitude, total_power, var_amplitude, wavelet_detail_energy_1, wavelet_detail_energy_2, wavelet_detail_energy_3, wavelet_detail_energy_4, wavelet_detail_energy_5, wavelet_detail_ratio_1, wavelet_detail_ratio_2, wavelet_detail_ratio_3, wavelet_detail_ratio_4, wavelet_detail_ratio_5, wavelet_energy_entropy, wavelet_energy_slope, wavelet_low_high_ratio, zero_crossing_rate, zero_crossing_rate_hz |
| RFE selection | rel_power_alpha, rel_power_gamma, spectral_flatness, log_power_gamma, abs_power_gamma, ratio_sigma_beta, petrosian_fractal_dimension, ratio_alpha_delta, permutation_entropy, spectral_rolloff_85, dfa_alpha1, wavelet_detail_ratio_3, spectral_asymmetry_index, ratio_theta_alpha, abs_power_alpha, log_power_delta, spectral_edge_95, wavelet_detail_ratio_2, rel_power_delta, ratio_delta_theta |

### Top mutual-information features

| feature | MI (nats) |
| --- | --- |
| rel_power_alpha | 1.3846 |
| rel_power_gamma | 1.3820 |
| spectral_flatness | 1.3646 |
| log_power_gamma | 1.3571 |
| abs_power_gamma | 1.3570 |
| ratio_sigma_beta | 1.3545 |
| wavelet_detail_ratio_1 | 1.3451 |
| rel_power_beta | 1.3385 |
| petrosian_fractal_dimension | 1.3153 |
| ratio_alpha_delta | 1.3133 |
| log_power_beta | 1.3070 |
| abs_power_beta | 1.3066 |
| permutation_entropy | 1.3053 |
| spectral_rolloff_85 | 1.2733 |
| ratio_delta_beta | 1.2630 |

### Interaction candidates (exploratory)

| pair | role | best parent MI | CV MI gain | passes screen |
| --- | --- | --- | --- | --- |
| interaction::rel_power_gamma*spectral_flatness | exploratory | 1.3820 | -0.0155 | no |
| interaction::rel_power_alpha*rel_power_gamma | exploratory | 1.3846 | -0.0177 | no |
| interaction::rel_power_alpha*spectral_flatness | exploratory | 1.3846 | -0.0209 | no |
| interaction::rel_power_alpha*abs_power_gamma | exploratory | 1.3846 | -0.0214 | no |
| interaction::rel_power_gamma*abs_power_gamma | exploratory | 1.3820 | -0.0217 | no |
| interaction::rel_power_gamma*log_power_gamma | exploratory | 1.3820 | -0.0322 | no |
| interaction::rel_power_alpha*log_power_gamma | exploratory | 1.3846 | -0.0682 | no |
| interaction::spectral_flatness*abs_power_gamma | exploratory | 1.3646 | -0.0873 | no |
| interaction::rel_power_alpha/ratio_sigma_beta | exploratory | 1.3846 | -0.0927 | no |
| interaction::rel_power_gamma/ratio_sigma_beta | exploratory | 1.3820 | -0.1238 | no |
| interaction::spectral_flatness/ratio_sigma_beta | exploratory | 1.3646 | -0.1307 | no |
| interaction::spectral_flatness*log_power_gamma | exploratory | 1.3646 | -0.1488 | no |
| interaction::rel_power_gamma*ratio_sigma_beta | exploratory | 1.3820 | -0.3451 | no |
| interaction::rel_power_alpha*ratio_sigma_beta | exploratory | 1.3846 | -0.3651 | no |
| interaction::rel_power_gamma/spectral_flatness | exploratory | 1.3820 | -0.3765 | no |

Interaction search is bounded to a prespecified candidate list and is scored by cross-validated mutual-information gain; any surviving pair is labelled *exploratory* until it is confirmed on held-out subjects in the predictive stage.

## 9. Figures
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
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/model_comparison_test.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/stage_distribution.png`
- `/Users/faraz/agentic-feature-exploration/artifacts/figures/subject_level_abs_power_alpha.png`

## 10. Interpretation limits
- A significant association is not a causal marker and is not by itself evidence of predictive value.
- The epoch-level p-values are pseudo-replicated and are strongly anti-conservative; use the primary analysis.
- Monte-Carlo p-values (permutation, bootstrap) have a floor of 1/(n_iterations+1); a p-value at that floor means `at least this small`.
- Effect sizes are reported with the test statistic because p-values depend on sample size.
- Statistical results here describe this corpus (synthetic); they do not transfer automatically.
