# Feature Catalog

> **Data provenance.** This run used a *synthetic* polysomnography surrogate (origin: `synthetic`), not human recordings. Every association reported below is a property of that simulation and must not be read as a statement about human sleep. Re-run with network access to use Sleep-EDF.

## Composition

| Field | Value |
| --- | --- |
| registered features | 107 |
| by novelty status | established=83, literature-derived=16, novel-hypothesis=8 |
| by family | spectral=32, temporal=19, nonlinear=7, novel=12, events=17, literature_derived=7, wavelet=13 |
| by validation status | reproduced-from-literature=43, unit-tested=47, synthetic-validated=17 |

## How validation status is assigned
- `unit-tested` - the implementation has a dedicated unit test.
- `synthetic-validated` - behaviour verified on synthetic signals with known properties (amplitude scaling, sampling-frequency sensitivity, expected range, numerical stability).
- `reproduced-from-literature` - the definition follows a published definition and the implementation reproduces the documented qualitative behaviour.
- `unvalidated` - implemented and exercised, but no dedicated validation evidence yet.
Validation status describes the *implementation*, never the *usefulness* of a feature. A synthetic-validated feature can still have no predictive value.

## Category A and B - established and literature-derived features

| feature | name | family | definition | units | cost | novelty | validation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| abs_power_alpha | Absolute alpha power | spectral | P_{alpha}=\int_{f_1}^{f_2}\mathrm{PSD}(f)\,\mathrm{d}f\quad[\mu V^2] | uV^2 | low | established | reproduced-from-literature |
| abs_power_beta | Absolute beta power | spectral | P_{beta}=\int_{f_1}^{f_2}\mathrm{PSD}(f)\,\mathrm{d}f\quad[\mu V^2] | uV^2 | low | established | reproduced-from-literature |
| abs_power_delta | Absolute delta power | spectral | P_{delta}=\int_{f_1}^{f_2}\mathrm{PSD}(f)\,\mathrm{d}f\quad[\mu V^2] | uV^2 | low | established | reproduced-from-literature |
| abs_power_gamma | Absolute gamma power | spectral | P_{gamma}=\int_{f_1}^{f_2}\mathrm{PSD}(f)\,\mathrm{d}f\quad[\mu V^2] | uV^2 | low | established | reproduced-from-literature |
| abs_power_sigma | Absolute sigma power | spectral | P_{sigma}=\int_{f_1}^{f_2}\mathrm{PSD}(f)\,\mathrm{d}f\quad[\mu V^2] | uV^2 | low | established | reproduced-from-literature |
| abs_power_theta | Absolute theta power | spectral | P_{theta}=\int_{f_1}^{f_2}\mathrm{PSD}(f)\,\mathrm{d}f\quad[\mu V^2] | uV^2 | low | established | reproduced-from-literature |
| amplitude_histogram_entropy | Amplitude distribution entropy | temporal | H_a=-\frac{1}{\ln B}\sum_b \hat p_b\ln\hat p_b,\quad \hat p_b=\text{histogram} | normalised | low | literature-derived | unit-tested |
| approximate_entropy | Approximate entropy | nonlinear | \mathrm{ApEn}(m,r)=\phi^m(r)-\phi^{m+1}(r),\ \phi^m=\frac{1}{N}\sum_i\ln C_i^m(r) | nats | high | established | reproduced-from-literature |
| band_power_variability_alpha | Alpha power variability (CV) | novel | \mathrm{CV}_{alpha}=\sigma(P_t)/\bar P_t | dimensionless | medium | literature-derived | unit-tested |
| band_power_variability_delta | Delta power variability (CV) | novel | \mathrm{CV}_{delta}=\sigma(P_t)/\bar P_t | dimensionless | medium | literature-derived | unit-tested |
| band_power_variability_sigma | Sigma power variability (CV) | novel | \mathrm{CV}_{sigma}=\sigma(P_t)/\bar P_t | dimensionless | medium | literature-derived | unit-tested |
| crest_factor | Crest factor | temporal | CF=\max\|x\|/x_{rms} | dimensionless | low | established | unit-tested |
| delta_sigma_coupling | Delta-sigma phase-amplitude coupling proxy | novel | \mathrm{MVL}=\left\|\frac1N\sum_t a_\sigma(t)e^{i\varphi_\delta(t)}\right\|\big/\overline{a_\sigma} | dimensionless | medium | literature-derived | unit-tested |
| dfa_alpha1 | DFA scaling exponent (short range) | nonlinear | F(s)\propto s^{\alpha_1},\quad \alpha_1=\text{slope for } 4\le s<16 | dimensionless | medium | established | reproduced-from-literature |
| dfa_alpha2 | DFA scaling exponent (long range) | nonlinear | \alpha_2=\text{slope of }\ln F(s)\ \text{vs}\ \ln s\ \text{for } s\ge16 | dimensionless | medium | established | reproduced-from-literature |
| higuchi_fractal_dimension | Higuchi fractal dimension | nonlinear | D_H=-\text{slope of }\ln L(k)\ \text{vs}\ \ln k | dimensionless | medium | established | reproduced-from-literature |
| hjorth_activity | Hjorth activity | temporal | A=\mathrm{Var}(x) | uV^2 | low | established | reproduced-from-literature |
| hjorth_complexity | Hjorth complexity | temporal | C=M(\dot x)/M(x) | dimensionless | low | established | reproduced-from-literature |
| hjorth_mobility | Hjorth mobility | temporal | M=\sqrt{\mathrm{Var}(\dot x)/\mathrm{Var}(x)} | dimensionless | low | established | reproduced-from-literature |
| iqr_amplitude | Interquartile range of amplitude | temporal | \mathrm{IQR}=Q_{0.75}-Q_{0.25} | uV | low | established | reproduced-from-literature |
| kcomplex_amplitude_mean | K-complex amplitude mean | events | \overline{A}_{KC} | uV | medium | established | synthetic-validated |
| kcomplex_count | K-complex count | events | N_{KC} | count | medium | established | synthetic-validated |
| kcomplex_density | K-complex density | events | \rho_{KC}=N_{KC}/T | 1/min | medium | established | synthetic-validated |
| kcomplex_duration_mean | K-complex duration mean | events | \overline{d}_{KC} | s | medium | established | synthetic-validated |
| kcomplex_occupancy | K-complex occupancy | events | o_{KC}=\sum d / T | fraction | medium | established | synthetic-validated |
| kurtosis | Amplitude excess kurtosis | temporal | \gamma_2=\mathrm{E}\left[(x-\mu)^4\right]/\sigma^4-3 | dimensionless | low | established | reproduced-from-literature |
| line_length | Line length | temporal | L=\sum_{i=1}^{N-1}\|x_{i+1}-x_i\| | uV | low | established | reproduced-from-literature |
| line_length_ratio | Normalised line length | temporal | L_{norm}=L\,/\,\left(N\sigma_x\right) | dimensionless | low | established | unit-tested |
| log_power_alpha | Log absolute alpha power | spectral | \ln\!\left(P_{alpha}+\epsilon\right),\ \epsilon=10^{-12} | ln(uV^2) | low | established | unit-tested |
| log_power_beta | Log absolute beta power | spectral | \ln\!\left(P_{beta}+\epsilon\right),\ \epsilon=10^{-12} | ln(uV^2) | low | established | unit-tested |
| log_power_delta | Log absolute delta power | spectral | \ln\!\left(P_{delta}+\epsilon\right),\ \epsilon=10^{-12} | ln(uV^2) | low | established | unit-tested |
| log_power_gamma | Log absolute gamma power | spectral | \ln\!\left(P_{gamma}+\epsilon\right),\ \epsilon=10^{-12} | ln(uV^2) | low | established | unit-tested |
| log_power_sigma | Log absolute sigma power | spectral | \ln\!\left(P_{sigma}+\epsilon\right),\ \epsilon=10^{-12} | ln(uV^2) | low | established | unit-tested |
| log_power_theta | Log absolute theta power | spectral | \ln\!\left(P_{theta}+\epsilon\right),\ \epsilon=10^{-12} | ln(uV^2) | low | established | unit-tested |
| log_total_power | Log total power | spectral | \ln(P_{tot}+\epsilon) | ln(uV^2) | low | established | unit-tested |
| mad_amplitude | Median absolute deviation of amplitude | temporal | \mathrm{MAD}=\mathrm{median}_i\|x_i-\mathrm{median}(x)\| | uV | low | established | unit-tested |
| mean_abs_derivative | Mean absolute first derivative | temporal | \overline{\|\dot x\|}=\frac{1}{N-1}\sum_i\|x_{i+1}-x_i\| | uV/sample | low | established | unit-tested |
| mean_amplitude | Mean amplitude | temporal | \bar{x}=\frac{1}{N}\sum_i x_i | uV | low | established | unit-tested |
| mean_frequency | Mean frequency (amplitude weighted) | spectral | f_{mean}=\sum_k f_k\,\mathrm{PSD}(f_k)\big/\sum_k\mathrm{PSD}(f_k) | Hz | low | established | unit-tested |
| peak_frequency | Peak frequency | spectral | f_{peak}=\arg\max_f p(f) | Hz | low | established | reproduced-from-literature |
| peak_to_peak | Peak-to-peak amplitude | temporal | A_{pp}=\max_i x_i-\min_i x_i | uV | low | established | reproduced-from-literature |
| permutation_entropy | Permutation entropy | nonlinear | H_p=-\frac{1}{\ln m!}\sum_\pi p(\pi)\ln p(\pi) | normalised | medium | established | reproduced-from-literature |
| petrosian_fractal_dimension | Petrosian fractal dimension | nonlinear | D_P=\frac{\log_{10}N}{\log_{10}N+\log_{10}\left(N/(N+0.4N_\Delta)\right)} | dimensionless | low | established | reproduced-from-literature |
| ratio_alpha_delta | Spectral ratio alpha/delta | literature_derived | P_\alpha/P_\delta | dimensionless | low | literature-derived | reproduced-from-literature |
| ratio_delta_beta | Spectral ratio delta/beta | literature_derived | \mathrm{DAR}=P_\delta/P_\beta | dimensionless | low | literature-derived | reproduced-from-literature |
| ratio_delta_theta | Spectral ratio delta/theta | literature_derived | P_\delta/P_\theta | dimensionless | low | literature-derived | reproduced-from-literature |
| ratio_sigma_beta | Spectral ratio sigma/beta | literature_derived | P_\sigma/P_\beta | dimensionless | low | literature-derived | reproduced-from-literature |
| ratio_sigma_delta | Spectral ratio sigma/delta | literature_derived | P_\sigma/P_\delta | dimensionless | low | literature-derived | reproduced-from-literature |
| ratio_theta_alpha | Spectral ratio theta/alpha | literature_derived | P_\theta/P_\alpha | dimensionless | low | literature-derived | reproduced-from-literature |
| rel_power_alpha | Relative alpha power | spectral | \rho_{alpha}=P_{alpha}\big/\textstyle\sum_b P_b | fraction | low | established | reproduced-from-literature |
| rel_power_beta | Relative beta power | spectral | \rho_{beta}=P_{beta}\big/\textstyle\sum_b P_b | fraction | low | established | reproduced-from-literature |
| rel_power_delta | Relative delta power | spectral | \rho_{delta}=P_{delta}\big/\textstyle\sum_b P_b | fraction | low | established | reproduced-from-literature |
| rel_power_gamma | Relative gamma power | spectral | \rho_{gamma}=P_{gamma}\big/\textstyle\sum_b P_b | fraction | low | established | reproduced-from-literature |
| rel_power_sigma | Relative sigma power | spectral | \rho_{sigma}=P_{sigma}\big/\textstyle\sum_b P_b | fraction | low | established | reproduced-from-literature |
| rel_power_theta | Relative theta power | spectral | \rho_{theta}=P_{theta}\big/\textstyle\sum_b P_b | fraction | low | established | reproduced-from-literature |
| rms_amplitude | Root mean square amplitude | temporal | x_{rms}=\sqrt{\frac{1}{N}\sum_i x_i^2} | uV | low | established | unit-tested |
| sample_entropy | Sample entropy | nonlinear | \mathrm{SampEn}(m,r)=-\ln\left(A/B\right),\ r=0.2\,\sigma_x,\ m=2 | nats | high | established | reproduced-from-literature |
| skewness | Amplitude skewness | temporal | \gamma_1=\mathrm{E}\left[(x-\mu)^3\right]/\sigma^3 | dimensionless | low | established | reproduced-from-literature |
| slow_wave_amplitude_mean | Slow wave amplitude mean | events | \overline{A}_{SW} | uV | medium | established | synthetic-validated |
| slow_wave_count | Slow wave count | events | N_{SW} | count | medium | established | synthetic-validated |
| slow_wave_density | Slow wave density | events | \rho_{SW}=N_{SW}/T | 1/min | medium | established | synthetic-validated |
| slow_wave_duration_mean | Slow wave duration mean | events | \overline{d}_{SW} | s | medium | established | synthetic-validated |
| slow_wave_occupancy | Slow wave occupancy | events | o_{SW}=\sum d / T | fraction | medium | established | synthetic-validated |
| slow_wave_slope_mean | Slow wave slope mean | events | \overline{s}_{SW} | uV/s | medium | established | synthetic-validated |
| spectral_asymmetry_index | Spectral asymmetry index | literature_derived | \mathrm{SAI}=\frac{(P_\delta+P_\theta)-(P_\alpha+P_\sigma+P_\beta)}{P_{0.5-30}} | dimensionless | low | literature-derived | reproduced-from-literature |
| spectral_centroid | Spectral centroid | spectral | f_c=\int f\,p(f)\,\mathrm{d}f,\quad p=\mathrm{PSD}/\int\mathrm{PSD} | Hz | low | established | reproduced-from-literature |
| spectral_coverage_fraction | Fraction of Welch power inside the configured bands | spectral | \kappa=\int_{0.5}^{45}\mathrm{PSD}\,\mathrm{d}f\big/\int_0^{f_N}\mathrm{PSD}\,\mathrm{d}f | fraction | low | established | unit-tested |
| spectral_edge_95 | Spectral edge frequency 95% | spectral | f_{95}=\min\{f:\int_{0.5}^{f}p(u)\,\mathrm{d}u\ge0.95\} | Hz | low | established | reproduced-from-literature |
| spectral_entropy | Spectral entropy | spectral | H_{spec}=-\frac{1}{\ln N}\sum_k p(f_k)\ln p(f_k) | normalised | low | established | reproduced-from-literature |
| spectral_flatness | Spectral flatness (Wiener entropy) | spectral | F=\exp\!\left(\tfrac1N\sum\ln p(f_k)\right)\big/\left(\tfrac1N\sum p(f_k)\right) | dimensionless | low | established | reproduced-from-literature |
| spectral_flux | Within-epoch spectral flux | spectral | \Phi=\frac{1}{T-1}\sum_t\sqrt{\frac1N\sum_k\big(\hat p_{t+1}(f_k)-\hat p_t(f_k)\big)^2} | dimensionless | medium | literature-derived | unit-tested |
| spectral_kurtosis | Spectral kurtosis | spectral | \gamma_2=\int\big((f-f_c)/\sigma_f\big)^4 p(f)\,\mathrm{d}f | dimensionless | low | established | unit-tested |
| spectral_rolloff_85 | Spectral roll-off 85% | spectral | f_{85}=\min\{f:\int_{0.5}^{f}p(u)\,\mathrm{d}u\ge0.85\} | Hz | low | established | reproduced-from-literature |
| spectral_skewness | Spectral skewness | spectral | \gamma_1=\int\big((f-f_c)/\sigma_f\big)^3 p(f)\,\mathrm{d}f | dimensionless | low | established | unit-tested |
| spectral_spread | Spectral bandwidth | spectral | \sigma_f=\sqrt{\int (f-f_c)^2 p(f)\,\mathrm{d}f} | Hz | low | established | unit-tested |
| spindle_amplitude_mean | Spindle amplitude mean | events | \overline{A}_{sp} | uV | medium | established | synthetic-validated |
| spindle_count | Spindle count | events | N_{sp} | count | medium | established | synthetic-validated |
| spindle_density | Spindle density | events | \rho_{sp}=N_{sp}/T_{epoch}\ \text{[min}^{-1}\text{]} | 1/min | medium | established | synthetic-validated |
| spindle_duration_mean | Spindle duration mean | events | \overline{d}_{sp} | s | medium | established | synthetic-validated |
| spindle_frequency_mean | Spindle frequency mean | events | \overline{f}_{sp} | Hz | medium | established | synthetic-validated |
| spindle_occupancy | Spindle occupancy | events | o_{sp}=\sum d / T | fraction | medium | established | synthetic-validated |
| std_amplitude | Standard deviation of amplitude | temporal | \sigma_x=\sqrt{\frac{1}{N}\sum_i (x_i-\bar{x})^2} | uV | low | established | reproduced-from-literature |
| total_power | Total power 0.5-45 Hz | spectral | P_{tot}=\int_{0.5}^{45}\mathrm{PSD}(f)\,\mathrm{d}f | uV^2 | low | established | reproduced-from-literature |
| var_amplitude | Variance of amplitude | temporal | \sigma_x^2 | uV^2 | low | established | unit-tested |
| wavelet_detail_energy_1 | Wavelet detail energy level 1 | wavelet | E_{1}=\sum_k \|c_{1,k}\|^2 | uV^2 | medium | established | unit-tested |
| wavelet_detail_energy_2 | Wavelet detail energy level 2 | wavelet | E_{2}=\sum_k \|c_{2,k}\|^2 | uV^2 | medium | established | unit-tested |
| wavelet_detail_energy_3 | Wavelet detail energy level 3 | wavelet | E_{3}=\sum_k \|c_{3,k}\|^2 | uV^2 | medium | established | unit-tested |
| wavelet_detail_energy_4 | Wavelet detail energy level 4 | wavelet | E_{4}=\sum_k \|c_{4,k}\|^2 | uV^2 | medium | established | unit-tested |
| wavelet_detail_energy_5 | Wavelet detail energy level 5 | wavelet | E_{5}=\sum_k \|c_{5,k}\|^2 | uV^2 | medium | established | unit-tested |
| wavelet_detail_ratio_1 | Relative wavelet energy level 1 | wavelet | q_{1}=E_{1}\big/\sum_j E_j | fraction | medium | established | unit-tested |
| wavelet_detail_ratio_2 | Relative wavelet energy level 2 | wavelet | q_{2}=E_{2}\big/\sum_j E_j | fraction | medium | established | unit-tested |
| wavelet_detail_ratio_3 | Relative wavelet energy level 3 | wavelet | q_{3}=E_{3}\big/\sum_j E_j | fraction | medium | established | unit-tested |
| wavelet_detail_ratio_4 | Relative wavelet energy level 4 | wavelet | q_{4}=E_{4}\big/\sum_j E_j | fraction | medium | established | unit-tested |
| wavelet_detail_ratio_5 | Relative wavelet energy level 5 | wavelet | q_{5}=E_{5}\big/\sum_j E_j | fraction | medium | established | unit-tested |
| wavelet_energy_entropy | Multiscale wavelet energy entropy | wavelet | M=-\sum_j q_j\ln(q_j+\epsilon)\big/\ln J,\quad q_j=E_j/\sum_i E_i | normalised | medium | literature-derived | unit-tested |
| wavelet_energy_slope | Wavelet energy scaling slope | wavelet | s=\text{slope of }\ln q_j\ \text{vs}\ \log_2(2^j) | dimensionless | medium | literature-derived | unit-tested |
| wavelet_low_high_ratio | Coarse/fine wavelet energy ratio | wavelet | R_{wh}=(E_4+E_5)/(E_1+E_2) | dimensionless | medium | literature-derived | unit-tested |
| zero_crossing_rate | Zero-crossing rate | temporal | Z=\frac{1}{N-1}\#\{i:\mathrm{sgn}(x_i)\ne\mathrm{sgn}(x_{i+1})\} | crossings/sample | low | established | reproduced-from-literature |
| zero_crossing_rate_hz | Zero-crossing rate (Hz equivalent) | temporal | f_{zc}=\frac{1}{2T}\#\{\text{sign changes}\} | Hz | low | established | unit-tested |

## Category C - novel hypotheses
The features below are **proposed by this project**, not taken from the literature. Each has an explicit equation, an executable implementation, unit tests and synthetic-signal validation. None of them is a validated biomarker, and a small p-value for one of them does not make it one.

### Beta power instability

| field | value |
| --- | --- |
| export label | Novel hypothesis - proposed by this project; NOT a validated finding |
| feature id | band_power_instability_beta |
| equation | I_{beta}=\frac{1}{T-1}\sum_t\left\|\ln(P_{t+1}+\epsilon)-\ln(P_t+\epsilon)\right\| |
| motivation | Within-epoch variability of band power.  Transitional stages (N1, REM) are expected to change faster than consolidated N2/N3, so variability may carry information that average power does not. |
| references | none |
| expected range | 0.0000, n/a |
| units | ln-ratio |
| computational cost | medium |
| validation | unit-tested |
| numerical notes | n/a |

### Delta power instability

| field | value |
| --- | --- |
| export label | Novel hypothesis - proposed by this project; NOT a validated finding |
| feature id | band_power_instability_delta |
| equation | I_{delta}=\frac{1}{T-1}\sum_t\left\|\ln(P_{t+1}+\epsilon)-\ln(P_t+\epsilon)\right\| |
| motivation | Within-epoch variability of band power.  Transitional stages (N1, REM) are expected to change faster than consolidated N2/N3, so variability may carry information that average power does not. |
| references | none |
| expected range | 0.0000, n/a |
| units | ln-ratio |
| computational cost | medium |
| validation | unit-tested |
| numerical notes | n/a |

### Sigma power instability

| field | value |
| --- | --- |
| export label | Novel hypothesis - proposed by this project; NOT a validated finding |
| feature id | band_power_instability_sigma |
| equation | I_{sigma}=\frac{1}{T-1}\sum_t\left\|\ln(P_{t+1}+\epsilon)-\ln(P_t+\epsilon)\right\| |
| motivation | Within-epoch variability of band power.  Transitional stages (N1, REM) are expected to change faster than consolidated N2/N3, so variability may carry information that average power does not. |
| references | none |
| expected range | 0.0000, n/a |
| units | ln-ratio |
| computational cost | medium |
| validation | unit-tested |
| numerical notes | n/a |

### Cross-band entropy imbalance

| field | value |
| --- | --- |
| export label | Novel hypothesis - proposed by this project; NOT a validated finding |
| feature id | cross_band_entropy |
| equation | H_{bands}=-\sum_b r_b\ln(r_b+\epsilon)\big/\ln B |
| motivation | How evenly power is distributed across the classic bands; low when a single rhythm dominates (N3 slow waves, W alpha), high for mixed-frequency stages (REM, N1). |
| references | none |
| expected range | 0.0000, 1.0000 |
| units | normalised |
| computational cost | low |
| validation | unit-tested |
| numerical notes | n/a |

### Sigma envelope transient index

| field | value |
| --- | --- |
| export label | Novel hypothesis - proposed by this project; NOT a validated finding |
| feature id | sigma_envelope_transient_index |
| equation | \mathrm{TIX}=\max_t\|e_\sigma(t)\|\big/\overline{\|e_\sigma(t)\|} |
| motivation | Differentiates spindles (transient sigma envelopes) from low-amplitude stationary sigma activity of the same mean power. |
| references | none |
| expected range | 1.0000, n/a |
| units | dimensionless |
| computational cost | medium |
| validation | unit-tested |
| numerical notes | n/a |

### Spectral concentration (alpha=2)

| field | value |
| --- | --- |
| export label | Novel hypothesis - proposed by this project; NOT a validated finding |
| feature id | spectral_concentration_a2 |
| equation | C_{2}=\int p(f)^{2}\,\mathrm{d}f,\quad p=\mathrm{PSD}\big/\int\mathrm{PSD} |
| motivation | High-order moments of the normalised spectrum respond quadratically/cubically to spectral concentration, so they weight the dominant rhythm far more strongly than band power does. |
| references | none |
| expected range | 0.0000, n/a |
| units | Hz^{1-2} |
| computational cost | low |
| validation | unit-tested |
| numerical notes | Hypothesis from the project research brief; not a published biomarker. |

### Spectral concentration (alpha=3)

| field | value |
| --- | --- |
| export label | Novel hypothesis - proposed by this project; NOT a validated finding |
| feature id | spectral_concentration_a3 |
| equation | C_{3}=\int p(f)^{3}\,\mathrm{d}f,\quad p=\mathrm{PSD}\big/\int\mathrm{PSD} |
| motivation | High-order moments of the normalised spectrum respond quadratically/cubically to spectral concentration, so they weight the dominant rhythm far more strongly than band power does. |
| references | none |
| expected range | 0.0000, n/a |
| units | Hz^{1-3} |
| computational cost | low |
| validation | unit-tested |
| numerical notes | Hypothesis from the project research brief; not a published biomarker. |

### Sub-window RMS instability

| field | value |
| --- | --- |
| export label | Novel hypothesis - proposed by this project; NOT a validated finding |
| feature id | subwindow_rms_instability |
| equation | \mathrm{NI}=\sigma(x_{rms}^{t})/\overline{x_{rms}^{t}} |
| motivation | Amplitude non-stationarity within the epoch, a proxy for stage transition. |
| references | none |
| expected range | 0.0000, n/a |
| units | dimensionless |
| computational cost | medium |
| validation | unit-tested |
| numerical notes | n/a |

## Requirements and failure modes

| feature | min sfreq (Hz) | max sfreq (Hz) | min samples | expected range | notes |
| --- | --- | --- | --- | --- | --- |
| abs_power_alpha | 24.0000 | n/a | 0 | 0.0000, n/a | Welch PSD integrated over the configured band boundaries. |
| abs_power_beta | 60.0000 | n/a | 0 | 0.0000, n/a | Welch PSD integrated over the configured band boundaries. |
| abs_power_delta | 8.0000 | n/a | 0 | 0.0000, n/a | Welch PSD integrated over the configured band boundaries. |
| abs_power_gamma | 90.0000 | n/a | 0 | 0.0000, n/a | Welch PSD integrated over the configured band boundaries. |
| abs_power_sigma | 32.0000 | n/a | 0 | 0.0000, n/a | Welch PSD integrated over the configured band boundaries. |
| abs_power_theta | 16.0000 | n/a | 0 | 0.0000, n/a | Welch PSD integrated over the configured band boundaries. |
| amplitude_histogram_entropy | 0.0000 | n/a | 0 | 0.0000, 1.0000 |  |
| approximate_entropy | 0.0000 | n/a | 64 | 0.0000, 3.0000 |  |
| band_power_instability_beta | 40.0000 | n/a | 400 | 0.0000, n/a |  |
| band_power_instability_delta | 40.0000 | n/a | 400 | 0.0000, n/a |  |
| band_power_instability_sigma | 40.0000 | n/a | 400 | 0.0000, n/a |  |
| band_power_variability_alpha | 40.0000 | n/a | 400 | 0.0000, n/a |  |
| band_power_variability_delta | 40.0000 | n/a | 400 | 0.0000, n/a |  |
| band_power_variability_sigma | 40.0000 | n/a | 400 | 0.0000, n/a |  |
| crest_factor | 0.0000 | n/a | 0 | 1.0000, 20.0000 |  |
| cross_band_entropy | 90.0000 | n/a | 0 | 0.0000, 1.0000 |  |
| delta_sigma_coupling | 40.0000 | n/a | 400 | 0.0000, 1.0000 | Not surrogate-normalised: a coupling *proxy*, not a significance-tested coupling value. |
| dfa_alpha1 | 0.0000 | n/a | 64 | 0.0000, 2.0000 |  |
| dfa_alpha2 | 0.0000 | n/a | 128 | 0.0000, 2.0000 |  |
| higuchi_fractal_dimension | 0.0000 | n/a | 64 | 1.0000, 2.0000 | Sampling-frequency dependent: compare only within a fixed sfreq. |
| hjorth_activity | 0.0000 | n/a | 0 | 0.0000, n/a |  |
| hjorth_complexity | 0.0000 | n/a | 0 | 1.0000, 6.0000 |  |
| hjorth_mobility | 0.0000 | n/a | 0 | 0.0000, 2.0000 |  |
| iqr_amplitude | 0.0000 | n/a | 0 | 0.0000, n/a |  |
| kcomplex_amplitude_mean | 20.0000 | n/a | 200 | n/a, n/a |  |
| kcomplex_count | 20.0000 | n/a | 200 | n/a, n/a |  |
| kcomplex_density | 20.0000 | n/a | 200 | n/a, n/a |  |
| kcomplex_duration_mean | 20.0000 | n/a | 200 | n/a, n/a |  |
| kcomplex_occupancy | 20.0000 | n/a | 200 | n/a, n/a |  |
| kurtosis | 0.0000 | n/a | 0 | -3.0000, 50.0000 |  |
| line_length | 0.0000 | n/a | 0 | 0.0000, n/a |  |
| line_length_ratio | 0.0000 | n/a | 0 | 0.0000, 1.0000 |  |
| log_power_alpha | 24.0000 | n/a | 0 | n/a, n/a |  |
| log_power_beta | 60.0000 | n/a | 0 | n/a, n/a |  |
| log_power_delta | 8.0000 | n/a | 0 | n/a, n/a |  |
| log_power_gamma | 90.0000 | n/a | 0 | n/a, n/a |  |
| log_power_sigma | 32.0000 | n/a | 0 | n/a, n/a |  |
| log_power_theta | 16.0000 | n/a | 0 | n/a, n/a |  |
| log_total_power | 90.0000 | n/a | 0 | n/a, n/a |  |
| mad_amplitude | 0.0000 | n/a | 0 | 0.0000, n/a |  |
| mean_abs_derivative | 0.0000 | n/a | 0 | 0.0000, n/a |  |
| mean_amplitude | 0.0000 | n/a | 0 | -100.0000, 100.0000 |  |
| mean_frequency | 0.0000 | n/a | 0 | 0.5000, 45.0000 |  |
| peak_frequency | 0.0000 | n/a | 0 | 0.5000, 45.0000 |  |
| peak_to_peak | 0.0000 | n/a | 0 | 0.0000, n/a |  |
| permutation_entropy | 0.0000 | n/a | 64 | 0.0000, 1.0000 |  |
| petrosian_fractal_dimension | 0.0000 | n/a | 32 | 1.0000, 1.2000 |  |
| ratio_alpha_delta | 0.0000 | n/a | 0 | 0.0000, n/a |  |
| ratio_delta_beta | 0.0000 | n/a | 0 | 0.0000, n/a |  |
| ratio_delta_theta | 0.0000 | n/a | 0 | 0.0000, n/a |  |
| ratio_sigma_beta | 0.0000 | n/a | 0 | 0.0000, n/a |  |
| ratio_sigma_delta | 0.0000 | n/a | 0 | 0.0000, n/a |  |
| ratio_theta_alpha | 0.0000 | n/a | 0 | 0.0000, n/a |  |
| rel_power_alpha | 24.0000 | n/a | 0 | 0.0000, 1.0000 | Denominator is the union of the configured bands; relative powers are compositional. |
| rel_power_beta | 60.0000 | n/a | 0 | 0.0000, 1.0000 | Denominator is the union of the configured bands; relative powers are compositional. |
| rel_power_delta | 8.0000 | n/a | 0 | 0.0000, 1.0000 | Denominator is the union of the configured bands; relative powers are compositional. |
| rel_power_gamma | 90.0000 | n/a | 0 | 0.0000, 1.0000 | Denominator is the union of the configured bands; relative powers are compositional. |
| rel_power_sigma | 32.0000 | n/a | 0 | 0.0000, 1.0000 | Denominator is the union of the configured bands; relative powers are compositional. |
| rel_power_theta | 16.0000 | n/a | 0 | 0.0000, 1.0000 | Denominator is the union of the configured bands; relative powers are compositional. |
| rms_amplitude | 0.0000 | n/a | 0 | 0.0000, 500.0000 |  |
| sample_entropy | 0.0000 | n/a | 64 | 0.0000, 3.0000 |  |
| sigma_envelope_transient_index | 40.0000 | n/a | 200 | 1.0000, n/a |  |
| skewness | 0.0000 | n/a | 0 | -5.0000, 5.0000 |  |
| slow_wave_amplitude_mean | 20.0000 | n/a | 200 | n/a, n/a |  |
| slow_wave_count | 20.0000 | n/a | 200 | n/a, n/a |  |
| slow_wave_density | 20.0000 | n/a | 200 | n/a, n/a |  |
| slow_wave_duration_mean | 20.0000 | n/a | 200 | n/a, n/a |  |
| slow_wave_occupancy | 20.0000 | n/a | 200 | n/a, n/a |  |
| slow_wave_slope_mean | 20.0000 | n/a | 200 | n/a, n/a |  |
| spectral_asymmetry_index | 0.0000 | n/a | 0 | -1.0000, 1.0000 |  |
| spectral_centroid | 0.0000 | n/a | 0 | 0.5000, 45.0000 |  |
| spectral_concentration_a2 | 90.0000 | n/a | 0 | 0.0000, n/a | Hypothesis from the project research brief; not a published biomarker. |
| spectral_concentration_a3 | 90.0000 | n/a | 0 | 0.0000, n/a | Hypothesis from the project research brief; not a published biomarker. |
| spectral_coverage_fraction | 0.0000 | n/a | 0 | 0.0000, 1.0000 |  |
| spectral_edge_95 | 0.0000 | n/a | 0 | 0.5000, 45.0000 |  |
| spectral_entropy | 0.0000 | n/a | 0 | 0.0000, 1.0000 |  |
| spectral_flatness | 0.0000 | n/a | 0 | 0.0000, 1.0000 |  |
| spectral_flux | 20.0000 | n/a | 400 | 0.0000, n/a |  |
| spectral_kurtosis | 0.0000 | n/a | 0 | n/a, n/a |  |
| spectral_rolloff_85 | 0.0000 | n/a | 0 | 0.5000, 45.0000 |  |
| spectral_skewness | 0.0000 | n/a | 0 | n/a, n/a |  |
| spectral_spread | 0.0000 | n/a | 0 | 0.0000, 30.0000 |  |
| spindle_amplitude_mean | 40.0000 | n/a | 200 | n/a, n/a |  |
| spindle_count | 40.0000 | n/a | 200 | n/a, n/a |  |
| spindle_density | 40.0000 | n/a | 200 | n/a, n/a |  |
| spindle_duration_mean | 40.0000 | n/a | 200 | n/a, n/a |  |
| spindle_frequency_mean | 40.0000 | n/a | 200 | n/a, n/a |  |
| spindle_occupancy | 40.0000 | n/a | 200 | n/a, n/a |  |
| std_amplitude | 0.0000 | n/a | 0 | 0.0000, 500.0000 |  |
| subwindow_rms_instability | 40.0000 | n/a | 400 | 0.0000, n/a |  |
| total_power | 90.0000 | n/a | 0 | 0.0000, n/a |  |
| var_amplitude | 0.0000 | n/a | 0 | 0.0000, n/a |  |
| wavelet_detail_energy_1 | 4 | n/a | 0 | 0.0000, n/a |  |
| wavelet_detail_energy_2 | 8 | n/a | 0 | 0.0000, n/a |  |
| wavelet_detail_energy_3 | 16 | n/a | 0 | 0.0000, n/a |  |
| wavelet_detail_energy_4 | 32 | n/a | 0 | 0.0000, n/a |  |
| wavelet_detail_energy_5 | 64 | n/a | 0 | 0.0000, n/a |  |
| wavelet_detail_ratio_1 | 4 | n/a | 0 | 0.0000, 1.0000 |  |
| wavelet_detail_ratio_2 | 8 | n/a | 0 | 0.0000, 1.0000 |  |
| wavelet_detail_ratio_3 | 16 | n/a | 0 | 0.0000, 1.0000 |  |
| wavelet_detail_ratio_4 | 32 | n/a | 0 | 0.0000, 1.0000 |  |
| wavelet_detail_ratio_5 | 64 | n/a | 0 | 0.0000, 1.0000 |  |
| wavelet_energy_entropy | 32.0000 | n/a | 0 | 0.0000, 1.0000 |  |
| wavelet_energy_slope | 32.0000 | n/a | 0 | -6.0000, 3.0000 |  |
| wavelet_low_high_ratio | 32.0000 | n/a | 0 | 0.0000, n/a |  |
| zero_crossing_rate | 0.0000 | n/a | 0 | 0.0000, 0.5000 |  |
| zero_crossing_rate_hz | 0.0000 | n/a | 0 | 0.0000, 50.0000 |  |

## Non-causal features
`feature_delta_norm` and `feature_delta_norm_sigma` compare an epoch with the preceding epoch of the same recording. They are *offline / non-causal* when the preceding epoch is unavailable (the first epoch of a recording is `n/a`), and they are excluded from every causal ablation and from every subject-grouped predictive evaluation.
