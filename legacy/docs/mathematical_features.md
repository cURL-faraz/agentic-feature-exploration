# Mathematical feature definitions

Every feature is registered in `eeg/feature_registry.py` with a formula, a
family, a novelty status, its units, the minimum sampling rate and minimum signal
length it needs.  This page documents the *definitions and the reasoning*; the
generated [`reports/feature_catalog.md`](../reports/feature_catalog.md) lists the
catalog exactly as the run produced it (107 features in the reference run).

Notation used throughout: `x[n]`, `n = 0 … N−1` is one epoch of the selected EEG
derivation sampled at `f_s`; `X(f)` is its Welch power spectral density; `P_b` is
the power in band `b`.  Amplitudes are stored in **volts**; band powers are
reported in **µV²**.

## Novelty and validation status

The registry distinguishes three statuses, and the reports never blur them:

| Status | Meaning |
| --- | --- |
| `established` | A textbook/shared feature (band power, Hjorth, entropy, spindle density, …). |
| `literature-derived` | Motivated by a published family of features but with our own parameterisation (spectral ratios, instability, coupling proxies). |
| `novel-hypothesis` | Our own mathematical proposal. It is a *hypothesis*, not a validated finding, and is labelled that way wherever it appears. |

Validation status is separate: `unit-tested` (exact-answer unit test),
`synthetic-validated` (behaviour verified on generated signals with known
properties), or `reproduced-from-literature`.

## 1. Frequency-domain features (established)

### Absolute and relative band power

$$P_b = \int_{f \in b} X(f)\,df, \qquad \text{rel}_b = \frac{P_b}{\sum_{b'} P_{b'}}$$

Bands follow the classical sleep-EEG tradition and are declared once in
`config/features.yaml`:

| band | range (Hz) |
| --- | --- |
| delta | 0.5 – 4.0 |
| theta | 4.0 – 8.0 |
| alpha | 8.0 – 12.0 |
| sigma | 12.0 – 16.0 |
| beta | 16.0 – 30.0 |
| gamma | 30.0 – 45.0 |

Power is estimated with Welch's method (4 s windows, 50 % overlap, constant
detrend) so a 30 s epoch yields ~13 averaged segments — a variance reduction that
matters because single-epoch periodograms are extremely noisy.

Because `f_s = 100 Hz` the Nyquist frequency is 50 Hz.  Features that need energy
above 24 Hz or above 32 Hz declare a sampling-rate requirement in the registry;
extraction skips them with a recorded reason rather than returning a meaningless
number.

*Literature motivation:* stage-dependent spectral power is the oldest and most
reproducible finding in sleep EEG scoring (delta for N3, sigma/spindle band for
N2, alpha for relaxed wake).

### Spectral shape

| Feature | Definition |
| --- | --- |
| `spectral_centroid` | `sum_f f*p(f)`, the "centre of mass" of the normalised spectrum `p(f) = X(f)/int X`. |
| `spectral_spread` | `sqrt(sum_f (f - centroid)^2 p(f))` — spectral bandwidth. |
| `spectral_skewness`, `spectral_kurtosis` | Third and fourth standardised moments of `p(f)`. |
| `spectral_entropy` | `-sum_f p(f) log p(f + eps)`; high for flat spectra, low for a single dominant rhythm. |
| `spectral_flatness` | Geometric mean / arithmetic mean of `X(f)` (the Wiener entropy). |
| `spectral_edge_95` | Frequency below which 95 % of the power lies. |
| `spectral_rolloff_85` | Frequency below which 85 % of the cumulative power lies. |
| `spectral_coverage_fraction` | Fraction of the band occupied by the dominant spectral mass. |
| `peak_frequency`, `mean_frequency` | Argmax and power-weighted mean of `X(f)`. |
| `spectral_flux` | L1 distance between the normalised spectra of consecutive sub-windows. |

### Spectral ratios (`literature-derived`)

`ratio_delta_theta`, `ratio_theta_alpha`, `ratio_delta_beta`,
`ratio_sigma_delta`, `ratio_sigma_beta`, `spectral_asymmetry_index`.

Ratios are *scale-invariant*: multiplying the whole epoch by a constant leaves
them unchanged, which makes them robust to electrode impedance and amplifier gain
differences between subjects — precisely the confound that makes absolute power
hard to generalise.

## 2. Time-domain features (established)

| Feature | Definition |
| --- | --- |
| `mean_amplitude`, `std_amplitude`, `var_amplitude` | First and second moments. |
| `rms_amplitude` | `sqrt(mean(x^2))`, i.e. `sqrt(P)`. |
| `peak_to_peak` | `max(x) - min(x)`. |
| `mean_abs_derivative` | `mean(|x[n+1]-x[n]|)` — cheap proxy for high-frequency content. |
| `zero_crossing_rate`, `zero_crossing_rate_hz` | Fraction of consecutive samples with a sign change, and its Hz equivalent. |
| `line_length` | `sum_n |x[n+1]-x[n]|`; rises with amplitude and frequency. |
| `line_length_ratio`, `crest_factor`, `kurtosis`, `skewness`, `iqr_amplitude`, `mad_amplitude` | Shape and amplitude-distribution statistics. |
| `amplitude_histogram_entropy` | Entropy of the amplitude histogram. |

### Hjorth parameters (established)

With `s0^2 = var(x)`, `s1^2 = var(dx/dt)` and `s2^2 = var(d2x/dt2)`:

$$\text{activity} = \sigma_0^2,\qquad
\text{mobility} = \sqrt{\sigma_1^2/\sigma_0^2},\qquad
\text{complexity} = \frac{\sqrt{\sigma_2^2/\sigma_1^2}}{\text{mobility}}$$

Activity is a variance, mobility estimates mean frequency, and complexity
estimates bandwidth.  Note that `hjorth_activity` is the same quantity as the
variance and `hjorth_mobility` correlates strongly with `mean_frequency`: the
redundancy screen in `feature_selection.py` is what keeps such near-duplicates
from inflating a model.

## 3. Nonlinear features (established)

| Feature | Definition |
| --- | --- |
| `sample_entropy` | `-log(A^m(r) / B^m(r))` with `m = 2`, `r = 0.2*sigma`; low in periodic N3 slow waves. |
| `approximate_entropy` | The same construction with the `log A - log B` (non-ratio) form. |
| `permutation_entropy` | Shannon entropy of the ordinal patterns of order 4, delay 1; amplitude-invariant. |
| `higuchi_fractal_dimension` | Slope of `log L(k)` vs `log(1/k)` for curve lengths at scale `k <= 10`. |
| `petrosian_fractal_dimension` | Closed form in the number of derivative sign changes. |
| `dfa_alpha1` | Detrended fluctuation analysis exponent over 4–16 samples. |
| `dfa_alpha2` | DFA exponent over 16–64 samples. |

Entropy estimators are capped at 6 000 samples per epoch
(`config/features.yaml: nonlinear.max_samples`) so runtime stays bounded; the cap
is part of the extraction configuration and therefore of the cache key.

**Interpretation warning.**  A fractal exponent is a *descriptor*, not a
biological mechanism.  Nothing in this project treats a change in DFA alpha as
evidence about a physiological process.

## 4. Wavelet features (established)

`eeg/wavelets.py` implements a Daubechies-4 DWT by hand (PyWavelets was
unavailable; see `docs/environment_notes.md`) and is verified by a
perfect-reconstruction test.  For 5 detail levels, `E_j = sum_k |c_jk|^2` and
`q_j = E_j / (sum_i E_i + eps)`:

| Feature | Definition |
| --- | --- |
| `wavelet_detail_energy_1…5` | Absolute per-level energy `E_j`. |
| `wavelet_detail_ratio_1…5` | Normalised per-level share `q_j`. |
| `wavelet_energy_entropy` | `-sum_j q_j log(q_j + eps)` — energy distribution across time scales. |
| `wavelet_energy_slope` | Slope of `log E_j` vs level: a compact summary of 1/f behaviour. |
| `wavelet_low_high_ratio` | Coarse-scale energy / fine-scale energy. |

## 5. Sleep-event features (established, with an explicit detector)

### Spindles (`spindle_*`)

`detect_spindles` is a documented, parameterised pipeline: band-pass 11–16 Hz →
Hilbert envelope → smoothing → threshold at 1.5 × moving RMS (2 s window) →
candidate intervals → constraints (0.5–3 s duration, 10–18 Hz zero-crossing rate)
→ 0.5 s refractory period.

Outputs: `spindle_count`, `spindle_density` (per minute), `spindle_duration_mean`,
`spindle_amplitude_mean`, `spindle_frequency_mean`, `spindle_occupancy`.

**A rise in `abs_power_sigma` is not a spindle.**  Sigma power integrates
everything in 12–16 Hz; spindle density counts discrete events that also satisfy
duration, frequency and refractory constraints.  Both are kept precisely because
they can disagree.

### K-complexes (`kcomplex_*`)

0.5–2 Hz band-pass → envelope → threshold → duration and amplitude constraints →
refractory period, giving `kcomplex_count`, `kcomplex_density`,
`kcomplex_amplitude_mean`, `kcomplex_duration_mean`, `kcomplex_occupancy`.

### Slow waves (`slow_wave_*`)

0.5–2 Hz activity with a negative-then-positive deflection and a measurable
slope: `slow_wave_count`, `slow_wave_density`, `slow_wave_amplitude_mean`,
`slow_wave_duration_mean`, `slow_wave_slope_mean`, `slow_wave_occupancy`.

Event detectors are heuristic by construction and their thresholds are declared
in `config/features.yaml`.  They are validated for *behaviour* on synthetic
spindle/slow-wave bursts, not against expert human annotations — the reports say
so explicitly.

## 6. Novel hypotheses (`novel-hypothesis`)

These are the project's own proposals.  Each one is labelled
*novel hypothesis — not a validated finding* in the catalog and the final report,
and none of them is allowed to support a biological claim on the strength of a
p-value.

### 6.1 Spectral concentration

With `p(f) = X(f) / int X`:

$$C_\alpha = \int_{f_{\min}}^{f_{\max}} p(f)^\alpha\,df$$

`spectral_concentration_a2` (alpha = 2) and `spectral_concentration_a3`
(alpha = 3).  Large values mean the spectrum is concentrated in a few bins: a
pure tone is the maximal case, white noise the minimal one.  Unit-tested: a pure
sinusoid scores strictly above white noise, and the feature is invariant to
amplitude scaling.

### 6.2 Temporal spectral instability

Splitting an epoch into `T` sub-windows with power `P_t(b)` in band `b`:

$$I_b = \frac{1}{T-1}\sum_{t=1}^{T-1}\left|\log(P_{t+1}(b)+\epsilon) - \log(P_t(b)+\epsilon)\right|$$

Implementations: `band_power_instability_delta`, `band_power_instability_sigma`,
`band_power_instability_beta`, and the amplitude-domain analogue
`subwindow_rms_instability`.  Rationale: transitional stages such as N1 are
hypothesised to be less stationary within an epoch than N2 or N3.  The log
transform makes the measure unit-free.

### 6.3 Cross-band entropy imbalance

$$\text{cross\_band\_entropy} = -\sum_b r_b \log(r_b + \epsilon),\qquad
r_b = \frac{P_b}{\sum_{b'} P_{b'}}$$

`cross_band_entropy` plus the per-band `band_power_variability_*` features.
These complement absolute power by describing the *shape* of the band-energy
distribution.

### 6.4 Multiscale energy variability

`wavelet_energy_entropy` is the wavelet-domain sibling of `cross_band_entropy`,
and `delta_sigma_coupling` is a cross-band coupling proxy.

### 6.5 Inter-epoch transition features (non-causal)

`feature_delta_norm` and `feature_delta_norm_sigma` measure
`||x_t - x_{t-1}||_2` over selected normalised features and over the sigma band
respectively.

**These are not available at inference time for the first epoch of a recording,
and they use temporal context.**  They are therefore:

* `NaN` on the first epoch of each recording by construction;
* excluded from every *causal* ablation;
* rejected by `leakage_checks.audit_non_causal_features` if they appear in a
  causal feature set.

### 6.6 Composite interactions

`feature_selection.evaluate_interactions` searches a bounded set of prespecified
pairwise products/ratios (bounded by `analysis.hypotheses.interaction_pairs_max`,
default 12) and reports each with a cross-validated mutual-information delta
against its parents.  The search space is fixed *before* test data is seen, and
the candidate list is stored in the feature-selection report so the number of
tests is auditable.  Weights are never tuned against the test set.

## 7. What is deliberately not claimed

* A small p-value is not evidence that a feature is a biological marker.
* A feature that improves held-out accuracy is a *predictive* result, not a
  causal one.
* Novel features are hypotheses.  In the reference run they appear with
  `novelty_status: novel-hypothesis` and `validation_status:
  synthetic-validated` at best.
* Results computed on the synthetic surrogate describe the surrogate.  Every
  report opens with that provenance caveat.
