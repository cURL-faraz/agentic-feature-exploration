# Statistical methodology

The brief's central methodological warning is that *a small p-value is not a
result*.  This page states exactly which test is used when, why, and what the
failure modes are.  Every rule below is implemented in code
(`statistics/hypothesis_tests.py`, `statistics/engine.py`,
`evaluation/leakage_checks.py`), not decided by a language model.

## 1. The unit of analysis is the subject, not the epoch

A 30 s sleep epoch is not an independent observation: consecutive epochs from one
night share the subject's anatomy, electrode placement, impedance and amplifier.
Treating every epoch as independent inflates the effective sample size by roughly
the number of epochs per subject and produces confident nonsense.

The project therefore reports **two** analyses, always clearly separated:

| Level | What it is | How it is labelled |
| --- | --- | --- |
| Primary | One value per subject per stage (the subject's mean for that stage). | `unit_of_analysis: "subject …"` |
| Epoch level | The raw per-epoch groups. | `notes: "… EPOCH LEVEL: pseudo-replicated …"` |

Only the primary analysis drives conclusions.  The epoch-level test is retained
for transparency and explicitly flagged as pseudo-replicated.

## 2. Test selection: a fixed decision table

Assumptions are checked first, then a deterministic table picks the test.  The
LLM has no role in this.

* **Normality** — D'Agostino-Pearson for n ≥ 8, Shapiro-Wilk below that.  A
  constant group is reported as non-normal with the reason `"constant group"`.
* **Variance homogeneity** — Levene, median-centred (Brown-Forsythe).

### Two samples

| normal(a) and normal(b) | equal variance | test |
| --- | --- | --- |
| yes | yes | Student's t |
| yes | no | Welch's t |
| otherwise | — | Mann-Whitney U (rank-biserial effect size) |

### *k* samples (omnibus)

| all groups normal | equal variance | test |
| --- | --- | --- |
| yes | yes | one-way ANOVA |
| yes | no | Welch's ANOVA (implemented in-project, not imported) |
| no | — | Kruskal-Wallis |

### Repeated measures

When the design is *complete* (at least four subjects contribute **every** stage)
and `statistics.use_subject_aggregation` is on, the primary analysis is a
**Friedman test** on the subject × stage mean matrix, with Kendall's W as the
effect size and **paired Wilcoxon** post-hoc comparisons (Holm-corrected).
Otherwise the unpaired path above runs on the same aggregates with
**Dunn** post-hoc comparisons.

Degenerate inputs never crash the pipeline:

* fewer than two usable groups → the comparison is skipped with a reason;
* fully tied (constant) data → skipped with `"feature is constant; there is no
  variance to compare across stages"` (SciPy's rank tests raise on this input);
* a stage with fewer than `analysis.statistics.min_group_size` finite epochs is
  excluded from the comparison, and the exclusion is recorded in the result.

## 3. Effect sizes and intervals

p-values are never reported alone.  Every comparison carries at least one effect
size with its interpretation:

| Situation | Effect size |
| --- | --- |
| Two groups, normal | Hedges' g (small-sample-corrected Cohen's d) |
| Two groups, rank-based | rank-biserial correlation |
| Two groups, any | Cliff's delta (dominance) |
| *k* groups, parametric | eta-squared / partial eta-squared from F |
| *k* groups, rank-based | epsilon-squared from H |
| Repeated measures | Kendall's W |

Confidence intervals are produced by bootstrap resampling of **subjects** (not
epochs), with `analysis.statistics.bootstrap_iterations` replicates at
`analysis.statistics.effect_size_ci` (default 95 %).

## 4. Multiple comparisons

The family of tests is every feature compared across stages in one run.  Raw and
adjusted p-values are both stored, and the method is recorded:

| Method | Use |
| --- | --- |
| Benjamini-Hochberg (`fdr_bh`) | Default: controls the false-discovery rate across many features. |
| Holm | Step-down FWER control, uniformly more powerful than Bonferroni. |
| Bonferroni | Available when strict FWER control is justified. |
| `none` | Only for a single prespecified confirmatory test. |

The implementation is verified against the textbook step-up formula and against
R's `p.adjust(..., method = "BH")`.  `family_correction.n_tests` records how many
tests were in the family, so a reader can always tell what the correction was
applied *to*.

## 5. Association measures for a multiclass label

Stage labels are categorical.  Treating `W=0, N1=1, N2=2, N3=3, REM=4` as a
continuous variable would be nonsense, so `correlations.py` never does it.
Instead it reports order-free measures, each with its interpretation:

| Measure | Interpretation |
| --- | --- |
| Kruskal epsilon-squared | Proportion of rank variance explained by stage. |
| Mutual information (nats) | Any dependence, linear or not; zero iff independent. |
| Point-biserial | Only for a genuine two-group split (e.g. N1 vs not-N1), never for the full multiclass label. |

## 6. Feature selection — training rows only

`feature_selection.py` is invoked with the **training** rows only.
`engine.py` slices the frame by the `split` column before calling it, and the
resulting report records `n_train_rows` and `n_train_subjects`.
`leakage_checks.audit_feature_selection_scope` then verifies that
`n_train_rows <= total_rows`; a mismatch fails the run.

Rankings produced: mutual information, ANOVA F, random-forest importance,
permutation importance, L1 (on the MI-selected subset) and RFE.

Redundancy is handled *before* modelling, because EEG band powers are strongly
collinear and a model fed 30 near-duplicates has unstable coefficients and
meaningless importances: Pearson correlation clustering at
`features_selection.correlation_threshold` (0.9) plus variance-inflation factors.

## 7. Interactions, with a bounded search space

The brief forbids "generate millions of combinations and report the best".
`evaluate_interactions` therefore works from a **prespecified** candidate list
(`interaction_pairs_max`, default 12) built from the top-ranked non-redundant
features, evaluates each candidate by a cross-validated mutual-information delta
against its parents, and stores the whole candidate list and the count of tests
in the feature-selection report so the multiple-comparison exposure is visible.

## 8. Predictive evaluation and leakage control

Splitting:

1. subjects (never epochs) are partitioned into train / validation / test with a
   saved manifest (`data/metadata/split_manifest.json`);
2. `verify_splits` fails if any subject appears under two split labels, if the
   declared and observed subject sets differ, or if a split is empty;
3. preprocessing, imputation, scaling and feature selection are fitted **inside
   the training fold** only;
4. the test set is read once, for the final numbers.

Models: majority-class baseline, logistic regression (multinomial), random
forest, gradient boosting and SVM.  Metrics: accuracy, **balanced accuracy**
and macro F1 (the primary metric on imbalanced sleep data), per-stage
precision/recall/F1, the confusion matrix, and Cohen's kappa.  N1, N2 and REM are
reported separately because they are the clinically difficult distinctions.

Ablations isolate feature families (`all_features`, `conventional_only`,
`literature_derived_only`, `novel_only`, `no_events`, `no_spectral`,
`top_k_mutual_information`) so a claim about a new feature has to survive
comparison against the conventional set.

## 9. Statistical verification

`verification/statistical_validation.py` checks the machinery rather than
trusting it:

* Monte-Carlo type-I error of the permutation test against nominal alpha;
* agreement with SciPy where a reference implementation exists;
* BH against the closed-form formula;
* effect sizes against hand-computed values and against known properties
  (identical samples give d = 0; perfect separation gives Cliff's delta = 1);
* that rank-based and parametric paths give the same qualitative answer on data
  with a known large effect.

## 10. Known limitations of this methodology

* No mixed-effects model: `statsmodels` is not available in the reference
  environment.  The subject-aggregation + Friedman/cluster-permutation
  combination is a defensible substitute, but it discards unequal epoch counts
  per stage within a subject.
* Assumption pre-testing (normality, Levene) is itself data-dependent and
  low-powered for small samples; a non-significant pre-test is not proof that
  the assumption holds.
* Cluster permutation tests resample subjects, so with a handful of subjects the
  achievable p-value granularity is limited.
* Any p-value computed on the synthetic surrogate is a property of the
  simulation, not of human sleep.
