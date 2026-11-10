# Architecture

## The one rule

**The LLM never computes a number.**  Every quantitative result in this project
is produced by plain Python in `src/eeg_research_agent/`, written to an artifact
under `artifacts/`, and then *described* — never re-derived — by the reporting
layer.  An LLM may choose which registered analysis to prioritise and how to
narrate it; it cannot execute code, cannot see a result it is allowed to change,
and cannot introduce a number that has no artifact behind it.

Everything else in this document follows from that rule.

## Layers

```
                    ┌───────────────────────────────────────────────┐
                    │  cli.py   |   agents/graph.py (LangGraph)     │
                    │  "what should run next, and how do we        │
                    │   describe it?"                              │
                    └──────────────┬────────────────────────────────┘
                                   │ calls
                    ┌──────────────▼────────────────────────────────┐
                    │  pipeline.ResearchPipeline                    │
                    │  the single source of truth for the science   │
                    └──────────────┬────────────────────────────────┘
        ┌──────────────┬───────────┼───────────┬───────────────┐
        ▼              ▼           ▼           ▼               ▼
  data/          eeg/        statistics/  evaluation/   verification/
  acquire        features    tests        models        synthetic +
  preprocess     registry    FDR          splits        statistical +
  annotate       extract     effects      leakage       citation checks
        │              │           │           │               │
        └──────────────┴───────────┴───────────┴───────────────┘
                                   │ writes
                    ┌──────────────▼────────────────────────────────┐
                    │  data/features/*.csv.gz, artifacts/tables/*   │
                    │  artifacts/figures/*, artifacts/execution_logs│
                    └──────────────┬────────────────────────────────┘
                                   │ reads (never recomputes)
                    ┌──────────────▼────────────────────────────────┐
                    │  reporting/  → reports/*.md + claims.json      │
                    └───────────────────────────────────────────────┘
```

### `data/` — getting from a file to labelled epochs

| Module | Responsibility |
| --- | --- |
| `acquisition.py` | Try Sleep-EDF; on failure fall back to the deterministic synthetic generator. Never fabricates a label; records `data_origin` on every recording. |
| `annotations.py` | Map dataset-specific annotation strings to canonical stages (`W, N1, N2, N3, REM`). Anything else becomes `UNKNOWN` and is *counted*, not coerced. |
| `preprocessing.py` | Resample → zero-phase Butterworth band-pass → optional mains notch → detrend, applied to the *continuous* signal so filter transients never land on an epoch boundary. |
| `segmentation.py` | 30 s epochs, quality gates, and an explicit exclusion record: every dropped epoch is written down with its reason. |
| `splits.py` | Subject-grouped train/validation/test partition with a saved manifest and a verifier. |

### `eeg/` — the feature engine

`feature_registry.py` is the contract.  A `FeatureDefinition` carries the id,
family, novelty status, formula, units, required sampling rate and required
signal length; `FeatureRegistry.validate_specification()` is the *only* boundary
an LLM may touch, and it accepts registered ids with declared parameters only.

`spectral.py`, `temporal.py`, `nonlinear.py`, `wavelets.py`, `sleep_events.py`
and `novel_features.py` implement the 107 registered features; `feature_extraction.py`
runs them over an epoch set, records per-feature failures instead of dropping
them, and caches the table by a hash of the extraction configuration.

### `statistics/` — association

`descriptive.py` (per-stage summaries), `hypothesis_tests.py` (assumption checks
and the deterministic test-selection table), `effect_sizes.py`,
`multiple_testing.py` (BH/Holm/Bonferroni), `correlations.py` (order-free
association measures) and `feature_selection.py` (MI, ANOVA-F, RF importance,
permutation importance, L1, RFE, VIF/clustering, bounded interaction search).
`engine.py` orchestrates them and, critically, **only ever sees training rows
when selecting features**.

### `evaluation/` — prediction and leakage

`classification.py`, `cross_validation.py` and `metrics.py` implement the
subject-grouped predictive evaluation; `leakage_checks.py` is the adversarial
audit that fails the run when a split overlaps, a row is duplicated, a
non-causal feature enters a causal ablation, or feature selection saw more than
the training rows.

### `verification/` — the agent that attacks the results

`synthetic_signals.py` builds signals whose correct answer is known by
construction; `feature_validation.py` checks that each feature responds as
theory demands; `statistical_validation.py` checks the statistics against
references and Monte-Carlo calibration; `research_validation.py` re-reads every
artifact and confirms each reported number and each citation still holds.

### `reporting/` and `llm/`

`reporting/context.py` loads the saved artifacts into a `ReportContext`;
each report module renders Markdown from that context and registers every number
it prints as a `Claim` (value + artifact locator + source path).  `claims.json`
is the evidence trail, and `verification/research_validation.validate_claims`
re-reads the artifacts to confirm the reports have not drifted.

`llm/provider.py` implements Anthropic- and DeepSeek-compatible clients that
verify the requested model id against the provider's model list (an unknown id
is a hard error, never a silent substitution) and redact keys in `__repr__`;
`llm/offline.py` is a deterministic template provider used by default.

## Data flow

```
raw file ──► Recording(signal, sfreq, ch_name, annotations)
          ──► preprocess            → PreprocessingReport
          ──► segment               → EpochSet + SegmentationReport (exclusions)
          ──► split (by subject)    → DatasetSplits + verification
          ──► extract features      → feature table (csv.gz) + cache key
          ──► statistics (train)    → statistical_results.json
          ──► select (train only)   → feature_selection report
          ──► evaluate (test rows)  → predictive_evaluation.json
          ──► verify                → verification_report.json
          ──► report                → reports/*.md + claims.json
```

## Reproducibility

* One seed (`settings.yaml: random_seed`, default `20240517`) drives the split,
  the permutation tests, the bootstrap intervals and every model.
* The feature table is cached on a hash of the extraction configuration; the
  cache key is logged, so a run states whether it recomputed or reused.
* `artifacts/execution_logs/execution_manifest.json` records the stage list,
  durations, metrics, git revision (when available) and the `data_origin`.
* `artifacts/execution_logs/events.jsonl` is an append-only structured trace of
  every stage, cache decision and failure.
* Reports can be regenerated from saved artifacts alone:
  `python -m eeg_research_agent.cli report`.

## Failure philosophy

A stage that fails does not silently degrade the result:

1. the exception is recorded in `state.errors` / `artifacts.errors` and in the
   event log;
2. `run_all` stops and re-raises, so a partial run never looks like a complete one;
3. in the LangGraph workflow a failed node takes an explicit conditional edge
   (for example `acquire_dataset → publish_artifacts`) instead of continuing
   with missing state;
4. any claim whose artifact cannot be re-read fails verification and is excluded
   from the "validated" summary.
