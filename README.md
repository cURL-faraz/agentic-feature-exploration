# featurelab

**A multi-agent system that invents new single-channel EEG features and proves - or disproves -
that they help sleep staging.**

Single-channel EEG sleep stagers confuse **N1, N2 and REM**, because the EOG and EMG channels that
human scorers rely on are missing. featurelab lets LLM agents read the literature, brainstorm
mathematical features aimed at exactly those confusions, write them as code, and then hands every
idea to a strict, deterministic statistical judge. A feature is *accepted* only if it adds
predictive value on top of a standard baseline, on people the model has never seen.

It is built for a team: ML people get a validated feature table, deep-learning people get extra
input channels/targets, and anyone can submit their own feature to the same judge.

```
literature ──► Scout (LLM) ──► inspiration cards
                                     │
          ┌──────────────────────────▼───────────────────────────┐
          │  Inventor (LLM)  formula + prediction + Python code  │◄─── coarse feedback
          └──────────────────────────┬───────────────────────────┘     from the ledger
                 Critic (LLM, other model family) reviews the code
                                     │
   ┌──────────────── deterministic judge (no LLM) ─────────────────────┐
   │ gate: AST check + sandboxed run   →  A: screen (cheap)             │
   │ → B: added value, subject-grouped CV, per-subject Wilcoxon         │
   │ → C: lockbox (unseen subjects) + external cohort                   │
   └──────────────────────────────┬────────────────────────────────────┘
                                  ▼
                   ledger (SQLite) → reports → exported feature tables
```

---

## 1. Setup

You need ~6 GB of free disk for the data and either **conda** or **Docker**.

### Option A - conda (what the maintainers use)

```bash
conda activate EEG_Sleep_Staging          # or: conda create -n featurelab python=3.12 && conda activate featurelab
pip install -e ".[dev]"
featurelab --help
```

### Option B - Docker (identical environment on every machine)

```bash
docker compose build
docker compose run --rm featurelab --help
```

Every command below works the same way with `docker compose run --rm featurelab <command>`.
Data is stored in `./data/featurelab` on your machine (mounted at `/data` in the container).

### API key (only for the LLM agents)

```bash
cp .env.example .env        # then paste your key from https://openrouter.ai/keys
featurelab models           # shows which of the configured models are free today
```

`.env` needs only two lines:

```bash
OPENROUTER_API_KEY=sk-or-v1-...                      # required for the agents
FEATURELAB_LLM_MODEL=nvidia/nemotron-3.5-super-vl:free  # optional: tried first by every agent
```

`FEATURELAB_LLM_MODEL` puts your model at the front of every role's list; the per-role lists in
`default_config.yaml` stay behind it as fallbacks. One model for all roles is simple, but the
critic then shares the inventor's blind spots; to keep a different family for the critic, leave
the variable empty and edit `llm.roles` in `featurelab.yaml` instead (section 5).

Free OpenRouter models are enough. They are rate limited (roughly 20 requests/minute and 50/day
without credits), so the client spaces calls, caches every answer, counts calls per day and falls
back to other free models automatically. Everything except the LLM agents works **without a key**.

## 2. Five-minute tour

```bash
featurelab controls                        # 1. prove the judge works (no data, no key; ~1 min)
featurelab download sc20 st sc_lockbox     # 2. Sleep-EDF from PhysioNet, checksum-verified (~5 GB, slow)
featurelab prepare                         # 3. EDF -> 30-s epochs, cached as numpy arrays
featurelab baseline                        # 4. how good is the standard feature set?
featurelab run --provider scripted         # 5. the full agent loop with canned answers (no key)
featurelab run --iterations 10             # 6. the real thing, with free LLMs
featurelab ledger                          # every hypothesis ever tested, and why it passed/failed
featurelab export --out exports/           # accepted features as CSV, per cohort
```

Each run writes a readable report to `data/featurelab/runs/<run_id>/report.md`.

### Test your own feature idea

Write one function in a file, e.g. `my_feature.py`:

```python
import numpy as np

def feature(x, fs, freqs, psd):
    """Ratio of 2-6 Hz power to 8-12 Hz power."""
    low = psd[(freqs >= 2) & (freqs < 6)].sum()
    alpha = psd[(freqs >= 8) & (freqs < 12)].sum()
    return float(np.log((low + 1e-12) / (alpha + 1e-12)))
```

```bash
featurelab evaluate my_feature.py --higher-in N1 --lower-in W
```

It goes through exactly the same gate and statistics as an LLM-invented feature, and is recorded in
the same ledger. `--higher-in/--lower-in` is your *pre-registered* prediction (optional).

**The contract**: `x` is one 30-s epoch (3000 samples, microvolts, 100 Hz), `freqs`/`psd` its Welch
spectrum (0-50 Hz, 0.25 Hz steps). Return one float. Only `numpy`, `scipy`, `math`. No randomness,
no files, no other epochs (the judge adds neighbouring-epoch context for you).

### Bring your own data (e.g. ear-EEG)

Any dataset becomes a cohort with one call - then every command works on it:

```python
from featurelab.data.cohort import write_cohort
write_cohort(root, "ear_eeg_pilot", X, y, subject, recording, epoch_idx, sfreq=100)
# X: (n_epochs, 3000) float, y: 0=W 1=N1 2=N2 3=N3 4=REM
```

and point a role at it in `featurelab.yaml`:

```yaml
data:
  cohorts: {external: ear_eeg_pilot}
```

## 3. Concepts in 10 minutes

**Agent vs. workflow.** An *agent* is an LLM that decides what to do next; a *workflow* is code that
decides. featurelab uses agents only where the task is open-ended (reading papers, inventing
formulas, reviewing code) and plain code everywhere else. The orchestrator (a LangGraph state
machine) owns control flow; the LLMs never decide whether their own idea passed.

**Generator / judge separation.** The inventor proposes; a deterministic judge decides. The
inventor never sees evaluation data, only coarse outcomes ("no reliable gain"). If it could see
detailed scores, the loop would slowly tune ideas to the noise of one dataset.

**Pre-registration.** Before any number is computed, the inventor must state *which stage the
feature should be higher in*. All acceptance thresholds live in
[`default_config.yaml`](src/featurelab/default_config.yaml), and the ledger stamps every result with
a hash of that protocol. Changing a threshold after seeing results means starting a new protocol.

**The subject is the unit.** 40 000 epochs from 20 people are not 40 000 independent samples.
Splits are by subject (both nights of a person stay together), and every test uses one number per
subject. Testing epochs as if independent gives tiny, meaningless p-values.

**Added value, not association.** Almost every EEG feature differs between stages somewhere.
The question is: *does it reduce N1/N2/REM errors beyond what standard features already achieve?*
So each candidate is compared to the same baseline model, on the same folds, with the gain measured
per subject.

**Epoch context.** Human scorers look at the epochs before and after; the model does too (the
previous and next epoch's features are inputs). A candidate gets credit only for information that
context does not already provide.

**Multiple testing.** Try 50 random features and a few will look "significant" at p < 0.05 by luck.
Each Stage-B test therefore uses α / 50 (Bonferroni over a declared budget), and the lockbox can be
opened only 10 times per protocol, each look using α / 10.

**Lockbox and external data.** Discovery happens on Sleep-EDF-20 (subjects 0-19). Features that
pass are confirmed *once* on subjects 20-39 (never seen during the search) and checked on Sleep
Telemetry (a different study and population). That is what turns "it worked here" into evidence.

**Controls.** `featurelab controls` hides a known effect in synthetic data and checks the judge
finds it, then checks the judge rejects pure noise. Never trust an evaluator you have not seen fail
and succeed on purpose.

More depth, including the design reasoning behind each decision:
[docs/AGENTIC_DESIGN_NOTES.md](docs/AGENTIC_DESIGN_NOTES.md).

## 4. Data

| Role | Cohort | Subjects | Why |
| --- | --- | --- | --- |
| discovery | Sleep-EDF SC 0-19 ("Sleep-EDF-20") | 20 | the standard single-channel benchmark |
| lockbox | Sleep-EDF SC 20-39 | 19 (no subject 39 in the dataset) | same population, unseen during search |
| external | Sleep-EDF Telemetry (ST) | 22 | different study, includes temazepam nights |

**Reference baseline** (Sleep-EDF-20, 42 307 epochs, 5-fold subject-grouped CV, 26 standard
features + neighbouring-epoch context, gradient boosting):

| accuracy | macro-F1 | Cohen's κ | F1 W | F1 N1 | F1 N2 | F1 N3 | F1 REM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.802 | 0.741 | 0.732 | 0.833 | **0.392** | 0.869 | 0.875 | 0.735 |

Biggest confusions (epochs, both directions): N1↔REM 1656, N2↔REM 1613, N1↔W 1208, N1↔N2 1048.
That is the gap the agents are trying to close.

**First real-data verdicts** (the two hand-written features in [`examples/`](examples/)): both target
N1 vs REM, and both were **rejected**. On real Fpz-Cz neither separates N1 from REM (AUC 0.52 and
0.48). They do separate other pairs (eye-movement index: N1 vs W, AUC 0.80), but their added value
over the baseline is small and unproven: +0.006 and +0.004 in N1/N2/REM F1, p = 0.053 and 0.066
against a threshold of 0.001. The theta-burstiness feature had *passed* on the synthetic world
where that effect was planted, so this is a reminder that the synthetic world tests the machinery,
not the physiology.

Channel Fpz-Cz at 100 Hz; R&K stages 3+4 merged into N3; movement/unknown epochs dropped; 30 min of
wake kept around sleep. Other common benchmarks (SHHS, MASS) require a data-use agreement - once you
have them, convert them with `write_cohort` and use them as extra external cohorts.

## 5. Configuration

Copy only what you want to change into `featurelab.yaml` in your working directory, e.g.

```yaml
loop: {max_iterations: 20, target_accepted: 2}
llm:
  roles:
    inventor: ["nvidia/nemotron-3.5-super-vl:free", "nvidia/nemotron-3-ultra-550b-a55b:free"]
    critic:   ["google/gemma-4-31b-it:free"]
```

Thresholds under `evaluation:` are the pre-registered protocol: changing them changes the protocol
hash, and results under different hashes are never mixed.

## 6. What each part does

```
src/featurelab/
  config.py + default_config.yaml   the protocol: data roles, thresholds, models, loop limits
  cli.py                            the `featurelab` command (one subcommand per step)
  data/
    download.py    fetches Sleep-EDF from PhysioNet, verifies SHA-256, resumes safely
    sleepedf.py    EDF + hypnogram -> 30-s Fpz-Cz epochs, stage mapping, wake trimming
    cohort.py      the on-disk cohort format (X.npy, labels, subjects, Welch PSD) - add your own data here
    synthetic.py   a fake cohort with a planted effect, for the positive/negative controls
  features/
    contract.py    what a feature function may receive, import and return
    baseline.py    the 26 standard features every candidate must beat
    gate.py        static check of generated code (imports, dunders, randomness, signature)
    runner.py      runs feature code in a separate process: no keys, CPU/time limits, probes
  evaluation/
    modeling.py    neighbouring-epoch context, subject-grouped folds, the classifier
    metrics.py     N1/N2/REM F1 per subject, confusion pairs, paired Wilcoxon test
    stages.py      the judge: A screen -> B added value + shuffled null -> C lockbox/external
  llm/client.py    OpenRouter access: per-role model lists, fallback, rate limit, daily cap, cache
  agents/
    literature.py  OpenAlex search: most-cited + recent sleep-EEG papers
    prompts.py     system prompts for each role
    schemas.py     pydantic shapes of every LLM answer (validated, repaired once)
    roles.py       Scout (papers -> inspiration cards), Inventor (card -> formula + code),
                   Critic (reviews the proposal before any compute is spent)
    scripted.py    canned answers so the loop runs offline (tests, demos)
  orchestrator/
    graph.py       the LangGraph state machine: who runs next, and when to stop
    steps.py       glue between graph nodes and the judge (gate+run, lockbox budget)
    ledger.py      SQLite memory: every hypothesis, verdict and LLM call; dedup; test budget
    report.py      run reports (markdown) and feature-table export
tests/             15 offline tests (~90 s): gate, parsing, judge controls, the full loop
examples/          two hand-written features to copy when writing your own
docs/              design notes: why each decision was made, interview Q&A
```

The first, single-agent version of this project is no longer in the tree; it is preserved in git
history at commit `3e73ecd` (`git show 3e73ecd:README.md`).

## 7. Limitations (read before presenting results)

* Fpz-Cz is a scalp proxy for ear-EEG, not ear-EEG; scale-invariant features are preferred for
  transfer and flagged in every report.
* Discovery uses 20 subjects: small gains are hard to prove. That is deliberate - it is better to
  miss a weak feature than to "discover" noise.
* The AST gate is not a security sandbox; run untrusted LLM code inside Docker.
* Free LLMs are slower and less reliable than paid ones; the loop is built to tolerate that, not to
  hide it (failed calls are logged in the ledger).
* Baseline model is gradient boosting on hand-crafted features; a deep sequence model would be a
  stronger baseline and is the natural next step for the DL side of the team.

MIT licence.
