# Designing an agentic system - notes from building featurelab

These notes walk through each major decision in the order it was made, with the
general principle behind it. The principles transfer to almost any multi-agent
system; the EEG details are the example.

---

## Step 0 - Audit before you build

**What happened.** The first version of this repo (commit `3e73ecd`, removed from the tree) looked
impressive: 22k lines, 18 LangGraph nodes, "83 checks, 0 failures". Tracing the
data flow showed that the LLM's output was discarded (`requested_by_model` was
only a flag; extraction always used the full fixed registry), no feature was
ever invented, and the only run used synthetic data with the answer built in.
The real-data path would have kept only the first 30 minutes of each recording,
which is almost all wake.

**Principle - verify behaviour, not descriptions.** Agent-written code optimises
for what is easy to check (tests pass, reports render), not for what you meant.
Follow one input from start to end and ask "does this value actually influence
the output?"

**Interview line.** *"Before extending a system I trace one request end to end.
In this project that showed the LLM's decisions never reached the computation -
the 'agent' was decorative."*

---

## Step 1 - Define "success" before designing agents

You defined "useful" as *reducing the N1/N2/REM confusions that single-channel
EEG suffers from*. That became a measurable endpoint:

* **primary metric**: per-subject macro-F1 over N1, N2, REM;
* **test**: does baseline + candidate beat baseline, paired per subject?
* **secondary**: confusion counts for N1-REM, N1-N2, N2-REM, N1-W.

**Principle - evaluation first.** The evaluator *is* the product; agents are
just a way to produce candidates for it. If you cannot write the acceptance test,
you are not ready to build agents.

**Interview line.** *"I write the eval before the agent. Otherwise you can't tell
improvement from noise, and you end up tuning prompts by vibes."*

---

## Step 2 - Decide what is an LLM and what is code

| Task | Who | Why |
| --- | --- | --- |
| retrieve papers | code (OpenAlex API) | deterministic, cheap, verifiable |
| read abstracts, extract ideas | LLM (Scout) | language understanding |
| invent formula + write code | LLM (Inventor) | open-ended creativity |
| review code vs formula | LLM (Critic) | reading comprehension; catches bugs early |
| check code safety / contract | code (AST gate + sandbox) | must be exact |
| decide if the feature works | code (the judge) | must be reproducible and unbiased |
| control flow, retries, stopping | code (LangGraph) | must be predictable and bounded |
| write the report | code (template) | a template cannot hallucinate a number |

Your original idea had "an agent measures the statistics". Here that is a
deterministic *tool*, not an LLM. Calling it an agent is fine; letting an LLM
compute or judge the numbers is not.

**Principle - LLMs for judgement and creativity, code for everything that must
be exact.** Every LLM call adds cost, latency and variance; spend them where
nothing else works.

---

## Step 3 - Separate the generator from the judge

The inventor never sees evaluation data. It gets the baseline's per-stage F1, the
confusion counts, inspiration cards, and a *coarse* history ("no reliable gain
(+0.003)"). It never sees per-subject scores or p-values.

**Why.** If a generator can see detailed scores, a loop of hundreds of attempts
becomes an optimiser that fits the noise of the evaluation set ("adaptive
overfitting" - the same reason Kaggle hides the private leaderboard).

**Defences in featurelab:** coarse feedback; Bonferroni over a declared budget of
50 Stage-B tests; a lockbox that can be opened at most 10 times per protocol; an
external cohort from a different study.

**Interview line.** *"Any loop that generates and evaluates many candidates is an
optimiser against its evaluation set. I limit what the generator sees and keep a
held-out set it can't touch."*

---

## Step 4 - Narrow interfaces between agents and code

The feature contract is one function: `feature(x, fs, freqs, psd) -> float`.

* It **can't leak**: no labels, no other epochs, no other subjects.
* It is **testable**: determinism, NaN rate, speed and scale-invariance are
  measured automatically.
* It is **easy for LLMs and teammates** to write: plain numpy/scipy.

Between LLM agents, messages are **pydantic objects** (`FeatureProposal`,
`CriticReview`, `InspirationCard`), validated on arrival, with one repair round
("your reply failed validation because ...") before the reply is dropped.

A practical detail: the inventor returns **JSON metadata plus a separate
` ```python ` block**. Forcing code through a JSON string causes escaping errors
with weaker models.

**Principle - contracts, not conversations.** Free-text hand-offs between agents
fail silently. Typed hand-offs fail loudly and early.

---

## Step 5 - Defence in depth for generated code

1. **Static gate** (`features/gate.py`): parse the AST and reject forbidden
   imports, `open`/`exec`/`eval`, dunder access, randomness, top-level logic.
   It also gives the inventor an exact error to repair.
2. **Process isolation** (`features/runner.py`): a separate process with a
   stripped environment (no API keys), a CPU-time limit and a timeout, plus
   restricted builtins and a guarded `__import__`.
3. **Container** (Docker): the real security boundary.

Be honest that (1) is not a sandbox. Python is too dynamic for that. It is a
filter for mistakes; (2) and (3) handle the rest.

**Interview line.** *"LLM-generated code is untrusted input. I validate
statically for fast feedback, isolate at runtime for safety, and treat the
container as the real boundary."*

---

## Step 6 - Test the tester (controls)

`featurelab controls` builds a synthetic cohort where N1 and REM have the same
average spectrum but REM theta is amplitude-modulated (the planted effect).

* The **positive control** (theta burstiness) must pass.
* The **negative control** (digits of a sample value) must fail.

This immediately found three real problems:

1. The first synthetic world was too easy (baseline F1 about 1.0), so nothing
   could show added value. Controls must be calibrated so the baseline is
   imperfect.
2. The negative control **passed the screen** on a smaller cohort: noise produced
   |AUC - 0.5| > 0.05 on a stage pair with few epochs. The fix was principled
   rather than a threshold tweak: a pair now counts only if the shift is more than
   3 null standard errors *and* the direction holds in at least 70 % of subjects.
3. With 14 lockbox subjects, a true effect failed confirmation at
   α = 0.005. That is a *power* issue. Tests now pre-declare fewer looks;
   real runs use 20 lockbox subjects.

**Principle - an evaluator you have never seen fail is an evaluator you cannot
trust.** Plant something you know is there; inject something you know is not.

---

## Step 7 - Orchestration: an explicit, bounded state machine

```
scout → baseline → invent → critique → gate → screen → incremental → confirm → record
                      ↑ revise ↙   ↑ repair ↙                                    │
                      └──────────────────── decide (stop rules) ◄─────────────────┘
```

* **Routing functions are small and pure** (`after_gate`, `after_screen`, ...).
  They read state; they never change it. In LangGraph, changes made inside a
  router are discarded (a bug I hit and fixed: stop rules now live in the
  `record` node).
* **Every loop has a bound**: repairs ≤ 2, critic revisions ≤ 1, iterations,
  daily LLM calls, Stage-B budget, lockbox looks.
* **State is small**: ids, short strings, file paths. Feature columns and
  reports live on disk.
* **One implementation of the science**: the agent loop and `featurelab evaluate`
  (human features) call the same `steps.py`.

**Principle - autonomy with bounds.** An agent loop without explicit stop
conditions is a bill, not a system.

---

## Step 8 - Memory: the ledger

A single SQLite table of every hypothesis provides:

* **memory** for the inventor (what was tried, why it failed);
* **dedup** (code hash), so the same idea is never paid for twice;
* **statistical accounting** (Stage-B tests and lockbox looks used);
* **an audit trail** for reports and teammates.

LLM calls are logged in a second table: role, model, latency, tokens, failures.
That is your observability: cost per accepted feature, failure rate per model.

**Principle - memory is a database, not a growing prompt.** Put structured
history in a store and inject a *summary* into the context.

---

## Step 9 - Living with free models

* **Per-role model lists** with automatic fallback to any model that is free
  today (the free catalogue changes weekly).
* **Rate limiting** (one call every 4 s) and a **daily cap** (45 calls).
* **On-disk cache** keyed by (model, prompt, temperature): re-runs are free.
* **Model diversity**: the critic uses a different family from the inventor, so
  they don't share blind spots.
* **Temperature by role**: inventor 0.9 (creative), critic 0.1, scout 0.2.

**Budget arithmetic.** One hypothesis costs about 2-5 calls (invent, critique,
plus any repair or revision). With 45 calls a day, that is about 10-15
hypotheses a day on the free tier.

---

## Step 10 - Retrieval quality beats model quality

The first literature search sorted a loose keyword search by citation count.
The top "paper" was a famous cognitive-science essay unrelated to sleep. The
fixes were a stricter `title_and_abstract.search` filter and a deterministic
relevance guard (it must mention sleep *and* EEG). The LLM can also only cite
paper ids it was given; any other card is dropped (a grounding check).

**Principle - in RAG-style agents, most quality problems are retrieval
problems.** Look at what you retrieved before tuning prompts.

---

## Step 11 - Production details that matter

* **Reproducibility**: one seed; the protocol hash is stamped on every ledger
  row; data files are SHA256-verified.
* **Resumability**: idempotent download (verified files are skipped);
  cached baselines; cached LLM answers.
* **Network realism**: S3 reset 80 % of parallel downloads. The fixes were
  exponential backoff, socket timeouts and fewer workers. A hung download had no
  timeout - always set one.
* **Team usability**: one CLI, Docker, `featurelab evaluate` for humans,
  `featurelab export` for ML/DL users, and a bring-your-own-cohort format.

---

## Reusable checklist for any multi-agent system

1. What is the measurable definition of success? Write the eval first.
2. Which steps need an LLM? Make everything else code.
3. Who generates and who judges? Keep them separate; limit what the generator sees.
4. What are the contracts between components? Use typed schemas and validate on arrival.
5. What can go wrong with LLM output? Plan for parsing, hallucination,
   unsafe code, and refusal, with a bounded repair for each.
6. Where are the loops, and what bounds each one?
7. Where does memory live? Use a store, and inject summaries.
8. How do you know the evaluator works? Use positive and negative controls.
9. Observability: can you answer "which model failed, how often, at what cost"?
10. What does a human do? Review, approve the lockbox, submit their own ideas.

---

## Likely interview questions (with short answers)

**Why multi-agent instead of one big prompt?** Different roles need different
context, temperature and even different models (the critic should not share the
inventor's blind spots). Separating them also lets me test each role with a
scripted stand-in. But I add an agent only when it fixes a measured failure; the
report writer is a template, not an agent.

**How do you stop hallucination?** By architecture rather than prompting: the
LLM never produces a number that reaches a report, citations must reference
retrieved ids, code must pass a gate and run, and acceptance is decided by
deterministic statistics.

**How do you evaluate the agent system itself?** At three levels: (1) unit
tests per role with scripted clients; (2) system tests in which the scripted
loop must accept the planted feature and reject noise; (3) run metrics from the
ledger: acceptance funnel, repair rate, cost per accepted feature, and how often
the inventor's pre-registered prediction held (a measure of its physiological
reasoning).

**What if the LLM writes malicious code?** AST gate, isolated process without
secrets and with resource limits, container as the boundary. I'd add
`--network none` for the runner container in production.

**How do you handle flaky or rate-limited APIs?** Per-role fallback chains,
backoff, a response cache, a daily cap and graceful degradation: if the critic
is down, the statistics still decide; if the scout is down, the inventor runs
without literature.

**What is the biggest risk in this system?** Adaptive overfitting: hundreds of
generated ideas tested against one dataset. Hence coarse feedback, a declared
testing budget, a lockbox with a limited number of looks, and an external cohort.

**Why LangGraph?** Explicit conditional edges, typed state, and the option of
checkpointing and human-in-the-loop interrupts. A plain `while` loop would work;
the graph makes routing inspectable and testable, which matters when the team
grows.

**What would you do next?**
* Human approval before each lockbox look (a LangGraph `interrupt`).
* Online FDR (LORD++) instead of Bonferroni, for more power.
* A deep sequence model as a stronger baseline.
* Run the gate container with `--network none`.
* Evaluate on ear-EEG data as the final external cohort.
