# Environment notes and known constraints

This page records **exactly** how the reference environment was built, what could
not be done, and every deviation from the requested specification.  It exists so
that a reader can reproduce the build and can tell an environment limitation
apart from a modelling choice.

## 1. Python version: requested 3.14, executed 3.13

The brief requests Python 3.14.  The environment was created with the requested
interpreter first:

```bash
conda create -n py314-probe python=3.14 -y     # succeeded: Python 3.14.8
```

The blocker is the scientific stack, not conda.  In this sandbox **PyPI and the
conda channels are unreachable** (DNS resolution fails; package downloads are
refused).  The interpreter can be created, but no `cp314` wheels for
`numpy`, `scipy`, `scikit-learn`, `pandas` or `matplotlib` exist in the local
channel or the on-disk wheelhouse, and they cannot be fetched.  The probes exit
with:

```
ModuleNotFoundError: No module named 'numpy'
```

Decision (documented rather than silent): the project runs on **Python 3.13.5**
and the requested 3.14 path is kept as a documented, one-line change:

* `environment.yml` → change `python=3.13` to `python=3.14`.
* `pyproject.toml` → `requires-python = ">=3.13,<3.15"` already permits 3.14.
* `[tool.ruff] target-version` / `[tool.mypy] python_version` → bump together.

The evidence for the 3.14 probe is kept at `.conda/envs/py314-probe/`
(git-ignored) so the claim above can be re-checked locally.

No source file uses a 3.13-only feature: the code is `__future__`-annotated,
avoids PEP 695 generics, and does not rely on `typing.TypeIs`.  Moving to 3.14
should therefore be a version bump, not a port.

## 2. Network

There is **no outbound network access** in the reference sandbox.  Consequences,
all handled explicitly by the code rather than hidden:

| Capability | Behaviour offline |
| --- | --- |
| Sleep-EDF download (PhysioNet) | Falls back to the deterministic synthetic PSG generator; every artifact is tagged `data_origin: synthetic`. |
| Literature retrieval (Semantic Scholar / OpenAlex / Crossref / PubMed / arXiv) | Skipped with a recorded blocker; the bibliography contains only the brief's anchor sources at `evidence_level: unverified-offline`. |
| Anthropic / DeepSeek API | Unreachable; the LLM layer uses the deterministic `OfflineProvider` and labels its output `generated_by: offline-template`. |
| GitHub push | Not possible; the repository is committed locally only. |

The pipeline is designed so that **none of these are prerequisites**: the whole
deterministic analysis runs with zero credentials and zero network.

## 3. Packages that are absent and their workarounds

Everything below is importable in the reference environment.  Where a required
package could not be obtained, the workaround is visible in the source tree
rather than papered over.

| Package | Status | Workaround |
| --- | --- | --- |
| `pyarrow` | absent | Feature tables are written as `CSV.gz` instead of Parquet. `feature_extraction` logs the downgrade. |
| `statsmodels` | absent | Mixed-effects models are not used; the repeated-measures requirement is met with Friedman / Wilcoxon on subject x stage means, plus cluster permutation tests (see `docs/statistical_methodology.md`). |
| `PyWavelets` | absent | `eeg/wavelets.py` implements a Daubechies-4 DWT/IDWT by hand and is verified by a perfect-reconstruction test (`max |x − idwt(dwt(x))| < 1e-9`). |
| `antropy` | absent | Entropy and fractal features are implemented directly in `eeg/nonlinear.py` and unit-tested against synthetic signals with known answers. |
| `ruff`, `mypy`, `pre-commit` | absent | Configuration is committed and CI-ready, but the tools cannot be run in this sandbox. Formatting was kept hand-clean; `make lint` documents the intended invocation. |
| `iniconfig` | absent | A minimal API-compatible shim was written into `site-packages` (documented in its own docstring) because `pytest` imports it at start-up. |
| `lazy_loader` | absent | Copied from a sibling environment; only imports the real module on first attribute access. |
| `tabulate` | absent | Tables are rendered by `reporting/markdown_utils.py` instead of via pandas' optional `tabulate` dependency. |

## 4. Exact versions

Frozen in `requirements-lock.txt`.  Core versions of the verified environment:

| Package | Version |
| --- | --- |
| Python | 3.13.5 (Anaconda build) |
| numpy | 2.3.1 |
| scipy | 1.16.0 |
| scikit-learn | 1.7.1 |
| pandas | 2.3.1 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
| mne | 1.6.1 |
| pydantic | 2.10.3 |
| httpx | 0.28.1 |
| langgraph | 1.0.7 |
| pytest | 9.1.1 |

## 5. Rebuilding the environment

When a package index *is* reachable, the standard path is:

```bash
conda env create -f environment.yml
conda activate eeg-research-agent
python -m pip install -e .
```

Offline, the reference environment was assembled from a local channel:

```bash
export CONDA_NO_PLUGINS=true CONDA_PKGS_DIRS=/tmp/condapkgs HOME=/tmp/condahome
conda create -p ./.conda/envs/eeg-research-agent -c file://$PWD/.conda/channel \
  python=3.13 numpy scipy scikit-learn pandas matplotlib seaborn pyyaml jinja2 pytest -y
./.conda/envs/eeg-research-agent/bin/python -m pip install -e . --no-build-isolation --no-deps --no-index
```

Two environment variables are worth exporting for any run, because matplotlib
and joblib otherwise try to write outside the sandbox:

```bash
export MPLCONFIGDIR=/tmp/mplcache
export LOKY_MAX_CPU_COUNT=4
```
