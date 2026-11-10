# Convenience targets.  The pipeline itself is driven by the CLI, so every
# target here is a thin wrapper over `python -m eeg_research_agent.cli`.
#
# The reference environment lives in ./.conda/envs/eeg-research-agent; when the
# environment is activated normally `PYTHON` is just `python`.

PYTHON ?= python
MODE   ?= standard

export MPLCONFIGDIR ?= /tmp/mplcache
export LOKY_MAX_CPU_COUNT ?= 4

.PHONY: help env install test test-all lint format run smoke agentic report verify \
        statistics features literature clean freeze

help:
	@echo "make env        create the conda environment from environment.yml"
	@echo "make install    editable install into the active environment"
	@echo "make test       run the offline test suite (no network, no API key)"
	@echo "make run        full deterministic pipeline (MODE=$(MODE))"
	@echo "make smoke      fast end-to-end run"
	@echo "make agentic    run the LangGraph workflow"
	@echo "make report     regenerate reports from saved artifacts"
	@echo "make verify     run the verification suite against saved artifacts"
	@echo "make freeze     write the exact installed versions"
	@echo "make clean      remove caches and generated artifacts"

env:
	conda env create -f environment.yml

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests/ -q

test-all:
	$(PYTHON) -m pytest tests/ -q -m "slow or not slow"

lint:
	# ruff and mypy are declared in pyproject.toml but are not installed in the
	# offline reference environment (see docs/environment_notes.md).
	-$(PYTHON) -m ruff check src tests
	-$(PYTHON) -m ruff format --check src tests
	-$(PYTHON) -m mypy src

format:
	-$(PYTHON) -m ruff format src tests
	-$(PYTHON) -m ruff check --fix src tests

run:
	$(PYTHON) -m eeg_research_agent.cli run --mode $(MODE)

smoke:
	$(PYTHON) -m eeg_research_agent.cli run --mode smoke

agentic:
	$(PYTHON) -m eeg_research_agent.cli run --agentic --mode $(MODE)

statistics:
	$(PYTHON) -m eeg_research_agent.cli statistics --mode $(MODE)

features:
	$(PYTHON) -m eeg_research_agent.cli features --mode $(MODE)

literature:
	$(PYTHON) -m eeg_research_agent.cli literature

verify:
	$(PYTHON) -m eeg_research_agent.cli verify --mode $(MODE)

report:
	$(PYTHON) -m eeg_research_agent.cli report --mode $(MODE)

freeze:
	$(PYTHON) -m pip freeze > requirements-lock.txt

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -f artifacts/execution_logs/*.jsonl artifacts/execution_logs/*.log
	find data/features -name '*.csv.gz' -delete
