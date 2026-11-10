"""Command-line interface.

The CLI is a thin shell over :class:`~.pipeline.ResearchPipeline` (deterministic
stages) and :func:`~.agents.graph.run_research_workflow` (the agentic workflow).
It never contains analysis logic, and it never prints a secret:
``llm-check`` reads a key with :func:`getpass` and writes it to ``.env`` with
owner-only permissions.

Examples
--------
``python -m eeg_research_agent.cli run --mode smoke``
``python -m eeg_research_agent.cli run --agentic``
``python -m eeg_research_agent.cli literature --offline``
``python -m eeg_research_agent.cli llm-check --provider anthropic``
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import ProjectConfig, load_config
from .logging_utils import get_logger
from .pipeline import ResearchPipeline

__all__ = ["main", "build_parser", "configure"]

logger = get_logger(__name__)

PROVIDER_KEY_VAR = {"anthropic": "ANTHROPIC_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def configure(args: argparse.Namespace) -> ProjectConfig:
    """Apply command-line overrides to the loaded configuration."""
    config = load_config()
    updates: dict[str, Any] = {}
    execution = config.settings.execution
    dataset = config.dataset
    changed_execution = False
    changed_dataset = False
    if getattr(args, "seed", None) is not None:
        updates["seed"] = int(args.seed)
    if getattr(args, "max_subjects", None) is not None:
        execution = execution.model_copy(update={"max_subjects": int(args.max_subjects)})
        changed_execution = True
    if getattr(args, "mode", None):
        execution = execution.model_copy(update={"mode": args.mode})
        changed_execution = True
    if getattr(args, "no_cache", False):
        execution = execution.model_copy(update={"use_cache": False})
        changed_execution = True
    if getattr(args, "channel", None):
        preference = [args.channel, *[c for c in dataset.signal.channel_preference if c != args.channel]]
        dataset = dataset.model_copy(update={"signal": dataset.signal.model_copy(update={"channel_preference": preference})})
        changed_dataset = True
    if changed_execution or changed_dataset:
        updates["settings"] = config.settings.model_copy(
            update={"execution": execution, "dataset": dataset}
        )
    return config.model_copy(update=updates) if updates else config


def _pipeline(args: argparse.Namespace) -> ResearchPipeline:
    config = configure(args)
    return ResearchPipeline(
        config=config,
        origin=getattr(args, "origin", None),
        mode=getattr(args, "mode", None),
        use_cache=None if not getattr(args, "no_cache", False) else False,
        max_subjects=getattr(args, "max_subjects", None),
        quiet=getattr(args, "quiet", False),
    )


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _prepare_stages(args: argparse.Namespace) -> list[str]:
    return ["initialise", "acquire", "inspect", "prepare"]


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_init(args: argparse.Namespace) -> int:
    """Create the directory layout, the event log and the execution manifest."""
    _emit(_pipeline(args).initialise())
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Acquire the dataset (Sleep-EDF when reachable, else the synthetic surrogate)."""
    pipeline = _pipeline(args)
    pipeline.initialise()
    pipeline.acquire()
    _emit(pipeline.inspect())
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print the dataset summary produced by the dataset-intelligence stage."""
    pipeline = _pipeline(args)
    pipeline.initialise()
    pipeline.acquire()
    _emit(pipeline.inspect())
    return 0


def cmd_literature(args: argparse.Namespace) -> int:
    """Run the literature stage (live search, or a labelled offline bibliography)."""
    pipeline = _pipeline(args)
    pipeline.initialise()
    payload = pipeline.literature(
        target_papers=getattr(args, "papers", None), force_offline=bool(getattr(args, "offline", False))
    )
    _emit(
        {
            "mode": payload["mode"],
            "n_papers": payload["n_papers"],
            "retrieval_available": payload["retrieval_available"],
            "retrieval_blockers": payload["retrieval_blockers"],
            "artifacts": payload["artifacts"],
        }
    )
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    """Preprocess, epoch, split and extract the registered feature catalog."""
    pipeline = _pipeline(args)
    pipeline.initialise()
    pipeline.acquire()
    pipeline.inspect()
    pipeline.prepare()
    _emit(pipeline.artifacts.results["features"])
    return 0


def cmd_statistics(args: argparse.Namespace) -> int:
    """Run the statistical comparisons over the extracted feature table."""
    pipeline = _pipeline(args)
    for stage in _prepare_stages(args):
        getattr(pipeline, stage)()
    _emit(pipeline.statistics())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run the subject-grouped predictive evaluation."""
    pipeline = _pipeline(args)
    for stage in _prepare_stages(args):
        getattr(pipeline, stage)()
    _emit(pipeline.evaluate())
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Run the full verification suite (synthetic, statistical, ML, leakage)."""
    pipeline = _pipeline(args)
    for stage in [*_prepare_stages(args), "statistics", "evaluate"]:
        getattr(pipeline, stage)()
    _emit(pipeline.verify())
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Regenerate every report from the saved artifacts."""
    pipeline = _pipeline(args)
    pipeline.initialise()
    pipeline.acquire()
    pipeline.inspect()
    pipeline.prepare()
    pipeline.literature()
    pipeline.statistics()
    pipeline.evaluate()
    pipeline.verify()
    _emit(pipeline.report())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the complete workflow: deterministically, or through the agent graph."""
    if getattr(args, "agentic", False):
        from .agents.graph import run_research_workflow

        summary = run_research_workflow(
            config=configure(args),
            mode=getattr(args, "mode", None) or "standard",
            origin=getattr(args, "origin", None),
        )
        _emit(summary)
        return 0 if summary.get("ok") else 1
    pipeline = _pipeline(args)
    summary = pipeline.run_all()
    _emit(summary)
    return 0 if not summary.get("errors") else 1


def cmd_tools(args: argparse.Namespace) -> int:
    """Print the agent tool table and its role permissions."""
    from .agents.tools import tool_table

    _emit(tool_table())
    return 0


def cmd_llm_check(args: argparse.Namespace) -> int:
    """Show the effective LLM configuration and optionally store an API key."""
    from .llm import build_llm_client, provider_status

    provider = getattr(args, "provider", None) or "offline"
    status = provider_status(provider, getattr(args, "model", None))
    if status["requires_api_key"] and not status["api_key_present"]:
        if getattr(args, "no_prompt", False):
            status["action"] = f"set {status['api_key_env_var']} in the environment or .env"
            _emit(status)
            return 1
        key = getpass.getpass(f"Enter {status['api_key_env_var']} (input hidden): ").strip()
        if not key:
            status["action"] = "no key entered; nothing was stored"
            _emit(status)
            return 1
        _store_secret(status["api_key_env_var"], key)
        os.environ[status["api_key_env_var"]] = key
        status["api_key_present"] = True
        status["action"] = "key stored in .env with owner-only permissions"
    if status["networked"]:
        client = build_llm_client(configure(args), provider=provider, model=getattr(args, "model", None))
        try:
            status["available_models"] = client.provider.list_models()[:25]
        except Exception as exc:  # noqa: BLE001 - report, do not pretend it worked
            status["error"] = f"{type(exc).__name__}: {exc}"
    _emit(status)
    return 0


def _store_secret(name: str, value: str) -> None:
    """Write ``NAME=value`` to ``.env`` (mode 600) without echoing the value."""
    env_path = Path(".env")
    lines: list[str] = []
    if env_path.is_file():
        lines = [
            line
            for line in env_path.read_text(encoding="utf-8").splitlines()
            if not line.strip().startswith(f"{name}=")
        ]
    lines.append(f"{name}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env_path.chmod(0o600)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def _shared_options() -> argparse.ArgumentParser:
    """Options accepted both before and after the subcommand.

    ``default=SUPPRESS`` matters: without it the sub-parser's default would
    overwrite a value already parsed by the top-level parser (``cli --mode smoke
    run``), silently discarding the flag.  Callers therefore read these with
    ``getattr(args, ..., default)``.
    """

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--mode", choices=["smoke", "standard", "full"], default=argparse.SUPPRESS, help="execution preset"
    )
    shared.add_argument(
        "--origin", default=argparse.SUPPRESS, help="dataset origin: sleep_edf | synthetic_psg | auto"
    )
    shared.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="override the random seed")
    shared.add_argument(
        "--max-subjects", type=int, default=argparse.SUPPRESS, dest="max_subjects", help="cap the subject count"
    )
    shared.add_argument(
        "--papers", type=int, default=argparse.SUPPRESS, help="target number of papers for the literature stage"
    )
    shared.add_argument("--channel", default=argparse.SUPPRESS, help="preferred EEG channel name")
    shared.add_argument(
        "--no-cache", action="store_true", default=argparse.SUPPRESS, help="ignore the feature-extraction cache"
    )
    shared.add_argument(
        "--offline", action="store_true", default=argparse.SUPPRESS, help="force offline behaviour (no network calls)"
    )
    shared.add_argument(
        "--quiet", action="store_true", default=argparse.SUPPRESS, help="reduce log output"
    )
    return shared


def build_parser() -> argparse.ArgumentParser:
    shared = _shared_options()
    parser = argparse.ArgumentParser(
        prog="python -m eeg_research_agent.cli",
        parents=[shared],
        description=(
            "Agentic single-channel EEG feature-discovery pipeline: acquisition, literature, "
            "feature extraction, statistics, predictive evaluation, verification and reporting."
        ),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler: Any, help_text: str) -> argparse.ArgumentParser:
        child = sub.add_parser(name, help=help_text, parents=[shared])
        child.set_defaults(handler=handler)
        return child

    add("init", cmd_init, "create directories, logging and the execution manifest")
    add("download", cmd_download, "acquire the dataset and print the dataset summary")
    add("inspect", cmd_inspect, "inspect channels, annotations, quality and class balance")
    add("literature", cmd_literature, "run the literature stage (offline-safe)")
    add("features", cmd_features, "preprocess, epoch, split and extract features")
    add("statistics", cmd_statistics, "run descriptive + inferential statistics with FDR correction")
    add("evaluate", cmd_evaluate, "run the subject-grouped predictive evaluation")
    add("verify", cmd_verify, "run synthetic-signal, statistical, ML and leakage verification")
    add("report", cmd_report, "regenerate every report from saved artifacts")
    run = add("run", cmd_run, "run the complete workflow")
    run.add_argument("--agentic", action="store_true", help="run through the LangGraph workflow instead")
    add("tools", cmd_tools, "print the agent tool permissions table")
    llm = add("llm-check", cmd_llm_check, "show LLM provider status / store an API key securely")
    llm.add_argument("--provider", choices=["anthropic", "deepseek", "offline"], default=None)
    llm.add_argument("--model", default=None, help="model id to verify against the provider's model list")
    llm.add_argument("--no-prompt", action="store_true", help="never prompt for a key (for CI)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = getattr(args, "handler", None)
    if handler is None:  # pragma: no cover - argparse enforces a subcommand
        parser.print_help()
        return 2
    try:
        return int(handler(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - the CLI reports the real error and exits non-zero
        logger.error("%s: %s", type(exc).__name__, exc)
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
