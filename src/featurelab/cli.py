"""featurelab command line.

    featurelab download sc20 st sc_lockbox   # raw Sleep-EDF files (checksummed)
    featurelab prepare                       # EDF -> epoch caches for all configured cohorts
    featurelab baseline                      # how good is the standard feature set?
    featurelab controls                      # prove the judge works (planted effect + noise)
    featurelab models                        # which OpenRouter models are free today
    featurelab scout                         # literature -> inspiration cards
    featurelab run [--provider scripted]     # the agent loop
    featurelab evaluate my_feature.py        # judge a human-written feature
    featurelab ledger                        # every hypothesis so far
    featurelab export                        # accepted features as CSV for ML/DL work
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _config(args):
    from .config import load_config

    return load_config(args.config)


def cmd_download(args):
    from .data.download import main as download

    return download([*args.cohorts, "--workers", str(args.workers)])


def cmd_prepare(args):
    from .data.sleepedf import build_cohort

    config = _config(args)
    roles = config["data"]["cohorts"]
    names = [roles.get(c, c) for c in (args.cohort or list(roles))]
    for name in names:
        try:
            summary = build_cohort(config, name, workers=args.workers)
            print(json.dumps(summary, indent=1))
        except FileNotFoundError as exc:
            print(f"skip {name}: {exc}")
    return 0


def cmd_baseline(args):
    from .evaluation.stages import Evaluator

    ev = Evaluator(_config(args))
    result = ev.baseline_cv()
    c = ev.cohort("discovery")
    print(f"discovery cohort: {c.name}, {len(c.subjects())} subjects, {c.n} epochs, stages {c.stage_counts()}")
    print(json.dumps({k: v for k, v in result["summary"].items()}, indent=1))
    print("confusions between stage pairs:", result["pair_errors"])
    return 0


def cmd_controls(args):
    """Validate the judge: a planted effect must pass, noise must fail."""
    import shutil
    import tempfile

    from .data import synthetic
    from .evaluation.stages import Evaluator

    config = _config(args)
    config["data"]["root"] = tempfile.mkdtemp(prefix="featurelab-controls-")
    config["data"]["cohorts"] = {"discovery": "synthetic"}
    root = Path(config["data"]["root"])
    synthetic.generate(root, n_subjects=20, epochs=150)
    ev = Evaluator(config)
    ok = True
    for label, code, should_pass in (("positive control (planted effect)", synthetic.POSITIVE_CONTROL, True),
                                     ("negative control (noise)", synthetic.NEGATIVE_CONTROL, False)):
        column, report = ev.compute_column(code, "discovery", root / label.split()[0])
        screen = ev.screen(column)
        inc = ev.incremental(column)
        passed = screen["passed"] and inc["passed"]
        verdict = "OK" if passed == should_pass else "WRONG"
        ok &= passed == should_pass
        print(f"{verdict:5s} {label}: screen={screen['passed']} incremental={inc['passed']} "
              f"gain={inc['mean_delta']:+.4f} p={inc['p_value']:.2g}")
    shutil.rmtree(root, ignore_errors=True)
    print("judge validated" if ok else "JUDGE FAILED ITS CONTROLS - do not trust its verdicts")
    return 0 if ok else 1


def cmd_models(args):
    from .llm.client import list_free_models

    config = _config(args)
    free = {m["id"]: m for m in list_free_models()}
    print(f"{len(free)} free models on OpenRouter right now")
    for role, models in config["llm"]["roles"].items():
        for model in models:
            print(f"  {role:9s} {model:55s} {'available' if model in free else 'NOT free/available'}")
    if args.all:
        for model in free.values():
            print(f"  {model['id']}  (context {model['context_length']})")
    return 0


def cmd_scout(args):
    from .agents.literature import build_inspiration
    from .llm.client import build_client

    config = _config(args)
    cards = build_inspiration(config, build_client(config, provider=args.provider), force=args.force)
    for card in cards:
        print(f"- [{card.paper_id}] {card.idea}  ({card.target_confusion})")
    return 0


def cmd_run(args):
    from .orchestrator.graph import run_loop

    config = _config(args)
    if args.iterations:
        config["loop"]["max_iterations"] = args.iterations
    if args.target:
        config["loop"]["target_accepted"] = args.target
    result = run_loop(config, provider=args.provider, skip_scout=args.no_scout)
    print(json.dumps(result, indent=1))
    return 0


def cmd_evaluate(args):
    from .orchestrator.graph import evaluate_human

    config = _config(args)
    code = Path(args.file).read_text(encoding="utf-8")
    meta = {}
    if args.higher_in and args.lower_in:
        meta["prediction"] = {"higher_in": args.higher_in, "lower_in": args.lower_in}
    meta["code"] = code
    row = evaluate_human(config, code, args.name or Path(args.file).stem, meta)
    print(f"\nverdict: {row['status']} - {row.get('reason')}")
    return 0


def cmd_ledger(args):
    from .config import protocol_hash
    from .orchestrator.ledger import Ledger

    config = _config(args)
    ledger = Ledger(Path(config["data"]["root"]) / "ledger.sqlite")
    for row in ledger.rows("protocol = ?", (protocol_hash(config),)):
        inc = row.get("incremental") if isinstance(row.get("incremental"), dict) else {}
        gain = f"{inc['mean_delta']:+.4f}" if inc else "      "
        print(f"#{row['id']:<4} {row['source']:6s} {row['status'] or '':20s} {gain}  {row['name']}  - "
              f"{(row.get('reason') or '')[:90]}")
    return 0


def cmd_export(args):
    from .orchestrator.report import export_features

    status = ("accepted", "confirm_failed", "incremental_failed") if args.all_tested else ("accepted",)
    export_features(_config(args), Path(args.out), include_status=status)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="featurelab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML file overriding the defaults (default: ./featurelab.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download Sleep-EDF cohorts")
    p.add_argument("cohorts", nargs="+", choices=["sc20", "sc_lockbox", "st", "sc78"])
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("prepare", help="build epoch caches")
    p.add_argument("--cohort", action="append", help="role (discovery/lockbox/external) or cohort name")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_prepare)

    sub.add_parser("baseline", help="cross-validate the baseline").set_defaults(fn=cmd_baseline)
    sub.add_parser("controls", help="validate the judge on planted/noise features").set_defaults(fn=cmd_controls)

    p = sub.add_parser("models", help="list free OpenRouter models")
    p.add_argument("--all", action="store_true")
    p.set_defaults(fn=cmd_models)

    for name, fn in (("scout", cmd_scout), ("run", cmd_run)):
        p = sub.add_parser(name)
        p.add_argument("--provider", choices=["openrouter", "scripted"])
        if name == "scout":
            p.add_argument("--force", action="store_true", help="re-fetch papers and re-extract cards")
        else:
            p.add_argument("--iterations", type=int)
            p.add_argument("--target", type=int, help="stop after this many accepted features")
            p.add_argument("--no-scout", action="store_true", help="skip the literature step")
        p.set_defaults(fn=fn)

    p = sub.add_parser("evaluate", help="judge a human-written feature file")
    p.add_argument("file")
    p.add_argument("--name")
    p.add_argument("--higher-in", help="pre-registered: stage where the feature should be higher")
    p.add_argument("--lower-in", help="pre-registered: stage where it should be lower")
    p.set_defaults(fn=cmd_evaluate)

    sub.add_parser("ledger", help="list all hypotheses").set_defaults(fn=cmd_ledger)

    p = sub.add_parser("export", help="export features as CSV for ML/DL")
    p.add_argument("--out", default="exports")
    p.add_argument("--all-tested", action="store_true", help="also export features that reached Stage B")
    p.set_defaults(fn=cmd_export)

    args = parser.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
