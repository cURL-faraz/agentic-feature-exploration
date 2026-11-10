"""The judge.  Three stages, ordered from cheap to expensive, no LLM anywhere.

Stage A - screen (discovery cohort, seconds)
    Is the column usable (few NaNs, not constant)?  Is it new (not a near-copy of a
    baseline feature)?  Does it separate at least one *confused* stage pair at all?
    Most ideas die here, cheaply.

Stage B - incremental value (discovery cohort, subject-grouped CV, minutes)
    Baseline model vs baseline + candidate, identical folds and seed.  The gain in
    target-stage F1 (N1/N2/REM) is computed *per subject* and tested with a one-sided
    Wilcoxon signed-rank test at alpha / hypothesis_budget (Bonferroni over the
    declared search budget).  The gain must also beat a shuffled copy of the same
    column - some models gain a little from any extra column.

Stage C - confirmation (lockbox + external cohorts, once per finalist)
    Fit on all discovery subjects, test on subjects never touched during the search.
    This is what protects against "the loop overfitted the validation data by trying
    many ideas" (adaptive overfitting).  The external cohort (Sleep Telemetry: other
    study, other population) is reported as a generalisation check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from ..config import protocol_hash
from ..data.cohort import STAGE_CODE, load_cohort
from ..features.baseline import load_or_compute_baseline
from ..features.runner import run_feature
from . import metrics
from .modeling import cross_val_predict, design, fit_predict


class Evaluator:
    def __init__(self, config: dict):
        self.config = config
        self.ev = config["evaluation"]
        self.root = Path(config["data"]["root"])
        self.cache = self.root / "cache" / protocol_hash(config)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._cohorts: dict = {}
        self._base: dict = {}

    # ------------------------------------------------------------------ data #
    def cohort(self, role_or_name: str):
        name = self.config["data"]["cohorts"].get(role_or_name, role_or_name)
        if name not in self._cohorts:
            self._cohorts[name] = load_cohort(self.root, name)
        return self._cohorts[name]

    def baseline(self, role: str) -> pd.DataFrame:
        cohort = self.cohort(role)
        if cohort.name not in self._base:
            self._base[cohort.name] = load_or_compute_baseline(cohort)
        return self._base[cohort.name]

    def has_cohort(self, role: str) -> bool:
        try:
            self.cohort(role)
            return True
        except FileNotFoundError:
            return False

    def compute_column(self, code: str, role: str, workdir: Path):
        sb = self.config["sandbox"]
        return run_feature(code, self.cohort(role).path, workdir / role, timeout_s=float(sb["timeout_s"]),
                           max_ms=float(sb["max_ms_per_epoch"]), workers=int(sb["workers"]))

    # -------------------------------------------------------- baseline (cached) #
    def baseline_cv(self) -> dict:
        path = self.cache / "baseline_cv.npz"
        c = self.cohort("discovery")
        if path.is_file():
            pred = np.load(path)["pred"]
        else:
            X = design(self.baseline("discovery"), c.recording, int(self.ev["context_epochs"]))
            pred = cross_val_predict(X, c.y, c.subject, self.ev)
            np.savez(path, pred=pred)
        targets = self.ev["target_stages"]
        return {
            "pred": pred,
            "per_subject": metrics.per_subject_scores(c.y, pred, c.subject, targets),
            "summary": metrics.summary(c.y, pred, targets),
            "pair_errors": metrics.confusion_pair_errors(c.y, pred, self.ev["confusion_pairs"]),
        }

    def baseline_holdout(self, role: str) -> np.ndarray:
        path = self.cache / f"baseline_{role}.npz"
        if path.is_file():
            return np.load(path)["pred"]
        k = int(self.ev["context_epochs"])
        d, t = self.cohort("discovery"), self.cohort(role)
        pred = fit_predict(design(self.baseline("discovery"), d.recording, k), d.y,
                           design(self.baseline(role), t.recording, k), self.ev)
        np.savez(path, pred=pred)
        return pred

    # --------------------------------------------------------------- stage A #
    def screen(self, column: np.ndarray, predicted: dict | None = None) -> dict:
        s = self.ev["screen"]
        c = self.cohort("discovery")
        base = self.baseline("discovery")
        finite = np.isfinite(column)
        out: dict = {"nan_fraction": float(1 - finite.mean())}
        reasons = []
        if out["nan_fraction"] > float(s["max_nan_fraction"]):
            reasons.append(f"too many NaN/failed epochs ({out['nan_fraction']:.1%})")
        n_unique = len(np.unique(np.round(column[finite], 9)))
        out["n_unique"] = int(n_unique)
        if n_unique < 10:
            reasons.append("feature is (nearly) constant")
        if reasons:
            return {**out, "passed": False, "reason": "; ".join(reasons)}

        rng = np.random.default_rng(int(self.ev["seed"]))
        idx = np.flatnonzero(finite)
        if idx.size > int(s["subsample"]):
            idx = np.sort(rng.choice(idx, int(s["subsample"]), replace=False))
        rhos = {col: abs(float(spearmanr(column[idx], base[col].to_numpy()[idx]).statistic))
                for col in base.columns}
        most = max(rhos, key=lambda k: rhos[k] if np.isfinite(rhos[k]) else -1)
        out["max_redundancy"] = {"feature": most, "abs_spearman": round(rhos[most], 4)}

        pairs = {}
        for a, b in self.ev["confusion_pairs"]:
            mask = finite & np.isin(c.y, [STAGE_CODE[a], STAGE_CODE[b]])
            if mask.sum() < 50 or len(np.unique(c.y[mask])) < 2:
                continue
            auc = float(roc_auc_score(c.y[mask] == STAGE_CODE[b], column[mask]))
            agree, total = 0, 0
            for subj in np.unique(c.subject[mask]):
                m = mask & (c.subject == subj)
                if len(np.unique(c.y[m])) == 2:
                    total += 1
                    agree += (roc_auc_score(c.y[m] == STAGE_CODE[b], column[m]) - 0.5) * (auc - 0.5) > 0
            n1, n2 = int((c.y[mask] == STAGE_CODE[a]).sum()), int((c.y[mask] == STAGE_CODE[b]).sum())
            null_se = float(np.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2)))  # SE of AUC under "no effect"
            consistency = agree / max(total, 1)
            shift = abs(auc - 0.5)
            pairs[f"{a}vs{b}"] = {
                "auc": round(auc, 4), "shift": round(shift, 4), "z": round(shift / null_se, 2),
                "subject_consistency": round(consistency, 3),
                # a pair counts only if the effect is big, beyond chance, AND holds in most subjects
                "counts": bool(shift >= float(s["min_auc_shift"]) and shift / null_se >= 3.0
                               and consistency >= float(s.get("min_subject_consistency", 0.7))),
            }
        out["pair_auc"] = pairs
        counting = [v["shift"] for v in pairs.values() if v["counts"]]
        best = max(counting, default=0.0)
        out["best_auc_shift"] = max((v["shift"] for v in pairs.values()), default=0.0)

        if predicted and predicted.get("higher_in") in STAGE_CODE and predicted.get("lower_in") in STAGE_CODE:
            hi = np.nanmedian(column[c.y == STAGE_CODE[predicted["higher_in"]]])
            lo = np.nanmedian(column[c.y == STAGE_CODE[predicted["lower_in"]]])
            out["preregistered_prediction_held"] = bool(hi > lo)

        if rhos[most] > float(s["max_redundancy_rho"]):
            reasons.append(f"redundant with baseline `{most}` (|rho|={rhos[most]:.3f})")
        if best == 0.0:
            reasons.append("does not reliably separate any confused pair "
                           f"(best |AUC-0.5|={out['best_auc_shift']:.3f}; needs >= {s['min_auc_shift']}, "
                           f"z >= 3 and the same direction in >= {s.get('min_subject_consistency', 0.7):.0%} of subjects)")
        return {**out, "passed": not reasons, "reason": "; ".join(reasons) or "passed"}

    # --------------------------------------------------------------- stage B #
    def incremental(self, column: np.ndarray, name: str = "candidate") -> dict:
        inc = self.ev["incremental"]
        c = self.cohort("discovery")
        k, targets = int(self.ev["context_epochs"]), self.ev["target_stages"]
        base = self.baseline_cv()
        X = design(self.baseline("discovery"), c.recording, k, column, name)
        pred = cross_val_predict(X, c.y, c.subject, self.ev)
        cand_scores = metrics.per_subject_scores(c.y, pred, c.subject, targets)
        test = metrics.paired_test(base["per_subject"], cand_scores)

        rng = np.random.default_rng(int(self.ev["seed"]) + 1)
        null_deltas = []
        for _ in range(int(inc["null_shuffles"])):
            shuffled = column.copy()
            for subj in np.unique(c.subject):
                where = np.flatnonzero(c.subject == subj)
                shuffled[where] = column[rng.permutation(where)]
            p_null = cross_val_predict(design(self.baseline("discovery"), c.recording, k, shuffled, name),
                                       c.y, c.subject, self.ev)
            null_deltas.append(metrics.paired_test(
                base["per_subject"], metrics.per_subject_scores(c.y, p_null, c.subject, targets))["mean_delta"])

        alpha_i = float(inc["alpha"]) / float(inc["hypothesis_budget"])
        cand_summary = metrics.summary(c.y, pred, targets)
        result = {
            **test,
            "alpha_per_test": alpha_i,
            "null_mean_deltas": [round(v, 5) for v in null_deltas],
            "baseline": {k2: base["summary"][k2] for k2 in ("kappa", "macro_f1", "target_f1", "per_class_f1")},
            "candidate": {k2: cand_summary[k2] for k2 in ("kappa", "macro_f1", "target_f1", "per_class_f1")},
            "pair_errors_baseline": base["pair_errors"],
            "pair_errors_candidate": metrics.confusion_pair_errors(c.y, pred, self.ev["confusion_pairs"]),
        }
        reasons = []
        if test["p_value"] >= alpha_i:
            reasons.append(f"p={test['p_value']:.2g} not below alpha/budget={alpha_i:.2g}")
        if test["mean_delta"] < float(inc["min_delta"]):
            reasons.append(f"mean gain {test['mean_delta']:+.4f} below minimum {inc['min_delta']}")
        if null_deltas and test["mean_delta"] <= max(null_deltas):
            reasons.append(f"gain does not beat a shuffled copy ({max(null_deltas):+.4f})")
        result["passed"] = not reasons
        result["reason"] = "; ".join(reasons) or "passed"
        return result

    # --------------------------------------------------------------- stage C #
    def confirm(self, columns: dict[str, np.ndarray], name: str = "candidate") -> dict:
        conf = self.ev["confirm"]
        k, targets = int(self.ev["context_epochs"]), self.ev["target_stages"]
        d = self.cohort("discovery")
        X_train = design(self.baseline("discovery"), d.recording, k, columns["discovery"], name)
        out = {}
        for role in ("lockbox", "external"):
            if role not in columns or columns[role] is None:
                continue
            t = self.cohort(role)
            base_pred = self.baseline_holdout(role)
            pred = fit_predict(X_train, d.y, design(self.baseline(role), t.recording, k, columns[role], name),
                               self.ev)
            test = metrics.paired_test(metrics.per_subject_scores(t.y, base_pred, t.subject, targets),
                                       metrics.per_subject_scores(t.y, pred, t.subject, targets))
            out[role] = {
                **test,
                "baseline": metrics.summary(t.y, base_pred, targets),
                "candidate": metrics.summary(t.y, pred, targets),
                "passed_raw": bool(test["p_value"] < float(conf["alpha"])
                                   and test["mean_delta"] > float(conf["min_delta"])),
            }
            for part in ("baseline", "candidate"):
                out[role][part].pop("confusion_matrix", None)
        return out


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
