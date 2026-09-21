#!/usr/bin/env python3
"""Cards-market evaluation gate (Paso 6).

Five arms on the same OOF predictions from train_cards:

  1. league_mean   — expanding league P(over) (past only; already in frame)
  2. referee_mean  — ref yellows/game → soft P(over) when n≥20, else league
  3. model_raw     — XGBoost OOF P(over)
  4. model_cal     — model_raw through the global Platt calibrator
  5. placebo       — same model re-fit OOF with CARD_FORM_BLOCK + REF block
                     row-permuted (width-matched control)

Acceptance (documented in the roadmap):
  - model_cal Brier ≤ best baseline (league or referee), with bootstrap 95% CI
    on ΔBrier preferring to exclude zero;
  - placebo must NOT beat the real feature block on Brier.

No odds → no ROI. Writes output/experiments/cards_<ts>.{json,txt}.

Usage:
    python3 scripts/experiment_cards.py
    python3 scripts/experiment_cards.py --refit-placebo   # slower; default ON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ml_project"))

from cards.calibration import (  # noqa: E402
    apply_cards_calibrator,
    fit_global_cards_calibrator,
    save_cards_calibration,
)
from cards.constants import (  # noqa: E402
    CARD_LINE,
    CARDS_FEATURES,
    CARD_FORM_BLOCK,
    REF_FEATURE_BLOCK,
    REFEREE_MIN_N,
)
from cards.train_cards import prepare_cards_frame, oof_binary_predictions  # noqa: E402

BOOTSTRAP_ITERS = 2000
PLACEBO_SEED = 11
PLACEBO_BLOCK = tuple(dict.fromkeys(list(CARD_FORM_BLOCK) + list(REF_FEATURE_BLOCK)))


def _clip(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


def brier_ci_delta(y, p_a, p_b, iters=BOOTSTRAP_ITERS, seed=0):
    """Bootstrap 95% CI on Brier(a) − Brier(b). Negative ⇒ a better than b."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p_a, p_b = np.asarray(p_a), np.asarray(p_b)
    n = len(y)
    deltas = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        ba = brier_score_loss(y[idx], p_a[idx])
        bb = brier_score_loss(y[idx], p_b[idx])
        deltas.append(ba - bb)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi)


def reliability_bins(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if not mask.any():
            continue
        rows.append({
            "bin_lo": round(float(edges[i]), 2),
            "bin_hi": round(float(edges[i + 1]), 2),
            "n": int(mask.sum()),
            "mean_pred": round(float(p[mask].mean()), 4),
            "mean_actual": round(float(y[mask].mean()), 4),
        })
    return rows


def score_arm(name, y, p):
    p = _clip(np.asarray(p, dtype=float))
    y = np.asarray(y, dtype=int)
    return {
        "arm": name,
        "n": int(len(y)),
        "brier": round(float(brier_score_loss(y, p)), 6),
        "log_loss": round(float(log_loss(y, p)), 6),
        "mean_pred": round(float(p.mean()), 4),
        "mean_actual": round(float(y.mean()), 4),
        "reliability": reliability_bins(y, p),
    }


def referee_baseline_probs(df: pd.DataFrame) -> np.ndarray:
    """Soft P(over) from ref mean yellows via Poisson CDF; else league_p_over."""
    out = df["league_p_over"].astype(float).values.copy()
    has_ref = (df["missing_ref"].values < 0.5) & (
        df["ref_n_with_cards"].values >= REFEREE_MIN_N)
    if has_ref.any():
        lam = df.loc[has_ref, "ref_yellows_pg"].astype(float).values
        # P(X > 3.5) = 1 - P(X <= 3) for integer counts.
        out[has_ref] = 1.0 - poisson.cdf(3, lam)
    return out


def placebo_oof(df: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """OOF preds after row-permuting the card-form + referee block as a unit."""
    d = df.copy()
    rng = np.random.default_rng(PLACEBO_SEED)
    cols = [c for c in PLACEBO_BLOCK if c in d.columns]
    order = rng.permutation(len(d))
    block = d[cols].to_numpy()[order]
    for i, c in enumerate(cols):
        d[c] = block[:, i]
    oof, _, _ = oof_binary_predictions(d, list(CARDS_FEATURES), n_splits=n_splits)
    return oof


def run(refit_placebo: bool = True, n_splits: int = 5) -> dict:
    # Prefer cached OOF from a prior train; else rebuild frame + OOF.
    oof_path = os.path.join(PROJECT_ROOT, "models", "oof_cards.npz")
    print("[exp] Preparing cards frame…")
    df = prepare_cards_frame()
    df = df.sort_values("date").reset_index(drop=True)

    if os.path.exists(oof_path):
        cached = np.load(oof_path, allow_pickle=True)
        # Align by length; if mismatch, recompute.
        if len(cached["oof"]) == len(df):
            oof_raw = cached["oof"]
            print(f"[exp] Reusing cached OOF from {oof_path}")
        else:
            print("[exp] Cached OOF length mismatch — refitting OOF…")
            oof_raw, df, _ = oof_binary_predictions(
                df, list(CARDS_FEATURES), n_splits=n_splits)
    else:
        print("[exp] No cached OOF — fitting…")
        oof_raw, df, _ = oof_binary_predictions(
            df, list(CARDS_FEATURES), n_splits=n_splits)

    valid = ~np.isnan(oof_raw)
    d = df.loc[valid].reset_index(drop=True)
    y = d["y_over"].astype(int).values
    p_model = oof_raw[valid]

    # Fit calibrator on OOF (same data the gate scores — documented as mildly
    # optimistic; chronological holdout would be C3-style, deferred for v1).
    cal_payload, reason = fit_global_cards_calibrator(p_model, y)
    if cal_payload is None:
        print(f"[exp] Calibrator rejected: {reason}")
        p_cal = p_model.copy()
        cal_accepted = False
    else:
        save_cards_calibration(cal_payload)
        p_cal = np.array([
            apply_cards_calibrator(float(p), cal_payload, enabled=True)[0]
            for p in p_model
        ])
        cal_accepted = True
        print(f"[exp] Calibrator accepted: ΔBrier={cal_payload['brier_delta']:+.5f}")

    p_league = d["league_p_over"].astype(float).values
    p_ref = referee_baseline_probs(d)

    # Optional simple Poisson blend of team rates (diagnostic, not a gate arm).
    proxy = d["expected_yellows_proxy"].astype(float).values
    p_poisson = np.where(
        np.isnan(proxy), p_league, 1.0 - poisson.cdf(3, np.clip(proxy, 0.1, 20)))

    arms = {
        "league_mean": score_arm("league_mean", y, p_league),
        "referee_mean": score_arm("referee_mean", y, p_ref),
        "poisson_proxy": score_arm("poisson_proxy", y, p_poisson),
        "model_raw": score_arm("model_raw", y, p_model),
        "model_cal": score_arm("model_cal", y, p_cal),
    }

    if refit_placebo:
        print("[exp] Fitting placebo OOF (shuffled card-form + ref block)…")
        # Placebo needs the full frame (incl. NaN-padded early rows) so fold
        # indices match; then slice with the same `valid` mask.
        oof_placebo_full = placebo_oof(df, n_splits=n_splits)
        p_placebo = oof_placebo_full[valid]
        arms["placebo"] = score_arm("placebo", y, p_placebo)
    else:
        p_placebo = None

    # Gate comparisons
    best_base_name = min(
        ("league_mean", "referee_mean"),
        key=lambda k: arms[k]["brier"])
    best_base = arms[best_base_name]["brier"]
    model_brier = arms["model_cal"]["brier"]
    delta, lo, hi = brier_ci_delta(y, p_cal, {
        "league_mean": p_league, "referee_mean": p_ref
    }[best_base_name])

    beats_baseline = model_brier <= best_base
    ci_excludes_zero = hi < 0  # model_cal − baseline < 0 ⇒ model better

    placebo_beats_model = False
    placebo_delta = None
    if p_placebo is not None:
        placebo_beats_model = arms["placebo"]["brier"] < arms["model_raw"]["brier"]
        placebo_delta, plo, phi = brier_ci_delta(y, p_placebo, p_model)

    # ENG/SCO subset (where referee features are dense)
    eng_sco = d["league"].astype(str).str.startswith(("ENG", "SCO"))
    subset = {}
    if eng_sco.any():
        ys, ps, pb = y[eng_sco], p_model[eng_sco], p_league[eng_sco]
        subset["eng_sco"] = {
            "n": int(eng_sco.sum()),
            "model_raw_brier": round(float(brier_score_loss(ys, ps)), 6),
            "league_brier": round(float(brier_score_loss(ys, pb)), 6),
            "n_with_ref": int(((d.loc[eng_sco, "missing_ref"] < 0.5)).sum()),
        }
    rest = ~eng_sco
    if rest.any():
        ys, ps, pb = y[rest], p_model[rest], p_league[rest]
        subset["non_eng_sco"] = {
            "n": int(rest.sum()),
            "model_raw_brier": round(float(brier_score_loss(ys, ps)), 6),
            "league_brier": round(float(brier_score_loss(ys, pb)), 6),
        }

    gate_pass = beats_baseline and not placebo_beats_model
    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "card_line": CARD_LINE,
        "n_oof": int(len(y)),
        "n_leagues": int(d["league"].nunique()),
        "leagues": sorted(d["league"].astype(str).unique().tolist()),
        "calibrator_accepted": cal_accepted,
        "calibrator_rejection": reason,
        "arms": arms,
        "gate": {
            "best_baseline": best_base_name,
            "best_baseline_brier": best_base,
            "model_cal_brier": model_brier,
            "delta_brier_model_minus_baseline": round(delta, 6),
            "delta_brier_ci95": [round(lo, 6), round(hi, 6)],
            "beats_baseline": beats_baseline,
            "ci_excludes_zero_improvement": ci_excludes_zero,
            "placebo_beats_model": placebo_beats_model,
            "placebo_minus_model_brier": (
                None if placebo_delta is None else round(placebo_delta, 6)),
            "pass": gate_pass,
            "rule": (
                "model_cal Brier ≤ best(league, referee) AND placebo does not "
                "beat model_raw on Brier"
            ),
        },
        "subset": subset,
        "note": (
            "No card odds in corpus → probability metrics only. "
            "No edge / ROI claim."
        ),
    }
    return report


def write_report(report: dict) -> tuple:
    out_dir = os.path.join(PROJECT_ROOT, "output", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jp = os.path.join(out_dir, f"cards_{ts}.json")
    tp = os.path.join(out_dir, f"cards_{ts}.txt")
    with open(jp, "w") as f:
        # Drop reliability arrays from arms for a cleaner JSON? Keep them.
        json.dump(report, f, indent=2, default=str)

    g = report["gate"]
    lines = [
        f"Cards experiment · {report['generated_at']}",
        f"n_oof={report['n_oof']}  leagues={report['n_leagues']}  "
        f"CARD_LINE={report['card_line']}",
        f"calibrator_accepted={report['calibrator_accepted']}",
        "",
        f"{'arm':16} {'Brier':>10} {'logloss':>10} {'mean_p':>8} {'actual':>8}",
    ]
    for name, arm in report["arms"].items():
        lines.append(
            f"{name:16} {arm['brier']:10.6f} {arm['log_loss']:10.6f} "
            f"{arm['mean_pred']:8.4f} {arm['mean_actual']:8.4f}")
    lines += [
        "",
        f"GATE: best_baseline={g['best_baseline']} "
        f"(Brier={g['best_baseline_brier']:.6f})",
        f"      model_cal Brier={g['model_cal_brier']:.6f}  "
        f"Δ={g['delta_brier_model_minus_baseline']:+.6f} "
        f"CI95={g['delta_brier_ci95']}",
        f"      beats_baseline={g['beats_baseline']}  "
        f"placebo_beats_model={g['placebo_beats_model']}",
        f"      PASS={g['pass']}",
        "",
        report["note"],
    ]
    with open(tp, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[+] Wrote {jp}")
    print(f"[+] Wrote {tp}")
    return jp, tp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-placebo", action="store_true",
                        help="Skip the (slower) placebo re-fit.")
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()
    t0 = time.time()
    report = run(refit_placebo=not args.no_placebo, n_splits=args.splits)
    write_report(report)
    print(f"[*] Done in {time.time() - t0:.1f}s")
    return 0 if report["gate"]["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
