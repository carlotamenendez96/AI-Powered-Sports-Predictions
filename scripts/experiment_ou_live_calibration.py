"""Is the live O/U 2.5 probability calibrated? (input to the stop_loss fix)

    python3 scripts/experiment_ou_live_calibration.py

Every `output/live_history/*.jsonl` snapshot is joined to the match's final
score (`verification_<date>.csv`, active or archived; the previous day is
tried too, since late kick-offs are snapshotted after midnight). For each
snapshot the CURRENT `LiveAdjuster.adjust_ou_probabilities` is re-run on the
stored stats, so the measurement describes the code that would be fixed, not
whichever adjuster version wrote the snapshot.

Why this exists: on 2026-10-03 the cashout backtest showed stop_loss losing
-92.10 on Over 2.5 bets (27 of 154 written-off Overs came in) while being
positive on 1X2 and Under. If the live P(Over) is too pessimistic, that is a
probability bug, not a threshold problem.

Reports, with 95% CIs bootstrapped BY MATCH (snapshots of one match are not
independent):
  1. bias (actual - predicted) by minute band x goals still needed;
  2. a reliability table by predicted-probability bin;
  3. the minute-90 bucket on its own: the scraper caps stoppage time to 90
     and also maps FINISHED to 90, and the adjuster locks Under at 0.99 there;
  4. two one-parameter fixes, fitted out-of-fold (5 folds grouped by match):
     a stoppage allowance S (remaining = 90 + S - minute), a multiplier k
     on remaining xG, and both jointly. Scored on Brier / log-loss against the current code.

Locked-Over snapshots (3+ goals already) are excluded: trivially right.
Writes output/experiments/ou_live_calibration_<ts>.{json,txt}.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import math
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO, os.path.join(_REPO, "ml_project")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ml_project.backtest.coverage import load_verif_row  # noqa: E402
from ml_project.live_adjuster import LiveAdjuster, _poisson_cdf  # noqa: E402

LIVE_DIR = os.path.join(_REPO, "output", "live_history")
OUT_DIR = os.path.join(_REPO, "output", "experiments")
N_BOOT = 2000
RNG = np.random.default_rng(42)
MINUTE_BANDS = ((0, 30), (30, 60), (60, 75), (75, 85), (85, 90), (90, 91))
PROB_BINS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.98, 1.0001)


def _final_goals(date, home, away, cache):
    key = (date, home, away)
    if key not in cache:
        goals = None
        for d in (date, (dt.date.fromisoformat(date) - dt.timedelta(days=1)).isoformat()):
            row = load_verif_row(d, home, away)
            if row is not None:
                try:
                    h, a = (int(x) for x in str(row["Score"]).split("-"))
                    goals = h + a
                except (ValueError, AttributeError, KeyError):
                    pass
                break
        cache[key] = goals
    return cache[key]


def load_snapshots():
    adj = LiveAdjuster()
    cache, rows, unjoined = {}, [], 0
    for path in sorted(glob.glob(os.path.join(LIVE_DIR, "live_history_*.jsonl"))):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not r.get("pre_ou_probs"):
                    continue
                final = _final_goals(r["date"], r["home_team"], r["away_team"], cache)
                if final is None:
                    unjoined += 1
                    continue
                try:
                    h, a = (int(x) for x in str(r["score"]).split("-"))
                except (ValueError, AttributeError):
                    continue
                cur = h + a
                if cur >= 3:
                    continue                      # Over already locked in
                if final < cur:
                    continue                      # inconsistent join, skip
                stats = r.get("stats") or {}
                p = adj.adjust_ou_probabilities(r["pre_ou_probs"], stats, int(r["minute"]), r["score"])["over"]
                rows.append({
                    "match": f'{r["date"]}|{r["home_team"]}|{r["away_team"]}',
                    "minute": int(r["minute"]), "need": 3 - cur, "p": p,
                    "pre_over": float(r["pre_ou_probs"].get("over", 0.5)),
                    "xg": float(stats.get("xg_home", 0) or 0) + float(stats.get("xg_away", 0) or 0),
                    "y": int(final >= 3),
                })
    return rows, unjoined


def boot_bias(rows):
    """Mean(y - p) with a match-clustered bootstrap CI."""
    by = {}
    for r in rows:
        by.setdefault(r["match"], []).append(r["y"] - r["p"])
    groups = list(by.values())
    sums = np.array([sum(g) for g in groups]); ns = np.array([len(g) for g in groups])
    point = sums.sum() / ns.sum()
    idx = RNG.integers(0, len(groups), size=(N_BOOT, len(groups)))
    bs = sums[idx].sum(1) / ns[idx].sum(1)
    return point, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def describe(rows):
    if not rows:
        return None
    b, lo, hi = boot_bias(rows) if len({r["match"] for r in rows}) > 1 else (float("nan"),) * 3
    return {"n": len(rows), "matches": len({r["match"] for r in rows}),
            "pred": float(np.mean([r["p"] for r in rows])),
            "actual": float(np.mean([r["y"] for r in rows])),
            "bias": b, "ci": [lo, hi]}


# ---- candidate fixes (one parameter each) --------------------------------

def p_variant(r, S=0.0, k=1.0, adj=LiveAdjuster()):
    """adjust_ou_probabilities with remaining time 90+S-minute and xG x k.
    S=0, k=1 reproduces the current code (asserted in main)."""
    minute, need = r["minute"], r["need"]
    if minute >= 90 and S <= 0:
        return 0.01
    rem_min = max(0.0, 90 + S - minute)
    if minute > 0 and r["xg"] > 0:
        rem_xg = min(r["xg"] / minute * rem_min, adj.OU_MAX_REMAINING_XG)
    else:
        rem_xg = 2.6 * rem_min / 90
    rem_xg *= k
    p_pace = 1.0 - _poisson_cdf(need - 1, rem_xg)
    w = 1.0 / (1.0 + math.exp(-(minute - adj.OU_PACE_CROSSOVER_MIN) / 15.0))
    p = w * p_pace + (1 - w) * r["pre_over"]
    return max(0.01, min(0.99, p))


def score(rows, ps):
    y = np.array([r["y"] for r in rows]); p = np.clip(np.array(ps), 1e-4, 1 - 1e-4)
    return float(np.mean((p - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


GRID = {"S": [0, 2, 4, 6, 8, 10, 12], "k": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4]}
GRID["S+k"] = [(S, k) for S in GRID["S"] for k in GRID["k"]]


def _kw(param, v):
    return {"S": v[0], "k": v[1]} if param == "S+k" else {param: v}


def cross_fit(rows, param):
    matches = sorted({r["match"] for r in rows})
    fold_of = {m: i % 5 for i, m in enumerate(RNG.permutation(matches))}
    oof, chosen = [None] * len(rows), []
    for f in range(5):
        train = [r for r in rows if fold_of[r["match"]] != f]
        best = min(GRID[param], key=lambda v: score(train, [p_variant(r, **_kw(param, v)) for r in train])[1])
        chosen.append(best)
        for i, r in enumerate(rows):
            if fold_of[r["match"]] == f:
                oof[i] = p_variant(r, **_kw(param, best))
    return oof, chosen


def main() -> int:
    rows, unjoined = load_snapshots()
    if not rows:
        print("[ou-cal] no joinable snapshots"); return 0
    cur = [p_variant(r) for r in rows]
    drift = max(abs(a - r["p"]) for a, r in zip(cur, rows))
    assert drift < 1e-9, f"p_variant(S=0,k=1) drifted from LiveAdjuster by {drift}"

    rep = {"snapshots": len(rows), "matches": len({r["match"] for r in rows}),
           "unjoined_snapshots": unjoined, "overall": describe(rows), "by_band": [], "reliability": []}
    for lo, hi in MINUTE_BANDS:
        for need in (1, 2, 3):
            d = describe([r for r in rows if lo <= r["minute"] < hi and r["need"] == need])
            if d:
                rep["by_band"].append({"minutes": f"{lo}-{hi - 1}", "need": need, **d})
    for lo, hi in zip(PROB_BINS, PROB_BINS[1:]):
        d = describe([r for r in rows if lo <= r["p"] < hi])
        if d:
            rep["reliability"].append({"bin": f"{lo:.2f}-{min(hi, 1):.2f}", **d})
    m90 = [r for r in rows if r["minute"] >= 90]
    rep["minute_90"] = {"n": len(m90), "matches": len({r["match"] for r in m90}),
                        "ended_over": sum(r["y"] for r in m90)}

    base_b, base_ll = score(rows, cur)
    rep["fits"] = {"current": {"brier": base_b, "logloss": base_ll}}
    for param in ("S", "k", "S+k"):
        oof, chosen = cross_fit(rows, param)
        b, ll = score(rows, oof)
        rep["fits"][param] = {"chosen_per_fold": chosen, "brier": b, "logloss": ll,
                              "overall_after": describe([{**r, "p": p} for r, p in zip(rows, oof)])}

    lines = [f"Live O/U 2.5 calibration — {rep['snapshots']} snapshots, {rep['matches']} matches "
             f"(locked-Over excluded; {unjoined} snapshots had no final score)", ""]
    o = rep["overall"]
    lines.append(f"Overall: predicted {o['pred']:.3f}  actual {o['actual']:.3f}  "
                 f"bias {o['bias']:+.3f} [{o['ci'][0]:+.3f}, {o['ci'][1]:+.3f}]")
    lines += ["", "By minute x goals needed (bias = actual - predicted; + means too pessimistic on Over)",
              f"{'minutes':<9}{'need':>5}{'snaps':>7}{'matches':>9}{'pred':>8}{'actual':>8}{'bias':>8}  95% CI"]
    for b in rep["by_band"]:
        lines.append(f"{b['minutes']:<9}{b['need']:>5}{b['n']:>7}{b['matches']:>9}{b['pred']:>8.3f}"
                     f"{b['actual']:>8.3f}{b['bias']:>+8.3f}  [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]")
    lines += ["", "Reliability by predicted P(Over)",
              f"{'bin':<12}{'snaps':>7}{'matches':>9}{'pred':>8}{'actual':>8}{'bias':>8}  95% CI"]
    for b in rep["reliability"]:
        lines.append(f"{b['bin']:<12}{b['n']:>7}{b['matches']:>9}{b['pred']:>8.3f}{b['actual']:>8.3f}"
                     f"{b['bias']:>+8.3f}  [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]")
    m = rep["minute_90"]
    lines += ["", f"Minute 90 (stoppage AND finished, both scraped as 90; adjuster locks Under at 0.99): "
              f"{m['n']} snapshots, {m['matches']} matches, {m['ended_over']} ended Over"]
    lines += ["", "One-parameter fixes, out-of-fold (5 folds grouped by match)",
              f"  current          Brier {base_b:.5f}  logloss {base_ll:.5f}"]
    for param, label in (("S", "stoppage S"), ("k", "xG mult k"), ("S+k", "both")):
        f = rep["fits"][param]; a = f["overall_after"]
        lines.append(f"  {label:<16} Brier {f['brier']:.5f}  logloss {f['logloss']:.5f}  "
                     f"per-fold {f['chosen_per_fold']}  bias after {a['bias']:+.3f} "
                     f"[{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}]")
    text = "\n".join(lines)
    print(text)

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(OUT_DIR, f"ou_live_calibration_{ts}.json"), "w") as f:
        json.dump(rep, f, indent=2)
    with open(os.path.join(OUT_DIR, f"ou_live_calibration_{ts}.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nSaved output/experiments/ou_live_calibration_{ts}.{{json,txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
