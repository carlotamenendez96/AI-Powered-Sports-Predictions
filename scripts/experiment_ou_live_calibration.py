"""Is the live O/U 2.5 probability calibrated? (input to the stop_loss fix)

    python3 scripts/experiment_ou_live_calibration.py

Every `output/live_history/*.jsonl` snapshot is joined to the match's final
score (`verification_<date>.csv`, active or archived; the previous day is
tried too, since late kick-offs are snapshotted after midnight). Each snapshot
is re-scored with `LiveAdjuster.adjust_ou_probabilities` on the stored stats --
once as configured (CURRENT) and once with the three 2026-10-03 corrections
switched off (LEGACY, the pre-fix behaviour) -- never with whichever adjuster
version wrote the snapshot.

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
  4. the correction constants -- stoppage allowance S, remaining-xG
     multiplier k, score-aware prior on/off -- fitted out-of-fold (5 folds
     grouped by match) and on the full sample, scored on Brier / log-loss
     against LEGACY and CURRENT. The full-sample best is what the
     LiveAdjuster defaults should be; the OOF score is the honest estimate.

Locked-Over snapshots (3+ goals already) are excluded: trivially right.
Writes output/experiments/ou_live_calibration_<ts>.{json,txt}.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO, os.path.join(_REPO, "ml_project")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ml_project.backtest.coverage import load_verif_row  # noqa: E402
from ml_project.live_adjuster import LiveAdjuster  # noqa: E402

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


LEGACY = {"S": 0, "k": 1.0, "prior": False}   # all three corrections off


def make_adjuster(S, k, prior):
    adj = LiveAdjuster()
    adj.OU_STOPPAGE_MIN, adj.OU_REMAINING_XG_MULT, adj.OU_SCORE_AWARE_PRIOR = S, k, prior
    return adj


def p_with(adj, r):
    return adj.adjust_ou_probabilities(r["pre_ou"], r["stats"], r["minute"], r["score"])["over"]


def load_snapshots():
    current, legacy = LiveAdjuster(), make_adjuster(**LEGACY)
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
                row = {"match": f'{r["date"]}|{r["home_team"]}|{r["away_team"]}',
                       "minute": int(r["minute"]), "need": 3 - cur, "score": r["score"],
                       "pre_ou": r["pre_ou_probs"], "stats": r.get("stats") or {},
                       "y": int(final >= 3)}
                row["p_legacy"] = p_with(legacy, row)
                row["p"] = p_with(current, row)
                rows.append(row)
    return rows, unjoined


def boot_bias(rows, key="p"):
    """Mean(y - p) with a match-clustered bootstrap CI."""
    by = {}
    for r in rows:
        by.setdefault(r["match"], []).append(r["y"] - r[key])
    groups = list(by.values())
    sums = np.array([sum(g) for g in groups]); ns = np.array([len(g) for g in groups])
    point = sums.sum() / ns.sum()
    idx = RNG.integers(0, len(groups), size=(N_BOOT, len(groups)))
    bs = sums[idx].sum(1) / ns[idx].sum(1)
    return point, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def describe(rows, key="p"):
    if not rows:
        return None
    b, lo, hi = boot_bias(rows, key) if len({r["match"] for r in rows}) > 1 else (float("nan"),) * 3
    return {"n": len(rows), "matches": len({r["match"] for r in rows}),
            "pred": float(np.mean([r[key] for r in rows])),
            "actual": float(np.mean([r["y"] for r in rows])),
            "bias": b, "ci": [lo, hi]}


# ---- fitting the correction constants ----------------------------------

def score(rows, ps):
    y = np.array([r["y"] for r in rows]); p = np.clip(np.array(ps), 1e-4, 1 - 1e-4)
    return float(np.mean((p - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


GRID = [{"S": S, "k": k, "prior": prior}
        for S in (0, 2, 4, 6, 8) for k in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0) for prior in (False, True)]


def _ll(rows, cfg):
    adj = make_adjuster(**cfg)
    return score(rows, [p_with(adj, r) for r in rows])[1]


def fit(rows):
    return min(GRID, key=lambda cfg: _ll(rows, cfg))


def cross_fit(rows):
    matches = sorted({r["match"] for r in rows})
    fold_of = {m: i % 5 for i, m in enumerate(RNG.permutation(matches))}
    oof, chosen = [None] * len(rows), []
    for f in range(5):
        best = fit([r for r in rows if fold_of[r["match"]] != f])
        chosen.append(best)
        adj = make_adjuster(**best)
        for i, r in enumerate(rows):
            if fold_of[r["match"]] == f:
                oof[i] = p_with(adj, r)
    return oof, chosen


def _cfg(c):
    return f"S={c['S']} k={c['k']} prior={'score-aware' if c['prior'] else 'pre-match'}"


def main() -> int:
    rows, unjoined = load_snapshots()
    if not rows:
        print("[ou-cal] no joinable snapshots"); return 0
    adj = LiveAdjuster()
    current_cfg = {"S": adj.OU_STOPPAGE_MIN, "k": adj.OU_REMAINING_XG_MULT, "prior": adj.OU_SCORE_AWARE_PRIOR}

    rep = {"snapshots": len(rows), "matches": len({r["match"] for r in rows}),
           "unjoined_snapshots": unjoined, "current_config": current_cfg,
           "overall": {"legacy": describe(rows, "p_legacy"), "current": describe(rows)},
           "by_band": [], "reliability": {"legacy": [], "current": []}}
    for lo, hi in MINUTE_BANDS:
        for need in (1, 2, 3):
            sub = [r for r in rows if lo <= r["minute"] < hi and r["need"] == need]
            if sub:
                rep["by_band"].append({"minutes": f"{lo}-{hi - 1}", "need": need,
                                       "legacy": describe(sub, "p_legacy"), "current": describe(sub)})
    for key, name in (("p_legacy", "legacy"), ("p", "current")):
        for lo, hi in zip(PROB_BINS, PROB_BINS[1:]):
            d = describe([r for r in rows if lo <= r[key] < hi], key)
            if d:
                rep["reliability"][name].append({"bin": f"{lo:.2f}-{min(hi, 1):.2f}", **d})
    m90 = [r for r in rows if r["minute"] >= 90]
    rep["minute_90"] = {"n": len(m90), "matches": len({r["match"] for r in m90}),
                        "ended_over": sum(r["y"] for r in m90),
                        "legacy_pred": float(np.mean([r["p_legacy"] for r in m90])) if m90 else None,
                        "current_pred": float(np.mean([r["p"] for r in m90])) if m90 else None}

    oof, chosen = cross_fit(rows)
    best = fit(rows)
    rep["fits"] = {
        "legacy": dict(zip(("brier", "logloss"), score(rows, [r["p_legacy"] for r in rows]))),
        "current": dict(zip(("brier", "logloss"), score(rows, [r["p"] for r in rows]))),
        "oof": {**dict(zip(("brier", "logloss"), score(rows, oof))), "chosen_per_fold": chosen,
                "overall_after": describe([{**r, "p": p} for r, p in zip(rows, oof)])},
        "full_sample_best": best,
    }

    def fmt(d):
        return f"{d['pred']:>7.3f}{d['bias']:>+8.3f}"

    o = rep["overall"]
    lines = [f"Live O/U 2.5 calibration — {rep['snapshots']} snapshots, {rep['matches']} matches "
             f"(locked-Over excluded; {unjoined} snapshots had no final score)",
             f"CURRENT = LiveAdjuster as configured ({_cfg(current_cfg)}); LEGACY = {_cfg(LEGACY)}", ""]
    for name in ("legacy", "current"):
        d = o[name]
        lines.append(f"Overall {name:<8} predicted {d['pred']:.3f}  actual {d['actual']:.3f}  "
                     f"bias {d['bias']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")
    lines += ["", "By minute x goals needed (bias = actual - predicted; + means too pessimistic on Over)",
              f"{'minutes':<9}{'need':>5}{'snaps':>7}{'matches':>9}{'actual':>8}"
              f"{'legacy':>8}{'bias':>8}{'current':>9}{'bias':>8}  current 95% CI"]
    for b in rep["by_band"]:
        L, C = b["legacy"], b["current"]
        lines.append(f"{b['minutes']:<9}{b['need']:>5}{C['n']:>7}{C['matches']:>9}{C['actual']:>8.3f}"
                     f"{fmt(L):>16} {fmt(C):>16}  [{C['ci'][0]:+.3f}, {C['ci'][1]:+.3f}]")
    for name in ("legacy", "current"):
        lines += ["", f"Reliability by predicted P(Over) — {name}",
                  f"{'bin':<12}{'snaps':>7}{'matches':>9}{'pred':>8}{'actual':>8}{'bias':>8}  95% CI"]
        for b in rep["reliability"][name]:
            lines.append(f"{b['bin']:<12}{b['n']:>7}{b['matches']:>9}{b['pred']:>8.3f}{b['actual']:>8.3f}"
                         f"{b['bias']:>+8.3f}  [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]")
    m = rep["minute_90"]
    if m["n"]:
        lines += ["", f"Minute 90 (stoppage AND finished are both scraped as 90): {m['n']} snapshots, "
                  f"{m['matches']} matches, {m['ended_over']} ended Over — mean P(Over) legacy "
                  f"{m['legacy_pred']:.3f}, current {m['current_pred']:.3f}"]
    f = rep["fits"]
    lines += ["", "Scores (lower is better)",
              f"  legacy             Brier {f['legacy']['brier']:.5f}  logloss {f['legacy']['logloss']:.5f}",
              f"  current            Brier {f['current']['brier']:.5f}  logloss {f['current']['logloss']:.5f}"
              "   (in-sample: constants were chosen on this data)",
              f"  out-of-fold refit  Brier {f['oof']['brier']:.5f}  logloss {f['oof']['logloss']:.5f}"
              f"   bias after {f['oof']['overall_after']['bias']:+.3f} "
              f"[{f['oof']['overall_after']['ci'][0]:+.3f}, {f['oof']['overall_after']['ci'][1]:+.3f}]",
              "  per-fold choice:   " + "; ".join(_cfg(c) for c in f["oof"]["chosen_per_fold"]),
              f"  full-sample best:  {_cfg(best)}"
              + ("   (= current)" if best == current_cfg else "   (differs from current — consider updating LiveAdjuster)")]
    text = "\n".join(lines)
    print(text)

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(OUT_DIR, f"ou_live_calibration_{ts}.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    with open(os.path.join(OUT_DIR, f"ou_live_calibration_{ts}.txt"), "w") as fh:
        fh.write(text + "\n")
    print(f"\nSaved output/experiments/ou_live_calibration_{ts}.{{json,txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
