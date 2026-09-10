"""Cashout backtest CLI.

Replays settled historical bets through synthetic (or real, where available)
per-minute trajectories under one or more candidate cashout rules, and prints
a P/L delta report. Saves machine-readable + human-readable artifacts under
output/backtests/.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --rules lock_in_profit,stop_loss --lanes value
    python scripts/run_backtest.py --start 2026-04-01 --end 2026-05-15 --paths 100

Self-validation: include the `null` rule (the default does); its `baseline_pnl`
column must match the actual stored slip P/L within rounding. Mismatch ⇒
simulator bug, don't trust other rules.
"""

import argparse
import datetime
import glob
import json
import os
import statistics
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_project'))

from ml_project.live_adjuster import LiveAdjuster
from ml_project.backtest.rules import RULES
from ml_project.backtest.report import aggregate, pretty_print
from ml_project.backtest.simulator import (
    Outcome, baseline_pnl_from_result, walk_bet,
)
from ml_project.backtest.trajectories import (
    RealTrajectory, SyntheticTrajectory,
)
from ml_project.backtest.coverage import (
    load_pred_row, load_verif_row, slips_in_range,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')


def build_trajectories(args, date: str, match_id: str, hg: int, ag: int):
    """Return a list of trajectories. Length 1 for real data; `paths` for synthetic."""
    if args.data in ('real', 'auto'):
        # Live history now lives in output/live_history/; fall back to the
        # legacy flat output/ path for any snapshots not yet migrated.
        hist = os.path.join(OUTPUT_DIR, 'live_history', f'live_history_{date}.jsonl')
        if not os.path.exists(hist):
            hist = os.path.join(OUTPUT_DIR, f'live_history_{date}.jsonl')
        t = RealTrajectory.from_jsonl(hist, match_id)
        if t:
            return [t], 'real'
    if args.data == 'real':
        return [], 'real'  # explicitly requested real, none available
    paths = [
        SyntheticTrajectory.generate(hg, ag, seed=args.seed + i)
        for i in range(args.paths)
    ]
    return paths, 'synthetic'


def reduce_paths(per_path_outcomes, sample_outcome):
    """Average rule_pnl/delta across Monte Carlo paths for a single (bet, rule)."""
    n = len(per_path_outcomes)
    triggered = [o for o in per_path_outcomes if o.triggered]
    avg_rule_pnl = sum(o.rule_pnl for o in per_path_outcomes) / n
    avg_delta = sum(o.delta for o in per_path_outcomes) / n
    trig_min = (
        int(statistics.mean(o.trigger_minute for o in triggered))
        if triggered else None
    )
    cashout = (
        round(statistics.mean(o.cashout_value for o in triggered), 2)
        if triggered else None
    )
    return Outcome(
        bet_id=sample_outcome.bet_id,
        lane=sample_outcome.lane,
        bet_type=sample_outcome.bet_type,
        rule_name=sample_outcome.rule_name,
        triggered=(len(triggered) / n) >= 0.5,
        trigger_minute=trig_min,
        cashout_value=cashout,
        baseline_pnl=sample_outcome.baseline_pnl,
        rule_pnl=round(avg_rule_pnl, 2),
        delta=round(avg_delta, 2),
        note=sample_outcome.note,
    )


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument('--start', default=(datetime.date.today() - datetime.timedelta(days=30)).isoformat(),
                    help='Start date (inclusive, YYYY-MM-DD). Default: 30 days ago.')
    ap.add_argument('--end', default=datetime.date.today().isoformat(),
                    help='End date (inclusive, YYYY-MM-DD). Default: today.')
    ap.add_argument('--rules', default='null,lock_in_profit,stop_loss,late_drift,momentum_fade',
                    help='Comma-separated rule names. Available: ' + ', '.join(RULES))
    ap.add_argument('--lanes', default='value,conviction,model',
                    help='Comma-separated lanes to include.')
    ap.add_argument('--paths', type=int, default=50,
                    help='Monte Carlo trajectories per bet (synthetic mode).')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data', choices=['synthetic', 'real', 'auto'], default='auto',
                    help='auto = real where available, synthetic otherwise.')
    ap.add_argument('--output-dir', default=os.path.join(OUTPUT_DIR, 'backtests'))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rules = [(name, RULES[name]) for name in args.rules.split(',') if name in RULES]
    if not rules:
        print(f"No valid rules. Available: {list(RULES)}", file=sys.stderr)
        sys.exit(2)
    lanes = {l.strip() for l in args.lanes.split(',') if l.strip()}

    adjuster = LiveAdjuster()
    outcomes = []
    counters = {'evaluated': 0, 'no_pred': 0, 'no_verif': 0,
                'unsettled': 0, 'lane_filtered': 0, 'real_used': 0, 'synth_used': 0}

    for slip_path, slip in slips_in_range(args.start, args.end):
        date = slip.get('date', '')
        for idx, bet in enumerate(slip.get('bets', [])):
            if bet.get('lane', 'value') not in lanes:
                counters['lane_filtered'] += 1
                continue
            result = bet.get('result', '')
            status = bet.get('status', '')
            # CASHED_OUT bets are deliberately excluded from this filter
            # (they pass neither tuple). Their P/L was set at cashout time,
            # so simulating "what if we held/cashed at min N" against them
            # would compare two cashouts rather than cashout-vs-hold. We
            # have plenty of WON/LOST bets to evaluate rules against.
            if result not in ('WON', 'LOST') and status not in ('WON', 'LOST', 'VOID'):
                counters['unsettled'] += 1
                continue

            home = bet.get('home') or (bet.get('match', '').split(' vs ')[0] if ' vs ' in bet.get('match', '') else None)
            away = bet.get('away') or (bet.get('match', '').split(' vs ')[1] if ' vs ' in bet.get('match', '') else None)
            if not home or not away:
                continue

            pred = load_pred_row(date, home, away)
            if pred is None:
                counters['no_pred'] += 1
                continue
            verif = load_verif_row(date, home, away)
            if verif is None:
                counters['no_verif'] += 1
                continue
            try:
                hg, ag = (int(x) for x in str(verif['Score']).split('-'))
            except (ValueError, AttributeError):
                continue

            bet_type = bet.get('type', '1X2')
            if bet_type == '1X2':
                pre_probs = {
                    'home': float(pred['Home Win %']),
                    'draw': float(pred['Draw %']),
                    'away': float(pred['Away Win %']),
                }
            elif bet_type == 'O/U':
                pre_probs = {
                    'over':  float(pred['Over %']),
                    'under': float(pred['Under %']),
                }
            else:
                continue

            bet['_bet_id'] = f"{os.path.basename(slip_path)}:{idx}"
            counters['evaluated'] += 1

            trajectories, mode = build_trajectories(args, date, bet.get('match_id', ''), hg, ag)
            if not trajectories:
                continue
            counters[('real_used' if mode == 'real' else 'synth_used')] += 1

            for rule_name, rule_fn in rules:
                per_path = [
                    walk_bet(bet, pre_probs, t, rule_fn, adjuster, rule_name=rule_name)
                    for t in trajectories
                ]
                outcomes.append(reduce_paths(per_path, per_path[0]))

    groups = aggregate(outcomes)
    title = (f"Backtest {args.start} → {args.end}   "
             f"data={args.data} paths={args.paths}   "
             f"bets={counters['evaluated']} (real={counters['real_used']}, synth={counters['synth_used']})")
    report = pretty_print(groups, title=title)
    print(report)

    skipped_msg = (f"\nSkipped — no_pred={counters['no_pred']} "
                   f"no_verif={counters['no_verif']} "
                   f"unsettled={counters['unsettled']} "
                   f"lane_filtered={counters['lane_filtered']}")
    print(skipped_msg)

    # Self-validation hint
    null_group = {k: v for k, v in groups.items() if k[0] == 'null'}
    if null_group:
        print("\nSelf-validation: null_rule's `Baseline P/L` column should "
              "equal the sum of actual stored P/L for the included slips. "
              "Cross-check against /football/betting if anything looks off.")

    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    json_out = os.path.join(args.output_dir, f'{ts}.json')
    txt_out  = os.path.join(args.output_dir, f'{ts}.txt')
    with open(json_out, 'w') as f:
        json.dump({
            'meta': {**vars(args), 'counters': counters},
            'outcomes': [vars(o) for o in outcomes],
            'aggregate': {f'{rule}|{lane}': g for (rule, lane), g in groups.items()},
        }, f, indent=2, default=str)
    with open(txt_out, 'w') as f:
        f.write(report + skipped_msg + '\n')
    print(f"\nSaved:\n  {json_out}\n  {txt_out}")

    # Prune older backtest outputs — keep latest 3 runs for short-term
    # Δ-trend comparison. The 2026-05-18 baseline values are recorded
    # in FOOTBALL_NEXT_STEPS.md ('The data wait' section) so dropping older files
    # doesn't lose the comparison anchor.
    _KEEP = 3
    for _patt in ('*.json', '*.txt'):
        _old = sorted(glob.glob(os.path.join(args.output_dir, _patt)))[:-_KEEP]
        for _f in _old:
            try: os.remove(_f)
            except OSError: pass
    print(f"Pruned older backtest outputs (kept latest {_KEEP}).")


if __name__ == '__main__':
    main()
