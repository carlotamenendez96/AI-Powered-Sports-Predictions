"""Is a cashout-backtest re-run worth it yet? Trigger on data, not on days.

The backtest's answer only moves when new *bound* trajectories accrue -- a
settled bet whose `match_id` has live_history snapshots and that resolves to
a predictions + verification row. Elapsed time moves nothing: collection ran
on 13 of the 115 days between 2026-05-18 and 2026-09-10 (11.3% uptime), so a
"every N days" trigger would have fired ~12 times through the 06-14..09-10
collection outage and emitted ~12 identical reports while the actual problem
went unreported. See `check_live_collection.py` for that half.

Counting goes through `ml_project.backtest.coverage`, which is the same
module `run_backtest.py` binds with, so this cannot drift from what the
harness would actually score.

The auto-run is `--data real` on purpose. The 2026-09-10 run showed the
synthetic arm is sign-inverted from the real one on every level-based rule,
so an unattended report that silently mixed in synthetic paths would be
worse than no report.

Usage:
    python scripts/check_backtest_due.py              # report status only
    python scripts/check_backtest_due.py --run        # run if due, else no-op
    python scripts/check_backtest_due.py --run --force
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_project'))

from ml_project.backtest.coverage import count_bound, trajectory_dates

STAMP = os.path.join(PROJECT_ROOT, 'output', 'backtests', '.last_run.json')
DEFAULT_THRESHOLD = 50


def read_stamp():
    try:
        with open(STAMP) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_stamp(bound, ran, window):
    os.makedirs(os.path.dirname(STAMP), exist_ok=True)
    with open(STAMP, 'w') as f:
        json.dump({
            'ts': datetime.datetime.now().isoformat(timespec='seconds'),
            'bound_count': bound,
            'ran_backtest': ran,
            'window': window,
        }, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--threshold', type=int, default=DEFAULT_THRESHOLD,
                    help=f'new bound trajectories required to re-run (default {DEFAULT_THRESHOLD})')
    ap.add_argument('--run', action='store_true',
                    help='actually run the backtest when due (otherwise just report)')
    ap.add_argument('--force', action='store_true',
                    help='run regardless of how little has accrued')
    args = ap.parse_args()

    bound = count_bound()
    stamp = read_stamp()
    dates = trajectory_dates()

    if not dates:
        print('[backtest] no live_history at all — nothing to score. '
              'Check collection: scripts/check_live_collection.py')
        return 0

    if stamp is None:
        # Don't fire on first sight: the existing corpus is not "new data".
        # Record it as the baseline so the next +threshold is measured from here.
        write_stamp(bound, ran=False, window=None)
        print(f'[backtest] baseline recorded at {bound} bound trajectories — '
              f'not running. Re-runs fire at +{args.threshold}. '
              f'Use --run --force to run now.')
        if not (args.run and args.force):
            return 0
        stamp = {'bound_count': bound}

    last = stamp.get('bound_count', 0)
    new = bound - last
    due = new >= args.threshold

    if not (due or args.force):
        print(f'[backtest] not due — {bound} bound trajectories, '
              f'+{new} since last run (need +{args.threshold}). '
              f'Last run: {stamp.get("ts", "never")}.')
        return 0

    why = 'forced' if (args.force and not due) else f'+{new} new bound trajectories'
    if not args.run:
        print(f'[backtest] DUE ({why}) — {bound} bound trajectories. '
              f'Run: python scripts/check_backtest_due.py --run')
        return 0

    window = {'start': dates[0], 'end': datetime.date.today().isoformat()}
    print(f'[backtest] running ({why}) — real arm over '
          f'{window["start"]} → {window["end"]}, {bound} bound trajectories')
    env = dict(os.environ)
    env['PYTHONPATH'] = (f'{PROJECT_ROOT}:{os.path.join(PROJECT_ROOT, "ml_project")}'
                         f':{env.get("PYTHONPATH", "")}')
    proc = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, 'scripts', 'run_backtest.py'),
         '--start', window['start'], '--end', window['end'], '--data', 'real'],
        cwd=PROJECT_ROOT, env=env,
    )
    if proc.returncode != 0:
        print(f'[backtest] run FAILED (exit {proc.returncode}) — stamp not advanced, '
              f'will retry next time', file=sys.stderr)
        return 0  # non-fatal: never break the caller's pipeline
    write_stamp(bound, ran=True, window=window)
    print(f'[backtest] done — stamp advanced to {bound}. '
          f'Record the Δ row in FOOTBALL_NEXT_STEPS.md if the sign moved.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
