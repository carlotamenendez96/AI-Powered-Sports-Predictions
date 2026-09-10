"""Which settled bets can the backtest actually score against real trajectories?

Split out of `scripts/run_backtest.py` so the "is a re-run worth it yet?"
check counts *exactly* what the harness would bind, rather than a second
implementation of the same eligibility rules that quietly drifts from it.
`run_backtest.py` imports these helpers instead of defining its own.

A bet is **bound** when every one of these holds -- the same gauntlet the
backtest's main loop walks:
  - it sits in a slip dated in range and in a requested lane
  - it is settled (WON/LOST, or VOID via `status`)
  - its match resolves to a predictions row and a verification row
  - its `match_id` has at least one snapshot in that date's live_history

Bound count is the only number that moves the backtest's answer, so it -- not
elapsed time -- is what should trigger a re-run.
"""

import glob
import json
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
LIVE_HISTORY_DIR = os.path.join(OUTPUT_DIR, 'live_history')

ALL_LANES = ('value', 'conviction', 'model')


def _resolve_artifact(name: str):
    """Locate an output artifact, falling back to the soft-delete archive.

    `/football/delete_file` moves artifacts to `output/history/`, so any date
    old enough to have been archived would otherwise look like missing data --
    which silently zeroed the real-trajectory arm of the backtest, since
    live_history only covers those older dates.
    """
    for d in (OUTPUT_DIR, os.path.join(OUTPUT_DIR, 'history')):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def slips_in_range(start: str, end: str):
    """Yield (path, slip_dict) for every bets_*.json in [start, end]."""
    seen = set()
    for d in (OUTPUT_DIR, os.path.join(OUTPUT_DIR, 'history')):
        for f in glob.glob(os.path.join(d, 'bets_*.json')):
            base = os.path.basename(f)
            if base in seen:
                continue
            seen.add(base)
            try:
                with open(f) as fh:
                    slip = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            date = slip.get('date', '')
            if start <= date <= end:
                yield f, slip


def load_pred_row(date: str, home: str, away: str):
    p = _resolve_artifact(f'predictions_{date}.csv')
    if p is None:
        return None
    df = pd.read_csv(p)
    m = df[(df['Home Team'] == home) & (df['Away Team'] == away)]
    return m.iloc[0] if not m.empty else None


def load_verif_row(date: str, home: str, away: str):
    p = _resolve_artifact(f'verification_{date}.csv')
    if p is None:
        return None
    df = pd.read_csv(p)
    m = df[(df['Home'] == home) & (df['Away'] == away)]
    return m.iloc[0] if not m.empty else None


def live_history_path(date: str):
    """Snapshot file for `date`, or None. Legacy flat path is the fallback."""
    for p in (os.path.join(LIVE_HISTORY_DIR, f'live_history_{date}.jsonl'),
              os.path.join(OUTPUT_DIR, f'live_history_{date}.jsonl')):
        if os.path.exists(p):
            return p
    return None


def live_history_match_ids(date: str):
    """Set of match_ids with at least one snapshot on `date`."""
    p = live_history_path(date)
    if p is None:
        return set()
    ids = set()
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid = d.get('match_id')
            if mid:
                ids.add(mid)
    return ids


def _teams(bet):
    home = bet.get('home')
    away = bet.get('away')
    if not (home and away):
        match = bet.get('match', '')
        if ' vs ' in match:
            home, away = match.split(' vs ', 1)
    return home, away


def bound_bets(start: str = '0000-00-00', end: str = '9999-99-99', lanes=ALL_LANES):
    """Yield one dict per bet the backtest could score on a real trajectory."""
    lanes = set(lanes)
    ids_by_date = {}
    for slip_path, slip in slips_in_range(start, end):
        date = slip.get('date', '')
        if date not in ids_by_date:
            ids_by_date[date] = live_history_match_ids(date)
        live_ids = ids_by_date[date]
        if not live_ids:
            continue  # no snapshots that day -- nothing can bind
        for idx, bet in enumerate(slip.get('bets', [])):
            if bet.get('lane', 'value') not in lanes:
                continue
            if bet.get('type', '1X2') not in ('1X2', 'O/U'):
                continue
            if (bet.get('result', '') not in ('WON', 'LOST')
                    and bet.get('status', '') not in ('WON', 'LOST', 'VOID')):
                continue
            match_id = bet.get('match_id', '')
            if match_id not in live_ids:
                continue
            home, away = _teams(bet)
            if not (home and away):
                continue
            if load_pred_row(date, home, away) is None:
                continue
            if load_verif_row(date, home, away) is None:
                continue
            yield {
                'date': date,
                'lane': bet.get('lane', 'value'),
                'match_id': match_id,
                'bet_id': f"{os.path.basename(slip_path)}:{idx}",
            }


def count_bound(start: str = '0000-00-00', end: str = '9999-99-99', lanes=ALL_LANES) -> int:
    return sum(1 for _ in bound_bets(start, end, lanes))


def trajectory_dates():
    """Sorted dates that have at least one live_history snapshot."""
    dates = set()
    for d in (LIVE_HISTORY_DIR, OUTPUT_DIR):
        for f in glob.glob(os.path.join(d, 'live_history_*.jsonl')):
            base = os.path.basename(f)
            dates.add(base[len('live_history_'):-len('.jsonl')])
    return sorted(dates)
