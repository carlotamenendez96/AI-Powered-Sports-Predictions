"""Is live-snapshot collection actually running? The failure mode is silence.

Collection feeds `output/live_history/*.jsonl`, which is the only real input
to the cashout backtest. It died unnoticed for three months (2026-06-14 ->
2026-09-10: `__main__` disarmed auto-cashout unconditionally at startup), and
nothing anywhere reported it -- the UI looked fine, the scheduler thread was
alive, the armed file just said false. Collection ran on 13 of 115 calendar
days in that span, 11.3% uptime.

This reports two things:
  - RETROSPECTIVE: per day, matches that had bets placed on them vs matches
    that actually got snapshots. A day with bets and zero snapshots is a GAP.
    Detectable after the fact because bets carry their kickoff in `date`.
  - NOW: armed/shadow flags, whether the server is up, and how stale the most
    recent snapshot is. Arming history isn't stored, so a past day's armed
    state is not recoverable -- only the bets-vs-snapshots comparison is.

Exit code is 0 even when gaps are found: this is a report, and it runs at the
tail of `bin/run_verification.sh` where it must never break the pipeline.

Usage:
    python scripts/check_live_collection.py            # last 14 days
    python scripts/check_live_collection.py --days 120
"""

import argparse
import collections
import datetime
import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_project'))

from ml_project.backtest.coverage import (
    live_history_match_ids, live_history_path, slips_in_range,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
ARMED_FILE = os.path.join(OUTPUT_DIR, 'auto_cashout_armed.json')


def bet_matches_by_kickoff_day(start, end):
    """{day: {match_id}} for bets whose kickoff falls on that day.

    Keyed on each bet's own kickoff (`date` = 'YYYY-MM-DD HH:MM'), not the
    slip date, because a slip can carry matches that kick off after midnight.
    """
    by_day = collections.defaultdict(set)
    # Widen the slip scan: a bet kicking off on `start` can sit in an earlier slip.
    slip_start = (datetime.date.fromisoformat(start) - datetime.timedelta(days=2)).isoformat()
    for _, slip in slips_in_range(slip_start, end):
        for bet in slip.get('bets', []):
            mid = bet.get('match_id')
            if not mid:
                continue
            raw = bet.get('date') or ''
            day = raw.split(' ')[0] if ' ' in raw else (raw or slip.get('date', ''))
            if start <= day <= end:
                by_day[day].add(mid)
    return by_day


# Mirrors `_live_window_active` / `_AUTO_CASHOUT_INTERVAL_S` / `_LIVE_MATCH_WINDOW_S`
# in web_ui/app.py. Reimplemented rather than imported because importing app.py
# builds the whole Flask app (blueprints, scheduler thread) just to read a
# predicate. app.py stays the source of truth -- drift here only mislabels a
# line in this report, it cannot change a cashout decision.
_LIVE_MATCH_WINDOW_S = 150 * 60
_SWEEP_INTERVAL_S = 10 * 60


def live_window_active(now=None):
    """Is any OPEN bet's match plausibly in play right now? Fails OPEN."""
    now = now or datetime.datetime.now()
    window = datetime.timedelta(seconds=_LIVE_MATCH_WINDOW_S)
    any_open = False
    parsed_any = False
    for _, slip in slips_in_range('0000-00-00', '9999-99-99'):
        for b in slip.get('bets', []):
            if b.get('status') != 'OPEN':
                continue
            any_open = True
            try:
                ko = datetime.datetime.strptime(b.get('date') or '', '%Y-%m-%d %H:%M')
            except (TypeError, ValueError):
                continue
            parsed_any = True
            if ko <= now <= ko + window:
                return True
    return any_open and not parsed_any


def server_running():
    try:
        out = subprocess.run(['ps', 'ax', '-o', 'pid=,command='],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if 'web_ui/app.py' in line and 'grep' not in line:
            return int(line.split()[0])
    return None


def last_snapshot_age(day):
    p = live_history_path(day)
    if p is None:
        return None, 0
    rows = 0
    latest = None
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                ts = json.loads(line).get('ts')
            except json.JSONDecodeError:
                continue
            if ts and (latest is None or ts > latest):
                latest = ts
    return latest, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--days', type=int, default=14, help='days back to audit (default 14)')
    args = ap.parse_args()

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=args.days - 1)).isoformat()
    end = today.isoformat()
    expected = bet_matches_by_kickoff_day(start, end)

    print(f'=== Live collection health   {start} → {end} ===\n')
    print(f'{"Day":<12}{"Bets on":>9}{"Snapshotted":>13}{"Covered":>9}   Status')
    print(f'{"":<12}{"matches":>9}{"matches":>13}{"":>9}')
    print('-' * 62)

    gaps, partials, clean, idle = [], [], 0, 0
    for i in range(args.days):
        day = (datetime.date.fromisoformat(start) + datetime.timedelta(days=i)).isoformat()
        if day > end:
            break
        want = expected.get(day, set())
        got = live_history_match_ids(day)
        covered = len(want & got)
        if day == end:
            # Today is still running: uncovered matches are mostly kickoffs that
            # haven't happened yet, so scoring it partial/GAP is a false alarm.
            status = 'in progress' + ('' if got else ' — nothing yet')
        elif not want:
            idle += 1
            status = 'no bets'
        elif not got:
            gaps.append(day)
            status = 'GAP — bets placed, nothing collected'
        elif covered < len(want):
            partials.append(day)
            status = f'partial — {len(want) - covered} match(es) uncovered'
        else:
            clean += 1
            status = 'ok'
        print(f'{day:<12}{len(want):>9}{len(got):>13}{covered:>9}   {status}')

    print('-' * 62)
    print(f'{clean} clean · {len(partials)} partial · {len(gaps)} GAP · {idle} idle (no bets)')
    if gaps:
        print(f'\n!! COLLECTION GAPS on {len(gaps)} day(s): {", ".join(gaps)}')
        print('   Bets were live and nothing was snapshotted. Check that the '
              'server is up and auto-cashout is armed.')

    # --- current state ---
    print('\n--- right now ---')
    try:
        with open(ARMED_FILE) as f:
            armed = json.load(f)
    except (OSError, json.JSONDecodeError):
        armed = {}
    on = armed.get('armed', False)
    shadow = armed.get('shadow', True)  # missing key defaults to shadow
    pid = server_running()
    latest, rows = last_snapshot_age(end)

    print(f'  armed      : {on}' + ('' if on else '   <-- no sweeps, so no collection'))
    print(f'  shadow     : {shadow}' + ('   (evaluates + logs, fires nothing)' if shadow else '   (REAL firing)'))
    print(f'  server     : ' + (f'running (pid {pid})' if pid else 'NOT RUNNING   <-- scheduler thread is dead'))
    print(f'  today      : {rows} snapshot(s)' + (f', last at {latest}' if latest else ''))
    if latest:
        try:
            age = (datetime.datetime.now() - datetime.datetime.fromisoformat(latest)).total_seconds()
            in_window = live_window_active()
            note = ('' if in_window
                    else '   (live window closed — staleness expected)')
            print(f'  last snap  : {age / 60:.0f} min ago{note}')
            if in_window and age > 2 * _SWEEP_INTERVAL_S:
                print(f'\n!! Live window is OPEN but the last snapshot is '
                      f'{age / 60:.0f} min old (sweeps are every '
                      f'{_SWEEP_INTERVAL_S // 60} min) — collection may be stalled. '
                      f'Check logs/ui.log.')
        except ValueError:
            pass
    if on and not pid:
        print('\n!! Armed but the server is down — nothing is collecting. '
              './bin/manage_server.sh start')
    elif not on:
        print('\n!! Not armed — collection is off. Tick "Auto-cashout" on the '
              'dashboard, or POST /football/auto_cashout/arm with on=1.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
