"""Aggregate Outcomes into per-(rule, lane) stats and pretty-print."""

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .simulator import Outcome


def aggregate(outcomes: Iterable[Outcome]) -> Dict[Tuple[str, str], dict]:
    """Group outcomes by (rule, lane); compute summary stats per group."""
    def _empty():
        return {
            'bets': 0, 'triggered': 0,
            'stake': 0.0, 'baseline_pnl': 0.0, 'rule_pnl': 0.0, 'delta': 0.0,
            'baseline_won': 0, 'baseline_lost': 0,
            # Triggered bets split by what really happened. On a bet that
            # really LOST, Δ = cashout received (>= 0: money saved). On one
            # that really WON, Δ = cashout - winnings (< 0: money forfeited).
            'trig_lost': 0, 'delta_lost': 0.0, 'trig_won': 0, 'delta_won': 0.0,
            'trigger_minutes': [],
            'note_counts': defaultdict(int),
        }
    groups = defaultdict(_empty)

    for o in outcomes:
        g = groups[(o.rule_name, o.lane)]
        g['bets'] += 1
        g['baseline_pnl'] += o.baseline_pnl
        g['rule_pnl'] += o.rule_pnl
        g['delta'] += o.delta
        if o.note:
            g['note_counts'][o.note] += 1
        if o.triggered:
            g['triggered'] += 1
            if o.trigger_minute is not None:
                g['trigger_minutes'].append(o.trigger_minute)
            if o.baseline_pnl < 0:
                g['trig_lost'] += 1
                g['delta_lost'] += o.delta
            elif o.baseline_pnl > 0:
                g['trig_won'] += 1
                g['delta_won'] += o.delta
        if o.baseline_pnl > 0:
            g['baseline_won'] += 1
        elif o.baseline_pnl < 0:
            g['baseline_lost'] += 1

    for g in groups.values():
        for f in ('baseline_pnl', 'rule_pnl', 'delta', 'stake', 'delta_lost', 'delta_won'):
            g[f] = round(g[f], 2)
        g['trigger_rate'] = round(g['triggered'] / g['bets'] * 100, 1) if g['bets'] else 0.0
        if g['trigger_minutes']:
            ms = sorted(g['trigger_minutes'])
            n = len(ms)
            g['trigger_min_p25'] = ms[max(0, n // 4)]
            g['trigger_min_med'] = ms[n // 2]
            g['trigger_min_p75'] = ms[min(n - 1, 3 * n // 4)]
        g['note_counts'] = dict(g['note_counts'])  # for JSON serialization

    return dict(groups)


def pretty_print(groups: Dict[Tuple[str, str], dict], title: str = '') -> str:
    lines = []
    if title:
        lines.append(f"=== {title} ===")
        lines.append('')
    header = (f"{'Rule':<18}{'Lane':<14}{'Bets':>6}{'Trig':>6}{'Trig%':>8}"
              f"{'Baseline P/L':>16}{'Rule P/L':>14}{'Δ':>12}"
              f"{'Saved on losers':>20}{'Lost on winners':>20}{'TrigMin (p25/med/p75)':>26}")
    lines.append(header)
    lines.append('-' * len(header))

    for (rule, lane), g in sorted(groups.items()):
        trig_min = ''
        if g.get('trigger_min_med') is not None:
            trig_min = f"{g['trigger_min_p25']}/{g['trigger_min_med']}/{g['trigger_min_p75']}"
        lines.append(
            f"{rule:<18}{lane:<14}{g['bets']:>6}{g['triggered']:>6}{g['trigger_rate']:>7.1f}%"
            f"{g['baseline_pnl']:>+16.2f}{g['rule_pnl']:>+14.2f}{g['delta']:>+12.2f}"
            f"{g['trig_lost']:>7} {g['delta_lost']:>+12.2f}{g['trig_won']:>7} {g['delta_won']:>+12.2f}"
            f"{trig_min:>26}"
        )

    lines.append('')
    lines.append("Saved on losers = triggers on bets that really LOST: count, and the cashout "
                 "money recovered (Δ > 0).")
    lines.append("Lost on winners = triggers on bets that really WON: count, and the winnings "
                 "given up (Δ < 0). Δ = the two summed.")

    # Notes summary (e.g., O/U skipped count)
    notes = defaultdict(int)
    for g in groups.values():
        for k, v in g.get('note_counts', {}).items():
            notes[k] += v
    if notes:
        lines.append('')
        lines.append(f"Notes: {dict(notes)}")

    return '\n'.join(lines)
