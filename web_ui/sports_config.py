"""
Sport-aware, lane-aware betting config access.

Each sport carries three independent bankrolls — `value`, `conviction`,
`model` — under `sports.<slug>.bankrolls.<lane>.{current,initial}`. All
routes go through these helpers so the JSON schema lives in exactly one
place.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'data_sets', 'betting_config.json')

# betting_config.json is gitignored deployment state (live bankroll balances),
# so a fresh clone has only the template. Seed it before the first read or the
# UI comes up with zero bankrolls and no obvious reason why. Path insert is
# defensive: app.py adds PROJECT_ROOT for us, but this module is also imported
# by CLI paths that have not.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
try:
    from ml_project.config_bootstrap import seed_all as _seed_all
    _seed_all()
except Exception:       # never let seeding break config access
    pass

LANES = ('value', 'conviction', 'model')

DEFAULT_LANE_BANKROLL = {'current': 0.0, 'initial': 0.0}

# Defaults merged with the per-sport overrides on read. Keeps callers free
# from None-checks for keys that haven't been set yet.
DEFAULT_SPORT_CONFIG = {
    'min_confidence': 0.45,
    'stake_multiplier': 0.4,
    'min_stake_eur': 2.0,
    'max_stake_pct': 0.03,
    # Clamp EV input to the value-lane sizing formula. Stops a single
    # high-EV pick (often from a low-data league where the model
    # overstates) from dominating the slip. Per-league recalibration is
    # the upstream fix — see FOOTBALL_NEXT_STEPS.md.
    'ev_cap_value': 0.05,
    # Phase C4: apply per-league Platt calibrators from
    # data_sets/league_calibration.json before the heuristic adjuster
    # in predict_matches.py. Toggle off to A/B against raw probs.
    'use_league_calibration': True,
    # Conviction lane
    'conviction_min_confidence': 0.65,
    'conviction_min_odds': 1.40,
    'conviction_stake_pct': 0.005,
    # Model lane — broad coverage, confidence/odds-aware sizing.
    # Uses its own lower min-stake floor so wide coverage isn't killed.
    'model_base_pct': 0.005,
    'model_max_stake_pct': 0.015,
    'model_min_stake_eur': 1.0,
    'model_ev_factor_min': 0.5,
    'model_ev_factor_max': 1.5,
    # Per-lane daily exposure caps (fraction of that lane's bankroll)
    'value_max_daily_exposure_pct': 0.10,
    'conviction_max_daily_exposure_pct': 0.10,
    'model_max_daily_exposure_pct': 0.15,
    # Where the displayed cashout value on live rows comes from:
    #   'synthetic' (default) — internal fair-value estimate
    #     `stake × odds × adj_prob × 0.95` computed in _attach_open_bets.
    #   'bookmaker' — read from output/real_betting/open_bets_snapshot.json
    #     when a fresh entry exists for the match; falls back to
    #     synthetic when the snapshot is stale/missing for that match.
    # Toggling to 'bookmaker' requires real-betting scenario #3B (scraper)
    # to be producing snapshots and at least one real bet on the
    # bookmaker site. Default 'synthetic' is the safe / pre-real-betting
    # value. See real_betting/test_case_scenarios.md.
    'cashout_source': 'synthetic',
    # How long (seconds) a bookmaker snapshot stays "fresh" before
    # _attach_open_bets falls back to synthetic. Default 600s = 10 min;
    # matches the Auto-5m refresh cadence with one cycle of slack.
    'cashout_snapshot_max_age_s': 600,
}


def _load():
    if not os.path.exists(CONFIG_PATH):
        return {'sports': {}}
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {'sports': {}}


def _save(config):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)


def _empty_bankrolls():
    return {lane: dict(DEFAULT_LANE_BANKROLL) for lane in LANES}


def get_sport_config(sport_slug):
    """Full per-sport config (defaults + overrides), including bankrolls."""
    cfg = _load()
    sport_cfg = cfg.get('sports', {}).get(sport_slug, {})
    out = dict(DEFAULT_SPORT_CONFIG)
    out.update({k: v for k, v in sport_cfg.items() if k != 'bankrolls'})
    bankrolls = sport_cfg.get('bankrolls', {})
    out['bankrolls'] = {
        lane: dict(DEFAULT_LANE_BANKROLL, **bankrolls.get(lane, {}))
        for lane in LANES
    }
    return out


def get_bankroll(sport_slug, lane='value'):
    """Current bankroll for one lane of a sport (0.0 if unknown)."""
    if lane not in LANES:
        raise ValueError(f"Unknown lane: {lane!r}. Valid lanes: {LANES}")
    return get_sport_config(sport_slug)['bankrolls'][lane]['current']


def update_bankroll(sport_slug, delta, lane='value'):
    """Add `delta` to one lane's current bankroll, return the new value.

    Negative delta when placing bets (debit); positive when settling
    wins / void refunds (credit).
    """
    if lane not in LANES:
        raise ValueError(f"Unknown lane: {lane!r}. Valid lanes: {LANES}")
    cfg = _load()
    sports = cfg.setdefault('sports', {})
    sport = sports.setdefault(sport_slug, {})
    bankrolls = sport.setdefault('bankrolls', _empty_bankrolls())
    bucket = bankrolls.setdefault(lane, dict(DEFAULT_LANE_BANKROLL))
    new_value = round(bucket.get('current', 0.0) + delta, 2)
    bucket['current'] = new_value
    _save(cfg)
    return new_value


def set_tunables(sport_slug, updates):
    """Persist a partial update of a sport's strategy tunables.

    `updates` is a {key: value} dict — only keys present in
    DEFAULT_SPORT_CONFIG (excluding bankroll state) are accepted; others
    are silently ignored to prevent the UI from writing arbitrary fields.
    Returns the post-update merged config (defaults + overrides).
    """
    allowed = set(DEFAULT_SPORT_CONFIG.keys())
    clean = {k: v for k, v in updates.items() if k in allowed}
    if not clean:
        return get_sport_config(sport_slug)
    cfg = _load()
    sports = cfg.setdefault('sports', {})
    sport = sports.setdefault(sport_slug, {})
    for k, v in clean.items():
        sport[k] = v
    _save(cfg)
    return get_sport_config(sport_slug)


def reset_tunable(sport_slug, key):
    """Remove a single tunable override so it reverts to its default.
    Returns True if anything was removed."""
    if key not in DEFAULT_SPORT_CONFIG:
        return False
    cfg = _load()
    sport = cfg.get('sports', {}).get(sport_slug)
    if not sport or key not in sport:
        return False
    del sport[key]
    _save(cfg)
    return True


def lane_bankrolls(sport_slug):
    """Return {lane: current_bankroll} for one sport."""
    cfg = get_sport_config(sport_slug)
    return {lane: cfg['bankrolls'][lane]['current'] for lane in LANES}


def sport_total(sport_slug):
    """Sum bankrolls across all lanes of one sport."""
    return round(sum(lane_bankrolls(sport_slug).values()), 2)


def all_bankrolls():
    """{sport_slug: sport_total} — flat per-sport view for navbar/landing."""
    cfg = _load()
    out = {}
    for slug in cfg.get('sports', {}).keys():
        out[slug] = sport_total(slug)
    return out


def all_lane_bankrolls():
    """{sport_slug: {lane: current}} — detailed view for portfolio breakdown."""
    cfg = _load()
    return {slug: lane_bankrolls(slug) for slug in cfg.get('sports', {}).keys()}


def total_bankroll():
    """Sum bankrolls across every lane of every sport."""
    return round(sum(all_bankrolls().values()), 2)
