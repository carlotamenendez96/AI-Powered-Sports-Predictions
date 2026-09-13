from flask import Flask, render_template, request, redirect, url_for, flash, Blueprint, g
import pandas as pd
import os
import subprocess
import datetime
import glob
import json
import shutil
import sys
import time

from sports_config import (
    get_sport_config,
    get_bankroll,
    update_bankroll,
    all_bankrolls,
    all_lane_bankrolls,
    lane_bankrolls,
    sport_total,
    total_bankroll,
    set_tunables,
    DEFAULT_SPORT_CONFIG,
    LANES,
)

# Betting backend abstraction (Phase 7 + transition to live betting —
# see docs/LIVE_BETTING_TRANSITION.md). Today only the virtual
# implementation is wired; live blueprint isn't registered.
from betting_backend import VirtualBettingBackend, make_bet_id

# In-UI docs renderer — converts the Markdown source under docs/ to HTML
# on demand for the 📚 Docs nav menu. Source files stay in Markdown.
import markdown as _md

# Sport blueprints — each sport's routes mount under /<sport>/.
# Sport-agnostic routes (/, /status, /stop/<task>, /server/<action>) stay
# on the app itself. Adding a new sport = create a blueprint, register it.
NBA_TASKS = {}  # kept as empty stub so /status responses don't break

# List of sports surfaced on the landing page. To activate NBA, flip 'active'
# to True, uncomment the import + register_blueprint call further down, and
# the landing page card automatically becomes a working link.
SPORTS = [
    {'slug': 'football',   'label': 'Football',   'icon': '⚽', 'icon_img': None,
     'active': True, 'bets_dir': 'output',
     'tagline': 'Daily 1X2 + Over/Under predictions, three-lane betting strategy.'},
    {'slug': 'euroleague', 'label': 'EuroLeague', 'icon': '🏆', 'icon_img': 'img/euroleague_logo_only.png',
     # Logo-only glyph + "EuroLeague" text label (renders like NBA: icon + name).
     # The full-wordmark svg squished badly next to a label, so the compact
     # contexts (navbar, cards, portfolio + cross-sport summary rows) use this
     # icon-only PNG; the betting-tab pane hero keeps the wordmark svg.
     'active': True, 'bets_dir': 'output_euroleague',
     'tagline': 'Daily moneyline predictions (Euroleague + EuroCup) on a combined XGBoost model with per-competition Platt calibration.'},
    {'slug': 'nba',        'label': 'NBA',        'icon': '🏀', 'icon_img': 'img/nba.svg',
     'active': True, 'bets_dir': 'output_basketball',
     'tagline': 'Daily moneyline + totals predictions on an enhanced-feature XGBoost model with Platt calibration.'},
]
# `icon_img` is an optional path relative to web_ui/static/ — when present,
# templates render <img> instead of the unicode emoji. Football keeps emoji
# because ⚽ is generic to the sport (no league mark applies).

app = Flask(__name__)
app.secret_key = 'super_secret_key_flashscore'

football_bp = Blueprint('football', __name__)


# Register Blueprints
from nba.routes import nba_bp, NBA_TASKS  # NBA reactivated 2026-05-28 (Phase 3)
app.register_blueprint(nba_bp, url_prefix='/nba')
from euroleague.routes import euroleague_bp, EUROLEAGUE_TASKS  # Euroleague Phase 3 (2026-05-29)
app.register_blueprint(euroleague_bp, url_prefix='/euroleague')

# Football blueprint registration happens at the bottom of this file,
# after all @football_bp.route handlers have been defined.

# Constants

# Constants
app.config['TEMPLATES_AUTO_RELOAD'] = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT) # Enable importing ml_project
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
HISTORY_DIR = os.path.join(OUTPUT_DIR, 'history')
DATA_SETS_DIR = os.path.join(PROJECT_ROOT, 'data_sets')
app.config['DATA_SETS_DIR'] = DATA_SETS_DIR
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


# Inject the appropriate BettingBackend on every football request.
# `g.backend` is the abstraction routes use for cashout (Phase 7) and,
# eventually, for balance + place + settle (Phase 9, when the
# `/football/live/` blueprint lands with the same hook returning
# `PamestoiximaBackend`). See docs/LIVE_BETTING_TRANSITION.md.
@football_bp.before_request
def _inject_backend():
    g.backend = VirtualBettingBackend(output_dir=OUTPUT_DIR)
    g.mode = 'virtual'


@app.template_filter('to_float')
def to_float_filter(value):
    try:
        return float(value)
    except:
        return 0.0

# Global dictionary to track running processes
# Format: {'task_name': {'process': Popen_obj, 'start_time': datetime}}
TASKS = {
    'predict': {'process': None, 'log': 'predict.log'},
    'verify': {'process': None, 'log': 'verify.log'},
    'live': {'process': None, 'log': 'live.log'},
    'update': {'process': None, 'log': 'update.log'},
    'leagues': {'process': None, 'log': 'leagues.log'},
    'retrain': {'process': None, 'log': 'retrain.log'},
    'backtest': {'process': None, 'log': 'backtest.log'},
}

@football_bp.route('/')
def index():
    # List prediction and verification files
    prediction_files = glob.glob(os.path.join(OUTPUT_DIR, 'predictions_*.csv'))
    verification_files = glob.glob(os.path.join(OUTPUT_DIR, 'verification_*.csv'))
    
    predictions = []
    for f in prediction_files:
        basename = os.path.basename(f)
        try:
            # Count matches (lines - header)
            with open(f, 'r') as fh:
                count = sum(1 for _ in fh) - 1
            count = max(0, count)

            # Extract date
            date_str = basename.replace('predictions_', '').replace('.csv', '')
            predictions.append({'filename': basename, 'date': date_str, 'type': 'Prediction', 'count': count})
        except:
             predictions.append({'filename': basename, 'date': 'Unknown', 'type': 'Prediction', 'count': 0})
             
    # matches_*.json (pre-match slate, carries odds) + results_*.json
    # (verification scrape output). Separate files since the verification
    # run stopped overwriting the slate — see bin/run_verification.sh.
    matches_files = (glob.glob(os.path.join(OUTPUT_DIR, 'matches_*.json'))
                     + glob.glob(os.path.join(OUTPUT_DIR, 'results_*.json')))
    scraped_data = []
    for f in matches_files:
        basename = os.path.basename(f)
        try:
            with open(f, 'r') as fh:
                match_data = json.load(fh)
                count = len(match_data)
            scraped_data.append({'filename': basename, 'count': count})
        except:
             scraped_data.append({'filename': basename, 'count': 0})
    scraped_data.sort(key=lambda x: x['filename'], reverse=True)
             
    verifications = []
    for f in verification_files:
        basename = os.path.basename(f)
        # Extract date from verification_2025-01-20.csv
        try:
             date_str = basename.replace('verification_', '').replace('.csv', '')
             verifications.append({'filename': basename, 'date': date_str, 'type': 'Verification'})
        except Exception as e:
             print(f"Error parsing verification filename {basename}: {e}")
             verifications.append({'filename': basename, 'date': 'Unknown', 'type': 'Verification'})
    verifications.sort(key=lambda x: x['date'], reverse=True)
        
    predictions.sort(key=lambda x: x['date'], reverse=True)
    
    # Load Cumulative Stats for Dashboard
    league_stats = []
    stats_file = os.path.join(PROJECT_ROOT, 'data_sets/league_analytics.json')
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r') as f:
                stats_data = json.load(f)
            
            for league, s in stats_data.items():
                total = s['total_matches']
                if total > 0:
                    acc_1x2 = round((s['correct_1x2'] / total) * 100, 2)
                    acc_ou = round((s['correct_ou'] / total) * 100, 2)
                else:
                    acc_1x2 = 0
                    acc_ou = 0
                    
                league_stats.append({
                    'League': league,
                    'Count': total,
                    'Acc_1X2': acc_1x2,
                    'Acc_OU': acc_ou
                })
            league_stats.sort(key=lambda x: x['Count'], reverse=True)
        except:
            pass

    # Join realized betting P/L per league as EXTRA columns. Accuracy above is
    # over ALL predicted+verified matches; the betting columns are over the
    # (usually smaller) bet subset — cashout-aware — and blank where we didn't
    # bet that league. Two different denominators, side by side on purpose.
    try:
        # Model lane only — it bets ~every prediction, so its per-league ROI
        # lines up with the (all-predictions) accuracy denominator above.
        _bet_by_league = {r['league']: r for r in compute_league_betting_summary(OUTPUT_DIR, lane='model')}
        for row in league_stats:
            b = _bet_by_league.get(row['League'])
            open_count = (b['bets'] - b['settled']) if b else 0
            if b and b.get('settled'):
                row['Bet_Settled'] = b['settled']
                # All four money columns are over the SETTLED subset so the row
                # reconciles: Return - Stake = P/L, and ROI = P/L / Stake. Using
                # the total committed stake here instead (which includes bets
                # still OPEN) made high-ROI leagues look like they returned less
                # than they staked. Unsettled money is reported as Open EUR.
                row['Bet_Stake'] = b['settled_stake']
                row['Bet_Returned'] = b['returned']
                row['Bet_PnL'] = b['pnl']
                row['Bet_ROI'] = b['roi']
            else:
                row['Bet_Settled'] = 0
                row['Bet_Stake'] = row['Bet_Returned'] = row['Bet_PnL'] = row['Bet_ROI'] = None
            row['Bet_Open_Stake'] = round(b['stake'] - b['settled_stake'], 2) if open_count else None
            row['Bet_Open_Count'] = open_count
    except Exception:
        pass

    # Load Live Live Data
    live_file = os.path.join(OUTPUT_DIR, "live_data.json")
    live_matches = []
    if os.path.exists(live_file):
        try:
            with open(live_file, 'r') as f:
                live_matches = json.load(f)
        except Exception as e:
            pass

    # Enrich each live match with any OPEN bets we have on it.
    # Lets the dashboard show stake / odds / fair-value cashout per bet
    # alongside the live stats. Read-only — actual cashout action is
    # gated behind FOOTBALL_NEXT_STEPS phase 7.
    _attach_open_bets(live_matches)

    return render_template('dashboard.html',
                          predictions=predictions,
                          verifications=verifications,
                          league_stats=league_stats,
                          live_matches=live_matches[:50],
                          scraped_data=scraped_data,
                          auto_cashout_armed=_auto_cashout_armed())


# Selection → adjusted-probs key for fair-value cashout estimation.
_SELECTION_TO_PROB_KEY = {
    '1': 'home', 1: 'home',
    'X': 'draw', 'x': 'draw',
    '2': 'away', 2: 'away',
    'Over 2.5':  'over',  'Over': 'over',
    'Under 2.5': 'under', 'Under': 'under',
}

_CASHOUT_HOUSE_HAIRCUT = 0.95

# Cashout decision thresholds — shared by the display badge in
# _attach_open_bets AND the auto-cashout executor (/auto_cashout) so the
# automatic action always matches what the UI shows.
#
# The decision is driven by `adj_prob` — the LiveAdjuster's synthesis of
# the live match statistics (score, minute, xG, shots, possession,
# dominance, red cards). So keying off adj_prob IS deciding from the live
# stats; we don't re-read raw stats here (that would double-count what
# already moved adj_prob).
#
#   lock_in   : the bet is IN PROFIT and either (a) near-certain to win
#               (adj_prob high — odds-independent, so it fires even on low
#               odds where the profit-ratio can never reach 1.5×), or
#               (b) the cashout is already a big multiple of stake (large
#               unrealized profit, typically a high-odds bet swinging our
#               way even before it's near-certain).
#   stop_loss : live win-probability has collapsed → cut the loss.
#   else      : hold.
# A minute floor suppresses BOTH before in-play stats are reliable: the
# LiveAdjuster only crosses over to trusting in-play data at ~min 30, so
# acting earlier means acting on noise.
_AUTO_CASHOUT_LOCK_IN_PROB = 0.85
_AUTO_CASHOUT_LOCK_IN_RATIO = 1.5
_AUTO_CASHOUT_STOP_LOSS_PROB = 0.20
_AUTO_CASHOUT_MIN_MINUTE = 30

# Bumped every time the server-side sweep actually cashes a bet. Exposed via
# /status so a live page can reload itself when an auto-cashout fires (the
# server can't push, and the daemon's scrape completes BEFORE the sweep, so the
# scrape-completion reload alone would miss the cashout).
_AUTO_CASHOUT_EPOCH = 0


def _parse_minute(live_match):
    """Best-effort current minute (int) from a live_data match dict.
    Accepts int, '67', "67'", '45+2', 'HT', 'FT'. Returns None when
    unknown (the decision then skips the minute floor rather than guess)."""
    raw = live_match.get('minute', live_match.get('time'))
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    s = str(raw).strip().upper()
    if 'HT' in s:
        return 45
    if 'FT' in s:
        return 90
    digits = ''
    for ch in s:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


def _cashout_decision(fair_cashout, stake, adj_prob, minute=None):
    """Return 'lock_in' | 'stop_loss' | 'hold' for an OPEN bet. Single
    source of truth for both the display badge and auto-cashout.
    `adj_prob` carries the live-stats signal (see thresholds note)."""
    # Too early to trust in-play stats → never auto-act.
    if minute is not None and minute < _AUTO_CASHOUT_MIN_MINUTE:
        return 'hold'
    if fair_cashout is not None and stake > 0:
        in_profit = fair_cashout >= stake
        near_certain = adj_prob is not None and adj_prob >= _AUTO_CASHOUT_LOCK_IN_PROB
        big_profit = fair_cashout / stake >= _AUTO_CASHOUT_LOCK_IN_RATIO
        if in_profit and (near_certain or big_profit):
            return 'lock_in'
        if adj_prob is not None and adj_prob < _AUTO_CASHOUT_STOP_LOSS_PROB:
            return 'stop_loss'
    return 'hold'


# Bookmaker cashout snapshot — scenario #3B in
# real_betting/test_case_scenarios.md produces this file. Schema (when
# the scraper is wired):
#   {
#     "ts": "<ISO UTC timestamp of the read>",
#     "bets": [
#       {"match_id": "<flashscore id>", "pamestoixima_uuid": "<uuid>",
#        "home": "...", "away": "...", "cashout_offer": <float|null>,
#        "paused": <bool>, ...},
#       ...
#     ]
#   }
# Missing file / unparseable JSON / stale `ts` → return empty, which
# silently falls back to synthetic fair_cashout in _attach_open_bets.
_BOOKMAKER_SNAPSHOT_PATH = os.path.join(
    OUTPUT_DIR, 'real_betting', 'open_bets_snapshot.json'
)


def _load_bookmaker_offers():
    """Load + index the scenario #3B snapshot. Returns a dict
        {'by_match_id': {<bookmaker_match_id>: entry, ...},
         'all':         [entry, ...],
         'age_s':       <float seconds since snapshot ts, or None>}
    or all-empty + age_s=None when the snapshot is missing/unparseable.

    Does NOT gate on staleness internally — the caller applies its own
    thresholds because link-existence and offer-value-freshness want
    DIFFERENT windows (a real bet stays real for the whole match; the
    €offer moves with the score). Two indexes because Flashscore and
    Pamestoixima use entirely different match-id schemes — Flashscore:
    8-char alphanumeric ('nFjvRRsQ'); Pamestoixima: 8-digit numeric
    ('11012505'). 'by_match_id' rarely fires; 'all' feeds the fuzzy
    team-name fallback in `_attach_open_bets`."""
    empty = {'by_match_id': {}, 'all': [], 'age_s': None}
    if not os.path.exists(_BOOKMAKER_SNAPSHOT_PATH):
        return empty
    try:
        with open(_BOOKMAKER_SNAPSHOT_PATH) as f:
            snap = json.load(f)
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(snap, dict):
        return empty

    age_s = None
    ts = snap.get('ts')
    if ts:
        try:
            # Snapshot ts is ISO UTC. Use timezone-aware comparison so
            # local-vs-UTC mismatches don't skew the age.
            snap_dt = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
            now_dt = datetime.datetime.now(datetime.timezone.utc)
            age_s = (now_dt - snap_dt).total_seconds()
        except (ValueError, TypeError):
            age_s = None

    by_match_id = {}
    all_entries = []
    for entry in snap.get('bets', []) or []:
        if not isinstance(entry, dict):
            continue
        all_entries.append(entry)
        mid = entry.get('match_id')
        if mid:
            by_match_id[str(mid)] = entry
    return {'by_match_id': by_match_id, 'all': all_entries, 'age_s': age_s}


# Link-existence window: how old the snapshot can be and still be
# trusted to tell us "a real bet exists on this match". Generous —
# a real bet doesn't disappear over a match's duration (it only
# changes via cashout/settlement, which a re-scrape would catch).
# 4 h comfortably covers a refresh-then-watch session.
_BOOKMAKER_LINK_MAX_AGE_S = 4 * 3600


def _match_offer_by_teams(home, away, offers_list, min_score=80):
    """Fuzzy-match (home, away) against a bookmaker-offers list. Returns
    the matched entry or None. Score is the *worse* of the two team-name
    matches — both home AND away must clear `min_score`. Lazy-imports
    rapidfuzz so the dashboard works even if it ever gets uninstalled."""
    if not home or not away or not offers_list:
        return None
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return None
    h_lo = home.lower().strip()
    a_lo = away.lower().strip()
    best_score = 0
    best = None
    for entry in offers_list:
        eh = (entry.get('home') or '').lower().strip()
        ea = (entry.get('away') or '').lower().strip()
        if not eh or not ea:
            continue
        s_home = fuzz.token_set_ratio(h_lo, eh)
        s_away = fuzz.token_set_ratio(a_lo, ea)
        worse = min(s_home, s_away)
        if worse >= min_score and worse > best_score:
            best_score = worse
            best = entry
    return best


_TERMINAL_BET_STATUSES = ('WON', 'LOST', 'VOID', 'CASHED_OUT')


def _attach_open_bets(live_matches):
    """Mutate each live match dict, adding `open_bets` (list).

    Each entry: {lane, type, selection, stake, odds, adj_prob,
    fair_cashout, status_badge}.

    Joined by match string ("Home vs Away"). Today's slip only;
    archived slips are skipped because OPEN bets can't be there.

    Also filters `live_matches` in place: any match whose bets are
    ALL in a terminal status (WON / LOST / VOID / CASHED_OUT) is
    dropped from the live panel — there's no actionable bet left on
    that fixture. Matches we never bet on are kept here as informational
    live rows, but the panel hides them by default behind its client-side
    "Bets only" toggle — they're only actionable once in-play betting is
    real (no pre-match bet needed to bet live).
    """
    today_slip = os.path.join(OUTPUT_DIR, f"bets_{datetime.date.today().isoformat()}.json")
    if not os.path.exists(today_slip):
        for m in live_matches:
            m['open_bets'] = []
        return

    try:
        with open(today_slip, 'r') as f:
            slip = json.load(f)
    except (OSError, json.JSONDecodeError):
        for m in live_matches:
            m['open_bets'] = []
        return

    # One-time backfill: stamp bet_id + mode='virtual' onto any pre-Phase-7
    # OPEN bets in this slip. Without bet_id the Cash Out button can't
    # target them (the cashout endpoint looks bets up by bet_id). Persist
    # the slip if we stamped anything.
    slip_date = slip.get('date') or datetime.date.today().isoformat()
    mutated = False
    for bet in slip.get('bets', []):
        if bet.get('status') != 'OPEN':
            continue
        if not bet.get('bet_id'):
            match_str = bet.get('match', '')
            home = bet.get('home') or (match_str.split(' vs ')[0].strip()
                                       if ' vs ' in match_str else '')
            away = bet.get('away') or (match_str.split(' vs ')[1].strip()
                                       if ' vs ' in match_str else '')
            if home and away:
                bet['bet_id'] = make_bet_id(
                    slip_date, home, away,
                    bet.get('type', '1X2'),
                    bet.get('selection', ''),
                )
                mutated = True
        if not bet.get('mode'):
            bet['mode'] = 'virtual'
            mutated = True
    if mutated:
        try:
            with open(today_slip, 'w') as f:
                json.dump(slip, f, indent=4)
        except OSError as e:
            print(f"Warning: could not persist bet_id backfill: {e}")

    # Build lookup: match_str → list of OPEN bets.
    # IMPORTANT: only OPEN bets show on live rows. Any terminal status
    # (WON, LOST, VOID, CASHED_OUT) is excluded — including a per-bet
    # CASHED_OUT inside a match that has other still-open bets on it
    # (different lane, different market). The earlier "skip if result
    # in WON/LOST/VOID" form leaked CASHED_OUT through because
    # result='CASHED_OUT' wasn't in that set. Be explicit instead.
    #
    # Also count terminal bets per match so we can suppress live rows
    # whose every bet is already settled / cashed out below.
    by_match = {}
    terminal_count = {}
    bets_total = {}
    for bet in slip.get('bets', []):
        match_str = bet.get('match', '').strip()
        if not match_str:
            continue
        bets_total[match_str] = bets_total.get(match_str, 0) + 1
        status = bet.get('status', '')
        if status in _TERMINAL_BET_STATUSES:
            terminal_count[match_str] = terminal_count.get(match_str, 0) + 1
            continue
        if status not in ('OPEN', ''):
            continue
        by_match.setdefault(match_str, []).append(bet)

    # Cashout-source flag (scenario #3A). When 'bookmaker', look up the
    # match in the scenario #3B snapshot and use the real offer; when
    # 'synthetic' (default) or the snapshot misses this match, fall back
    # to the internal fair_cashout formula.
    sport_cfg = get_sport_config('football')
    cashout_source_pref = sport_cfg.get('cashout_source', 'synthetic')
    value_max_age_s = float(sport_cfg.get('cashout_snapshot_max_age_s', 600))
    # Always load the bookmaker snapshot (no internal staleness gate).
    # We apply TWO windows below:
    #   - link existence: trusted up to _BOOKMAKER_LINK_MAX_AGE_S (4h) —
    #     a real bet doesn't vanish mid-match, so the `🔗 linked` badge
    #     should persist through a watch session on a single refresh.
    #   - offer value: trusted only up to value_max_age_s (default 600s,
    #     per-sport configurable) because the €offer moves with the score.
    bookmaker_offers = _load_bookmaker_offers()
    snap_age_s = bookmaker_offers.get('age_s')
    link_fresh = (snap_age_s is not None
                  and snap_age_s <= _BOOKMAKER_LINK_MAX_AGE_S)
    # Value freshness == link freshness. The snapshot value on disk only
    # changes when read-open-bets re-runs (manual button press) — nothing
    # drifts it in the background — so a separate short value window just
    # downgraded a still-best-known real offer to synthetic for no gain.
    # Show the real value whenever linked; surface the age so the user
    # can judge staleness (a 3h-old offer mid-match has moved, but it's
    # still the most recent REAL data we have, labelled with its age).
    value_fresh = link_fresh
    snap_age_min = int(snap_age_s // 60) if snap_age_s is not None else None

    # Tracks whether we stamped a new persisted link onto any bet this
    # pass (so we re-save the slip once at the end). Persisting the link
    # makes it survive snapshot staleness — it stays until the bet
    # itself resolves (terminal status → filtered off the live panel),
    # which is the "link until the bet is resolved" behaviour. Also
    # records pamestoixima_uuid for future settlement reconciliation.
    link_mutated = False

    for m in live_matches:
        if m.get('message'):
            m['open_bets'] = []
            continue
        match_str = m.get('match', '').strip()
        related = by_match.get(match_str, [])

        # Bookmaker offer join — TWO paths, tried in order:
        #
        # 1. Direct match_id lookup. Cheap O(1), but Flashscore and
        #    Pamestoixima use entirely different ID schemes (verified
        #    2026-05-25: Flashscore = 8-char alphanumeric "nFjvRRsQ";
        #    Pamestoixima = 8-digit numeric "11012505"). So this almost
        #    never fires in practice.
        #
        # 2. Fuzzy team-name match against the full snapshot list. Uses
        #    rapidfuzz token_set_ratio with a min-score-80 floor on the
        #    worse of (home, away). This is the path that actually
        #    surfaces real offers on the dashboard today.
        #
        # When the snapshot is missing / too old for link / flag=='synthetic',
        # the relevant path yields None and the synthetic formula wins.
        #
        # Link detection uses the generous link-freshness window; the
        # offer-value path additionally requires value-freshness below.
        m_id = str(m.get('match_id') or '').strip()
        bk_entry = None
        if link_fresh:
            if m_id:
                bk_entry = bookmaker_offers.get('by_match_id', {}).get(m_id)
            if bk_entry is None:
                # Fall back to team-name fuzzy match. The match string is
                # "Home vs Away" — split and pass to the helper.
                ms = match_str
                if ' vs ' in ms:
                    fb_home, fb_away = ms.split(' vs ', 1)
                    bk_entry = _match_offer_by_teams(
                        fb_home, fb_away,
                        bookmaker_offers.get('all', []),
                    )
        bk_offer = None
        # Offer value only when the snapshot is value-fresh AND not paused.
        if bk_entry and value_fresh and not bk_entry.get('paused'):
            try:
                bk_offer = float(bk_entry.get('cashout_offer'))
            except (TypeError, ValueError):
                bk_offer = None

        enriched = []
        for bet in related:
            stake = float(bet.get('stake_units', 0))
            odds = float(bet.get('odds', 0))
            bet_type = bet.get('type', '1X2')
            selection = bet.get('selection')
            prob_key = _SELECTION_TO_PROB_KEY.get(selection)

            # 1X2 reads adj_probs; O/U reads adj_ou_probs (persisted by
            # run_live_analysis.py via LiveAdjuster.adjust_ou_probabilities).
            adj_prob = None
            synthetic_cashout = None
            if bet_type == '1X2' and prob_key in ('home', 'draw', 'away'):
                adj_prob = float(m.get('adj_probs', {}).get(prob_key, 0))
                synthetic_cashout = round(stake * odds * adj_prob * _CASHOUT_HOUSE_HAIRCUT, 2)
            elif bet_type == 'O/U' and prob_key in ('over', 'under'):
                adj_prob = float(m.get('adj_ou_probs', {}).get(prob_key, 0))
                synthetic_cashout = round(stake * odds * adj_prob * _CASHOUT_HOUSE_HAIRCUT, 2)

            # Pick the displayed cashout value + tag the source.
            #   'bookmaker' wins iff (a) the flag is on, (b) a snapshot
            #     entry exists for this match, (c) the offer is a usable
            #     float, and (d) the entry isn't paused.
            #   'synthetic' otherwise (default; missing snapshot; paused).
            use_bookmaker_value = (
                cashout_source_pref == 'bookmaker' and bk_offer is not None
            )
            if use_bookmaker_value:
                fair_cashout = round(bk_offer, 2)
                cashout_source = 'bookmaker'
            else:
                fair_cashout = synthetic_cashout
                cashout_source = 'synthetic'

            # Status badge — same decision the auto-cashout executor uses
            # (so a 🟢/🔴 badge is exactly what /auto_cashout would act on).
            badge = _cashout_decision(fair_cashout, stake, adj_prob, _parse_minute(m))

            # Link existence — True if either a fresh snapshot match was
            # found this pass (bk_entry) OR the bet was already linked on
            # a previous pass (persisted flag). Persisting means the link
            # survives snapshot staleness and stays until the bet itself
            # resolves (terminal status → filtered off this panel) —
            # i.e. "link until the bet is resolved".
            persisted_link = bool(bet.get('linked_to_bookmaker'))
            linked = (bk_entry is not None) or persisted_link
            uuid = ((bk_entry or {}).get('pamestoixima_uuid')
                    or bet.get('pamestoixima_uuid'))

            # First-time establishment: stamp the link onto the slip bet
            # so it persists across future loads (and snapshot expiry).
            if bk_entry is not None and not persisted_link:
                bet['linked_to_bookmaker'] = True
                if uuid:
                    bet['pamestoixima_uuid'] = uuid
                link_mutated = True

            enriched.append({
                'lane': bet.get('lane', 'value'),
                'type': bet_type,
                'selection': str(selection),
                'stake': round(stake, 2),
                'odds': odds,
                'adj_prob': round(adj_prob, 3) if adj_prob is not None else None,
                'fair_cashout': fair_cashout,
                'cashout_source': cashout_source,
                # Persisted-or-fresh link (see above). Drives the
                # `🔗 linked` chip in _open_bets_fragment.html and the
                # live_analysis page's linked-only filter.
                'linked_to_bookmaker': linked,
                'pamestoixima_uuid': uuid,
                # Snapshot age in minutes (None when no snapshot) — shown
                # in the cashout-value tooltip so a stale real offer is
                # honestly labelled rather than silently downgraded.
                'cashout_age_min': snap_age_min if cashout_source == 'bookmaker' else None,
                'badge': badge,
                # Pass bet_id through so the dashboard's Cash Out button
                # can target the right bet via /football/cashout/<bet_id>.
                # Pre-Phase-7 slips don't carry bet_id; the template
                # hides the button in that case.
                'bet_id': bet.get('bet_id'),
            })
        m['open_bets'] = enriched

    # Persist any newly-established links back to the slip (one write).
    if link_mutated:
        try:
            with open(today_slip, 'w') as f:
                json.dump(slip, f, indent=4)
        except OSError as e:
            print(f"Warning: could not persist bookmaker links: {e}")

    # In-place filter: drop matches whose every bet is terminal. A
    # match with no bets at all stays on the panel (informational live
    # row, hidden by default by the template's "Bets only" toggle); a
    # match where SOME bets are terminal but others still OPEN also
    # stays (the open ones still need eyes on them).
    def _all_terminal(m):
        match_str = (m.get('match') or '').strip()
        total = bets_total.get(match_str, 0)
        return total > 0 and terminal_count.get(match_str, 0) >= total

    live_matches[:] = [m for m in live_matches if not _all_terminal(m)]


@app.route('/status')
def get_status():
    status = {}
    
    # Merge NBA Tasks into checking logic
    # NBA_TASKS format: {'retrain': Popen, 'verify': Popen}
    # We prefix them to distinction: 'nba_retrain', 'nba_verify'
    
    # Check Standard Tasks
    for task_name, task_info in TASKS.items():
        if task_info and task_info.get('process'):
            poll = task_info['process'].poll()
            if poll is None:
                status[task_name] = {'state': 'running'}
                if task_info.get('target_date'):
                    status[task_name]['target_date'] = task_info['target_date']
            elif poll == 0:
                status[task_name] = {'state': 'completed'}
            else:
                # Capture last lines of log directly
                log_file = os.path.join(LOG_DIR, f"{task_name}.log")
                error_msg = 'Unknown error'
                if os.path.exists(log_file):
                     try:
                         # Get last 3 lines
                         lines = subprocess.check_output(['tail', '-n', '3', log_file]).decode('utf-8')
                         error_msg = lines.strip()
                     except:
                         pass
                status[task_name] = {'state': 'error', 'msg': error_msg}
        elif task_info and task_info.get('state'): # Thread tasks wrapper
             status[task_name] = {'state': task_info['state'], 'msg': task_info.get('msg', '')}
        else:
            status[task_name] = {'state': 'idle'}
            
    # Check NBA + Euroleague Tasks (same Popen-dict shape, slug-prefixed keys).
    for prefix, task_dict in (("nba", NBA_TASKS), ("euroleague", EUROLEAGUE_TASKS)):
        for task_name, proc in task_dict.items():
            key = f"{prefix}_{task_name}"
            if proc:
                poll = proc.poll()
                if poll is None:
                    status[key] = {'state': 'running'}
                elif poll == 0:
                    status[key] = {'state': 'completed'}
                else:
                    status[key] = {'state': 'error'}
            else:
                status[key] = {'state': 'idle'}

    # Auto-cashout epoch — bumps when the server-side sweep cashes a bet, so a
    # live page can reload itself (the cashout is server-side, no push).
    status['auto_cashout_epoch'] = _AUTO_CASHOUT_EPOCH
    return status

@app.route('/stop/<task_name>', methods=['POST'])
def stop_task(task_name):
    if task_name in TASKS and TASKS[task_name] and TASKS[task_name]['process']:
        try:
            # Send SIGTERM
            TASKS[task_name]['process'].terminate()
            time.sleep(1)
            # If still alive, SIGKILL
            if TASKS[task_name]['process'].poll() is None:
                TASKS[task_name]['process'].kill()
            
            TASKS[task_name]['process'] = None # Set process to None to indicate it's stopped
            flash(f'Task {task_name} stopped.', 'warning')
        except Exception as e:
            flash(f'Error stopping task: {str(e)}', 'danger')
    else:
        flash(f'No running {task_name} task found.', 'secondary')
    return redirect(url_for('football.index'))

# NOTE: `process_bet_verification` used to live here. It was deleted on
# 2026-05-21 after being identified as long-time dead code (called with
# the wrong filename, silently no-op'd). Bet settlement is the canonical
# responsibility of `ml_project/resolve_daily_bets.py:resolve_all_bets`,
# which is invoked by `bin/run_verification.sh` at the end of the
# verification flow. `VirtualBettingBackend.settle_bets` also delegates
# to that function for the Phase 7+ backend abstraction.

@football_bp.route('/predict', methods=['POST'])
def run_prediction():
    if TASKS['predict'] and TASKS['predict']['process'] and TASKS['predict']['process'].poll() is None:
         flash('Prediction is already running!', 'warning')
         return redirect(url_for('football.index'))

    try:
        # Execute run_predictions.sh from project root
        script_path = os.path.join(PROJECT_ROOT, 'bin', 'run_predictions.sh')
        log_file = open(os.path.join(LOG_DIR, 'predict.log'), 'w')
        
        cmd = ['/bin/bash', script_path]
        
        # Add Date Arg
        date_arg = request.form.get('date')
        if date_arg:
            cmd.append(date_arg)
            target_date = date_arg
        else:
            # Default matches run_predictions.sh: tomorrow.
            target_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

        if request.form.get('force'):
            cmd.append('--force')

        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT)

        TASKS['predict'] = {'process': proc, 'start_time': datetime.datetime.now(), 'target_date': target_date}
        
        flash('Prediction pipeline started! Check <a href="/logs/predict.log">logs</a> for status.', 'success')
    except Exception as e:
        flash(f'Error starting prediction: {e}', 'danger')
        
    return redirect(url_for('football.index'))

@football_bp.route('/verify', methods=['POST'])
def run_verification():
    if TASKS['verify'] and TASKS['verify']['process'] and TASKS['verify']['process'].poll() is None:
         flash('Verification is already running!', 'warning')
         return redirect(url_for('football.index'))

    # 1. Retrieve Date
    date_arg = request.form.get('date')
    if date_arg:
        target_date = date_arg
    else:
        # Default to yesterday
        target_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        
    pred_file = os.path.join(OUTPUT_DIR, f"predictions_{target_date}.csv")
    bets_file = os.path.join(OUTPUT_DIR, f"bets_{target_date}.json")

    # Allow verification when EITHER predictions or open bets exist for
    # the target date. `bin/run_verification.sh` falls back to bet-
    # derived match IDs when predictions are missing — same UI gate
    # should reflect that.
    if not os.path.exists(pred_file) and not os.path.exists(bets_file):
        flash(f'Error: neither predictions ({os.path.basename(pred_file)}) nor '
              f'bets ({os.path.basename(bets_file)}) exists for {target_date}. '
              f'Nothing to verify.', 'danger')
        return redirect(url_for('football.index'))
    if not os.path.exists(pred_file):
        flash(f'No predictions file for {target_date}; verifying against open '
              f'bets only (no prediction-accuracy report will be produced).',
              'info')

    try:
        script_path = os.path.join(PROJECT_ROOT, 'bin', 'run_verification.sh')
        log_file = open(os.path.join(LOG_DIR, 'verify.log'), 'w')
        
        cmd = ['/bin/bash', script_path]
        if date_arg:
            cmd.append(date_arg)
            
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT)

        TASKS['verify'] = {'process': proc, 'start_time': datetime.datetime.now(), 'target_date': target_date}

        flash('Verification pipeline started! Check <a href="/logs/verify.log">logs</a> for status.', 'success')
    except Exception as e:
        flash(f'Error starting verification: {e}', 'danger')
        
    return redirect(url_for('football.index'))

@football_bp.route('/logs/<filename>')
def view_log(filename):
    filepath = os.path.join(LOG_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
        return f"<pre>{content}</pre>"
    else:
        return "Log file not found."


# --- Docs renderer ---------------------------------------------------------
# Whitelist of doc names → source paths. Restricting to a known set avoids
# any path-traversal concerns and means we control which docs the UI
# exposes (FOOTBALL_NEXT_STEPS, internal planning docs, etc. don't show up here).
_DOCS = {
    'betting_strategy': {
        'title': 'Betting Strategy',
        'icon': '💸',
        'path': os.path.join(PROJECT_ROOT, 'docs', 'betting_strategy.md'),
    },
    'ui_manual': {
        'title': 'UI Manual',
        'icon': '🖥️',
        'path': os.path.join(PROJECT_ROOT, 'docs', 'ui_manual.md'),
    },
}


# --- Strategy tunables editor (Phase 1) ----------------------------------
# Lane-grouped form for the editable knobs in DEFAULT_SPORT_CONFIG.
# Bankroll state (current/initial) is NOT editable here — that's separate.
#
# Each entry: {key, label, lane, unit, step, min, max, kind, description}.
# - `kind`: 'pct' (UI shows %, stored as fraction), 'num' (raw decimal),
#           'eur' (EUR), 'odds' (decimal odds ≥ 1), 'bool' (checkbox)
# - `min`/`max` are in UI units; for `pct` they're 0-100, internally 0-1.
_TUNABLE_SPEC = [
    # Value lane
    {'key': 'min_confidence', 'label': 'Min confidence', 'lane': 'value',
     'kind': 'num', 'min': 0, 'max': 1, 'step': 0.01,
     'description': 'Entry filter — model confidence must be ≥ this to consider a Value-lane bet.'},
    {'key': 'stake_multiplier', 'label': 'Stake multiplier', 'lane': 'value',
     'kind': 'num', 'min': 0, 'max': 2, 'step': 0.05,
     'description': 'Multiplier in the Value-lane stake formula: stake = bankroll × EV × Conf × multiplier.'},
    {'key': 'min_stake_eur', 'label': 'Min stake', 'lane': 'value',
     'kind': 'eur', 'min': 0, 'max': 100, 'step': 0.5,
     'description': 'Picks whose computed stake falls below this floor are dropped from the slip.'},
    {'key': 'max_stake_pct', 'label': 'Max stake per bet', 'lane': 'value',
     'kind': 'pct', 'min': 0, 'max': 100, 'step': 0.1,
     'description': 'Hard cap per Value-lane bet, as a percentage of the lane bankroll.'},
    {'key': 'ev_cap_value', 'label': 'EV cap', 'lane': 'value',
     'kind': 'num', 'min': 0, 'max': 1, 'step': 0.01,
     'description': 'Clamp EV input to the stake formula. Stops a single high-EV pick from dominating the slip.'},
    {'key': 'value_max_daily_exposure_pct', 'label': 'Daily exposure cap', 'lane': 'value',
     'kind': 'pct', 'min': 0, 'max': 100, 'step': 0.5,
     'description': 'Max fraction of the Value-lane bankroll that can be staked in one day.'},

    # Conviction lane
    {'key': 'conviction_min_confidence', 'label': 'Min confidence', 'lane': 'conviction',
     'kind': 'num', 'min': 0, 'max': 1, 'step': 0.01,
     'description': 'Entry filter — confidence floor for Conviction lane (typically higher than Value).'},
    {'key': 'conviction_min_odds', 'label': 'Min odds', 'lane': 'conviction',
     'kind': 'odds', 'min': 1, 'max': 50, 'step': 0.05,
     'description': 'Conviction picks must have odds ≥ this. Filters out heavy favourites where stake is silly.'},
    {'key': 'conviction_stake_pct', 'label': 'Stake %', 'lane': 'conviction',
     'kind': 'pct', 'min': 0, 'max': 100, 'step': 0.1,
     'description': 'Flat stake fraction of the Conviction-lane bankroll, applied to every qualifying pick.'},
    {'key': 'conviction_max_daily_exposure_pct', 'label': 'Daily exposure cap', 'lane': 'conviction',
     'kind': 'pct', 'min': 0, 'max': 100, 'step': 0.5,
     'description': 'Max fraction of the Conviction-lane bankroll that can be staked in one day.'},

    # Model lane
    {'key': 'model_base_pct', 'label': 'Base stake %', 'lane': 'model',
     'kind': 'pct', 'min': 0, 'max': 100, 'step': 0.1,
     'description': 'Base stake fraction. Model lane: stake = bankroll × base_pct × Conf × ev_factor.'},
    {'key': 'model_max_stake_pct', 'label': 'Max stake per bet', 'lane': 'model',
     'kind': 'pct', 'min': 0, 'max': 100, 'step': 0.1,
     'description': 'Hard cap per Model-lane bet, as a percentage of the lane bankroll.'},
    {'key': 'model_min_stake_eur', 'label': 'Min stake', 'lane': 'model',
     'kind': 'eur', 'min': 0, 'max': 100, 'step': 0.5,
     'description': 'Model lane has its own (typically lower) floor so broad coverage isn\'t killed by the Value €2 floor.'},
    {'key': 'model_ev_factor_min', 'label': 'EV factor min', 'lane': 'model',
     'kind': 'num', 'min': 0, 'max': 5, 'step': 0.1,
     'description': 'Lower clamp for the Model-lane `ev_factor = clamp(Conf × odds, min, max)`.'},
    {'key': 'model_ev_factor_max', 'label': 'EV factor max', 'lane': 'model',
     'kind': 'num', 'min': 0, 'max': 5, 'step': 0.1,
     'description': 'Upper clamp for the Model-lane ev_factor.'},
    {'key': 'model_max_daily_exposure_pct', 'label': 'Daily exposure cap', 'lane': 'model',
     'kind': 'pct', 'min': 0, 'max': 100, 'step': 0.5,
     'description': 'Max fraction of the Model-lane bankroll that can be staked in one day.'},

    # Global (cross-lane)
    {'key': 'use_league_calibration', 'label': 'Apply per-league calibration', 'lane': 'global',
     'kind': 'bool',
     'description': 'When ON, per-league Platt scaling adjusts raw 1X2 / O/U probabilities before the heuristic adjuster.'},
]


def _ui_value(kind, fraction_value):
    """Convert stored value → UI value. Only `pct` needs conversion
    (stored as fraction 0-1, displayed as 0-100)."""
    if kind == 'pct':
        return round(float(fraction_value) * 100, 2)
    if kind == 'bool':
        return bool(fraction_value)
    return fraction_value


def _stored_value(kind, ui_value):
    """Convert UI value → stored value (inverse of _ui_value)."""
    if kind == 'pct':
        return float(ui_value) / 100.0
    if kind == 'bool':
        # Checkbox value: present in form when checked, absent when unchecked.
        return bool(ui_value)
    if kind in ('num', 'eur', 'odds'):
        return float(ui_value)
    return ui_value


@football_bp.route('/settings', methods=['GET', 'POST'])
def settings():
    """Strategy tunables editor (Phase 1 — read/write, bankroll state
    untouched). On POST: parse, validate, save valid fields; on GET:
    render the form populated with current values + defaults shown."""
    sport = 'football'

    if request.method == 'POST':
        updates = {}
        errors = []
        for spec in _TUNABLE_SPEC:
            key = spec['key']
            kind = spec['kind']
            # Only update fields whose form was actually rendered + submitted.
            # A hidden `_seen_<key>=1` is rendered next to every input —
            # without this guard, an unchecked checkbox would silently flip
            # to False on ANY POST (since absent checkbox == no value sent),
            # which is how an earlier round-trip flipped
            # use_league_calibration to False unintentionally.
            if request.form.get(f'_seen_{key}') != '1':
                continue
            if kind == 'bool':
                # Checkbox present means the field was rendered. Value 'on'
                # if checked, missing if not — translate to True/False.
                updates[key] = (request.form.get(key) == 'on')
                continue
            raw = request.form.get(key, '').strip()
            if raw == '':
                continue  # left blank → no update, keep existing
            try:
                ui_val = float(raw)
            except ValueError:
                errors.append(f"{spec['label']} ({spec['lane']}): '{raw}' is not a number.")
                continue
            # Range check (UI units).
            if 'min' in spec and ui_val < spec['min']:
                errors.append(f"{spec['label']} ({spec['lane']}): {ui_val} < {spec['min']} (allowed min).")
                continue
            if 'max' in spec and ui_val > spec['max']:
                errors.append(f"{spec['label']} ({spec['lane']}): {ui_val} > {spec['max']} (allowed max).")
                continue
            updates[key] = _stored_value(kind, ui_val)

        # Cross-field validation: ev_factor_min < ev_factor_max.
        new_min = updates.get('model_ev_factor_min',
                              get_sport_config(sport).get('model_ev_factor_min'))
        new_max = updates.get('model_ev_factor_max',
                              get_sport_config(sport).get('model_ev_factor_max'))
        if new_min is not None and new_max is not None and new_min >= new_max:
            errors.append('Model EV factor min must be strictly less than max.')

        if errors:
            for e in errors:
                flash(e, 'danger')
            return redirect(url_for('football.settings'))

        set_tunables(sport, updates)
        flash(f'Saved {len(updates)} tunable(s).', 'success')
        return redirect(url_for('football.settings'))

    # GET — render form.
    cfg = get_sport_config(sport)
    fields = []
    for spec in _TUNABLE_SPEC:
        key = spec['key']
        kind = spec['kind']
        current_stored = cfg.get(key, DEFAULT_SPORT_CONFIG[key])
        default_stored = DEFAULT_SPORT_CONFIG[key]
        fields.append({
            **spec,
            'current_ui': _ui_value(kind, current_stored),
            'default_ui': _ui_value(kind, default_stored),
            'is_default': (current_stored == default_stored),
        })

    # Group fields by lane for the template.
    lanes = {'value': [], 'conviction': [], 'model': [], 'global': []}
    for f in fields:
        lanes[f['lane']].append(f)

    return render_template('settings.html', lanes=lanes)


@football_bp.route('/docs/')
@football_bp.route('/docs/<name>')
def docs(name=None):
    """Render one of the user-facing Markdown docs from `docs/` as HTML.

    Source files stay in Markdown; conversion happens on demand. Only docs
    in the `_DOCS` whitelist are reachable — random doc files in the repo
    aren't auto-exposed.
    """
    if name is None:
        # Index → redirect to the first doc.
        first_name = next(iter(_DOCS))
        return redirect(url_for('football.docs', name=first_name))

    entry = _DOCS.get(name)
    if entry is None:
        flash(f'Unknown doc: {name!r}.', 'warning')
        return redirect(url_for('football.index'))

    if not os.path.exists(entry['path']):
        flash(f'Doc source missing on disk: {entry["path"]}.', 'danger')
        return redirect(url_for('football.index'))

    with open(entry['path'], 'r', encoding='utf-8') as f:
        src = f.read()

    # `tables` enables GitHub-style pipe tables (we use these heavily).
    # `fenced_code` for ``` blocks. `toc` injects a table of contents.
    html = _md.markdown(
        src,
        extensions=['extra', 'tables', 'fenced_code', 'toc', 'sane_lists'],
        output_format='html5',
    )

    # Side-nav: list of (name, title, icon, is_current).
    side_nav = [
        {'name': k, 'title': v['title'], 'icon': v['icon'], 'current': (k == name)}
        for k, v in _DOCS.items()
    ]

    return render_template('docs.html',
                           html_content=html,
                           current=entry,
                           current_name=name,
                           side_nav=side_nav)

def _archive_file(filepath, filename):
    """
    Soft-delete: move filepath to output/history/, returning the archived path.
    On name collision in history, append a timestamp suffix so nothing gets clobbered.
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    target = os.path.join(HISTORY_DIR, filename)
    if os.path.exists(target):
        base, ext = os.path.splitext(filename)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        target = os.path.join(HISTORY_DIR, f"{base}.{ts}{ext}")
    shutil.move(filepath, target)
    return target


# --- Phase 7: manual cashout endpoint -------------------------------------
# Lets the user commit a cashout on an OPEN bet via the dashboard's
# Cash Out button. Delegates to `g.backend.execute_cashout()` so the
# call site is mode-agnostic — Phase 9 will register an identical
# endpoint under `/football/live/cashout/<bet_id>` against the live
# backend. See docs/LIVE_BETTING_TRANSITION.md for the full design.

def _find_bet_and_live_match(bet_id: str):
    """Locate a bet in any output/bets_*.json by its bet_id, plus the
    matching live_data entry. Returns (bet_dict, live_match_dict) or
    (None, None) if either can't be found."""
    # The bet_id starts with `<YYYY-MM-DD>:` — fast-path the right slip.
    date_prefix = bet_id.split(':', 1)[0] if ':' in bet_id else ''
    slip_candidates = []
    if date_prefix and len(date_prefix) == 10:
        slip_candidates.append(os.path.join(OUTPUT_DIR, f'bets_{date_prefix}.json'))
    # Fallback: any slip in output/ (NOT history — cashed-out bets can't
    # exist there because OPEN-only is the soft-delete invariant).
    slip_candidates.extend(sorted(glob.glob(os.path.join(OUTPUT_DIR, 'bets_*.json'))))

    target_bet = None
    for slip_path in slip_candidates:
        if not os.path.exists(slip_path):
            continue
        try:
            with open(slip_path) as f:
                slip = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for b in slip.get('bets', []):
            if b.get('bet_id') == bet_id:
                target_bet = b
                # Ensure the bet record carries a `date` field for the
                # backend (older records may not have one explicitly).
                if 'date' not in target_bet:
                    target_bet['date'] = slip.get('date') or date_prefix
                break
        if target_bet is not None:
            break
    if target_bet is None:
        return (None, None)

    # Find the matching live match (by 'match' string equality).
    # live_data.json is a top-level list (run_live_analysis writes
    # `json.dump(final_results, ...)` where final_results is a list).
    # Be defensive in case the shape ever changes — accept either a
    # list or a dict with a 'matches' key.
    live_path = os.path.join(OUTPUT_DIR, 'live_data.json')
    if not os.path.exists(live_path):
        return (target_bet, None)
    try:
        with open(live_path) as f:
            live_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return (target_bet, None)

    if isinstance(live_data, list):
        matches = live_data
    elif isinstance(live_data, dict):
        matches = live_data.get('matches') or []
    else:
        matches = []

    match_str = (target_bet.get('match') or '').strip()
    live_match = None
    for m in matches:
        if not isinstance(m, dict):
            continue
        if (m.get('match') or '').strip() == match_str:
            live_match = m
            break
    return (target_bet, live_match)


@football_bp.route('/cashout/<bet_id>', methods=['POST'])
def cashout(bet_id):
    """Cash out the bet identified by `bet_id`. Sets the Phase 3 schema
    fields (status='CASHED_OUT', cashout_amount, cashout_profit,
    cashout_timestamp), credits the lane bankroll with the cashout
    amount, persists the slip."""
    # Safety: bet_id is opaque text, but URL-pathing might be exploited.
    if '/' in bet_id or '..' in bet_id:
        flash('Invalid bet_id.', 'danger')
        return redirect(request.referrer or url_for('football.index'))

    bet, live_match = _find_bet_and_live_match(bet_id)
    if bet is None:
        flash(f'Bet not found: {bet_id}.', 'warning')
        return redirect(request.referrer or url_for('football.index'))
    # NOTE: do NOT gate on the representative bet's own status. The
    # cashout cascades across all lanes' OPEN bets with this bet_id —
    # the representative might already be CASHED_OUT but sibling lanes
    # could still be OPEN. Let execute_cashout decide.
    if live_match is None:
        flash(f'No live data for this match — cashout requires the match '
              f'to be in-play with current adjusted probabilities.', 'warning')
        return redirect(request.referrer or url_for('football.index'))

    # Snapshot the slip BEFORE the cashout so we can report exactly
    # what was cashed out (which lanes, totals).
    pre_open = []
    try:
        slip_date = bet_id.split(':', 1)[0] if ':' in bet_id else ''
        slip_path = os.path.join(OUTPUT_DIR, f'bets_{slip_date}.json')
        if os.path.exists(slip_path):
            with open(slip_path) as f:
                slip_snapshot = json.load(f)
            for b in slip_snapshot.get('bets', []):
                if b.get('bet_id') == bet_id and b.get('status') == 'OPEN':
                    pre_open.append({
                        'lane': b.get('lane', 'value'),
                        'stake': float(b.get('stake_units', 0) or 0),
                    })
    except Exception:
        pass

    ok = g.backend.execute_cashout(bet, live_match)
    if not ok:
        flash('No OPEN bets found to cash out (sibling lanes may have '
              'been cashed out already).', 'info')
        return redirect(request.referrer or url_for('football.index'))

    # Re-read slip to compute the actual cashed totals from the
    # post-state — execute_cashout writes the per-bet cashout_amount
    # and we want to report the sum.
    total_amount = 0.0
    total_profit = 0.0
    lanes_credited = set()
    try:
        with open(slip_path) as f:
            slip_after = json.load(f)
        for b in slip_after.get('bets', []):
            if b.get('bet_id') == bet_id and b.get('status') == 'CASHED_OUT':
                total_amount += float(b.get('cashout_amount', 0) or 0)
                total_profit += float(b.get('cashout_profit', 0) or 0)
                lanes_credited.add(b.get('lane', 'value'))
    except Exception:
        pass

    sign = '+' if total_profit >= 0 else ''
    lanes_label = ', '.join(sorted(lanes_credited)) if lanes_credited else 'unknown'
    flash(f'Cashed out €{total_amount:.2f} across {len(lanes_credited)} lane(s) '
          f'({lanes_label}); net P/L {sign}{total_profit:.2f}.',
          'success')
    return redirect(request.referrer or url_for('football.index'))


def _run_auto_cashout_sweep(backend, shadow=None):
    """Evaluate every OPEN bet currently on a live match and cash out (via
    the lane-cascading `execute_cashout`) those whose decision is non-`hold`
    (`_cashout_decision`, same thresholds as the display badge). Pricing is
    the synthetic fair-value estimate, so this exercises cashout TIMING/
    MECHANISM, not real bookmaker economics — VIRTUAL money only.

    `shadow` (None = read the persisted flag) evaluates and logs decisions
    but fires nothing. Fair-value cashout is EV-negative by exactly its 0.95
    haircut, so shadow is how decision data accrues at zero cost — and each
    entry also records the real bookmaker offer, which is what lets us later
    test whether adj_prob beats the real live price.

    Pure function (no Flask request context) so BOTH the POST endpoint and
    the autonomous scheduler thread can call it; pass the backend in. Every
    evaluation (fired, would-have-fired, or held) is appended to
    `output/auto_cashout_log.jsonl`. Returns a summary dict."""
    shadow = _auto_cashout_shadow() if shadow is None else bool(shadow)
    live_by_match = {}
    live_path = os.path.join(OUTPUT_DIR, 'live_data.json')
    try:
        with open(live_path) as f:
            live_data = json.load(f)
        matches = live_data if isinstance(live_data, list) else (live_data.get('matches') or [])
        for m in matches:
            if isinstance(m, dict) and m.get('match'):
                live_by_match[m['match'].strip()] = m
    except (OSError, json.JSONDecodeError):
        return {'evaluated': 0, 'shadow': shadow, 'cashed_count': 0, 'cashed': [],
                'would_fire_count': 0, 'would_fire': [], 'note': 'no live snapshot'}

    # One representative per conceptual wager (execute_cashout cascades
    # across lanes), so we don't evaluate/fire the same bet_id twice.
    seen = set()
    representatives = []
    for slip_path in sorted(glob.glob(os.path.join(OUTPUT_DIR, 'bets_*.json'))):
        try:
            with open(slip_path) as f:
                slip = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for b in slip.get('bets', []):
            if b.get('status') != 'OPEN':
                continue
            key = b.get('bet_id') or (b.get('match'), b.get('type'), b.get('selection'))
            if key in seen:
                continue
            seen.add(key)
            if 'date' not in b:
                b['date'] = slip.get('date')
            representatives.append(b)

    # Real bookmaker offers, loaded once per sweep. These are recorded for
    # every evaluation but never used to price or decide — the decision still
    # runs off the synthetic fair value, so shadow logs let us later ask
    # whether adj_prob actually beats the real live price.
    bk = _load_bookmaker_offers() or {}
    bk_all = bk.get('all') or []
    bk_age_s = bk.get('age_s')

    evaluated = 0
    cashed = []
    would_fire = []
    log_lines = []
    now_iso = datetime.datetime.now().isoformat(timespec='seconds')
    for bet in representatives:
        match_str = (bet.get('match') or '').strip()
        lm = live_by_match.get(match_str)
        if lm is None or lm.get('message'):
            continue  # not in play / no adjusted probs → can't price or decide
        stake = float(bet.get('stake_units', bet.get('stake', 0)) or 0)
        prob_key = _SELECTION_TO_PROB_KEY.get(bet.get('selection'))
        if bet.get('type') == 'O/U':
            adj_prob = float((lm.get('adj_ou_probs') or {}).get(prob_key, 0) or 0)
        else:
            adj_prob = float((lm.get('adj_probs') or {}).get(prob_key, 0) or 0)
        fair = backend.get_cashout_amount(bet, lm)
        minute = _parse_minute(lm)
        decision = _cashout_decision(fair, stake, adj_prob if adj_prob > 0 else None, minute)
        evaluated += 1

        bk_offer = None
        if bk_all:
            hit = _match_offer_by_teams(bet.get('home'), bet.get('away'), bk_all)
            if hit:
                bk_offer = hit.get('cashout_offer')

        entry = {
            'ts': now_iso, 'bet_id': bet.get('bet_id'), 'match': match_str,
            'minute': minute, 'type': bet.get('type'),
            'selection': str(bet.get('selection')), 'stake': round(stake, 2),
            'odds': bet.get('odds'), 'adj_prob': round(adj_prob, 3),
            'fair_cashout': fair,
            'ratio': round(fair / stake, 3) if (fair and stake) else None,
            'decision': decision, 'shadow': shadow,
            'executed': False, 'would_fire': False, 'amount': None,
            'bookmaker_offer': bk_offer, 'bookmaker_age_s': bk_age_s,
        }
        if decision != 'hold':
            if shadow:
                # Record the counterfactual and leave the bet OPEN. It will be
                # re-evaluated next sweep, so scoring must take the FIRST
                # non-hold entry per bet_id as the would-have-fired moment.
                entry['would_fire'] = True
                entry['amount'] = fair
                would_fire.append({'bet_id': bet.get('bet_id'), 'match': match_str,
                                   'decision': decision, 'amount': fair,
                                   'selection': str(bet.get('selection'))})
            else:
                ok = backend.execute_cashout(bet, lm)
                if ok:
                    entry['executed'] = True
                    entry['would_fire'] = True
                    entry['amount'] = fair
                    cashed.append({'bet_id': bet.get('bet_id'), 'match': match_str,
                                   'decision': decision, 'amount': fair,
                                   'selection': str(bet.get('selection'))})
        log_lines.append(entry)

    if log_lines:
        try:
            with open(os.path.join(OUTPUT_DIR, 'auto_cashout_log.jsonl'), 'a') as f:
                for e in log_lines:
                    f.write(json.dumps(e) + '\n')
        except OSError:
            pass

    if cashed:
        global _AUTO_CASHOUT_EPOCH
        _AUTO_CASHOUT_EPOCH += 1   # signal live pages to reload (see /status)

    return {'evaluated': evaluated, 'shadow': shadow,
            'cashed_count': len(cashed), 'cashed': cashed,
            'would_fire_count': len(would_fire), 'would_fire': would_fire}


# --- Auto-cashout arming (server-side, browser-independent) ----------------
# The autonomous scheduler thread (started in __main__) refreshes Flashscore
# + runs the sweep every _AUTO_CASHOUT_INTERVAL_S while armed. Arming is
# persisted to disk so it survives a server restart and does NOT depend on a
# browser tab being open — the cashout actually executes, it isn't just a
# badge a human has to click.
_AUTO_CASHOUT_ARM_PATH = os.path.join(OUTPUT_DIR, 'auto_cashout_armed.json')
_AUTO_CASHOUT_INTERVAL_S = 10 * 60
# How long after kickoff a match can still be in play: 90' + half-time +
# stoppage + a buffer for late kickoffs.
_LIVE_MATCH_WINDOW_S = 150 * 60


def _read_auto_cashout_state():
    try:
        with open(_AUTO_CASHOUT_ARM_PATH) as f:
            return json.load(f) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _auto_cashout_armed():
    return bool(_read_auto_cashout_state().get('armed'))


def _auto_cashout_shadow():
    """Shadow mode: evaluate + log decisions but fire no cashout.

    Defaults to True when the key is absent so an existing arm file (or a
    fresh one) never starts moving money on its own — re-enabling real
    firing has to be an explicit choice. Cashout is priced at fair value
    (stake x odds x adj_prob x 0.95), which is EV-negative by exactly the
    haircut, so shadow is how decision data accrues at zero cost.
    """
    state = _read_auto_cashout_state()
    return True if 'shadow' not in state else bool(state.get('shadow'))


def _set_auto_cashout_armed(on: bool, shadow=None):
    """Persist arm state. `shadow=None` preserves the stored shadow flag."""
    keep = _auto_cashout_shadow() if shadow is None else bool(shadow)
    try:
        with open(_AUTO_CASHOUT_ARM_PATH, 'w') as f:
            json.dump({'armed': bool(on), 'shadow': keep,
                       'changed': datetime.datetime.now().isoformat(timespec='seconds')}, f)
    except OSError:
        pass


def _live_window_active(now=None):
    """Is any OPEN bet's match plausibly in play right now?

    The scheduler scrapes Flashscore on every tick, but a day's slate leaves
    long dead stretches (a typical day has an ~12h gap between the early
    matches finishing and the evening kickoffs), and scraping through those
    is pure waste. A match counts as in play from its kickoff until
    `_LIVE_MATCH_WINDOW_S` after it.

    Fails OPEN — if there are OPEN bets but no parseable kickoff among them,
    return True. Silently suppressing collection is a worse failure than an
    unnecessary scrape, and a stalled collector is exactly what went unnoticed
    for three months.
    """
    now = now or datetime.datetime.now()
    window = datetime.timedelta(seconds=_LIVE_MATCH_WINDOW_S)
    any_open = False
    parsed_any = False
    for slip_path in glob.glob(os.path.join(OUTPUT_DIR, 'bets_*.json')):
        try:
            with open(slip_path) as f:
                slip = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for b in slip.get('bets', []):
            if b.get('status') != 'OPEN':
                continue
            any_open = True
            raw = b.get('date') or ''
            try:
                ko = datetime.datetime.strptime(raw, '%Y-%m-%d %H:%M')
            except (TypeError, ValueError):
                continue
            parsed_any = True
            if ko <= now <= ko + window:
                return True
    return any_open and not parsed_any


@football_bp.route('/auto_cashout', methods=['POST'])
def auto_cashout():
    """Run ONE auto-cashout sweep now (manual/diagnostic trigger). The
    scheduler thread runs this autonomously when armed; this endpoint is
    handy for an immediate sweep against the current snapshot. JSON out.

    Defaults to the persisted shadow flag; pass `shadow=0|1` to override
    for this one sweep (e.g. a deliberate real firing while collection
    otherwise runs in shadow)."""
    raw = (request.form.get('shadow') or request.args.get('shadow') or '').strip().lower()
    shadow = None if raw == '' else raw in ('1', 'true', 'on', 'yes')
    return jsonify(_run_auto_cashout_sweep(g.backend, shadow=shadow)), 200


@football_bp.route('/auto_cashout/arm', methods=['POST'])
def auto_cashout_arm():
    """Arm/disarm the autonomous server-side auto-cashout loop. Body/query
    `on=1|0`. While armed, the background thread refreshes Flashscore and
    sweeps every 10 min regardless of whether any browser tab is open.

    `shadow=1|0` (optional) sets whether those sweeps fire for real or only
    log what they would have done.

    BOTH fields are optional and an omitted one is PRESERVED. Each checkbox
    posts only the field it owns, so a stale browser tab can never implicitly
    turn shadow off (which would silently re-enable real firing) by echoing
    back the state it happened to render with."""
    def _tri(name):
        raw = (request.form.get(name) or request.args.get(name) or '').strip().lower()
        return None if raw == '' else raw in ('1', 'true', 'on', 'yes')

    on = _tri('on')
    shadow = _tri('shadow')
    if on is None:
        on = _auto_cashout_armed()
    _set_auto_cashout_armed(on, shadow=shadow)
    return jsonify({'armed': _auto_cashout_armed(),
                    'shadow': _auto_cashout_shadow()}), 200


@football_bp.route('/void_bet/<bet_id>', methods=['POST'])
def void_bet(bet_id):
    """Mark an OPEN bet VOID — for matches that won't settle (postponed,
    cancelled, abandoned). Stake refunded to the lane. Uses the same
    backend abstraction as cashout, so the Phase 9 PamestoiximaBackend
    will get the same operation when it ships.
    """
    if '/' in bet_id or '..' in bet_id:
        flash('Invalid bet_id.', 'danger')
        return redirect(request.referrer or url_for('football.index'))

    bet, _ = _find_bet_and_live_match(bet_id)
    if bet is None:
        flash(f'Bet not found: {bet_id}.', 'warning')
        return redirect(request.referrer or url_for('football.index'))
    # Cascade across lanes — let void_bet check status itself.

    ok = g.backend.void_bet(bet)
    if not ok:
        flash('No OPEN bets found to void (sibling lanes may already '
              'be settled).', 'info')
        return redirect(request.referrer or url_for('football.index'))

    # Report what got refunded by re-reading the slip.
    total_refund = 0.0
    lanes_refunded = set()
    try:
        slip_date = bet_id.split(':', 1)[0] if ':' in bet_id else ''
        slip_path = os.path.join(OUTPUT_DIR, f'bets_{slip_date}.json')
        if os.path.exists(slip_path):
            with open(slip_path) as f:
                slip_after = json.load(f)
            for b in slip_after.get('bets', []):
                if b.get('bet_id') == bet_id and b.get('status') == 'VOID':
                    total_refund += float(b.get('stake_units', 0) or 0)
                    lanes_refunded.add(b.get('lane', 'value'))
    except Exception:
        pass

    lanes_label = ', '.join(sorted(lanes_refunded)) if lanes_refunded else 'unknown'
    flash(f'Voided across {len(lanes_refunded)} lane(s) ({lanes_label}); '
          f'€{total_refund:.2f} total refunded.', 'success')
    return redirect(request.referrer or url_for('football.index'))


@football_bp.route('/cancel_slip/<date>', methods=['POST'])
def cancel_slip(date):
    """Cancel a virtual slip while every bet on it is still OPEN —
    refunds each lane's stakes and closes the slip. Virtual-mode only;
    the live (Pamestoixima) backend has no analogue."""
    if '/' in date or '..' in date or len(date) != 10:
        flash('Invalid slip date.', 'danger')
        return redirect(request.referrer or url_for('football.betting_page'))

    if getattr(g, 'mode', 'virtual') != 'virtual':
        flash('Slip cancellation is only available in virtual mode.', 'warning')
        return redirect(request.referrer or url_for('football.betting_page'))

    ok, message = g.backend.cancel_slip(date)
    if not ok:
        flash(f'Could not cancel slip {date}: {message}', 'warning')
        return redirect(request.referrer or url_for('football.betting_page'))

    filename = f'bets_{date}.json'
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        if os.path.exists(filepath):
            _archive_file(filepath, filename)
        flash(f'Slip {date} cancelled and archived. {message}', 'success')
    except OSError as e:
        flash(f'Slip {date} cancelled ({message}), but archiving failed: {e}',
              'warning')
    return redirect(request.referrer or url_for('football.betting_page'))


@football_bp.route('/delete_file/<filename>', methods=['POST'])
def delete_file(filename):
    """
    Soft-delete: archive the file to output/history/ instead of removing.
    Bet slips remain readable by the cumulative comparison aggregator;
    they just disappear from the visible UI lists.
    """
    if os.path.sep in filename or '..' in filename:
        flash('Invalid filename!', 'danger')
        return redirect(url_for('football.index'))

    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        try:
            _archive_file(filepath, filename)
            flash(f'Archived {filename} (moved to history/).', 'success')
        except OSError as e:
            flash(f'Error archiving file: {e}', 'danger')
    else:
        flash('File not found.', 'warning')

    return redirect(request.referrer or url_for('football.index'))


# Bulk soft-delete: archive every file of the given kind. Each kind maps
# to one or more glob patterns. (`report_*.txt` used to live alongside
# predictions but those files are no longer generated by
# bin/run_verification.sh — the verification CSV is the source of truth.)
_ARCHIVE_ALL_PATTERNS = {
    'predictions':   ('predictions_*.csv',),
    'verifications': ('verification_*.csv',),
    # results_*.json is the verification scrape's own output. It used to be
    # written over matches_*.json; both are listed so neither is invisible.
    'scraped':       ('matches_*.json', 'results_*.json'),
}


@football_bp.route('/archive_all/<kind>', methods=['POST'])
def archive_all(kind):
    """Archive every file of `kind` (predictions / verifications / scraped)
    to output/history/. Per-file failures are collected and reported but
    don't abort the batch."""
    patterns = _ARCHIVE_ALL_PATTERNS.get(kind)
    if not patterns:
        flash(f'Unknown archive kind: {kind!r}.', 'danger')
        return redirect(url_for('football.index'))

    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(OUTPUT_DIR, pattern)))
    files = sorted(set(files))

    if not files:
        flash(f'No {kind} files to archive.', 'info')
        return redirect(request.referrer or url_for('football.index'))

    archived = 0
    errors = []
    for f in files:
        try:
            _archive_file(f, os.path.basename(f))
            archived += 1
        except OSError as e:
            errors.append(f"{os.path.basename(f)}: {e}")

    if archived:
        flash(f'Archived {archived} {kind} file(s) to history/.', 'success')
    if errors:
        # Cap displayed errors so a wholesale failure doesn't flood the UI.
        shown = '; '.join(errors[:3])
        suffix = '' if len(errors) <= 3 else f' (+{len(errors) - 3} more)'
        flash(f'Errors: {shown}{suffix}', 'warning')

    return redirect(request.referrer or url_for('football.index'))

@football_bp.route('/view/<filename>')
def view_file(filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        flash('File not found!', 'danger')
        return redirect(url_for('football.index'))
        
    try:
        df = pd.read_csv(filepath)
        df = df.fillna('')  # Ensure NaNs are empty strings so template .split() works
        
        # Load Cumulative Stats from JSON
        league_stats = []
        stats_file = os.path.join(PROJECT_ROOT, 'data_sets/league_analytics.json')
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)
                    
                # Convert dict to list for template
                for league, s in stats_data.items():
                    # Calculate percentages
                    total = s['total_matches']
                    if total > 0:
                        acc_1x2 = round((s['correct_1x2'] / total) * 100, 2)
                        acc_ou = round((s['correct_ou'] / total) * 100, 2)
                    else:
                        acc_1x2 = 0
                        acc_ou = 0
                        
                    league_stats.append({
                        'League': league,
                        'Count': total,
                        'Acc_1X2': acc_1x2,
                        'Acc_OU': acc_ou
                    })
                # Sort by Count desc
                league_stats.sort(key=lambda x: x['Count'], reverse=True)
            except Exception as e:
                print(f"Error loading league stats json: {e}")

        # Convert to list of dicts for easy iteration
        data = df.to_dict(orient='records')
        
        # Define Preferred Order
        # prediction_cols starts with basic info
        # prediction_cols starts with basic info
        base_cols = ['Date', 'League', 'Home Team', 'Away Team', 'Home', 'Away', 'Score']
        
        # User requested 1x2 cluster then O/U cluster
        target_cols = [
            'Prediction 1X2', 'Prediction 1X2 Odd', 'Conf 1X2', 'EV 1X2', 'Kelly 1X2', 'Home Win %', 'Draw %', 'Away Win %', 
            'Pred 1X2', 'Actual 1X2', 'Correct 1X2 Label', # Verification variants
            
            'Prediction O/U', 'Prediction O/U Odd', 'Conf O/U', 'EV O/U', 'Kelly O/U', 'Over %', 'Under %',
            'Pred O/U', 'Actual O/U', 'Correct O/U Label' # Verification variants
        ]
        
        final_cols = []
        # Add base columns if they exist
        for c in base_cols:
            if c in df.columns:
                final_cols.append(c)
        
        # Add target columns if they exist
        for c in target_cols:
             if c in df.columns:
                 final_cols.append(c)
                 
        # Add any remaining columns (e.g. Form), but exclude user-requested removals
        # Excluding: Home ELO, Away ELO, Home Form, Away Form, Match, and redundant booleans
        exclude_cols = ['Home ELO', 'Away ELO', 'Home Form', 'Away Form', 'Adj Logs', 'Match', 'Correct 1X2', 'Correct O/U']
        existing = set(final_cols)
        for c in df.columns:
            if c not in existing and c not in exclude_cols:
                final_cols.append(c)
        
        columns = final_cols
        count = len(data)
        return render_template('results.html', filename=filename, columns=columns, data=data, league_stats=league_stats, count=count)
    except Exception as e:
        flash(f'Error reading file: {e}', 'danger')
        return redirect(url_for('football.index'))

@football_bp.route('/live_analysis')
def live_analysis():
    """Retired: the live analysis now lives on the standalone sport-tabbed
    /live page (football tab). This route 301-redirects so old bookmarks /
    deep-links keep working. The former bookmaker-linked-only filtered view
    is gone — /live shows the full football live panel."""
    return redirect(url_for('live_tabbed', tab='football'), code=301)

def _launch_live_refresh(with_bookmaker=False):
    """Start the Flashscore live scrape (`scripts/run_live_analysis.py`).
    Returns the Popen, or None if a live scrape is already running. Shared
    by the /refresh_live route and the autonomous auto-cashout scheduler."""
    if TASKS.get('live') and TASKS['live'].get('process') and TASKS['live']['process'].poll() is None:
        return None
    script_path = os.path.join(PROJECT_ROOT, 'scripts', 'run_live_analysis.py')
    log_file = open(os.path.join(LOG_DIR, 'live.log'), 'w')
    # ml_project imports need both repo root and ml_project/ on PYTHONPATH.
    env = os.environ.copy()
    ml_paths = [PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'ml_project')]
    env['PYTHONPATH'] = os.pathsep.join([p for p in ml_paths + [env.get('PYTHONPATH', '')] if p])
    cmd = ['venv/bin/python', script_path]
    if with_bookmaker:
        cmd.append('--with-bookmaker')
    proc = subprocess.Popen(
        cmd, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    TASKS['live'] = {'process': proc, 'start_time': datetime.datetime.now()}
    return proc


def _live_panel_context():
    """Context for `_live_football_panel.html`: the live snapshot enriched
    with any OPEN bets on each match, plus the auto-cashout arm state.
    Shared by the /live page and the /football/live_fragment swap endpoint
    so both render the panel from an identical context."""
    live_file = os.path.join(OUTPUT_DIR, "live_data.json")
    live_matches = []
    if os.path.exists(live_file):
        try:
            with open(live_file, 'r') as f:
                live_matches = json.load(f)
        except Exception:
            pass
    _attach_open_bets(live_matches)
    return {
        'live_matches': live_matches[:50],
        'auto_cashout_armed': _auto_cashout_armed(),
        'auto_cashout_shadow': _auto_cashout_shadow(),
    }


@football_bp.route('/live_fragment')
def live_fragment():
    """Render the live panel alone, for in-place swapping by the browser.

    The panel used to be refreshed with `location.reload()`, which threw
    away scroll position and all in-page state (collapse toggles, the
    Auto-10m checkbox) on every 10-minute auto-refresh. The client now
    fetches this and replaces `#livePanelBody`. The panel's <script> sits
    OUTSIDE that wrapper, so a swap never re-inserts it — the client calls
    `wireLivePanel()` to rebind the fresh nodes instead."""
    return render_template('_live_football_panel.html', **_live_panel_context())


@football_bp.route('/refresh_live', methods=['POST'])
def refresh_live():
    """Kicks off `scripts/run_live_analysis.py` (Flashscore live scrape).
    When the request carries `with_bookmaker=1` (manual-button path),
    passes `--with-bookmaker` to the script — which, after the Flashscore
    scrape finishes, conditionally chains the Pamestoixima open-bets
    scrape ONLY IF a live match has an open bet on it (otherwise there's
    nothing for a cashout offer to attach to, so the slow ~25s headed
    Chromium scrape is skipped). The auto-refresh polling omits the
    flag entirely (Pamestoixima needs headed mode — Akamai blocks
    headless — so popping a Chromium window every refresh is off the
    table). See PAMESTOIXIMA_NOTES.md "Corrections"."""
    with_bookmaker = (request.args.get('with_bookmaker')
                      or request.form.get('with_bookmaker'))
    try:
        proc = _launch_live_refresh(with_bookmaker=bool(with_bookmaker))
        if proc is None:
            flash('Live analysis is already running!', 'warning')
        elif with_bookmaker:
            flash('Live refresh started — bookmaker offers will refresh too '
                  'if a live match has an open bet (brief Chromium window).', 'info')
        else:
            flash('Live analysis started! Auto-refreshing...', 'info')
    except Exception as e:
        flash(f"Error starting live analysis: {e}", 'danger')

    return redirect(url_for('football.index'))

@football_bp.route('/clear_live', methods=['POST'])
def clear_live():
    live_file = os.path.join(OUTPUT_DIR, "live_data.json")
    try:
        with open(live_file, 'w') as f:
            json.dump([], f)
        flash('Live data cleared.', 'success')
    except Exception as e:
        flash(f'Error clearing live data: {e}', 'danger')
        
    return redirect(url_for('football.index'))

@football_bp.route('/update_leagues', methods=['POST'])
def update_leagues():
    script_path = os.path.join(PROJECT_ROOT, 'bin', 'update_leagues_data.sh')
    try:
        if TASKS.get('leagues') and TASKS['leagues']['process'] and TASKS['leagues']['process'].poll() is None:
             flash('Leagues update is already running!', 'warning')
             return redirect(url_for('football.index'))
             
        log_file = open(os.path.join(LOG_DIR, 'leagues.log'), 'w')
        # Use simple bash execution
        proc = subprocess.Popen(['/bin/bash', script_path], cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
        
        TASKS['leagues'] = {'process': proc, 'start_time': datetime.datetime.now()}
        
        flash('Leagues data update started! Check <a href="/logs/leagues.log">logs</a> for status.', 'success')
    except Exception as e:
        flash(f"Error starting leagues update: {e}", 'danger')
        
    return redirect(url_for('football.index'))

@app.route('/server/<action>', methods=['POST'])
def server_control(action):
    if action not in ['restart', 'stop']:
        return "Invalid action", 400
        
    try:
        # Use nohup to ensure the script survives the server killing itself
        # We need to detach properly.
        script = os.path.join(PROJECT_ROOT, 'bin', 'manage_server.sh')
        cmd = ['nohup', '/bin/bash', script, action]
        
        # Popen with start_new_session=True is key
        subprocess.Popen(cmd, cwd=PROJECT_ROOT, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if action == 'stop':
            return "Server stopping...", 200
        else:
             # Return HTML that auto-redirects after 5 seconds
             return """
             <html>
             <head>
                <meta http-equiv="refresh" content="5;url=/" />
                <title>Restarting...</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
             </head>
             <body class="d-flex justify-content-center align-items-center vh-100 bg-light">
               <div class="text-center">
                 <h1 class="display-4">🔄 Restarting...</h1>
                 <p class="lead">The server is rebooting. You will be redirected to the dashboard in 5 seconds.</p>
                 <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                 </div>
               </div>
             </body>
             </html>
             """, 200
             
    except Exception as e:
        return f"Error: {e}", 500


@football_bp.route('/place_bets', methods=['POST'])
def place_bets():
    try:
        data = request.get_json()
        bets = data.get('bets', [])

        extracted_date = None
        if bets:
            first_date = bets[0].get('date', '')
            if first_date:
                try:
                    extracted_date = first_date.split(' ')[0]
                except: pass

        date_str = extracted_date if extracted_date else data.get('date', datetime.datetime.now().strftime('%Y-%m-%d'))

        if not bets:
            return jsonify({'error': 'No bets provided.'}), 400

        # Group stakes by lane, validate funds per lane, then debit each lane.
        # Also stamp each bet with a canonical bet_id + mode='virtual' so the
        # Phase 7 cashout endpoint can find it later. (Per the BettingBackend
        # contract — docs/LIVE_BETTING_TRANSITION.md "Bet ID format".)
        stake_by_lane = {lane: 0.0 for lane in LANES}
        for b in bets:
            lane = b.get('lane', 'value')
            if lane not in LANES:
                lane = 'value'
                b['lane'] = lane
            stake_by_lane[lane] += float(b.get('stake_units', 0))
            # Stamp the canonical bet_id + mode. Won't overwrite an
            # explicitly-set bet_id (e.g. on tests or future Phase 9 paths).
            if not b.get('bet_id'):
                b['bet_id'] = make_bet_id(
                    date_str,
                    b.get('home') or (b.get('match', '').split(' vs ')[0]
                                      if ' vs ' in b.get('match', '') else ''),
                    b.get('away') or (b.get('match', '').split(' vs ')[1]
                                      if ' vs ' in b.get('match', '') else ''),
                    b.get('type', '1X2'),
                    b.get('selection', ''),
                )
            b.setdefault('mode', 'virtual')
            # Normalise the live-betting mark to a bool and persist it on
            # the slip. Set by the per-bet "Live" checkbox in the slip
            # preview. This records *intent* only — no real bet is placed
            # here. The dormant /football/place_real_bets route (and a
            # future real-betting backend) reads this flag.
            b['mark_for_real'] = bool(b.get('mark_for_real', False))

        current_lane_br = lane_bankrolls('football')
        for lane, stake in stake_by_lane.items():
            if stake > current_lane_br[lane] + 1e-6:
                return jsonify({'error': (
                    f"Insufficient {lane} funds. Stake ({stake:.2f}) exceeds "
                    f"{lane} bankroll ({current_lane_br[lane]:.2f})."
                )}), 400

        new_lane_br = {}
        for lane, stake in stake_by_lane.items():
            if stake > 0:
                new_lane_br[lane] = update_bankroll('football', -stake, lane=lane)
            else:
                new_lane_br[lane] = current_lane_br[lane]

        total_stake = sum(stake_by_lane.values())

        filename = f"bets_{date_str}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, 'w') as f:
            json.dump({
                'date': date_str,
                'count': len(bets),
                'bets': bets,
                'total_stake': total_stake,
                'stake_by_lane': {k: round(v, 2) for k, v in stake_by_lane.items()},
                'status': 'OPEN',
                'pnl': 0.0,
                'settled': False
            }, f, indent=4)

        return jsonify({
            'message': f"Successfully placed {len(bets)} virtual bets! Deducted {total_stake:.2f} across lanes.",
            'file': filename,
            'new_balance': round(sum(new_lane_br.values()), 2),
            'lane_bankrolls': {k: round(v, 2) for k, v in new_lane_br.items()},
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@football_bp.route('/place_real_bets', methods=['POST'])
def place_real_bets():
    """DORMANT real-betting hook. Receives the bets the user ticked
    'Live' in the slip preview and reports them back — but DOES NOT
    place any real bet. The real-betting backend (Pamestoixima
    placement via real_betting/) is out of scope per FOOTBALL_NEXT_STEPS until
    a separate re-evaluation; this route exists so the UI affordance
    and the request path are in place now, ready to wire to the real
    flow later.

    When real placement is eventually wired, this is where it would
    dispatch to real_betting (likely after a confirmation modal +
    per-bet stake caps + the EXECUTE_* gating used by the dryrun
    scripts). For now it's a no-op acknowledgement."""
    try:
        data = request.get_json(silent=True) or {}
        marked = data.get('bets', []) or []
        n = len(marked)
        # Build a short human summary of what WOULD be placed.
        lines = []
        for b in marked[:10]:
            lines.append(f"{b.get('match', '?')} — {b.get('type', '?')} "
                         f"{b.get('selection', '?')} @ {b.get('odds', '?')} "
                         f"(€{b.get('stake_units', '?')})")
        preview = '; '.join(lines) + (f" (+{n - 10} more)" if n > 10 else '')
        print(f"[place_real_bets] DORMANT — {n} bet(s) flagged for live "
              f"betting, NOT placed: {preview}")
        return jsonify({
            'message': (f"Real betting is DORMANT — recorded intent for "
                        f"{n} bet(s) but placed nothing. The marks are saved "
                        f"on the slip (mark_for_real=true). Wiring to the "
                        f"bookmaker is out of scope until re-evaluation "
                        f"(see FOOTBALL_NEXT_STEPS.md)."),
            'count': n,
            'placed': 0,
            'dormant': True,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _load_slip(filepath):
    """Load a single bets_*.json with backfilled fields."""
    try:
        with open(filepath, 'r') as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    data['filename'] = os.path.basename(filepath)
    if 'total_stake' not in data:
        data['total_stake'] = sum(float(b.get('stake_units', 0)) for b in data.get('bets', []))
    if 'total_return' not in data:
        data['total_return'] = (
            data.get('total_stake', 0) + data.get('pnl', 0)
            if data.get('status') == 'CLOSED' else 0.0
        )
    return data


def _empty_bet_stats():
    return {'bets': 0, 'settled': 0, 'won': 0, 'lost': 0, 'void': 0,
            'cashed_out': 0, 'stake': 0.0, 'settled_stake': 0.0,
            'returned': 0.0, 'pnl': 0.0}


def _accumulate_bet(s, bet):
    """Fold one bet into a stats dict `s` (the canonical status taxonomy +
    realized money). Shared by the per-lane (compute_sport_summary) and
    per-league (compute_league_betting_summary) aggregations so cashout / void /
    won / lost are counted identically. Cashed-out bets are realized by their
    stored cashout_amount / pnl (a cashout > stake is a win, < stake a loss)."""
    stake = float(bet.get('stake_units', 0) or 0)
    status = bet.get('status', 'OPEN')
    result = bet.get('result', '')
    s['bets'] += 1
    s['stake'] += stake
    if status == 'OPEN':
        return
    s['settled'] += 1
    s['settled_stake'] += stake  # ROI denominator: only stake of resolved bets
    if status == 'CASHED_OUT' or result == 'CASHED_OUT':
        s['cashed_out'] += 1
        cashout_amount = float(bet.get('cashout_amount', stake))
        pnl_v = float(bet.get('pnl', cashout_amount - stake))
        s['returned'] += cashout_amount
        s['pnl'] += pnl_v
        if pnl_v > 0:
            s['won'] += 1
        elif pnl_v < 0:
            s['lost'] += 1
    elif result == 'WON' or status == 'WON':
        s['won'] += 1
        s['returned'] += stake + float(bet.get('pnl', 0))
        s['pnl'] += float(bet.get('pnl', 0))
    elif result == 'LOST' or status == 'LOST':
        s['lost'] += 1
        s['pnl'] += float(bet.get('pnl', -stake))
    else:  # VOID (or unrecognised terminal status)
        s['void'] += 1
        s['returned'] += stake


def _finalize_bet_stats(s):
    """Add win_rate + roi and round the money fields. Mutates + returns `s`."""
    decided = s['won'] + s['lost']
    s['win_rate'] = round((s['won'] / decided * 100) if decided > 0 else 0.0, 1)
    # ROI is realized: divide P/L by stake of SETTLED bets only, so open
    # (unrealized) bets don't deflate the figure. `stake` stays the total
    # committed amount (used for exposure display).
    s['roi'] = round((s['pnl'] / s['settled_stake'] * 100) if s['settled_stake'] > 0 else 0.0, 1)
    s['stake'] = round(s['stake'], 2)
    s['settled_stake'] = round(s['settled_stake'], 2)
    s['returned'] = round(s['returned'], 2)
    s['pnl'] = round(s['pnl'], 2)
    return s


def compute_sport_summary(bets_dir):
    """
    Aggregate stats for one sport from its bets directory.
    Reads ACTIVE slips (bets_dir/bets_*.json) AND ARCHIVED slips
    (bets_dir/history/bets_*.json) so soft-deleted slips still count.

    Returns: {
        'history': [active slips, sorted desc by date],
        'lane_stats': {lane: {bets, settled, won, lost, void, stake,
                              returned, pnl, win_rate, roi}},
        'totals': {bets, settled, stake, returned, pnl, roi},
    }
    """
    abs_dir = bets_dir if os.path.isabs(bets_dir) else os.path.join(PROJECT_ROOT, bets_dir)
    history_dir = os.path.join(abs_dir, 'history')

    history = []
    for f in glob.glob(os.path.join(abs_dir, "bets_*.json")):
        s = _load_slip(f)
        if s: history.append(s)
    history.sort(key=lambda x: x.get('date', ''))  # chronological: earliest slip first

    # Order the bets inside each slip for display: first by lane (canonical
    # value → conviction → model order), then by kickoff start time.
    _lane_rank = {lane: i for i, lane in enumerate(LANES)}
    for s in history:
        s.get('bets', []).sort(
            key=lambda b: (_lane_rank.get(b.get('lane', 'value'), len(LANES)),
                           str(b.get('date', ''))))

    archived_slips = []
    for f in glob.glob(os.path.join(history_dir, "bets_*.json")):
        s = _load_slip(f)
        if s: archived_slips.append(s)

    lane_stats = {lane: _empty_bet_stats() for lane in LANES}

    for slip in (history + archived_slips):
        for bet in slip.get('bets', []):
            lane = bet.get('lane', 'value')
            _accumulate_bet(lane_stats.setdefault(lane, _empty_bet_stats()), bet)

    for s in lane_stats.values():
        _finalize_bet_stats(s)

    totals = {
        'bets': sum(s['bets'] for s in lane_stats.values()),
        'settled': sum(s['settled'] for s in lane_stats.values()),
        'stake': round(sum(s['stake'] for s in lane_stats.values()), 2),
        'settled_stake': round(sum(s['settled_stake'] for s in lane_stats.values()), 2),
        'returned': round(sum(s['returned'] for s in lane_stats.values()), 2),
        'pnl': round(sum(s['pnl'] for s in lane_stats.values()), 2),
    }
    totals['roi'] = round((totals['pnl'] / totals['settled_stake'] * 100) if totals['settled_stake'] > 0 else 0.0, 1)

    return {'history': history, 'lane_stats': lane_stats, 'totals': totals}


def compute_league_betting_summary(bets_dir, lane=None):
    """Per-LEAGUE realized betting performance (money, cashout-aware).

    Distinct from the dashboard's "Cumulative League Performance" (which is pure
    model prediction-accuracy vs the final result). This answers "which leagues
    am I making/losing money on?" — aggregating settled bets (active + archived)
    by `league`, with cashed-out bets realized by their cashout P/L. Returns a
    list of rows sorted by P/L desc.

    ``lane`` restricts to one lane (e.g. 'model'). The dashboard joins the
    **model** lane only: it bets ~every prediction, so its per-league ROI is the
    apples-to-apples companion to the prediction accuracy (same broad coverage),
    whereas value/conviction are filtered subsets. ``None`` = all lanes.
    """
    abs_dir = bets_dir if os.path.isabs(bets_dir) else os.path.join(PROJECT_ROOT, bets_dir)
    slips = []
    for d in (abs_dir, os.path.join(abs_dir, 'history')):
        for f in glob.glob(os.path.join(d, "bets_*.json")):
            s = _load_slip(f)
            if s:
                slips.append(s)

    by_league = {}
    for slip in slips:
        for bet in slip.get('bets', []):
            if lane is not None and bet.get('lane', 'value') != lane:
                continue
            league = bet.get('league') or '(unknown)'
            _accumulate_bet(by_league.setdefault(league, _empty_bet_stats()), bet)

    rows = []
    for league, s in by_league.items():
        _finalize_bet_stats(s)
        rows.append({'league': league, **s})
    # Most-staked / most-impactful first; settled leagues before untouched ones.
    rows.sort(key=lambda r: (r['settled'] > 0, r['pnl']), reverse=True)
    return rows


@football_bp.route('/betting')
def betting_page():
    summary = compute_sport_summary(OUTPUT_DIR)
    # Per-lane defaults for the override form (so the placeholder shows
    # the actual saved value, not the generic word "default").
    cfg = get_sport_config('football')
    lane_br = lane_bankrolls('football')
    lane_defaults = {
        'value':      {'bankroll': lane_br['value'],
                       'cap_pct':  cfg['value_max_daily_exposure_pct'] * 100},
        'conviction': {'bankroll': lane_br['conviction'],
                       'cap_pct':  cfg['conviction_max_daily_exposure_pct'] * 100},
        'model':      {'bankroll': lane_br['model'],
                       'cap_pct':  cfg['model_max_daily_exposure_pct'] * 100},
    }
    return render_template('betting.html',
                           history=summary['history'],
                           lane_stats=summary['lane_stats'],
                           lane_defaults=lane_defaults,
                           sport_label='Football')

@football_bp.route('/update_data', methods=['POST'])
def update_data():
    if TASKS.get('update') and TASKS['update']['process'] and TASKS['update']['process'].poll() is None:
         flash('Data update is already running!', 'warning')
         return redirect(url_for('football.index'))

    try:
        script_path = os.path.join(PROJECT_ROOT, 'scripts', 'update_football_data.py')
        log_file = open(os.path.join(LOG_DIR, 'update_data.log'), 'w')
        proc = subprocess.Popen(['venv/bin/python', script_path], cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
        
        TASKS['update'] = {'process': proc, 'start_time': datetime.datetime.now()}
        
        flash('Data update started! Check <a href="/logs/update_data.log">logs</a> for status.', 'success')
    except Exception as e:
        flash(f"Error starting update: {e}", 'danger')
        
    return redirect(url_for('football.index'))

@football_bp.route('/retrain_model', methods=['POST'])
def retrain_model():
    if TASKS.get('retrain') and TASKS['retrain']['process'] and TASKS['retrain']['process'].poll() is None:
         flash('Model Retraining is already running!', 'warning')
         return redirect(url_for('football.index'))

    try:
        script_path = os.path.join(PROJECT_ROOT, 'bin', 'retrain_pipeline.sh')
        log_file = open(os.path.join(LOG_DIR, 'retrain.log'), 'w')
        # Using bash directly
        proc = subprocess.Popen(['/bin/bash', script_path], cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
        
        TASKS['retrain'] = {'process': proc, 'start_time': datetime.datetime.now()}
        
        flash('Full Retrain Pipeline started! Check <a href="/logs/retrain.log">logs</a> for progress.', 'success')
    except Exception as e:
        flash(f"Error starting retrain: {e}", 'danger')
        
    return redirect(url_for('football.index'))

import threading

def run_verify_task_thread(date_arg):
    try:
        cmd = ["/bin/bash", "run_verification.sh"]
        if date_arg:
            cmd.extend(["-d", date_arg])
            
        # Run process synchronously in this thread
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )
        
        # We store the process in global TASKS so we can kill it if needed
        TASKS['verify']['process'] = process
        TASKS['verify']['state'] = 'running'
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            TASKS['verify']['state'] = 'completed'
            TASKS['verify']['msg'] = 'Verification completed successfully.'
            # Note: bin/run_verification.sh already invokes
            # ml_project/resolve_daily_bets.py at the end of its flow,
            # which is now the canonical settlement path (per-lane
            # bankroll updates via sports_config, idempotent across
            # re-runs, doesn't VOID matches that haven't finished).
            # No extra settle call needed here — this was previously a
            # buggy duplicate using a 'verify_' filename that never
            # matched the script's 'verification_' output.
        else:
            TASKS['verify']['state'] = 'error'
            TASKS['verify']['msg'] = f"Error: {stderr}"
            
    except Exception as e:
        TASKS['verify']['state'] = 'error'
        TASKS['verify']['msg'] = str(e)
    finally:
        TASKS['verify']['process'] = None

@football_bp.route('/verify', methods=['POST'])
def run_verify():
    if TASKS['verify'] and TASKS['verify']['process'] and TASKS['verify']['process'].poll() is None:
        flash('Verification task is already running.', 'warning')
        return redirect(url_for('football.index'))

    # Reset state
    TASKS['verify'] = {'state': 'running', 'process': None, 'msg': ''}
    
    # Launch in thread
    thread = threading.Thread(target=run_verify_task_thread, args=(None,))
    thread.start()
    
    flash('Verification task started! Bankroll will update upon completion.', 'success')
    return redirect(url_for('football.index'))

import glob
import pandas as pd
from flask import jsonify

def _available_prediction_dates():
    """Dates that (a) have a predictions_<date>.csv, (b) have NO *live* bets
    slip — active OR archived — yet, and (c) are today or later. A slip counts
    as live only if it holds at least one non-VOID bet; a fully-cancelled slip
    (every bet VOID) means that date is effectively un-bet and IS re-offered, so
    the user can re-bet after cancelling. A date whose slip was archived to
    output/history/ with real (WON/LOST/OPEN) bets still counts as done. Past
    dates are excluded so the picker can't surface fixtures that already kicked
    off. Sorted descending so today (the most common pick) is first."""
    today_str = datetime.date.today().isoformat()
    pred_dates = set()
    for p in glob.glob(os.path.join(OUTPUT_DIR, 'predictions_*.csv')):
        name = os.path.basename(p)
        if name.startswith('predictions_') and name.endswith('.csv'):
            pred_dates.add(name[len('predictions_'):-len('.csv')])
    bet_dates = set()
    # Active slips (output/) AND archived slips (output/history/) can both mark a
    # date as already-bet — but only if they still hold a non-VOID bet. A slip
    # whose bets were all voided/cancelled no longer blocks the date.
    slip_paths = (glob.glob(os.path.join(OUTPUT_DIR, 'bets_*.json'))
                  + glob.glob(os.path.join(OUTPUT_DIR, 'history', 'bets_*.json')))
    for b in slip_paths:
        name = os.path.basename(b)
        if not (name.startswith('bets_') and name.endswith('.json')):
            continue
        # Strip optional `.<ts>` archive suffix some old slips carry
        # (e.g. bets_2026-05-21.20260522_124718.json).
        stem = name[len('bets_'):-len('.json')]
        date_key = stem.split('.', 1)[0]
        try:
            data = json.load(open(b))
        except (ValueError, OSError):
            # Unreadable slip: be conservative and treat the date as bet.
            bet_dates.add(date_key)
            continue
        bets = data if isinstance(data, list) else data.get('bets', [])
        if any(str(bet.get('status', '')).upper() != 'VOID' for bet in bets):
            bet_dates.add(date_key)
    return sorted(
        (d for d in pred_dates - bet_dates if d >= today_str),
        reverse=True,
    )


@football_bp.route('/predictions/available')
def predictions_available():
    """List prediction dates that don't already have a bets slip on disk."""
    return jsonify({'dates': _available_prediction_dates()})


@football_bp.route('/auto_wager')
def auto_wager():
    try:
        # Date selection: explicit ?date=YYYY-MM-DD wins; otherwise pick
        # the most recent prediction CSV without a corresponding bets
        # slip; otherwise fall back to the most recent prediction CSV
        # (so the page still renders something if every date is bet).
        date_arg = (request.args.get('date') or '').strip()
        if date_arg:
            cand = os.path.join(OUTPUT_DIR, f'predictions_{date_arg}.csv')
            if not os.path.isfile(cand):
                return jsonify({'error': f'No predictions file for {date_arg}.'}), 404
            latest_file = cand
        else:
            available = _available_prediction_dates()
            if available:
                latest_file = os.path.join(OUTPUT_DIR, f'predictions_{available[0]}.csv')
            else:
                pred_files = glob.glob(os.path.join(OUTPUT_DIR, 'predictions_*.csv'))
                if not pred_files:
                    return jsonify({'error': 'No prediction files found.'})
                pred_files.sort(reverse=True)
                latest_file = pred_files[0]
        df = pd.read_csv(latest_file)

        def _to_float(v):
            try:
                if isinstance(v, str):
                    v = v.strip().rstrip('%')
                    if not v:
                        return 0.0
                return float(v)
            except (ValueError, TypeError):
                return 0.0

        def parse_kelly(k_str):
            v = _to_float(k_str)
            return v / 100.0 if isinstance(k_str, str) and '%' in k_str else v

        config = get_sport_config('football')
        lane_br = lane_bankrolls('football')

        # Shared
        min_stake_eur            = config['min_stake_eur']
        # Value lane
        min_confidence           = config['min_confidence']
        stake_multiplier         = config['stake_multiplier']
        max_stake_pct            = config['max_stake_pct']
        ev_cap_value             = config['ev_cap_value']
        # Conviction lane
        conv_min_confidence      = config['conviction_min_confidence']
        conv_min_odds            = config['conviction_min_odds']
        conv_stake_pct           = config['conviction_stake_pct']
        # Model lane
        model_base_pct           = config['model_base_pct']
        model_max_stake_pct      = config['model_max_stake_pct']
        model_min_stake_eur      = config['model_min_stake_eur']
        ev_factor_min            = config['model_ev_factor_min']
        ev_factor_max            = config['model_ev_factor_max']

        def _override_bankroll(lane, default):
            """Allow ?bankroll_<lane>=N as a per-session override, capped at saved balance."""
            raw = request.args.get(f'bankroll_{lane}')
            if raw is None or raw == '':
                return default, None
            try:
                v = float(raw)
            except ValueError:
                return default, None
            if v <= 0:
                raise ValueError(f"{lane} session bankroll must be positive.")
            if v > default:
                raise ValueError(
                    f"{lane} session bankroll ({v}) cannot exceed saved balance ({default})."
                )
            return v, v

        def _override_cap_pct(lane, default):
            """?cap_<lane>=0.15 overrides this lane's daily exposure pct."""
            raw = request.args.get(f'cap_{lane}')
            if raw is None or raw == '':
                return default
            try:
                v = float(raw)
            except ValueError:
                return default
            if v <= 0 or v > 1.0:
                return default
            return v

        try:
            value_br, value_session     = _override_bankroll('value', lane_br['value'])
            conv_br, conv_session       = _override_bankroll('conviction', lane_br['conviction'])
            model_br, model_session     = _override_bankroll('model', lane_br['model'])
        except ValueError as ve:
            return jsonify({'error': str(ve)}), 400

        value_cap_pct = _override_cap_pct('value', config['value_max_daily_exposure_pct'])
        conv_cap_pct  = _override_cap_pct('conviction', config['conviction_max_daily_exposure_pct'])
        model_cap_pct = _override_cap_pct('model', config['model_max_daily_exposure_pct'])

        max_value_per_bet = value_br * max_stake_pct
        max_model_per_bet = model_br * model_max_stake_pct
        conv_flat_stake   = conv_br * conv_stake_pct

        def _common_fields(row, bet_type, selection_col, odd_col, conf_col, ev_col, kelly_col):
            _date = str(row.get('Date', '') or '')
            # Kickoff time (HH:MM) split out for display; '' when the source
            # only carried a date (legacy predictions / unknown start_time).
            _time = _date.split(' ', 1)[1] if ' ' in _date and not _date.endswith('00:00') else ''
            return {
                'date': _date,
                'time': _time,
                'match': f"{row['Home Team']} vs {row['Away Team']}",
                'home': row['Home Team'],
                'away': row['Away Team'],
                'match_id': row.get('match_id', ''),
                'league': row['League'],
                'type': bet_type,
                'selection': row[selection_col],
                'odds': row[odd_col],
                'odd': row[odd_col],
                'conf': f"{_to_float(row.get(conf_col, 0)):.2f}",
                'ev': f"{_to_float(row.get(ev_col, 0)):+.2f}",
                'kelly': f"{parse_kelly(row.get(kelly_col, '0%')):.2%}",
                'status': 'OPEN',
            }

        def build_value_bet(row, bet_type, selection_col, odd_col, conf_col, ev_col, kelly_col):
            """Option B: EV-gated, stake = bankroll * min(EV, cap) * conf * multiplier, capped & floored.

            The min(ev, ev_cap_value) clamp prevents a single high-EV pick (often
            from a low-data league) from dominating the slip. Real fix is
            per-league probability recalibration upstream — see FOOTBALL_NEXT_STEPS.md.
            """
            ev = _to_float(row.get(ev_col, 0))
            conf = _to_float(row.get(conf_col, 0))
            if ev <= 0 or conf < min_confidence:
                return None
            ev_for_sizing = min(ev, ev_cap_value)
            raw_stake = value_br * ev_for_sizing * conf * stake_multiplier
            stake = min(raw_stake, max_value_per_bet)
            if stake < min_stake_eur:
                return None
            bet = _common_fields(row, bet_type, selection_col, odd_col, conf_col, ev_col, kelly_col)
            bet.update({'lane': 'value', 'stake_units': round(stake, 2), 'stake': round(stake, 2),
                        '_rank': ev})
            return bet

        def build_conviction_bet(row, bet_type, selection_col, odd_col, conf_col, ev_col, kelly_col):
            """Conviction lane: high conf, odds ≥ floor, EV ignored. Flat stake."""
            conf = _to_float(row.get(conf_col, 0))
            odd = _to_float(row.get(odd_col, 0))
            if conf < conv_min_confidence or odd < conv_min_odds:
                return None
            stake = conv_flat_stake
            if stake < min_stake_eur:
                return None
            bet = _common_fields(row, bet_type, selection_col, odd_col, conf_col, ev_col, kelly_col)
            bet.update({'lane': 'conviction', 'stake_units': round(stake, 2), 'stake': round(stake, 2),
                        '_rank': conf})
            return bet

        def build_model_bet(row, bet_type, selection_col, odd_col, conf_col, ev_col, kelly_col):
            """Model lane: bet every prediction with valid odds. Sized by conf × ev_factor."""
            conf = _to_float(row.get(conf_col, 0))
            odd = _to_float(row.get(odd_col, 0))
            if conf <= 0 or odd <= 1.0:
                return None
            ev_factor = max(ev_factor_min, min(ev_factor_max, conf * odd))
            raw_stake = model_br * model_base_pct * conf * ev_factor
            stake = min(raw_stake, max_model_per_bet)
            if stake < model_min_stake_eur:
                return None
            bet = _common_fields(row, bet_type, selection_col, odd_col, conf_col, ev_col, kelly_col)
            bet.update({'lane': 'model', 'stake_units': round(stake, 2), 'stake': round(stake, 2),
                        '_rank': conf * ev_factor})
            return bet

        value_bets, conviction_bets, model_bets = [], [], []
        for _, row in df.iterrows():
            for bet_type, sel, odd, conf, ev, kelly in [
                ('1X2', 'Prediction 1X2', 'Prediction 1X2 Odd', 'Conf 1X2', 'EV 1X2', 'Kelly 1X2'),
                ('O/U', 'Prediction O/U', 'Prediction O/U Odd', 'Conf O/U', 'EV O/U', 'Kelly O/U'),
            ]:
                vb = build_value_bet(row, bet_type, sel, odd, conf, ev, kelly)
                if vb: value_bets.append(vb)
                cb = build_conviction_bet(row, bet_type, sel, odd, conf, ev, kelly)
                if cb: conviction_bets.append(cb)
                mb = build_model_bet(row, bet_type, sel, odd, conf, ev, kelly)
                if mb: model_bets.append(mb)

        def _enforce_daily_cap(bets, bankroll, cap_pct, lane_name):
            """Per-lane daily cap: rank-and-truncate — keep the best bets at full
            stake until the cap is exhausted, drop the rest.

            The previous policy scaled every stake pro-rata to fit the cap, which
            broke down on large slates: with ~128 fixtures the value/model lanes
            produced 57/254 candidates, the scale factor fell to 0.12/0.24, and
            *every* scaled stake landed under the lane's min-stake floor — so both
            lanes emptied out completely (see bets_2026-08-29.json). Truncation
            keeps the surviving bets at their intended size and honours the cap
            exactly. Each builder stamps a `_rank` (value: EV, conviction: conf,
            model: conf x ev_factor); a bet that doesn't fit the remaining room is
            skipped rather than ending the fill, so smaller bets further down can
            still use it up. Stakes are never scaled, and builders already reject
            sub-floor stakes, so no floor check is needed here.
            """
            cap = bankroll * cap_pct
            total = sum(b['stake_units'] for b in bets)
            action = None
            if cap > 0 and total > cap:
                ranked = sorted(bets, key=lambda b: b['_rank'], reverse=True)
                kept, room = [], cap
                for b in ranked:
                    if b['stake_units'] <= room:
                        kept.append(b)
                        room -= b['stake_units']
                action = f"{lane_name}-truncated {len(kept)}/{len(bets)}"
                # Restore the original (CSV) ordering for display.
                kept_ids = {id(b) for b in kept}
                bets = [b for b in bets if id(b) in kept_ids]
            for b in bets:
                b.pop('_rank', None)
            return bets, cap, action

        value_bets, value_cap, value_action     = _enforce_daily_cap(value_bets, value_br, value_cap_pct, 'value')
        conviction_bets, conv_cap, conv_action  = _enforce_daily_cap(conviction_bets, conv_br, conv_cap_pct, 'conviction')
        model_bets, model_cap, model_action     = _enforce_daily_cap(model_bets, model_br, model_cap_pct, 'model')

        value_total      = sum(b['stake_units'] for b in value_bets)
        conviction_total = sum(b['stake_units'] for b in conviction_bets)
        model_total      = sum(b['stake_units'] for b in model_bets)
        all_bets = value_bets + conviction_bets + model_bets
        total_stake = value_total + conviction_total + model_total

        return jsonify({
            'filename': os.path.basename(latest_file),
            'bankroll': round(value_br + conv_br + model_br, 2),
            'count': len(all_bets),
            'total_stake': round(total_stake, 2),
            'bets': all_bets,
            'lanes': {
                'value': {
                    'count': len(value_bets),
                    'total_stake': round(value_total, 2),
                    'bankroll': round(value_br, 2),
                    'bets': value_bets,
                },
                'conviction': {
                    'count': len(conviction_bets),
                    'total_stake': round(conviction_total, 2),
                    'bankroll': round(conv_br, 2),
                    'bets': conviction_bets,
                },
                'model': {
                    'count': len(model_bets),
                    'total_stake': round(model_total, 2),
                    'bankroll': round(model_br, 2),
                    'bets': model_bets,
                },
            },
            'guardrails': {
                'value': {
                    'min_confidence': min_confidence,
                    'stake_multiplier': stake_multiplier,
                    'max_stake_per_bet': round(max_value_per_bet, 2),
                    'daily_cap_pct': value_cap_pct,
                    'daily_cap_eur': round(value_cap, 2),
                    'cap_action': value_action,
                },
                'conviction': {
                    'min_confidence': conv_min_confidence,
                    'min_odds': conv_min_odds,
                    'flat_stake_eur': round(conv_flat_stake, 2),
                    'daily_cap_pct': conv_cap_pct,
                    'daily_cap_eur': round(conv_cap, 2),
                    'cap_action': conv_action,
                },
                'model': {
                    'base_pct': model_base_pct,
                    'max_stake_per_bet': round(max_model_per_bet, 2),
                    'ev_factor_range': [ev_factor_min, ev_factor_max],
                    'daily_cap_pct': model_cap_pct,
                    'daily_cap_eur': round(model_cap, 2),
                    'cap_action': model_action,
                },
                'shared': {
                    'min_stake_eur': min_stake_eur,
                },
            },
        })

    except Exception as e:
        return jsonify({'error': f"Internal Error: {str(e)}"})

@app.context_processor
def inject_bankroll():
    """
    Sport-aware navbar bankroll. On a sport page, show that sport's
    balance; on the landing page (or any sport-agnostic page), show the
    portfolio total. Templates also get `bankrolls` (per-sport map) so
    they can render a breakdown if they want.
    """
    bankrolls = all_bankrolls()
    bp = request.blueprint  # 'football', 'nba', or None
    if bp and bp in bankrolls:
        active = bankrolls[bp]
        label = bp.capitalize()
    else:
        active = total_bankroll()
        label = 'Total'
    return dict(bankroll=active, bankroll_label=label, bankrolls=bankrolls)


@app.context_processor
def inject_sports():
    """Make the SPORTS list available to every template (for navbar picker)."""
    return dict(sports=SPORTS)


# --- Sport-agnostic landing page ---
# Cashout backtest is a weekly LOCAL cadence (output/ is gitignored — a cloud
# agent sees nothing), so instead of a scheduler we surface a staleness reminder
# + the latest momentum_fade Δ (phase 8d) on the landing page. See
# FOOTBALL_NEXT_STEPS.md "Weekly backtest re-run".
_BACKTEST_STALE_DAYS = 7


def _latest_backtest_summary():
    """Read the newest output/backtests/*.json for the landing-page reminder.

    Returns None if no backtest has ever run, else a dict: last-run date,
    days-since + stale flag, sample counters, and momentum_fade Δ per lane
    (phase 8d's accruing signal). Best-effort — any read error → None."""
    bt_dir = os.path.join(OUTPUT_DIR, 'backtests')
    try:
        files = glob.glob(os.path.join(bt_dir, '*.json'))
        if not files:
            return None
        newest = max(files, key=os.path.getmtime)
        with open(newest) as f:
            data = json.load(f)
    except Exception:
        return None

    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(newest))
    days_since = (datetime.datetime.now() - mtime).days
    meta = data.get('meta', {})
    counters = meta.get('counters', {})
    agg = data.get('aggregate', {})

    # momentum_fade Δ per lane (phase 8d). Keys are "<rule>|<lane>".
    momentum = []
    for key, row in agg.items():
        if key.startswith('momentum_fade|'):
            momentum.append({
                'lane': key.split('|', 1)[1],
                'delta': row.get('delta', 0.0),
                'bets': row.get('bets', 0),
                'trigger_rate': row.get('trigger_rate', 0.0),
            })
    momentum.sort(key=lambda r: r['lane'])

    return {
        'last_run': mtime.strftime('%Y-%m-%d %H:%M'),
        'days_since': days_since,
        'stale': days_since >= _BACKTEST_STALE_DAYS,
        'window': f"{meta.get('start', '?')} → {meta.get('end', '?')}",
        'evaluated': counters.get('evaluated', 0),
        'real_used': counters.get('real_used', 0),
        'synth_used': counters.get('synth_used', 0),
        'momentum': momentum,
    }


@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """Launch the cashout backtest (scripts/run_backtest.py) server-side, same
    subprocess+TASKS pattern as Predict/Verify/Retrain. Self-service for the
    weekly LOCAL re-run nudged by the landing-page reminder. Interpreting the
    Δ (sign-stability, whether momentum_fade is ripe to tune — phase 8d) is a
    separate, human/Claude judgment step, not this button."""
    if TASKS.get('backtest') and TASKS['backtest'].get('process') and TASKS['backtest']['process'].poll() is None:
        flash('Backtest is already running!', 'warning')
        return redirect(url_for('landing'))
    try:
        script_path = os.path.join(PROJECT_ROOT, 'scripts', 'run_backtest.py')
        log_file = open(os.path.join(LOG_DIR, 'backtest.log'), 'w')
        env = os.environ.copy()
        ml_paths = [PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'ml_project')]
        env['PYTHONPATH'] = os.pathsep.join([p for p in ml_paths + [env.get('PYTHONPATH', '')] if p])
        # Default rules (includes momentum_fade) over the default 30-day window,
        # data=auto (real live_history trajectories where available).
        proc = subprocess.Popen(
            ['venv/bin/python', script_path], cwd=PROJECT_ROOT,
            stdout=log_file, stderr=subprocess.STDOUT, env=env)
        TASKS['backtest'] = {'process': proc, 'start_time': datetime.datetime.now()}
        flash('Cashout backtest started — refreshing when it finishes. '
              'Nudge me to evaluate the momentum_fade Δ once it lands.', 'info')
    except Exception as e:
        flash(f"Error starting backtest: {e}", 'danger')
    return redirect(url_for('landing'))


@app.route('/')
def landing():
    """Sport picker + portfolio summary. Active sports link to their dashboards.
    Per-sport stats are aggregated from each sport's bets directory (active +
    archived slips), so the portfolio totals stay correct after soft-deletes."""
    sport_summaries = {}
    for sport in SPORTS:
        bets_dir = sport.get('bets_dir')
        if bets_dir:
            sport_summaries[sport['slug']] = compute_sport_summary(bets_dir)['totals']
    return render_template('landing.html', sport_summaries=sport_summaries,
                           backtest=_latest_backtest_summary())


@app.route('/betting')
def betting_tabbed():
    """Sport-tabbed consolidated betting dashboard.

    Renders templates/betting_tabbed.html with four tabs (order matches
    landing-page card order: football, euroleague, nba):
      - **All sports**: cross-sport summary row per active sport (bankroll, bets,
        settled, stake, P/L, ROI) + a TOTAL footer.
      - **Football**: includes `_betting_football_panel.html` verbatim — the
        same partial /football/betting renders, with the same context vars
        (history, lane_stats, lane_defaults, sport_label). Football's full
        per-bet UI, lane-comparison table, and place-bets JS all work here
        unchanged.
      - **Euroleague**: placeholder card — onboarding in progress, roadmap
        in EUROLEAGUE_NEXT_STEPS.md.
      - **NBA**: placeholder card directing operators to /nba/ for the slim
        v1 NBA betting flow; full port into this tab is a Phase-3 follow-up
        tracked in FOOTBALL_NEXT_STEPS.md.

    The /football/betting route stays in place (renders the football-only
    shell) so deep-links keep working; the navbar's primary Betting Dashboard
    link points to this consolidated /betting page.

    Initial tab via ?tab=all|football|euroleague|nba (default 'all').
    """
    active_tab = (request.args.get('tab') or 'all').strip().lower()
    if active_tab not in ('all', 'football', 'nba', 'euroleague'):
        active_tab = 'all'

    # Per-sport summary + bankroll for the All tab + each sport panel's context.
    sport_rows = []
    totals_all = {'bets': 0, 'settled': 0, 'stake': 0.0, 'settled_stake': 0.0,
                  'returned': 0.0, 'pnl': 0.0}
    bank_by_sport = all_lane_bankrolls()
    total_bankroll = 0.0
    summary_by_slug = {}
    for sport in SPORTS:
        if not sport.get('bets_dir'):
            continue
        s_lane_br = bank_by_sport.get(sport['slug'], {})
        s_bankroll = round(sum(s_lane_br.values()), 2) if s_lane_br else 0.0
        total_bankroll += s_bankroll
        s_summary = compute_sport_summary(sport['bets_dir'])
        summary_by_slug[sport['slug']] = s_summary
        s_totals = s_summary['totals']
        for k in ('bets', 'settled', 'stake', 'settled_stake', 'returned', 'pnl'):
            totals_all[k] += s_totals.get(k, 0)
        sport_rows.append({
            'slug':     sport['slug'],
            'label':    sport['label'],
            'icon':     sport['icon'],
            'icon_img': sport.get('icon_img'),
            'active':   sport.get('active', False),
            'bankroll': s_bankroll,
            'totals':   s_totals,
        })
    # ROI as a percentage, matching compute_sport_summary's convention (the
    # template renders it directly, no extra ×100).
    totals_all['roi'] = (totals_all['pnl'] / totals_all['settled_stake'] * 100) if totals_all['settled_stake'] > 0 else 0.0

    # Football panel context — must match /football/betting exactly so the
    # included partial renders identically.
    summary = summary_by_slug.get('football', compute_sport_summary(OUTPUT_DIR))

    # Per-sport lane defaults (bankroll + daily-exposure cap) for the panels.
    def _lane_defaults_for(slug):
        c = get_sport_config(slug)
        lb = lane_bankrolls(slug)
        return {lane: {'bankroll': lb.get(lane, 0.0),
                       'cap_pct': c.get(f'{lane}_max_daily_exposure_pct', 0.0) * 100}
                for lane in ('value', 'conviction', 'model')}

    lane_defaults = _lane_defaults_for('football')

    return render_template(
        'betting_tabbed.html',
        active_tab=active_tab,
        sport_rows=sport_rows,
        totals_all=totals_all,
        total_bankroll=round(total_bankroll, 2),
        # Football tab (passed to the included partial)
        history=summary['history'],
        lane_stats=summary['lane_stats'],
        lane_defaults=lane_defaults,
        sport_label='Football',
        # NBA + Euroleague tab panels (Phase B — shared basketball panel)
        nba_summary=summary_by_slug.get('nba'),
        nba_lane_defaults=_lane_defaults_for('nba'),
        euroleague_summary=summary_by_slug.get('euroleague'),
        euroleague_lane_defaults=_lane_defaults_for('euroleague'),
        nba_bankrolls=bank_by_sport.get('nba', {}),
        euroleague_bankrolls=bank_by_sport.get('euroleague', {}),
    )


@app.route('/live')
def live_tabbed():
    """Sport-tabbed standalone Live page (parallels /betting).

    Renders templates/live_tabbed.html with three tabs (order matches the
    landing page: football, euroleague, nba):
      - **Football**: includes `_live_football_panel.html` — the in-play
        analysis panel moved off the football dashboard (live stats, Pre→Live
        1X2 + O/U adjustments, open-bet cashout, refresh / auto-10m /
        auto-cashout controls).
      - **EuroLeague / NBA**: placeholders — no in-play feed yet (in-play
        scraping is football-only). Each lights up by adding a per-sport
        live-data source + a `_live_<sport>_panel.html`.

    Initial tab via ?tab=football|euroleague|nba (default 'football').
    """
    active_tab = (request.args.get('tab') or 'football').strip().lower()
    if active_tab not in ('football', 'euroleague', 'nba'):
        active_tab = 'football'

    # Football is the only sport with a live feed today. Same read + enrich
    # path the football dashboard used before the panel was relocated here,
    # shared with /football/live_fragment so a swap matches a fresh load.
    return render_template(
        'live_tabbed.html',
        active_tab=active_tab,
        **_live_panel_context(),
    )


# Register sport blueprints after their routes have been defined.
app.register_blueprint(football_bp, url_prefix='/football')


def _auto_cashout_scheduler():
    """Autonomous background loop: while armed (`auto_cashout_armed.json`),
    refresh Flashscore then run the cashout sweep every
    `_AUTO_CASHOUT_INTERVAL_S` — independent of any browser tab, so the
    cashout actually EXECUTES rather than waiting on a manual user action.
    Daemon thread (started below). Each tick is guarded so one failure
    can't kill the loop. Virtual money only — no real bet is touched."""
    import time
    import logging
    log = logging.getLogger('auto_cashout')
    last_run = 0.0
    in_window = None            # None = unknown; log only on transitions
    while True:
        time.sleep(20)  # cheap when idle; responsive to arm/disarm
        try:
            if not _auto_cashout_armed():
                continue
            now = time.time()
            if now - last_run < _AUTO_CASHOUT_INTERVAL_S:
                continue
            # Don't scrape when nothing can be in play — a day's slate leaves
            # long dead stretches and Flashscore gains us nothing across them.
            active = _live_window_active()
            if active != in_window:
                log.warning('auto-cashout live window %s',
                            'OPEN — scraping resumes' if active else 'CLOSED — scraping paused')
                in_window = active
            if not active:
                continue
            last_run = now
            # Use a running scrape if one exists, else launch one; then wait.
            existing = (TASKS.get('live') or {}).get('process')
            proc = existing if (existing is not None and existing.poll() is None) \
                else _launch_live_refresh(with_bookmaker=False)
            if proc is not None:
                for _ in range(120):           # wait up to ~4 min for the scrape
                    if proc.poll() is not None:
                        break
                    time.sleep(2)
            res = _run_auto_cashout_sweep(VirtualBettingBackend(output_dir=OUTPUT_DIR))
            if res.get('cashed_count'):
                log.warning('auto-cashout fired: %s', res)
            elif res.get('would_fire_count'):
                log.warning('auto-cashout SHADOW (nothing fired): %s', res)
        except Exception as e:
            log.warning('auto-cashout tick failed: %s', e)


if __name__ == '__main__':
    # Auto-cashout is armed via the dashboard checkbox and persisted to disk.
    # Disarm REAL firing on every startup so money never moves from a stale
    # flag left over from a prior session (it must be a deliberate, in-session
    # opt-in). Shadow arming survives restarts on purpose: it fires nothing,
    # and silently dropping it on each restart is what stalled collection for
    # three months.
    if _auto_cashout_armed() and not _auto_cashout_shadow():
        _set_auto_cashout_armed(False)
    # Autonomous auto-cashout loop (executes server-side, no browser needed).
    import threading
    threading.Thread(target=_auto_cashout_scheduler, daemon=True).start()
    # Disable debug for performance
    app.run(debug=False, port=5001, host='0.0.0.0')
