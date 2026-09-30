"""Euroleague/EuroCup blueprint — Phase 3 v1 (predictions + moneyline paper betting).

Ports the NBA blueprint (``web_ui/nba/routes.py``) one-to-one — same moneyline-only
3-lane paper-betting flow, fully slug-separated storage (writes
``output_euroleague/bets_*.json``, debits ``sports.euroleague.bankrolls``). A
Euroleague bet can never touch a football or NBA bankroll/slip. Football code is
untouched (additive blueprint, same as NBA).

v1 surface
----------
* ``/euroleague/`` dashboard — latest predictions, bankrolls, recent slips, actions.
* ``/euroleague/auto_wager`` (GET JSON) — 3-lane moneyline slip preview.
* ``/euroleague/place_bets`` (POST JSON) — validates + debits per lane, writes the slip.
* ``/euroleague/{predict,verify,retrain}`` — trigger the bin scripts.

Season-gated (empty until then, by design — same as NBA waited on ESPN odds)
----------------------------------------------------------------------------
* **Odds → EV/Kelly**: ``auto_wager`` joins ``output_euroleague/euroleague_odds_<date>.json``
  (written by the future Flashscore Euroleague odds probe at season start). Until
  that file exists, games have no odds → slips come back empty. The dashboard +
  predictions + bin triggers all work now.
* **Totals (Over/Under)**: predictor emits a point estimate only, no P(Over) —
  same follow-up as NBA's totals market.
* **Cashout / void / live**: no basketball live feed (Phase-7 football-only).
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import subprocess
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from flask import (
    Blueprint, current_app, flash, g, jsonify, redirect, render_template,
    request, url_for,
)

from betting_backend import EuroleagueBettingBackend, make_bet_id
from sport_archive import archive_file
# Shared with predict_euroleague.py so the displayed pick and the staked
# pick are computed by the same code path.
from ml_project.euroleague import euroleague_totals as el_totals
from sports_config import LANES, get_sport_config, lane_bankrolls, update_bankroll


euroleague_bp = Blueprint('euroleague', __name__)
EUROLEAGUE_TASKS = {}   # {'predict'|'verify'|'retrain': Popen} — checked by /status
# Per-run metadata for the dashboard status bar: {'target_date', 'start_time'}.
# Kept beside EUROLEAGUE_TASKS (not inside it) so /status's shared Popen-dict
# loop over NBA + Euroleague keeps one shape.
EUROLEAGUE_TASK_META = {}

EUROLEAGUE_OUTPUT_DIR = 'output_euroleague'


# ---------------------------------------------------------------------------
# Path / helper utilities
# ---------------------------------------------------------------------------

def _project_root() -> str:
    return os.path.dirname(current_app.root_path)


def _out_dir() -> str:
    return os.path.join(_project_root(), EUROLEAGUE_OUTPUT_DIR)


@euroleague_bp.before_request
def _attach_backend():
    g.backend = EuroleagueBettingBackend(output_dir=EUROLEAGUE_OUTPUT_DIR)


def _latest_predictions_path() -> Optional[str]:
    files = sorted(glob.glob(os.path.join(_out_dir(), "predictions_euroleague_*.csv")),
                   key=os.path.getctime)
    return files[-1] if files else None


def _load_predictions(path: Optional[str]) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_odds_by_pair(date_str: str) -> dict:
    """Index Euroleague odds by (home_team, away_team). Empty until the
    season-start Flashscore odds probe writes euroleague_odds_<date>.json."""
    if not date_str:
        return {}
    path = os.path.join(_out_dir(), f"euroleague_odds_{date_str}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = json.load(f) or []
    return {(r.get("home_team"), r.get("away_team")): r for r in rows}


def _prediction_files(limit: int = 8) -> list:
    """[{filename, date, count}] newest first — same shape football's dashboard
    uses, so the two pages can render file lists identically.

    The dashboard lists these instead of the rows themselves: the full table
    lives behind /euroleague/view/<filename>, opened on demand.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(_out_dir(), "predictions_euroleague_*.csv")),
                       key=os.path.getmtime, reverse=True)[:limit]:
        name = os.path.basename(path)
        try:
            n = max(0, sum(1 for _ in open(path)) - 1)       # rows minus header
        except OSError:
            n = 0
        out.append({"filename": name,
                    "date": name.replace("predictions_euroleague_", "").replace(".csv", ""),
                    "count": n})
    return out


def _verification_files(limit: int = 8) -> list:
    """[{filename, date, count}] newest first, from the prediction-vs-result
    reports that bin/run_euroleague_verification.sh writes."""
    out = []
    for path in sorted(glob.glob(os.path.join(_out_dir(), "verification_euroleague_*.csv")),
                       key=os.path.getmtime, reverse=True)[:limit]:
        name = os.path.basename(path)
        try:
            df = pd.read_csv(path)
            n, hits = len(df), int(df.get('Winner Correct', pd.Series(dtype=int)).sum())
        except Exception:
            n, hits = 0, 0
        out.append({"filename": name,
                    "date": name.replace("verification_euroleague_", "").replace(".csv", ""),
                    "count": n, "hits": hits})
    return out


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@euroleague_bp.route('/')
def index():
    pred_path = _latest_predictions_path()
    bankrolls = lane_bankrolls('euroleague')
    total_bankroll = round(sum(bankrolls.values()), 2)
    return render_template(
        'euroleague/index.html',
        prediction_files=_prediction_files(),
        pred_file=(os.path.basename(pred_path) if pred_path else None),
        bankrolls=bankrolls,
        total_bankroll=total_bankroll,
        verification_files=_verification_files(),
    )


# ---------------------------------------------------------------------------
# Auto-wager (JSON) — moneyline slip generator (v1)
# ---------------------------------------------------------------------------

def _to_float(v) -> float:
    try:
        if isinstance(v, str):
            v = v.strip().rstrip('%')
            if not v:
                return 0.0
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _kelly(odd: float, prob: float) -> float:
    if odd <= 1.0 or prob <= 0.0 or prob >= 1.0:
        return 0.0
    b = odd - 1.0
    q = 1.0 - prob
    return max(0.0, ((b * prob - q) / b) * 0.25)


# The Euroleague feed's `startime` is Central European time (Panathinaikos'
# 21:15 Athens tip-off is listed as 20:15), not the machine's local zone.
_TIPOFF_TZ = ZoneInfo('Europe/Berlin')


def _now() -> datetime.datetime:
    return datetime.datetime.now(_TIPOFF_TZ)


def _tipoffs(date_str: str) -> dict:
    """{gameId: aware tip-off datetime} from fixtures_<date>.json. Games with
    no parseable time are left out, so callers fall back to the date."""
    path = os.path.join(_project_root(), 'data_sets', 'Euroleague', f'fixtures_{date_str}.json')
    try:
        with open(path) as f:
            rows = json.load(f) or []
    except (OSError, ValueError):
        return {}
    out = {}
    for r in rows:
        try:
            t = datetime.datetime.strptime(str(r.get('tipoff', '')), '%Y-%m-%d %H:%M')
        except ValueError:
            continue
        out[str(r.get('gameId'))] = t.replace(tzinfo=_TIPOFF_TZ)
    return out


def _has_started(date_str: str, game_id, tipoffs: dict, now: datetime.datetime) -> bool:
    """True once a game can no longer be bet: past its tip-off when known,
    otherwise once its date is before today."""
    t = tipoffs.get(str(game_id))
    if t is not None:
        return t <= now
    return bool(date_str) and date_str < now.date().isoformat()


def _available_prediction_dates() -> list:
    """Prediction dates the slip generator may bet: a predictions file exists,
    no live slip covers the date yet, and at least one game has not tipped off.

    Mirrors football's `_available_prediction_dates`. A slip counts as live if
    it holds a non-VOID bet, so a fully cancelled date is re-offered — but only
    while it still has games to bet; past slates never come back.
    """
    out = _out_dir()
    pred = {os.path.basename(p).replace('predictions_euroleague_', '').replace('.csv', '')
            for p in glob.glob(os.path.join(out, 'predictions_euroleague_*.csv'))}
    bet = set()
    # Active AND archived slips — an archived slip with real bets still means
    # the date was bet. Strip the optional `.<ts>` archive-collision suffix.
    for p in (glob.glob(os.path.join(out, 'bets_*.json'))
              + glob.glob(os.path.join(out, 'history', 'bets_*.json'))):
        stem = os.path.basename(p)[len('bets_'):-len('.json')].split('.', 1)[0]
        try:
            data = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            bet.add(stem)
            continue
        bets = data if isinstance(data, list) else data.get('bets', [])
        if any(str(b.get('status', '')).upper() != 'VOID' for b in bets):
            bet.add(stem)
    today = datetime.date.today().isoformat()
    now = _now()
    live = []
    for d in pred - bet:
        if d < today:
            continue
        tip = _tipoffs(d)
        # No fixture times on disk → date-level check only (d >= today).
        if tip and all(t <= now for t in tip.values()):
            continue
        live.append(d)
    return sorted(live, reverse=True)


@euroleague_bp.route('/predictions/available')
def predictions_available():
    """Dates the slip generator can bet (prediction file present, no slip yet)."""
    return jsonify({'dates': _available_prediction_dates()})


@euroleague_bp.route('/auto_wager')
def auto_wager():
    """JSON: 3-lane Euroleague moneyline slip preview (parity with NBA's)."""
    try:
        date_arg = (request.args.get('date') or '').strip()
        if date_arg:
            pred_path = os.path.join(_out_dir(), f"predictions_euroleague_{date_arg}.csv")
            if not os.path.isfile(pred_path):
                return jsonify({'error': f"No Euroleague predictions for {date_arg}."}), 404
        else:
            pred_path = _latest_predictions_path()
            if not pred_path:
                return jsonify({'error': "No Euroleague prediction files found."}), 404

        df = _load_predictions(pred_path)
        if df.empty:
            return jsonify({'error': f"Predictions file is empty: {os.path.basename(pred_path)}."}), 400

        target_date = str(df['Date'].iloc[0]) if 'Date' in df.columns else None

        # Never offer a game that has already tipped off (a cancelled past
        # slip used to re-open yesterday's slate for betting).
        tipoffs, now = _tipoffs(target_date), _now()
        started = df.apply(lambda r: _has_started(target_date, r.get('gameId'), tipoffs, now), axis=1)
        if started.all():
            return jsonify({'error': f"All Euroleague games on {target_date} have already tipped off."}), 400
        df = df[~started]
        odds_by_pair = _load_odds_by_pair(target_date)

        config = get_sport_config('euroleague')
        lane_br = lane_bankrolls('euroleague')
        if sum(lane_br.values()) < 1.0:
            return jsonify({
                'error': "Euroleague bankrolls are zero — fund them in betting_config.json "
                         "(sports.euroleague.bankrolls.<lane>) before generating slips."
            }), 400

        min_stake_eur    = config['min_stake_eur']
        min_confidence   = config['min_confidence']
        stake_multiplier = config['stake_multiplier']
        max_stake_pct    = config['max_stake_pct']
        ev_cap_value     = config['ev_cap_value']
        conv_min_conf    = config['conviction_min_confidence']
        conv_min_odds    = config['conviction_min_odds']
        conv_stake_pct   = config['conviction_stake_pct']
        model_base_pct   = config['model_base_pct']
        model_max_pct    = config['model_max_stake_pct']
        model_min_stake  = config['model_min_stake_eur']
        ev_factor_min    = config['model_ev_factor_min']
        ev_factor_max    = config['model_ev_factor_max']

        def _override_br(lane: str, default: float) -> float:
            raw = request.args.get(f'bankroll_{lane}')
            if not raw:
                return default
            try:
                v = float(raw)
            except ValueError:
                return default
            if v <= 0 or v > default:
                raise ValueError(f"{lane} session bankroll must be in (0, {default:.2f}]")
            return v

        def _override_cap(lane: str, default: float) -> float:
            raw = request.args.get(f'cap_{lane}')
            if not raw:
                return default
            try:
                v = float(raw)
            except ValueError:
                return default
            return v if 0 < v <= 1.0 else default

        try:
            value_br = _override_br('value',      lane_br['value'])
            conv_br  = _override_br('conviction', lane_br['conviction'])
            model_br = _override_br('model',      lane_br['model'])
        except ValueError as ve:
            return jsonify({'error': str(ve)}), 400

        value_cap_pct = _override_cap('value',      config['value_max_daily_exposure_pct'])
        conv_cap_pct  = _override_cap('conviction', config['conviction_max_daily_exposure_pct'])
        model_cap_pct = _override_cap('model',      config['model_max_daily_exposure_pct'])

        max_value_per_bet = value_br * max_stake_pct
        max_model_per_bet = model_br * model_max_pct
        conv_flat_stake   = conv_br * conv_stake_pct

        def _disp(row):
            """Short display names ('Panathinaikos') over the API's
            sponsor-laden ones. Applied to home/away/selection TOGETHER: the
            resolver settles a moneyline by comparing `selection` to the bet's
            own `home`/`away`, so they must agree with each other. The odds
            join still keys on the canonical `Home Team`/`Away Team`."""
            return (row.get('Home Short') or row.get('Home Team'),
                    row.get('Away Short') or row.get('Away Team'))

        def _ml_candidate(row) -> Optional[dict]:
            home, away = _disp(row)
            if not home or not away:
                return None
            odds_row = odds_by_pair.get((row.get('Home Team'), row.get('Away Team')))
            if not odds_row:
                return None  # no odds → skip (season-gated until the odds probe lands)
            p_home = _to_float(row.get('Home Win Prob'))
            predicted = (row.get('Predicted Winner') or '').upper()
            if predicted == 'HOME':
                conf, odds_dec, selection = p_home, odds_row.get('home_ml_decimal'), home
            elif predicted == 'AWAY':
                conf, odds_dec, selection = 1.0 - p_home, odds_row.get('away_ml_decimal'), away
            else:
                return None
            if odds_dec in (None, 0):
                return None
            odds_dec = float(odds_dec)
            ev = conf * odds_dec - 1.0
            return {
                'date': target_date, 'match': f"{home} vs {away}",
                'home': home, 'away': away, 'match_id': row.get('gameId') or '',
                'type': 'ML', 'selection': selection,
                'odds': round(odds_dec, 3), 'odd': round(odds_dec, 3),
                'conf': f"{conf:.3f}", 'ev': f"{ev:+.3f}", 'kelly': f"{_kelly(odds_dec, conf):.2%}",
                'status': 'OPEN',
                '_conf': conf, '_odds': odds_dec, '_ev': ev,
            }

        def _total_candidate(row) -> Optional[dict]:
            """O/U candidate chosen from the whole TOTALS LADDER, not one line.

            Each book posts its own total and its own prices, so the bet is a
            (line, side, price) triple that must come from a single row. The
            selection is `euroleague_totals.best_ev` — the same helper the
            predictor displays from, so the dashboard and the slip can never
            disagree — and every bet also carries `counterfactual`: the
            main-line bet we did NOT place. That pairing is the whole point;
            without it a losing month tells us nothing about whether
            ladder-shopping or the model was at fault.
            """
            home, away = _disp(row)
            odds_row = odds_by_pair.get((row.get('Home Team'), row.get('Away Team')))
            if not odds_row:
                return None
            # Training-contract gate (see predict_euroleague.py). The model is
            # never trained on games with NaN L10/venue features -- the trainer
            # dropna's them -- and serving those was the whole EuroCup totals
            # bias (+8.63 vs market at t=4.13; +4.13 at t=1.11 once gated).
            # Checked here too, not just in the predictor, because auto_wager
            # can be pointed at any CSV on disk. A CSV predating the column has
            # no flag and is allowed through, same as before it existed.
            if str(row.get('Totals Eligible', '1')).strip() in ('0', '0.0', 'False'):
                return None
            pred_total, sigma = _to_float(row.get('Predicted Total')), _to_float(row.get('Total Sigma'))
            ladder = odds_row.get('totals') or []
            if not pred_total or not sigma or not ladder:
                return None
            pick = el_totals.best_ev(pred_total, sigma, ladder)
            if not pick:
                return None
            cf = el_totals.counterfactual(pred_total, sigma, ladder)
            conf, odds_dec = pick['prob'], pick['odds']
            selection = f"{pick['side']} {pick['line']}"
            ev = pick['ev']
            return {
                'date': target_date, 'match': f"{home} vs {away}",
                'home': home, 'away': away, 'match_id': row.get('gameId') or '',
                'type': 'O/U', 'selection': selection,
                'odds': round(odds_dec, 3), 'odd': round(odds_dec, 3),
                'conf': f"{conf:.3f}", 'ev': f"{ev:+.3f}", 'kelly': f"{_kelly(odds_dec, conf):.2%}",
                'status': 'OPEN',
                'book': pick['book'], 'line': pick['line'], 'side': pick['side'],
                'ladder_size': len(ladder),
                # The main-line bet we did NOT place, settled later against the
                # same final score to A/B ladder-shopping vs the consensus line.
                'counterfactual': ({'line': cf['line'], 'side': cf['side'],
                                    'odds': round(cf['odds'], 3),
                                    'prob': round(cf['prob'], 4),
                                    'ev': round(cf['ev'], 4), 'book': cf['book']}
                                   if cf else None),
                '_conf': conf, '_odds': odds_dec, '_ev': ev,
            }

        def _build_value(c: dict) -> Optional[dict]:
            if c['_ev'] <= 0 or c['_conf'] < min_confidence:
                return None
            stake = min(value_br * min(c['_ev'], ev_cap_value) * c['_conf'] * stake_multiplier, max_value_per_bet)
            if stake < min_stake_eur:
                return None
            b = {k: v for k, v in c.items() if not k.startswith('_')}
            b.update({'lane': 'value', 'stake_units': round(stake, 2), 'stake': round(stake, 2)})
            return b

        def _build_conviction(c: dict) -> Optional[dict]:
            if c['_conf'] < conv_min_conf or c['_odds'] < conv_min_odds:
                return None
            stake = conv_flat_stake
            if stake < min_stake_eur:
                return None
            b = {k: v for k, v in c.items() if not k.startswith('_')}
            b.update({'lane': 'conviction', 'stake_units': round(stake, 2), 'stake': round(stake, 2)})
            return b

        def _build_model(c: dict) -> Optional[dict]:
            if c['_conf'] <= 0 or c['_odds'] <= 1.0:
                return None
            ev_factor = max(ev_factor_min, min(ev_factor_max, c['_conf'] * c['_odds']))
            stake = min(model_br * model_base_pct * c['_conf'] * ev_factor, max_model_per_bet)
            if stake < model_min_stake:
                return None
            b = {k: v for k, v in c.items() if not k.startswith('_')}
            b.update({'lane': 'model', 'stake_units': round(stake, 2), 'stake': round(stake, 2)})
            return b

        value_bets, conviction_bets, model_bets = [], [], []
        no_odds = 0
        for _, row in df.iterrows():
            cands = [c for c in (_ml_candidate(row), _total_candidate(row)) if c]
            if not cands:
                if odds_by_pair.get((row.get('Home Team'), row.get('Away Team'))) is None:
                    no_odds += 1
                continue
            for c in cands:
                vb = _build_value(c)
                if vb: value_bets.append(vb)
                cb = _build_conviction(c)
                if cb: conviction_bets.append(cb)
                mb = _build_model(c)
                if mb: model_bets.append(mb)

        def _cap(bets, br, cap_pct, floor):
            cap = br * cap_pct
            total = sum(b['stake_units'] for b in bets)
            scaled = None
            if cap > 0 and total > cap:
                scale = cap / total
                for b in bets:
                    b['stake_units'] = round(b['stake_units'] * scale, 2)
                    b['stake'] = b['stake_units']
                bets = [b for b in bets if b['stake_units'] >= floor]
                scaled = True
            return bets, cap, scaled

        value_bets, value_cap, value_scaled    = _cap(value_bets,      value_br, value_cap_pct, min_stake_eur)
        conviction_bets, conv_cap, conv_scaled = _cap(conviction_bets, conv_br,  conv_cap_pct,  min_stake_eur)
        model_bets, model_cap, model_scaled    = _cap(model_bets,      model_br, model_cap_pct, model_min_stake)

        all_bets = value_bets + conviction_bets + model_bets
        return jsonify({
            'date': target_date,
            'pred_file': os.path.basename(pred_path),
            'odds_present': len(odds_by_pair),
            'odds_missing_for_games': no_odds,
            'lanes': {
                'value':      {'count': len(value_bets),      'total_stake': round(sum(b['stake_units'] for b in value_bets), 2),
                               'bankroll': value_br, 'cap': round(value_cap, 2), 'scaled': bool(value_scaled)},
                'conviction': {'count': len(conviction_bets), 'total_stake': round(sum(b['stake_units'] for b in conviction_bets), 2),
                               'bankroll': conv_br,  'cap': round(conv_cap, 2),  'scaled': bool(conv_scaled)},
                'model':      {'count': len(model_bets),      'total_stake': round(sum(b['stake_units'] for b in model_bets), 2),
                               'bankroll': model_br, 'cap': round(model_cap, 2), 'scaled': bool(model_scaled)},
            },
            'bets': all_bets,
            'total_stake': round(sum(b['stake_units'] for b in all_bets), 2),
        })
    except Exception as e:
        return jsonify({'error': f"Internal Error: {e}"}), 500


# ---------------------------------------------------------------------------
# Place bets (JSON POST) — debits bankrolls + writes the slip
# ---------------------------------------------------------------------------

@euroleague_bp.route('/place_bets', methods=['POST'])
def place_bets():
    try:
        data = request.get_json(force=True) or {}
        bets = data.get('bets') or []
        if not bets:
            return jsonify({'error': "No bets provided."}), 400

        date_str = None
        first = bets[0].get('date', '')
        if first:
            try:
                date_str = str(first).split(' ')[0]
            except Exception:
                pass
        if not date_str:
            date_str = data.get('date') or datetime.date.today().strftime('%Y-%m-%d')

        # Server-side guard: the preview may be stale (page left open past
        # tip-off), so re-check every bet rather than trusting the client.
        tipoffs, now = _tipoffs(date_str), _now()
        late = [b.get('match', b.get('match_id', '?')) for b in bets
                if _has_started(str(b.get('date') or date_str).split(' ')[0],
                                b.get('match_id'), tipoffs, now)]
        if late:
            return jsonify({'error': "These games have already tipped off — regenerate the slip: "
                                     + ", ".join(sorted(set(late)))}), 400

        stake_by_lane = {lane: 0.0 for lane in LANES}
        for b in bets:
            lane = b.get('lane', 'value')
            if lane not in LANES:
                lane = 'value'
                b['lane'] = lane
            stake_by_lane[lane] += float(b.get('stake_units', 0))
            if not b.get('bet_id'):
                b['bet_id'] = make_bet_id(
                    date_str,
                    b.get('home') or (b.get('match', '').split(' vs ')[0] if ' vs ' in b.get('match', '') else ''),
                    b.get('away') or (b.get('match', '').split(' vs ')[1] if ' vs ' in b.get('match', '') else ''),
                    b.get('type', 'ML'),
                    b.get('selection', ''),
                )
            b.setdefault('mode', 'virtual')

        # Never overwrite a slip that still holds real bets — the old write
        # replaced it wholesale, orphaning its stakes from the record.
        _existing = os.path.join(_out_dir(), f"bets_{date_str}.json")
        if os.path.exists(_existing):
            try:
                with open(_existing) as f:
                    _prev = json.load(f)
                _prev_bets = _prev if isinstance(_prev, list) else _prev.get('bets', [])
            except (OSError, ValueError):
                _prev_bets = [{}]   # unreadable: be conservative
            if any(str(b.get('status', '')).upper() != 'VOID' for b in _prev_bets):
                return jsonify({'error': f"A slip for {date_str} already exists "
                                         f"(bets_{date_str}.json). Cancel or settle it first."}), 409

        current = lane_bankrolls('euroleague')
        for lane, stake in stake_by_lane.items():
            if stake > current[lane] + 1e-6:
                return jsonify({
                    'error': f"Insufficient Euroleague {lane} funds. Stake ({stake:.2f}) > "
                             f"bankroll ({current[lane]:.2f})."
                }), 400

        new_br = dict(current)
        for lane, stake in stake_by_lane.items():
            if stake > 0:
                new_br[lane] = update_bankroll('euroleague', -stake, lane=lane)

        total_stake = sum(stake_by_lane.values())
        filepath = os.path.join(_out_dir(), f"bets_{date_str}.json")
        os.makedirs(_out_dir(), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                'date': date_str,
                'count': len(bets),
                'bets': bets,
                'total_stake': round(total_stake, 2),
                'stake_by_lane': {k: round(v, 2) for k, v in stake_by_lane.items()},
                'status': 'OPEN',
                'pnl': 0.0,
                'settled': False,
            }, f, indent=4)

        return jsonify({
            'message': f"Placed {len(bets)} Euroleague virtual bets — debited {total_stake:.2f} across lanes.",
            'file': os.path.basename(filepath),
            'new_balance': round(sum(new_br.values()), 2),
            'lane_bankrolls': {k: round(v, 2) for k, v in new_br.items()},
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Bin-script task triggers
# ---------------------------------------------------------------------------

def _kick(task: str, script: str, args: list, success_msg: str,
          target_date: Optional[str] = None) -> None:
    if EUROLEAGUE_TASKS.get(task) and EUROLEAGUE_TASKS[task].poll() is None:
        flash(f"Euroleague {task} is already running.", "warning")
        return
    project_root = _project_root()
    script_path = os.path.join(project_root, 'bin', script)
    os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
    log_path = os.path.join(project_root, 'logs', f"euroleague_{task}.log")
    try:
        log_f = open(log_path, 'w')
        proc = subprocess.Popen(['/bin/bash', script_path, *args], cwd=project_root,
                                stdout=log_f, stderr=subprocess.STDOUT)
        EUROLEAGUE_TASKS[task] = proc
        EUROLEAGUE_TASK_META[task] = {'target_date': target_date,
                                      'start_time': datetime.datetime.now()}
        flash(success_msg, "success")
    except Exception as e:
        flash(f"Failed to start Euroleague {task}: {e}", "danger")


@euroleague_bp.route('/stop/<task>', methods=['POST'])
def stop_task(task):
    proc = EUROLEAGUE_TASKS.get(task)
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
            # Drop the handle so /status reports idle rather than a
            # signal-exit 'error' for a run the user stopped on purpose.
            EUROLEAGUE_TASKS[task] = None
            flash(f"Euroleague {task} stopped.", "warning")
        except Exception as e:
            flash(f"Error stopping Euroleague {task}: {e}", "danger")
    else:
        flash(f"No running Euroleague {task} task found.", "secondary")
    return redirect(url_for('euroleague.index'))


@euroleague_bp.route('/predict', methods=['POST'])
def predict():
    date = (request.form.get('date') or '').strip()
    args = [date] if date else []
    target = date or (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    _kick('predict', 'run_euroleague_predictions.sh', args,
          f"Started Euroleague prediction pipeline ({date or 'tomorrow'}). Check logs.",
          target_date=target)
    return redirect(url_for('euroleague.index'))


@euroleague_bp.route('/verify', methods=['POST'])
def verify():
    date = (request.form.get('date') or '').strip()
    args = [date] if date else []
    target = date or (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    _kick('verify', 'run_euroleague_verification.sh', args,
          f"Started Euroleague verification ({date or 'yesterday'}).",
          target_date=target)
    return redirect(url_for('euroleague.index'))


@euroleague_bp.route('/retrain', methods=['POST'])
def retrain():
    _kick('retrain', 'retrain_euroleague_pipeline.sh', [], "Started Euroleague retrain pipeline (full).")
    return redirect(url_for('euroleague.index'))


def _find_bet(bet_id: str):
    """Locate one bet by bet_id across this sport's slips. Football's
    equivalent also resolves a live match; basketball has no in-play feed, so
    this is the plain lookup."""
    for path in sorted(glob.glob(os.path.join(_out_dir(), 'bets_*.json')), reverse=True):
        try:
            slip = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        for bet in (slip.get('bets') or []):
            if bet.get('bet_id') == bet_id:
                return bet
    return None


@euroleague_bp.route('/void_bet/<bet_id>', methods=['POST'])
def void_bet(bet_id):
    """Mark an OPEN bet VOID — postponed / cancelled games that will never
    settle. The backend cascades across every lane holding the same bet_id and
    refunds each lane's stake, exactly as football's does."""
    if '/' in bet_id or '..' in bet_id:
        flash('Invalid bet_id.', 'danger')
        return redirect(request.referrer or url_for('euroleague.index'))
    bet = _find_bet(bet_id)
    if bet is None:
        flash(f'Bet not found: {bet_id}.', 'warning')
        return redirect(request.referrer or url_for('euroleague.index'))
    if not g.backend.void_bet(bet):
        flash('No OPEN bets found to void (sibling lanes may already be settled).', 'info')
        return redirect(request.referrer or url_for('euroleague.index'))

    refund, lanes = 0.0, set()
    slip_date = bet_id.split(':', 1)[0] if ':' in bet_id else ''
    path = os.path.join(_out_dir(), f'bets_{slip_date}.json')
    if os.path.exists(path):
        try:
            for b in (json.load(open(path)).get('bets') or []):
                if b.get('bet_id') == bet_id and b.get('status') == 'VOID':
                    refund += float(b.get('stake_units', 0) or 0)
                    lanes.add(b.get('lane', 'value'))
        except (json.JSONDecodeError, OSError):
            pass
    flash(f"Voided across {len(lanes)} lane(s) ({', '.join(sorted(lanes)) or 'unknown'}); "
          f"€{refund:.2f} refunded.", 'success')
    return redirect(request.referrer or url_for('euroleague.index'))


@euroleague_bp.route('/cancel_slip/<date>', methods=['POST'])
def cancel_slip(date):
    """Cancel a whole slip while every bet on it is still OPEN — refunds each
    lane and closes it. Virtual money only."""
    if '/' in date or '..' in date or len(date) != 10:
        flash('Invalid slip date.', 'danger')
        return redirect(request.referrer or url_for('euroleague.index'))
    ok, message = g.backend.cancel_slip(date)
    if not ok:
        flash(f'Could not cancel slip {date}: {message}', 'warning')
        return redirect(request.referrer or url_for('euroleague.index'))

    # Archive so a cancelled slip stops cluttering the history, mirroring
    # football. Non-fatal: the refund already happened and is what matters.
    # Collision-safe: a plain os.replace here once overwrote an already
    # archived, SETTLED slip for the same date with this cancelled one.
    ok_arch, arch_msg = archive_file(_out_dir(), f'bets_{date}.json', ('bets_*.json',))
    if ok_arch:
        flash(f'Slip {date} cancelled and archived. {message}', 'success')
    else:
        flash(f'Slip {date} cancelled ({message}), but archiving failed: {arch_msg}', 'warning')
    return redirect(request.referrer or url_for('euroleague.index'))


@euroleague_bp.route('/archive/<filename>', methods=['POST'])
def archive(filename):
    """Soft-delete a predictions CSV or a CLOSED bet slip to history/."""
    ok, message = archive_file(_out_dir(), filename,
                               ('predictions_euroleague_*.csv', 'verification_euroleague_*.csv',
                                'bets_*.json'))
    flash(message, 'success' if ok else 'warning')
    return redirect(request.referrer or url_for('euroleague.index'))


_ARCHIVE_ALL_PATTERNS = {
    'predictions':   'predictions_euroleague_*.csv',
    'verifications': 'verification_euroleague_*.csv',
}


@euroleague_bp.route('/archive_all/<kind>', methods=['POST'])
def archive_all(kind):
    """Archive every predictions or verification report (football parity)."""
    pattern = _ARCHIVE_ALL_PATTERNS.get(kind)
    if not pattern:
        flash(f'Unknown archive kind: {kind}', 'danger')
        return redirect(url_for('euroleague.index'))
    done, failed = 0, []
    for path in glob.glob(os.path.join(_out_dir(), pattern)):
        ok, msg = archive_file(_out_dir(), os.path.basename(path), (pattern,))
        if ok:
            done += 1
        else:
            failed.append(msg)
    flash(f'Archived {done} {kind} file(s) to history/.'
          + (f' {len(failed)} failed: ' + '; '.join(failed) if failed else ''),
          'success' if not failed else 'warning')
    return redirect(url_for('euroleague.index'))


@euroleague_bp.route('/view/<filename>')
def view_file(filename):
    """Render one predictions CSV. The dashboard links here rather than
    inlining the table (football does the same via /football/view/<f>)."""
    safe = os.path.basename(filename)                      # no path traversal
    path = os.path.join(_out_dir(), safe)
    is_verif = safe.startswith('verification_euroleague_')
    if not (safe.startswith('predictions_euroleague_') or is_verif) or not os.path.exists(path):
        flash('File not found.', 'danger')
        return redirect(url_for('euroleague.index'))
    df = _load_predictions(path).fillna('')
    if is_verif:
        return render_template('euroleague/verification.html',
                               filename=safe, rows=df.to_dict(orient='records'))
    return render_template('euroleague/view.html',
                           filename=safe,
                           rows=df.to_dict(orient='records'))
