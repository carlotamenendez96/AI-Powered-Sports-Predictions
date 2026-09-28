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

import pandas as pd
from flask import (
    Blueprint, current_app, flash, g, jsonify, redirect, render_template,
    request, url_for,
)

from betting_backend import EuroleagueBettingBackend, make_bet_id
# Shared with predict_euroleague.py so the displayed pick and the staked
# pick are computed by the same code path.
from ml_project.euroleague import euroleague_totals as el_totals
from sports_config import LANES, get_sport_config, lane_bankrolls, update_bankroll


euroleague_bp = Blueprint('euroleague', __name__)
EUROLEAGUE_TASKS = {}   # {'predict'|'verify'|'retrain': Popen} — checked by /status

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


def _recent_slips(limit: int = 5) -> list:
    files = sorted(glob.glob(os.path.join(_out_dir(), "bets_*.json")),
                   key=os.path.getctime, reverse=True)
    out = []
    for f in files[:limit]:
        try:
            with open(f) as fh:
                s = json.load(fh)
                out.append({
                    "date": s.get("date"),
                    "file": os.path.basename(f),
                    "count": s.get("count"),
                    "total_stake": s.get("total_stake"),
                    "status": s.get("status"),
                })
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@euroleague_bp.route('/')
def index():
    pred_path = _latest_predictions_path()
    df = _load_predictions(pred_path)
    bankrolls = lane_bankrolls('euroleague')
    total_bankroll = round(sum(bankrolls.values()), 2)
    return render_template(
        'euroleague/index.html',
        predictions=df.to_dict(orient='records') if not df.empty else [],
        pred_file=(os.path.basename(pred_path) if pred_path else None),
        bankrolls=bankrolls,
        total_bankroll=total_bankroll,
        recent_slips=_recent_slips(),
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

        def _ml_candidate(row) -> Optional[dict]:
            home, away = row.get('Home Team'), row.get('Away Team')
            if not home or not away:
                return None
            odds_row = odds_by_pair.get((home, away))
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
            home, away = row.get('Home Team'), row.get('Away Team')
            odds_row = odds_by_pair.get((home, away))
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

def _kick(task: str, script: str, args: list, success_msg: str) -> None:
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
        flash(success_msg, "success")
    except Exception as e:
        flash(f"Failed to start Euroleague {task}: {e}", "danger")


@euroleague_bp.route('/predict', methods=['POST'])
def predict():
    date = (request.form.get('date') or '').strip()
    args = [date] if date else []
    _kick('predict', 'run_euroleague_predictions.sh', args,
          f"Started Euroleague prediction pipeline ({date or 'tomorrow'}). Check logs.")
    return redirect(url_for('euroleague.index'))


@euroleague_bp.route('/verify', methods=['POST'])
def verify():
    date = (request.form.get('date') or '').strip()
    args = [date] if date else []
    _kick('verify', 'run_euroleague_verification.sh', args,
          f"Started Euroleague verification ({date or 'yesterday'}).")
    return redirect(url_for('euroleague.index'))


@euroleague_bp.route('/retrain', methods=['POST'])
def retrain():
    _kick('retrain', 'retrain_euroleague_pipeline.sh', [], "Started Euroleague retrain pipeline (full).")
    return redirect(url_for('euroleague.index'))
