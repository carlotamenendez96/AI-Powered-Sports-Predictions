"""E4 — GBDT x Dixon-Coles hybrid (Karlis-Ntzoufras).

NOT an attempt to beat the market at 1X2. D3 already refuted that premise: a
single bivariate Poisson reproduces the production 1X2 and O/U heads to a
joint-fit residual of 0.011 RMS with 0% of matches above 5pp, so the heads are
already coherent and there is no incoherence to fix. E0/E1/E3/E5 then closed the
edge question from four more directions.

What a scoreline model buys that nothing in this repo currently has is
DERIVATIVE MARKETS. The production stack emits 1X2 and Over/Under 2.5 and
nothing else; it structurally cannot price Asian handicap, BTTS or correct
score, because those need a joint distribution over (home goals, away goals)
rather than two independent heads. This experiment builds that distribution and
asks whether the prices coming out of it are good enough to trade.

METHOD (Karlis-Ntzoufras hybrid)
    1. Two XGBoost Poisson regressors predict the rates lambda_home (target
       FTHG) and lambda_away (target FTAG) from the full production feature set
       — so unlike native Dixon-Coles this keeps ELO, form, xG and the odds.
    2. A Dixon-Coles low-score correction rho is fitted by MLE on each training
       fold, perturbing the (0,0), (1,0), (0,1), (1,1) cells where independent
       Poisson is known to misprice.
    3. The resulting scoreline matrix yields every market at once, mutually
       consistent by construction.

SCORING, per market, against whatever benchmark exists:
    1X2    vs the production multiclass head AND the market. This is a GUARD,
           not a target: the gate is that 1X2 RPS must not regress more than
           0.5%. A hybrid that prices AH well but wrecks 1X2 is not shippable.
    O/U    vs the production Poisson head AND the market.
    AH     vs the real Asian-handicap market. Restricted to HALF-LINES
           (|AHh| mod 1 == 0.5), where settlement is a clean Bernoulli — integer
           lines can push and quarter lines split the stake, and neither has an
           unambiguous binary outcome to score a probability against. Dropping
           pushes from integer lines instead would condition on the outcome.
    BTTS   calibration only. football-data carries no BTTS prices, so there is
           no market to beat — the question is whether the number is honest.
    CS     correct score: log-loss and top-1 hit rate over the 11x11 grid.

The vectorised scoreline builder is verified against
`ml_project/dixon_coles/dc_scoreline.scoreline_matrix` on every run, so the fast
path cannot silently drift from the reference implementation.

Writes output/experiments/dc_hybrid_<ts>.{json,txt}. Never writes into models/.

Usage:
    python3 scripts/experiment_dc_hybrid.py
    python3 scripts/experiment_dc_hybrid.py --splits 5 --max-goals 10
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_project'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from experiment_odds_free import prepare, add_1x2_market, stack_test   # noqa: E402
from experiment_h2h import arm_features, feats_required                # noqa: E402
from model_registry import get_spec                                    # noqa: E402
from dixon_coles.dc_scoreline import scoreline_matrix                  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'experiments')
K = 3
RHO_BOUNDS = (-0.25, 0.25)
GATE_1X2_REGRESSION = 0.005          # 1X2 RPS may not get >0.5% worse


# --------------------------------------------------------------------------- #
# vectorised scoreline
# --------------------------------------------------------------------------- #
def poisson_pmf_grid(lam, max_goals):
    """(n, max_goals+1) Poisson pmf, computed in log space."""
    ks = np.arange(max_goals + 1)
    lam = np.clip(np.asarray(lam, dtype=float), 1e-6, None)[:, None]
    return np.exp(-lam + ks * np.log(lam) - gammaln(ks + 1))


def scoreline_grid(lam_h, lam_a, rho=0.0, max_goals=10):
    """(n, G+1, G+1) joint P(home=i, away=j), Dixon-Coles corrected."""
    ph = poisson_pmf_grid(lam_h, max_goals)
    pa = poisson_pmf_grid(lam_a, max_goals)
    mat = ph[:, :, None] * pa[:, None, :]
    if rho != 0.0:
        lh = np.clip(np.asarray(lam_h, float), 1e-6, None)
        la = np.clip(np.asarray(lam_a, float), 1e-6, None)
        mat[:, 0, 0] *= 1.0 - lh * la * rho
        mat[:, 0, 1] *= 1.0 + lh * rho
        mat[:, 1, 0] *= 1.0 + la * rho
        mat[:, 1, 1] *= 1.0 - rho
    mat = np.clip(mat, 1e-15, None)
    return mat / mat.sum(axis=(1, 2), keepdims=True)


def markets_from_grid(mat, ou_line=2.5):
    G = mat.shape[1]
    i = np.arange(G)
    diff = i[:, None] - i[None, :]
    tot = i[:, None] + i[None, :]
    flat = mat.reshape(len(mat), -1)
    return {
        'home': flat[:, (diff > 0).ravel()].sum(1),
        'draw': flat[:, (diff == 0).ravel()].sum(1),
        'away': flat[:, (diff < 0).ravel()].sum(1),
        'over': flat[:, (tot > ou_line).ravel()].sum(1),
        'btts': mat[:, 1:, 1:].sum(axis=(1, 2)),
    }


def ah_home_cover(mat, lines):
    """P(home covers) at a per-row Asian handicap line (home handicap `h`).

    Home covers when (home_goals - away_goals) + h > 0. Only called on
    half-lines, where that inequality can never be an equality, so there is no
    push mass to allocate.
    """
    G = mat.shape[1]
    i = np.arange(G)
    diff = (i[:, None] - i[None, :]).ravel()
    flat = mat.reshape(len(mat), -1)
    out = np.empty(len(mat))
    for u in np.unique(lines):
        sel = lines == u
        out[sel] = flat[sel][:, diff + u > 0].sum(1)
    return out


# --------------------------------------------------------------------------- #
# rho by MLE
# --------------------------------------------------------------------------- #
def fit_rho(lam_h, lam_a, hg, ag):
    """Dixon-Coles rho by maximum likelihood on the observed scorelines."""
    lh = np.clip(lam_h, 1e-6, None)
    la = np.clip(lam_a, 1e-6, None)
    base = (-lh + hg * np.log(lh) - gammaln(hg + 1)
            - la + ag * np.log(la) - gammaln(ag + 1))
    c00 = (hg == 0) & (ag == 0)
    c01 = (hg == 0) & (ag == 1)
    c10 = (hg == 1) & (ag == 0)
    c11 = (hg == 1) & (ag == 1)

    def nll(rho):
        tau = np.ones_like(lh)
        tau[c00] = 1.0 - lh[c00] * la[c00] * rho
        tau[c01] = 1.0 + lh[c01] * rho
        tau[c10] = 1.0 + la[c10] * rho
        tau[c11] = 1.0 - rho
        if np.any(tau <= 0):
            return 1e12
        return -float((base + np.log(tau)).sum())

    res = minimize_scalar(nll, bounds=RHO_BOUNDS, method='bounded',
                          options={'xatol': 1e-5})
    return float(res.x) if res.success else 0.0


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def rps(P, y):
    Y = np.eye(K)[y]
    return float((((np.cumsum(P, 1)[:, :-1] - np.cumsum(Y, 1)[:, :-1]) ** 2).sum(1).mean())
                 / (K - 1))


def brier3(P, y):
    return float(((P - np.eye(K)[y]) ** 2).sum(1).mean())


def mll3(P, y):
    return float(-np.log(np.clip(P[np.arange(len(P)), y], 1e-9, 1)).mean())


def bin_brier(p, y):
    return float(((p - y) ** 2).mean())


def bin_ll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 30:
            out.append({'bin': f'{lo:.1f}-{hi:.1f}', 'n': int(m.sum()),
                        'pred': float(p[m].mean()), 'actual': float(y[m].mean())})
    return out


def devig2(a, b):
    ia, ib = 1.0 / a, 1.0 / b
    return ia / (ia + ib)


# --------------------------------------------------------------------------- #
def build(df, splits, max_goals):
    import xgboost as xgb
    spec = get_spec('1x2', 'xgboost')
    feats = arm_features('1x2', 'base', spec)
    dense = feats_required(feats)
    need = dense + ['target_1x2', 'mkt_H', 'mkt_D', 'mkt_A', 'FTHG', 'FTAG']
    d = df.dropna(subset=need).copy().sort_values('date').reset_index(drop=True)
    d['league_cat'] = d['league_cat'].astype('category')

    y = d['target_1x2'].values.astype(int)
    hg = d['FTHG'].values.astype(int)
    ag = d['FTAG'].values.astype(int)
    mkt = d[['mkt_H', 'mkt_D', 'mkt_A']].values
    X = d[feats]

    rep = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'), 'rows': int(len(d)),
           'splits': splits, 'max_goals': max_goals}

    # ---- OOF lambdas + rho + the production controls ----------------------
    lam_h = np.full(len(d), np.nan)
    lam_a = np.full(len(d), np.nan)
    P_ctrl = np.full((len(d), K), np.nan)
    ou_ctrl = np.full(len(d), np.nan)
    rhos = []
    pois = dict(objective='count:poisson', n_estimators=400, learning_rate=0.05,
                max_depth=4, subsample=0.9, colsample_bytree=0.8,
                tree_method='hist', enable_categorical=True, seed=42)
    tscv = TimeSeriesSplit(n_splits=splits)
    for fold, (tr, te) in enumerate(tscv.split(d)):
        mh = xgb.XGBRegressor(**pois).fit(X.iloc[tr], hg[tr])
        ma = xgb.XGBRegressor(**pois).fit(X.iloc[tr], ag[tr])
        lam_h[te], lam_a[te] = mh.predict(X.iloc[te]), ma.predict(X.iloc[te])
        r = fit_rho(mh.predict(X.iloc[tr]), ma.predict(X.iloc[tr]), hg[tr], ag[tr])
        rhos.append(r)
        # controls, same folds
        c = spec.build().fit(X.iloc[tr], y[tr])
        P_ctrl[te] = c.predict_proba(X.iloc[te])
        ou = xgb.XGBRegressor(**pois).fit(X.iloc[tr], (hg + ag)[tr])
        lam_tot = np.clip(ou.predict(X.iloc[te]), 1e-6, None)
        ou_ctrl[te] = 1.0 - np.exp(-lam_tot) * (1 + lam_tot + lam_tot ** 2 / 2)
        print(f'  fold {fold + 1}/{splits}  rho={r:+.4f}')
    rho = float(np.mean(rhos))
    rep['rho_per_fold'] = [float(r) for r in rhos]
    rep['rho_mean'] = rho

    ok = np.isfinite(lam_h) & np.isfinite(lam_a)
    mat = scoreline_grid(lam_h[ok], lam_a[ok], rho=rho, max_goals=max_goals)

    # ---- the fast path must match the reference implementation ------------
    idx = np.linspace(0, mat.shape[0] - 1, 25).astype(int)
    errs = [np.abs(mat[t] - scoreline_matrix(lam_h[ok][t], lam_a[ok][t],
                                             rho=rho, max_goals=max_goals)).max()
            for t in idx]
    rep['vectorised_vs_reference_max_abs_error'] = float(max(errs))
    rep['vectorised_matches_reference'] = bool(max(errs) < 1e-12)

    mk = markets_from_grid(mat, ou_line=2.5)
    P_dc = np.column_stack([mk['home'], mk['draw'], mk['away']])
    P_dc = P_dc / P_dc.sum(1, keepdims=True)
    yo, hgo, ago = y[ok], hg[ok], ag[ok]
    mkt_o, ctrl_o, ouctrl_o = mkt[ok], P_ctrl[ok], ou_ctrl[ok]

    # ---- 1X2: a guard, not a target ---------------------------------------
    rep['market_1x2'] = {'rps': rps(mkt_o, yo), 'brier': brier3(mkt_o, yo)}
    rep['x12'] = {}
    for nm, P in (('dc_hybrid', P_dc), ('production', ctrl_o), ('market', mkt_o)):
        e = {'rps': rps(P, yo), 'brier': brier3(P, yo), 'logloss': mll3(P, yo),
             'accuracy': float((P.argmax(1) == yo).mean())}
        pick = P.argmax(1)
        r = np.arange(len(P))
        e['nats_over_market'] = stack_test(P[r, pick], mkt_o[r, pick],
                                           (pick == yo).astype(int))['nats_added_over_market']
        rep['x12'][nm] = e
    reg = (rep['x12']['dc_hybrid']['rps'] - rep['x12']['production']['rps']) \
        / rep['x12']['production']['rps']
    rep['x12']['regression_vs_production'] = float(reg)
    rep['x12']['gate_no_1x2_regression'] = bool(reg <= GATE_1X2_REGRESSION)

    # ---- O/U 2.5 ----------------------------------------------------------
    y_over = ((hgo + ago) > 2.5).astype(int)
    rep['ou'] = {'n': int(len(y_over))}
    for nm, p in (('dc_hybrid', mk['over']), ('production', ouctrl_o)):
        rep['ou'][nm] = {'brier': bin_brier(p, y_over), 'logloss': bin_ll(p, y_over),
                         'accuracy': float(((p >= 0.5) == y_over).mean())}
    om = pd.to_numeric(d.loc[ok, 'B365>2.5'], errors='coerce').values
    um = pd.to_numeric(d.loc[ok, 'B365<2.5'], errors='coerce').values
    hasou = np.isfinite(om) & np.isfinite(um) & (om > 1) & (um > 1)
    if hasou.sum() > 500:
        pm = devig2(om[hasou], um[hasou])
        rep['ou']['market'] = {'n': int(hasou.sum()), 'brier': bin_brier(pm, y_over[hasou]),
                               'logloss': bin_ll(pm, y_over[hasou])}
        rep['ou']['dc_nats_over_market'] = stack_test(
            mk['over'][hasou], pm, y_over[hasou])['nats_added_over_market']

    # ---- Asian handicap, half-lines only ----------------------------------
    ahh = pd.to_numeric(d.loc[ok, 'AHh'], errors='coerce').values
    ha = pd.to_numeric(d.loc[ok, 'B365AHH'], errors='coerce').values
    aa = pd.to_numeric(d.loc[ok, 'B365AHA'], errors='coerce').values
    half = (np.isfinite(ahh) & np.isclose(np.abs(ahh) % 1, 0.5)
            & np.isfinite(ha) & np.isfinite(aa) & (ha > 1) & (aa > 1))
    rep['ah'] = {'n_half_line': int(half.sum()),
                 'n_any_line': int((np.isfinite(ahh) & np.isfinite(ha)).sum())}
    if half.sum() > 500:
        p_dc = ah_home_cover(mat[half], ahh[half])
        y_cov = ((hgo[half] - ago[half]) + ahh[half] > 0).astype(int)
        p_mk = devig2(ha[half], aa[half])
        rep['ah']['dc_hybrid'] = {'brier': bin_brier(p_dc, y_cov),
                                  'logloss': bin_ll(p_dc, y_cov),
                                  'accuracy': float(((p_dc >= 0.5) == y_cov).mean())}
        rep['ah']['market'] = {'brier': bin_brier(p_mk, y_cov),
                               'logloss': bin_ll(p_mk, y_cov),
                               'accuracy': float(((p_mk >= 0.5) == y_cov).mean())}
        st = stack_test(p_dc, p_mk, y_cov)
        rep['ah']['nats_over_market'] = st['nats_added_over_market']
        rep['ah']['blend_weight_model'] = st['blend_weight_model']
        # The AH blend weight is the only positive one anywhere in the E
        # programme, and it rests on a few thousand rows — so it gets an error
        # bar before anyone reads anything into it.
        rng = np.random.default_rng(0)
        nats_bs, w_bs = [], []
        for _ in range(400):
            ix = rng.integers(0, len(y_cov), len(y_cov))
            if y_cov[ix].min() == y_cov[ix].max():
                continue
            try:
                s = stack_test(p_dc[ix], p_mk[ix], y_cov[ix])
            except Exception:
                continue
            nats_bs.append(s['nats_added_over_market'])
            w_bs.append(s['blend_weight_model'])
        if nats_bs:
            rep['ah']['nats_ci'] = [float(v) for v in np.percentile(nats_bs, [2.5, 97.5])]
            rep['ah']['blend_weight_ci'] = [float(v) for v in np.percentile(w_bs, [2.5, 97.5])]
            rep['ah']['blend_weight_beats_zero'] = bool(np.percentile(w_bs, 2.5) > 0)
        rep['ah']['reliability'] = reliability(p_dc, y_cov)
        rep['ah']['base_rate'] = float(y_cov.mean())

    # ---- BTTS: calibration only (no market in the corpus) -----------------
    y_btts = ((hgo >= 1) & (ago >= 1)).astype(int)
    rep['btts'] = {'n': int(len(y_btts)), 'base_rate': float(y_btts.mean()),
                   'pred_mean': float(mk['btts'].mean()),
                   'brier': bin_brier(mk['btts'], y_btts),
                   'logloss': bin_ll(mk['btts'], y_btts),
                   'accuracy': float(((mk['btts'] >= 0.5) == y_btts).mean()),
                   'reliability': reliability(mk['btts'], y_btts),
                   'note': 'football-data carries no BTTS prices — calibration only'}

    # ---- correct score ----------------------------------------------------
    G = mat.shape[1]
    hgc, agc = np.clip(hgo, 0, G - 1), np.clip(ago, 0, G - 1)
    p_true = mat[np.arange(len(mat)), hgc, agc]
    flat = mat.reshape(len(mat), -1)
    top = flat.argmax(1)
    rep['correct_score'] = {
        'logloss': float(-np.log(np.clip(p_true, 1e-12, 1)).mean()),
        'top1_accuracy': float(((top // G == hgc) & (top % G == agc)).mean()),
        'most_likely_share': float(np.bincount(top, minlength=G * G).max() / len(top)),
        'note': 'no correct-score market in the corpus — calibration only',
    }
    return rep


def render(rep):
    L = ['=' * 100, f'E4 — GBDT x DIXON-COLES HYBRID — {rep["generated"]}', '=' * 100, '',
         f'rows {rep["rows"]}   folds {rep["splits"]}   max_goals {rep["max_goals"]}',
         f'Dixon-Coles rho (MLE per fold): '
         f'{[round(r, 4) for r in rep["rho_per_fold"]]}  mean {rep["rho_mean"]:+.4f}',
         f'vectorised scoreline vs reference implementation: max abs error '
         f'{rep["vectorised_vs_reference_max_abs_error"]:.2e} '
         f'(matches={rep["vectorised_matches_reference"]})', '']

    L += ['', '=' * 100, '1X2 — A GUARD, NOT A TARGET', '=' * 100,
          f'{"":14} {"RPS":>9} {"Brier":>9} {"logloss":>9} {"acc":>8} {"nats/mkt":>10}']
    for nm, e in rep['x12'].items():
        if not isinstance(e, dict):
            continue
        L.append(f'{nm:14} {e["rps"]:9.5f} {e["brier"]:9.5f} {e["logloss"]:9.5f} '
                 f'{100 * e["accuracy"]:7.2f}% {e["nats_over_market"]:+10.5f}')
    L += ['', f'  1X2 RPS vs production: {100 * rep["x12"]["regression_vs_production"]:+.2f}%'
              f'   gate (<= +{100 * GATE_1X2_REGRESSION:.1f}%): '
              f'{"PASS" if rep["x12"]["gate_no_1x2_regression"] else "FAIL"}']

    o = rep['ou']
    L += ['', '', '=' * 100, 'OVER/UNDER 2.5', '=' * 100,
          f'{"":14} {"Brier":>9} {"logloss":>9} {"acc":>8}']
    for nm in ('dc_hybrid', 'production', 'market'):
        if nm in o:
            e = o[nm]
            acc = f'{100 * e["accuracy"]:7.2f}%' if 'accuracy' in e else f'{"-":>8}'
            L.append(f'{nm:14} {e["brier"]:9.5f} {e["logloss"]:9.5f} {acc}')
    if 'dc_nats_over_market' in o:
        L.append(f'  nats added over market: {o["dc_nats_over_market"]:+.5f}')

    a = rep['ah']
    L += ['', '', '=' * 100,
          'ASIAN HANDICAP — the market the production stack cannot price at all',
          '=' * 100,
          f'  half-line rows {a["n_half_line"]} of {a["n_any_line"]} with any AH line',
          '  (integer lines can push, quarter lines split the stake — neither has an',
          '   unambiguous binary outcome to score a probability against)']
    if 'dc_hybrid' in a:
        L += ['', f'{"":14} {"Brier":>9} {"logloss":>9} {"acc":>8}']
        for nm in ('dc_hybrid', 'market'):
            e = a[nm]
            L.append(f'{nm:14} {e["brier"]:9.5f} {e["logloss"]:9.5f} '
                     f'{100 * e["accuracy"]:7.2f}%')
        nci = (f'  [{a["nats_ci"][0]:+.5f},{a["nats_ci"][1]:+.5f}]'
               if 'nats_ci' in a else '')
        wci = (f'  [{100 * a["blend_weight_ci"][0]:+.0f}%,'
               f'{100 * a["blend_weight_ci"][1]:+.0f}%]'
               f'  CI excludes 0: {a["blend_weight_beats_zero"]}'
               if 'blend_weight_ci' in a else '')
        L += ['',
              f'  base rate (home covers) {100 * a["base_rate"]:.1f}%',
              f'  nats added over AH market  {a["nats_over_market"]:+.5f}{nci}',
              f'  blend weight on model      {100 * a["blend_weight_model"]:+.0f}%{wci}',
              '', '  reliability (DC hybrid):',
              f'    {"bin":10} {"n":>6} {"pred":>8} {"actual":>8}']
        for b in a['reliability']:
            L.append(f'    {b["bin"]:10} {b["n"]:6d} {100 * b["pred"]:7.1f}% '
                     f'{100 * b["actual"]:7.1f}%')

    b = rep['btts']
    L += ['', '', '=' * 100, 'BTTS — calibration only (no market in the corpus)', '=' * 100,
          f'  n {b["n"]}   predicted mean {100 * b["pred_mean"]:.1f}%   '
          f'actual {100 * b["base_rate"]:.1f}%',
          f'  Brier {b["brier"]:.5f}   logloss {b["logloss"]:.5f}   '
          f'acc {100 * b["accuracy"]:.2f}%', '',
          f'    {"bin":10} {"n":>6} {"pred":>8} {"actual":>8}']
    for r in b['reliability']:
        L.append(f'    {r["bin"]:10} {r["n"]:6d} {100 * r["pred"]:7.1f}% '
                 f'{100 * r["actual"]:7.1f}%')

    c = rep['correct_score']
    L += ['', '', '=' * 100, 'CORRECT SCORE — calibration only', '=' * 100,
          f'  logloss over the {rep["max_goals"] + 1}x{rep["max_goals"] + 1} grid '
          f'{c["logloss"]:.5f}   top-1 hit rate {100 * c["top1_accuracy"]:.2f}%',
          f'  single most-likely scoreline chosen on '
          f'{100 * c["most_likely_share"]:.1f}% of matches']

    L += ['', '=' * 100,
          'READ: 1X2 and O/U are guards — the hybrid has to not break what already',
          'works. The verdict is whether AH/BTTS/correct-score come out calibrated',
          'enough to trade, because those are markets the production stack cannot',
          'price at all. Beating the AH market is a separate and much higher bar than',
          'pricing it honestly, and given E0/E1/E5 it should not be expected.',
          '=' * 100]
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--max-goals', type=int, default=10)
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    df = add_1x2_market(prepare(cache=not args.no_cache))
    rep = build(df, args.splits, args.max_goals)
    report = render(rep)
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUT_DIR, f'dc_hybrid_{ts}.json'), 'w') as f:
        json.dump(rep, f, indent=2, default=float)
    with open(os.path.join(OUT_DIR, f'dc_hybrid_{ts}.txt'), 'w') as f:
        f.write(report)
    print(report)
    print(f'\nWrote output/experiments/dc_hybrid_{ts}.{{json,txt}}')


if __name__ == '__main__':
    main()
