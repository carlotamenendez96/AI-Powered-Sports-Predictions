"""Model registry — the swappable-estimator seam.

Single source of truth for the model families the pipeline can train and
serve, across football heads:

  '1x2'         — multiclass P(H/D/A)        (predict_proba -> (n, 3))
  'ou'          — Over/Under 2.5 goal count  (predict      -> lambda, Poisson mean)
  'draw'        — binary P(draw)             (predict_proba -> (n, 2))   [trained only]
  'cards'       — binary P(HY+AY > 3.5)      (predict_proba -> (n, 2))   [Paso 6]
  'cards_total' — Poisson total yellows      (predict      -> lambda)   [Paso 6 diagnostic]

Each family is a `ModelSpec` exposing a uniform contract: build a fresh
estimator, plus the metadata the pipeline needs to prepare features for it —
whether it consumes the categorical `league_cat` column (XGBoost) or a
numeric-only feature list with imputation/scaling folded into its Pipeline.

The point of the seam: code that evaluates / trains / serves a model
references only this contract, never `xgboost` directly, so the model family
becomes a pluggable choice. The production default for every head stays
byte-identical XGBoost.

Note: this is a single module (not an `ml_project/models/` package) on purpose
— a `models` package would shadow the repo-root `models/` artifacts directory
on PYTHONPATH.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


@dataclass
class ModelSpec:
    """Describes one swappable model family for a given market.

    `factory()` returns a fresh, unfitted estimator. The expected predict
    contract depends on `task`:
      - 'multiclass' / 'binary' -> predict_proba(X)
      - 'poisson'               -> predict(X) (the Poisson mean / lambda)
    """
    name: str            # family, e.g. 'xgboost'
    market: str          # '1x2' | 'ou' | 'draw'
    task: str            # 'multiclass' | 'binary' | 'poisson'
    factory: Callable[[], Any]
    uses_categorical: bool = False  # consumes league_cat as a category dtype
    description: str = ''

    def build(self) -> Any:
        return self.factory()


def _tuned(market: str) -> dict:
    """Load models/best_params_<market>.json if present, else {}.
    early_stopping_rounds is stripped — every seam fit is a plain fit
    (the tuned files carry no early stopping, so the prior eval_set only
    logged a metric and never altered tree building)."""
    pf = os.path.join(_ROOT, 'models', f'best_params_{market}.json')
    if not os.path.exists(pf):
        return {}
    with open(pf) as f:
        t = json.load(f)
    t.pop('early_stopping_rounds', None)
    return t


# --------------------------------------------------------------------------- #
# 1X2 factories (multiclass)
# --------------------------------------------------------------------------- #
def _xgb_1x2():
    import xgboost as xgb
    params = dict(
        objective='multi:softprob', num_class=3, n_estimators=100,
        learning_rate=0.1, max_depth=5, eval_metric='mlogloss',
        tree_method='hist', enable_categorical=True,
    )
    params.update(_tuned('1x2'))
    return xgb.XGBClassifier(**params)


def _logreg_1x2():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, C=1.0)),
    ])


def _rf_1x2():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=12,
                                       n_jobs=-1, random_state=0)),
    ])


# --------------------------------------------------------------------------- #
# O/U factories (Poisson regression on total goals; predict -> lambda)
# --------------------------------------------------------------------------- #
def _xgb_ou():
    import xgboost as xgb
    # Mirrors train_ou exactly: load best_params_ou but FORCE a Poisson
    # objective (the tuned file carries a stale binary objective) and the
    # matching eval metric.
    params = dict(
        objective='count:poisson', n_estimators=100, learning_rate=0.1,
        max_depth=5, eval_metric='poisson-nloglik', tree_method='hist',
        enable_categorical=True,
    )
    tuned = _tuned('ou')
    if tuned:
        tuned['objective'] = 'count:poisson'
        if tuned.get('eval_metric') in ('logloss', 'error'):
            tuned['eval_metric'] = 'poisson-nloglik'
        params.update(tuned)
    return xgb.XGBRegressor(**params)


def _poisson_glm_ou():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('reg', PoissonRegressor(max_iter=1000)),
    ])


# --------------------------------------------------------------------------- #
# Draw factories (binary P(draw); trained but not served at inference)
# --------------------------------------------------------------------------- #
def _xgb_draw():
    import xgboost as xgb
    # Mirrors train_draw's inline params exactly (no tuned file for draw).
    return xgb.XGBClassifier(
        objective='binary:logistic', n_estimators=100, learning_rate=0.05,
        max_depth=4, eval_metric='logloss', tree_method='hist',
        enable_categorical=True,
    )


def _logreg_draw():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, C=1.0)),
    ])


# --------------------------------------------------------------------------- #
# Cards factories (Paso 6 — isolated from 1X2 / O/U production)
# --------------------------------------------------------------------------- #
def _xgb_cards():
    import xgboost as xgb
    # Conservative binary config — no 6-stage tune in v1.
    return xgb.XGBClassifier(
        objective='binary:logistic', n_estimators=120, learning_rate=0.05,
        max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
        gamma=1.0, eval_metric='logloss', tree_method='hist',
        enable_categorical=True, random_state=0,
    )


def _logreg_cards():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, C=1.0)),
    ])


def _xgb_cards_total():
    import xgboost as xgb
    return xgb.XGBRegressor(
        objective='count:poisson', n_estimators=120, learning_rate=0.05,
        max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
        gamma=1.0, eval_metric='poisson-nloglik', tree_method='hist',
        enable_categorical=True, random_state=0,
    )


def _poisson_glm_cards_total():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('reg', PoissonRegressor(max_iter=1000)),
    ])


# --------------------------------------------------------------------------- #
# Registry — REGISTRY[market][family]
# --------------------------------------------------------------------------- #
REGISTRY: dict[str, dict[str, ModelSpec]] = {
    '1x2': {
        'xgboost': ModelSpec('xgboost', '1x2', 'multiclass', _xgb_1x2, True,
                             'Production multi:softprob GBT (with league_cat).'),
        'logreg': ModelSpec('logreg', '1x2', 'multiclass', _logreg_1x2, False,
                            'Impute+scale+multinomial logistic regression (baseline).'),
        'rf': ModelSpec('rf', '1x2', 'multiclass', _rf_1x2, False,
                        'Impute + random forest.'),
    },
    'ou': {
        'xgboost': ModelSpec('xgboost', 'ou', 'poisson', _xgb_ou, True,
                             'Production count:poisson GBT regressor (with league_cat).'),
        'poisson_glm': ModelSpec('poisson_glm', 'ou', 'poisson', _poisson_glm_ou, False,
                                 'Impute+scale+Poisson GLM (baseline count model).'),
    },
    'draw': {
        'xgboost': ModelSpec('xgboost', 'draw', 'binary', _xgb_draw, True,
                             'Production binary:logistic GBT (with league_cat).'),
        'logreg': ModelSpec('logreg', 'draw', 'binary', _logreg_draw, False,
                            'Impute+scale+logistic regression (baseline).'),
    },
    'cards': {
        'xgboost': ModelSpec('xgboost', 'cards', 'binary', _xgb_cards, True,
                             'Paso 6 binary:logistic GBT for P(HY+AY > 3.5).'),
        'logreg': ModelSpec('logreg', 'cards', 'binary', _logreg_cards, False,
                            'Impute+scale+logistic baseline for cards over-line.'),
    },
    'cards_total': {
        'xgboost': ModelSpec('xgboost', 'cards_total', 'poisson',
                             _xgb_cards_total, True,
                             'Paso 6 diagnostic Poisson GBT on total yellows.'),
        'poisson_glm': ModelSpec('poisson_glm', 'cards_total', 'poisson',
                                 _poisson_glm_cards_total, False,
                                 'Impute+scale+Poisson GLM baseline for total yellows.'),
    },
}


def get_spec(market: str, family: str) -> ModelSpec:
    try:
        return REGISTRY[market][family]
    except KeyError:
        known = REGISTRY.get(market)
        if known is None:
            raise KeyError(f'unknown market {market!r}; known: {", ".join(REGISTRY)}')
        raise KeyError(f'unknown {market} family {family!r}; known: {", ".join(known)}')


def available(market: str) -> list[str]:
    return list(REGISTRY.get(market, {}))


# --------------------------------------------------------------------------- #
# Persistence — a per-market meta sidecar records which family/artifact serves
# a head; loaders fall back to the legacy XGBoost JSON so a pre-seam checkout
# is unchanged.
# --------------------------------------------------------------------------- #
_LEGACY = {
    '1x2': 'xgb_model_1x2.json',
    'ou': 'xgb_model_ou.json',
    'draw': 'xgb_model_draw.json',
    'cards': 'xgb_model_cards.json',
    'cards_total': 'xgb_model_cards_total.json',
}


def _meta_name(market: str) -> str:
    return f'model_meta_{market}.json'


def save_meta(market: str, family: str, artifact: str,
              models_dir: str = 'models') -> None:
    """Record which family/artifact serves `market`. Additive sidecar —
    does not touch the model bytes."""
    with open(os.path.join(models_dir, _meta_name(market)), 'w') as f:
        json.dump({'family': family, 'artifact': artifact, 'market': market},
                  f, indent=2)


# Back-compat alias (1X2 callers added in an earlier step).
def save_meta_1x2(family: str, artifact: str, models_dir: str = 'models') -> None:
    save_meta('1x2', family, artifact, models_dir)


def _load_estimator(market: str, models_dir: str):
    """Return (estimator, family) for a market, honouring the meta sidecar and
    falling back to the legacy XGBoost artifact."""
    meta_path = os.path.join(models_dir, _meta_name(market))
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        family = meta.get('family', 'xgboost')
        artifact = os.path.join(models_dir, meta.get('artifact', _LEGACY[market]))
    else:
        family, artifact = 'xgboost', os.path.join(models_dir, _LEGACY[market])

    if family == 'xgboost':
        import xgboost as xgb
        # Classifiers: 1x2 / draw / cards. Regressors: ou / cards_total.
        is_regressor = market in ('ou', 'cards_total')
        est = xgb.XGBRegressor() if is_regressor else xgb.XGBClassifier()
        est.load_model(artifact)
    else:
        import joblib
        est = joblib.load(artifact)
    return est, family


def load_1x2_model(models_dir: str = 'models'):
    """Load the serving 1X2 estimator (predict_proba -> (n, 3)). For 'xgboost'
    the caller must feed league_cat as a category dtype; other families consume
    the numeric-only feature list written into features_1x2.json."""
    return _load_estimator('1x2', models_dir)


def load_ou_model(models_dir: str = 'models'):
    """Load the serving O/U estimator (predict -> Poisson lambda). Same
    family/categorical contract as the 1X2 loader."""
    return _load_estimator('ou', models_dir)


def load_cards_model(models_dir: str = 'models'):
    """Load the serving cards over-line estimator (predict_proba -> (n, 2))."""
    return _load_estimator('cards', models_dir)
