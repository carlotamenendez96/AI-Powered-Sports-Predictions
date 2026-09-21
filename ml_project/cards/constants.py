"""Single source of truth for the cards-market feature / target contract.

Train and serve both import from here so a window present in training cannot
silently vanish at serve time (same spirit as FORM_WINDOWS in
feature_engineering.py).
"""

# Book-like line for the primary binary product.
CARD_LINE = 3.5

# Rolling card-form windows: (n_matches, column_suffix). L5 keeps unsuffixed
# names so the feature list stays readable; L10 is the slower baseline.
CARDS_FORM_WINDOWS = ((5, ""), (10, "_l10"))

# Referee rates are only used when the catalog has this many PRIOR matches
# WITH card data (hy/ay non-null). Below the gate → NaN + missing_ref=1.
REFEREE_MIN_N = 20

# Feature columns the binary cards head trains on, in serve/train order.
# Referee block is NaN-gated; XGBoost learns a split direction for NaN.
# Availability / suspensions are deliberately ABSENT (no historical backfill).
CARDS_FEATURES = (
    # Team card form — home side (overall + venue-home L5)
    "H_card_for", "H_card_against",
    "H_card_for_l10", "H_card_against_l10",
    "H_home_card_for", "H_home_card_against",
    "H_red_rate",
    # Team card form — away side (overall + venue-away L5)
    "A_card_for", "A_card_against",
    "A_card_for_l10", "A_card_against_l10",
    "A_away_card_for", "A_away_card_against",
    "A_red_rate",
    # Combined proxy
    "expected_yellows_proxy",
    # League priors (expanding, shift-1)
    "league_mean_yellows", "league_p_over",
    # Referee block (gated)
    "ref_yellows_pg", "ref_red_rate", "ref_n_with_cards", "missing_ref",
    # Match context
    "league_cat",
    "elo_diff", "abs_elo_diff",
)

# Sub-blocks used by the placebo arm in experiment_cards.py.
REF_FEATURE_BLOCK = (
    "ref_yellows_pg", "ref_red_rate", "ref_n_with_cards", "missing_ref",
)
CARD_FORM_BLOCK = (
    "H_card_for", "H_card_against",
    "H_card_for_l10", "H_card_against_l10",
    "H_home_card_for", "H_home_card_against",
    "H_red_rate",
    "A_card_for", "A_card_against",
    "A_card_for_l10", "A_card_against_l10",
    "A_away_card_for", "A_away_card_against",
    "A_red_rate",
    "expected_yellows_proxy",
)

# Serve-time + virtual-betting config (isolated from sports_config LANES).
# Cards has no book odds in the scraper yet — betting uses synthetic_odd for
# conviction/model sizing only. Value lane is skipped until real odds exist.
CARDS_CONFIG_DEFAULTS = {
    "enabled": True,
    "use_calibration": True,
    # Documented but unused in v1 — no historical availability backfill.
    "use_availability_at_serve": False,
    # Virtual betting (same three bankroll lanes; type="Cards")
    "include_in_betting": True,
    "synthetic_odd": 1.90,
    "min_confidence": 0.55,
    # Value needs real EV/odds — do not invent edge with a flat synthetic price.
    "bet_value": False,
    "bet_conviction": True,
    "bet_model": True,
}
