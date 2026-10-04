"""Isolated config reader for the cards market.

Does NOT go through web_ui.sports_config (no bankroll / lane coupling).
Reads optional `sports.football.cards` from data_sets/betting_config.json.
"""
from __future__ import annotations

import json
import os

from .constants import CARDS_CONFIG_DEFAULTS

_DEFAULT_PATH = "data_sets/betting_config.json"

_BOOL_KEYS = (
    "enabled", "use_calibration", "use_availability_at_serve",
    "include_in_betting", "allow_synthetic_fallback",
    "bet_value", "bet_conviction", "bet_model",
)
_FLOAT_KEYS = ("synthetic_odd", "min_confidence")


def get_cards_config(path: str = _DEFAULT_PATH) -> dict:
    out = dict(CARDS_CONFIG_DEFAULTS)
    if not os.path.exists(path):
        return out
    try:
        with open(path) as f:
            cfg = json.load(f)
        block = (cfg.get("sports", {})
                   .get("football", {})
                   .get("cards", {}))
        if not isinstance(block, dict):
            return out
        for k in _BOOL_KEYS:
            if k in block:
                out[k] = bool(block[k])
        for k in _FLOAT_KEYS:
            if k in block:
                try:
                    out[k] = float(block[k])
                except (TypeError, ValueError):
                    pass
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return out
