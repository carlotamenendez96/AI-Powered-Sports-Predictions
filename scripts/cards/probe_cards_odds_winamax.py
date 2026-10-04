#!/usr/bin/env python3
"""Probe Winamax for cards O/U 3.5 (READ-ONLY, plain HTTPS).

Pamestoixima is Akamai-blocked headless; Flashscore/BE/OP have no cards O/U.
Winamax.es embeds ``PRELOADED_STATE`` with betType 2603
("Número total de tarjetas", ``specialBetValue=total=3.5``).

Gate: ≥2 upcoming matches with Over+Under > 1.0.

Usage:
    python3 scripts/cards/probe_cards_odds_winamax.py
    python3 scripts/cards/probe_cards_odds_winamax.py --limit 40
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ml_project"))

from cards.winamax_odds import main  # noqa: E402

if __name__ == "__main__":
    if "--probe" not in sys.argv:
        sys.argv.insert(1, "--probe")
    raise SystemExit(main())
