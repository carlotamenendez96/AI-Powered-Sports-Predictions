"""Single source of truth for which scraped competitions are NATIONAL-TEAM
football (routed to the NT model) vs club football (the existing predictor).

Used by both the club predictor's skip-guard (predict_matches.py) and the NT
batch predictor, so the routing can never disagree.

Base name = the league label with playoff/group/division suffixes stripped
(mirrors predict_matches.py's canonicalisation), so e.g.
  "EUROPE: UEFA Nations League - League A" -> "EUROPE: UEFA Nations League".
"""

# National-team competitions (base names) — these route to the NT model.
# NOTE: Flashscore labels the FIFA World Cup finals as "World Championship"
# (not "World Cup"), so both spellings are registered to keep routing robust.
INTERNATIONAL_BASES = {
    "WORLD: World Cup",
    "WORLD: World Cup Qualification",
    "WORLD: World Cup Final Tournament",
    "WORLD: World Cup Play Offs",
    "WORLD: World Championship",
    "WORLD: World Championship Qualification",
    "WORLD: World Championship Final Tournament",
    "WORLD: World Championship Play Offs",
    "EUROPE: Euro",
    "EUROPE: Euro Qualification",
    "EUROPE: Euro Final Tournament",
    "EUROPE: UEFA Nations League",
    # Added 2026-09-26. AFCON was whitelisted in target_leagues.json (15d1c08)
    # but never registered here, so it was national-team football that routed
    # nowhere: is_international() said False, and the club predictor then threw
    # it away because 'AFRICA' is absent from its SUPPORTED_COUNTRIES set.
    # Measured across three slates: 8 + 14 + 2 AFCON fixtures scraped, 0 rows
    # in the matching predictions_<date>.csv. base_name() strips the
    # " - Qualification" suffix, so this one entry covers the qualifiers too.
    # The NT corpus supports them — all 24 scraped fixtures resolve to
    # eloratings teams, and it carries "African Nations Cup [qualifier]" comps.
    "AFRICA: Africa Cup of Nations",
}

# Subset played at neutral venues (finals tournaments) → predict orientation-
# averaged with no home advantage. Qualifiers + Nations League are home/away.
# TODO: host-nation home advantage in finals (USA/Can/Mex for WC 2026) is a
# refinement — v1 treats all finals matches as neutral.
NEUTRAL_BASES = {
    "WORLD: World Cup",
    "WORLD: World Cup Final Tournament",
    "WORLD: World Championship",
    "WORLD: World Championship Final Tournament",
    "EUROPE: Euro",
    "EUROPE: Euro Final Tournament",
}


def base_name(league_name: str) -> str:
    if ":" in league_name:
        country, rest = league_name.split(":", 1)
        return f"{country.strip()}: {rest.split(' - ', 1)[0].strip()}"
    return league_name.strip()


def is_international(league_name: str) -> bool:
    """True if this competition is national-team football (route to NT model)."""
    return base_name(league_name) in INTERNATIONAL_BASES


def is_neutral_competition(league_name: str) -> bool:
    """True if matches are at neutral venues (finals tournaments)."""
    return base_name(league_name) in NEUTRAL_BASES
