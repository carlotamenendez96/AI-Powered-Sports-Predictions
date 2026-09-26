import json
import os
import re
import pandas as pd
from entity_resolver import EntityResolver

# --- League-label reconciliation (2026-09-26) -------------------------------
# Standings rows are labelled from the hand-typed LEAGUE column of
# data_sets/standings_form_flashscore_direct_links.csv, while the daily slate
# is labelled from Flashscore's live DOM. The two have drifted, so the exact
# "COUNTRY|League" key lookup below missed 9 whitelisted leagues and silently
# returned the neutral (0.0, 1.0, 1.0) strength for every team in them —
# measured 2026-09-26 at 69 fixtures, 24% of all matches on record.
#
# The old substring fallback cannot rescue these: it needs the slate label to
# be contained in the standings key, and the slate label is usually the LONGER
# one ("Jupiler Pro League" is not inside "Jupiler Pro").
#
# Most of the drift is cosmetic, so normalise both sides instead of
# enumerating pairs — that also absorbs the next cosmetic rename:
#     "League Two"         -> league2     == "League 2"
#     "LaLiga" / "LaLiga 2"-> laliga(2)   == "La Liga" / "La Liga 2"
#     "2. Bundesliga"      -> bundesliga2 == "Bundesliga 2"  (leading number moved)
#     "Jupiler Pro League" -> jupilerpro  == "Jupiler Pro"   (trailing "league" cut)
_NUM_WORDS = (("one", "1"), ("two", "2"), ("three", "3"), ("four", "4"))


def _norm_league(name):
    """Canonical form of a league label for cross-source comparison."""
    s = name.lower()
    for word, digit in _NUM_WORDS:
        s = re.sub(rf"\b{word}\b", digit, s)
    s = re.sub(r"[^a-z0-9]", "", s)
    m = re.match(r"^(\d+)(.+)$", s)          # "2bundesliga" -> "bundesliga2"
    if m:
        s = m.group(2) + m.group(1)
    if s.endswith("league") and s != "league":
        s = s[:-len("league")]
    return s


# Irreducible cases — a genuinely different name, not a different spelling,
# so no normalisation can bridge them. Keyed COUNTRY|<normalised slate label>.
STANDINGS_LEAGUE_ALIASES = {
    "PORTUGAL|ligaportugal": "Liga 1",       # slate "Liga Portugal"
}


def _norm_index(keys):
    """{"COUNTRY|normalised": "COUNTRY|Actual League"} over a set of keys.

    A normalised form claimed by two different leagues of the same country is
    dropped rather than guessed — an ambiguous match would silently attach one
    league's table to another's fixtures, which is worse than no match at all.
    """
    index, clashes = {}, set()
    for k in keys:
        country, _, league = k.partition("|")
        nk = f"{country}|{_norm_league(league)}"
        if nk in index and index[nk] != k:
            clashes.add(nk)
        index[nk] = k
    for nk in clashes:
        index.pop(nk, None)
    return index


class _LeagueLookup(dict):
    """{"COUNTRY|League": {team: row}} plus a `.norm` index over its keys.

    A dict subclass so every existing caller keeps working unchanged.
    """
    norm = {}


class HeuristicAdjuster:
    def __init__(self, data_dir="data_sets/standings"):
        self.data_dir = data_dir
        self.resolver = EntityResolver()
        
        # Load Data
        self.standings = self._load_json("standings_overall.json")
        self.form_overall = self._load_json("last_5_matches_overall.json")
        self.standings_home = self._load_json("standings_home.json")
        self.standings_away = self._load_json("standings_away.json")
        self.form_home = self._load_json("last_5_matches_home.json")
        self.form_away = self._load_json("last_5_matches_away.json")
        
        # Load Last 10 Data
        self.form_overall_10 = self._load_json("last_10_matches_overall.json")
        
        # Create lookups based on "Country: League" keys for faster access
        self.standings_lookup = self._build_lookup(self.standings)
        self.form_lookup = self._build_lookup(self.form_overall)
        self.home_table_lookup = self._build_lookup(self.standings_home)
        self.away_table_lookup = self._build_lookup(self.standings_away)
        self.form_home_lookup = self._build_lookup(self.form_home)
        self.form_away_lookup = self._build_lookup(self.form_away)
        
        self.form_lookup_10 = self._build_lookup(self.form_overall_10)
        
        # Calibration Data
        self.league_stats = self._calculate_league_stats(self.standings)
        # Same reconciliation for the per-league baseline used by
        # get_team_strength — without it a team could resolve through the
        # normalised index while its league's avg_gf silently fell back to the
        # 1.3 cross-league prior, leaving att/def scaled against the wrong base.
        self.league_stats_norm = _norm_index(self.league_stats.keys())

    def _calculate_league_stats(self, standings_data):
        """
        Calculates average stats per league (Draw Rate, Avg Goals For per team-game).
        Returns: { "Country|League": { "draw_rate": 0.25, "avg_gf": 1.45 } }
        """
        stats = {}
        # Group by league
        leagues = {}
        for entry in standings_data:
            c = entry.get('country', '').upper()
            l = entry.get('league', '')
            key = f"{c}|{l}"

            if key not in leagues: leagues[key] = []
            leagues[key].append(entry)

        for key, entries in leagues.items():
            total_matches = 0
            total_draws = 0
            total_gf = 0

            for t in entries:
                try:
                    # 'draws' field in standings
                    d = int(t.get('draws', t.get('draw', 0)))
                    mp = int(t.get('matches_played', 0))

                    # Each draw and match is counted twice (both teams). Ratios stay valid.
                    total_draws += d
                    total_matches += mp

                    # Goals For (format "GF:GA")
                    goals_str = t.get('goals', '0:0')
                    if ':' in goals_str:
                        gf = int(goals_str.split(':')[0])
                        total_gf += gf
                except (KeyError, ValueError, TypeError, AttributeError):
                    continue

            draw_rate = (total_draws / total_matches) if total_matches > 0 else 0.26
            if draw_rate < 0.10: draw_rate = 0.15
            if draw_rate > 0.40: draw_rate = 0.35

            # League avg goals scored per team-game. 1.3 is a reasonable cross-league prior.
            avg_gf = (total_gf / total_matches) if total_matches > 0 else 1.3

            stats[key] = {
                'draw_rate': draw_rate,
                'avg_gf': avg_gf,
            }

        return stats

    def get_team_strength(self, league_name, team_name):
        """
        Season-to-date strength for a team, derived from current standings.
        Mirrors the training-time features built in
        FeatureEngineer._add_ppg_strength_features so model inputs match.

        Returns (ppg, att_strength, def_weakness).
        Defaults to (0.0, 1.0, 1.0) when the team or league is unavailable.
        """
        entry = self.find_team_stats(self.standings_lookup, "", league_name, team_name)
        if not entry:
            return 0.0, 1.0, 1.0

        try:
            mp = int(entry.get('matches_played', 0))
            if mp <= 0:
                return 0.0, 1.0, 1.0
            pts = int(entry.get('points', 0))
            gf_str, ga_str = entry.get('goals', '0:0').split(':')[:2]
            gf, ga = int(gf_str), int(ga_str)
        except (KeyError, ValueError, TypeError, AttributeError):
            return 0.0, 1.0, 1.0

        ppg = pts / mp
        avg_gf = gf / mp
        avg_ga = ga / mp

        baseline = 1.3
        if ":" in league_name:
            parts = league_name.split(":", 1)
            c_in = parts[0].strip().upper()
            l_in = parts[1].strip()
            key = f"{c_in}|{l_in}"
            league_stat = self.league_stats.get(key)
            if not league_stat:
                # Same three-step reconciliation as find_team_stats.
                norm_key = f"{c_in}|{_norm_league(l_in)}"
                alias = STANDINGS_LEAGUE_ALIASES.get(norm_key)
                resolved = (f"{c_in}|{alias}" if alias
                            else self.league_stats_norm.get(norm_key))
                league_stat = self.league_stats.get(resolved) if resolved else None
            if not league_stat:
                for k in self.league_stats:
                    if c_in in k and l_in in k:
                        league_stat = self.league_stats[k]
                        break
            if league_stat:
                baseline = league_stat.get('avg_gf', 1.3) or 1.3

        if baseline <= 0:
            baseline = 1.3

        return ppg, avg_gf / baseline, avg_ga / baseline

    def _load_json(self, filename):
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return []

    def _build_lookup(self, data):
        """
        Builds a dict: { "Country|League": { "TeamName": {stats...} } }
        """
        lookup = _LeagueLookup()
        for entry in data:
            c = entry.get('country', '').upper()
            l = entry.get('league', '')
            key = f"{c}|{l}"

            if key not in lookup:
                lookup[key] = {}
            
            team = entry.get('team_name')
            if team:
                # Normalize team name for lookup key?
                # Using resolver might be overkill here, we'll fuzzy match at query time
                # Store by raw name for iteration
                lookup[key][team] = entry
        lookup.norm = _norm_index(lookup.keys())
        return lookup

    def find_team_stats(self, lookup, country, league_name, team_name):
        """
        Finds stats for a team in a specific league lookup.
        Tries exact match first, then fuzzy via EntityResolver logic if needed.
        """
        # 1. Construct Key
        # Scraper league: "ENGLAND: Premier League" -> Country="ENGLAND", League="Premier League"
        # Adjuster lookup key: "ENGLAND|Premier League"
        
        # Normalize League Name input
        if ":" in league_name:
            parts = league_name.split(":")
            c_in = parts[0].strip().upper()
            l_in = parts[1].strip()
        else:
            # Maybe just league name? heuristic check
            # For now assume scraper provides full string
            return None 

        key = f"{c_in}|{l_in}"
        
        league_data = lookup.get(key)
        if not league_data:
            # The slate label and the standings label disagree. Try, in order
            # of decreasing confidence (see _norm_league above):
            #   1. an explicit alias for a genuinely different name
            #   2. the normalised index ("LaLiga" == "La Liga"), which refuses
            #      to resolve anything ambiguous
            #   3. the original substring fallback, kept for back-compat
            norm_key = f"{c_in}|{_norm_league(l_in)}"
            alias = STANDINGS_LEAGUE_ALIASES.get(norm_key)
            if alias:
                league_data = lookup.get(f"{c_in}|{alias}")

            if not league_data:
                resolved = getattr(lookup, 'norm', {}).get(norm_key)
                if resolved:
                    league_data = lookup.get(resolved)

            if not league_data:
                found_key = None
                for k in lookup.keys():
                    if c_in in k and l_in in k:
                        found_key = k
                        break
                if found_key:
                    league_data = lookup[found_key]
                else:
                    return None

        # 2. Find Team
        if team_name in league_data:
            return league_data[team_name]
        
        # Fuzzy Match Team
        best_match = None
        best_score = 0
        from difflib import SequenceMatcher
        
        for t_key in league_data.keys():
            ratio = SequenceMatcher(None, team_name.lower(), t_key.lower()).ratio()
            if ratio > 0.8 and ratio > best_score:
                best_score = ratio
                best_match = t_key
        
        if best_match:
            return league_data[best_match]
            
        return None

    # Tier 1 safety-net tunables
    MAX_TOTAL_BOOST_PER_CLASS = 0.15  # cap on cumulative H1-H6 delta per outcome
    FADE_TO_OPPONENT_SHARE = 0.7      # when a team fades, share of lift to opponent (rest to Draw)

    def adjust_probabilities(self, match_info, probs_1x2, probs_ou):
        """
        match_info: dict with 'League', 'Home Team', 'Away Team'
        probs_1x2: [Home%, Draw%, Away%] (0.0-1.0)
        probs_ou: [Under%, Over%]

        Returns: (adj_1x2, adj_ou, logs)
        """
        logs = []
        league = match_info.get('League', '')
        home = match_info.get('Home Team', '')
        away = match_info.get('Away Team', '')

        adj_1x2 = list(probs_1x2)
        adj_ou = list(probs_ou)

        H, D, A = 0, 1, 2  # outcome indices

        league_stat = self._lookup_league_stat(league)

        # --- CALIBRATION: League-aware draw shrinkage ---
        if league_stat:
            try:
                target_draw = league_stat.get('draw_rate', 0.26)
                current_draw = adj_1x2[D]
                alpha = 0.15
                new_draw = (1 - alpha) * current_draw + alpha * target_draw
                old_not_draw = max(1.0 - current_draw, 0.001)
                ratio = (1.0 - new_draw) / old_not_draw
                adj_1x2[H] *= ratio
                adj_1x2[D] = new_draw
                adj_1x2[A] *= ratio
                scaler = sum(adj_1x2)
                if scaler > 0:
                    adj_1x2 = [x / scaler for x in adj_1x2]
                logs.append(f"Calibration: Draw {current_draw:.2f}->{new_draw:.2f} (Target {target_draw:.2f})")
            except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
                logs.append(f"Calib Error: {type(e).__name__}: {e}")

            # --- DRAW CAP: cap at base + 5% ---
            try:
                base_draw_rate = league_stat.get('draw_rate', 0.26)
                draw_cap = base_draw_rate + 0.05
                if adj_1x2[D] > draw_cap:
                    old_draw = adj_1x2[D]
                    diff = old_draw - draw_cap
                    adj_1x2[D] = draw_cap
                    prob_not_draw = adj_1x2[H] + adj_1x2[A]
                    if prob_not_draw > 0:
                        adj_1x2[H] += diff * (adj_1x2[H] / prob_not_draw)
                        adj_1x2[A] += diff * (adj_1x2[A] / prob_not_draw)
                    else:
                        adj_1x2[H] += diff / 2
                        adj_1x2[A] += diff / 2
                    logs.append(f"Draw Cap: {old_draw:.2f}->{draw_cap:.2f} (Base {base_draw_rate:.2f})")
            except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
                logs.append(f"DrawCap Error: {type(e).__name__}: {e}")

        # --- LOOKUPS ---
        s_home = self.find_team_stats(self.standings_lookup, "", league, home)
        s_away = self.find_team_stats(self.standings_lookup, "", league, away)
        f_home = self.find_team_stats(self.form_lookup, "", league, home)
        f_away = self.find_team_stats(self.form_lookup, "", league, away)
        s_home_spec = self.find_team_stats(self.home_table_lookup, "", league, home)
        s_away_spec = self.find_team_stats(self.away_table_lookup, "", league, away)
        f_home_spec = self.find_team_stats(self.form_home_lookup, "", league, home)
        f_away_spec = self.find_team_stats(self.form_away_lookup, "", league, away)

        if not s_home or not s_away:
            return adj_1x2, adj_ou, logs + ["No Standings Data"]

        # --- HEURISTIC ACCUMULATOR (H1-H6 propose deltas; capped before applying) ---
        delta = [0.0, 0.0, 0.0]

        def _boost(idx, magnitude, label):
            """Single-outcome lift (e.g. team in winning form)."""
            delta[idx] += magnitude
            logs.append(f"{label} (+{magnitude:.2f})")

        def _fade(opponent_idx, magnitude, label):
            """Team is fading: split lift between opponent and Draw to avoid pro-home bias."""
            opp_share = magnitude * self.FADE_TO_OPPONENT_SHARE
            draw_share = magnitude * (1.0 - self.FADE_TO_OPPONENT_SHARE)
            delta[opponent_idx] += opp_share
            delta[D] += draw_share
            logs.append(f"{label} (+{opp_share:.2f} opp, +{draw_share:.2f} D)")

        # --- HEURISTIC 1: Overall rank differential ---
        try:
            h_rank = int(s_home['rank'])
            a_rank = int(s_away['rank'])
            diff = a_rank - h_rank
            if diff >= 5:
                _boost(H, min(0.02 * (diff / 5), 0.10), f"Rank Boost Home H#{h_rank} vs A#{a_rank}")
            elif diff <= -5:
                _boost(A, min(0.02 * (abs(diff) / 5), 0.10), f"Rank Boost Away H#{h_rank} vs A#{a_rank}")
        except (KeyError, ValueError, TypeError) as e:
            logs.append(f"H1 Error: {type(e).__name__}: {e}")

        # --- HEURISTIC 2: Specific rank (Home Table vs Away Table) ---
        if s_home_spec and s_away_spec:
            try:
                h_rank_spec = int(s_home_spec['rank'])
                a_rank_spec = int(s_away_spec['rank'])
                diff_spec = a_rank_spec - h_rank_spec
                if diff_spec >= 5:
                    _boost(H, min(0.03 * (diff_spec / 5), 0.10),
                           f"Spec Rank Home H_h#{h_rank_spec} vs A_a#{a_rank_spec}")
                elif diff_spec <= -5:
                    _boost(A, min(0.03 * (abs(diff_spec) / 5), 0.10),
                           f"Spec Rank Away H_h#{h_rank_spec} vs A_a#{a_rank_spec}")
            except (KeyError, ValueError, TypeError) as e:
                logs.append(f"H2 Error: {type(e).__name__}: {e}")

        # --- HEURISTIC 3: Form momentum (overall) ---
        # Symmetric: hot team -> boost; cold team -> fade split between opponent and Draw.
        if f_home:
            res = f_home.get('last_5_results', '')
            if res.count('W') >= 4:
                _boost(H, 0.05, f"Form Boost Home (W={res.count('W')})")
            elif res.count('L') >= 4:
                _fade(A, 0.05, f"Form Fade Home (L={res.count('L')})")
        if f_away:
            res = f_away.get('last_5_results', '')
            if res.count('W') >= 4:
                _boost(A, 0.05, f"Form Boost Away (W={res.count('W')})")
            elif res.count('L') >= 4:
                _fade(H, 0.05, f"Form Fade Away (L={res.count('L')})")

        # --- HEURISTIC 4: Form momentum (venue-specific) ---
        if f_home_spec:
            res = f_home_spec.get('last_5_results', '')
            if res.count('W') >= 4:
                _boost(H, 0.06, f"Spec Form Home Boost (W={res.count('W')})")
            elif res.count('L') >= 4:
                _fade(A, 0.06, f"Spec Form Home Fade (L={res.count('L')})")
        if f_away_spec:
            res = f_away_spec.get('last_5_results', '')
            if res.count('W') >= 4:
                _boost(A, 0.06, f"Spec Form Away Boost (W={res.count('W')})")
            elif res.count('L') >= 4:
                _fade(H, 0.06, f"Spec Form Away Fade (L={res.count('L')})")

        # --- HEURISTIC 6: Form trend (L5 vs L10) ---
        def _win_rate(form_entry):
            if not form_entry:
                return 0.0
            res = form_entry.get('last_5_results', '')
            w, d, l = res.count('W'), res.count('D'), res.count('L')
            total = w + d + l
            return (w / total) if total > 0 else 0.0

        f_home_10 = self.find_team_stats(self.form_lookup_10, "", league, home)
        if f_home and f_home_10:
            wr_5 = _win_rate(f_home)
            wr_10 = _win_rate(f_home_10)
            if wr_5 >= (wr_10 + 0.3):
                _boost(H, 0.04, f"Home Heating Up L5:{wr_5:.0%}/L10:{wr_10:.0%}")
            elif wr_5 <= (wr_10 - 0.3):
                _fade(A, 0.03, f"Home Cooling L5:{wr_5:.0%}/L10:{wr_10:.0%}")
            elif wr_5 >= 0.70 and wr_10 >= 0.60:
                _boost(H, 0.03, f"Home Consistent L5:{wr_5:.0%}/L10:{wr_10:.0%}")

        f_away_10 = self.find_team_stats(self.form_lookup_10, "", league, away)
        if f_away and f_away_10:
            wr_5 = _win_rate(f_away)
            wr_10 = _win_rate(f_away_10)
            if wr_5 >= (wr_10 + 0.3):
                _boost(A, 0.04, f"Away Heating Up L5:{wr_5:.0%}/L10:{wr_10:.0%}")
            elif wr_5 <= (wr_10 - 0.3):
                _fade(H, 0.03, f"Away Cooling L5:{wr_5:.0%}/L10:{wr_10:.0%}")
            elif wr_5 >= 0.70 and wr_10 >= 0.60:
                _boost(A, 0.03, f"Away Consistent L5:{wr_5:.0%}/L10:{wr_10:.0%}")

        # --- APPLY CAP, ADD DELTAS, NORMALIZE ---
        cap = self.MAX_TOTAL_BOOST_PER_CLASS
        capped = [max(-cap, min(cap, d)) for d in delta]
        if capped != delta:
            logs.append(
                f"Boost Cap: H{delta[H]:+.2f}->{capped[H]:+.2f} "
                f"D{delta[D]:+.2f}->{capped[D]:+.2f} "
                f"A{delta[A]:+.2f}->{capped[A]:+.2f}"
            )

        adj_1x2 = [max(0.0, adj_1x2[i] + capped[i]) for i in (H, D, A)]
        total = sum(adj_1x2)
        if total > 0:
            adj_1x2 = [x / total for x in adj_1x2]

        # --- HEURISTIC 5: High-scoring teams (O/U) ---
        try:
            h_mp = int(s_home['matches_played'])
            a_mp = int(s_away['matches_played'])
            h_gf = int(s_home['goals'].split(':')[0])
            a_gf = int(s_away['goals'].split(':')[0])
            h_avg = h_gf / h_mp
            a_avg = a_gf / a_mp
            if (h_avg + a_avg) > 3.5:
                adj_ou[1] += 0.05
                logs.append(f"Goal Fest Boost (Avg GF: {h_avg + a_avg:.2f})")
            total_ou = sum(adj_ou)
            if total_ou > 0:
                adj_ou = [x / total_ou for x in adj_ou]
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
            logs.append(f"H5 Error: {type(e).__name__}: {e}")

        # --- HEURISTIC 7: Value bet logging ---
        # adj_ou layout is fixed by caller (predict_matches.py): [Under, Over].
        odds = match_info.get('Odds', {})
        try:
            o_h = float(odds.get('1', 0.0))
            o_d = float(odds.get('X', 0.0))
            o_a = float(odds.get('2', 0.0))
            imp_h = (1.0 / o_h) if o_h > 1.0 else 0.0
            imp_d = (1.0 / o_d) if o_d > 1.0 else 0.0
            imp_a = (1.0 / o_a) if o_a > 1.0 else 0.0
            if adj_1x2[H] - imp_h > 0.05: logs.append(f"Value 1(+{adj_1x2[H] - imp_h:.2%})")
            if adj_1x2[D] - imp_d > 0.05: logs.append(f"Value X(+{adj_1x2[D] - imp_d:.2%})")
            if adj_1x2[A] - imp_a > 0.05: logs.append(f"Value 2(+{adj_1x2[A] - imp_a:.2%})")
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
            logs.append(f"H7 1X2 Value Error: {type(e).__name__}: {e}")

        try:
            o_o = float(odds.get('O', 0.0))
            o_u = float(odds.get('U', 0.0))
            imp_o = (1.0 / o_o) if o_o > 1.0 else 0.0
            imp_u = (1.0 / o_u) if o_u > 1.0 else 0.0
            if adj_ou[1] - imp_o > 0.05: logs.append(f"Value O(+{adj_ou[1] - imp_o:.2%})")
            if adj_ou[0] - imp_u > 0.05: logs.append(f"Value U(+{adj_ou[0] - imp_u:.2%})")
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
            logs.append(f"H7 OU Value Error: {type(e).__name__}: {e}")

        return adj_1x2, adj_ou, logs

    def _lookup_league_stat(self, league_name):
        """Resolve a 'COUNTRY: League' string to its league_stats entry, with fuzzy fallback."""
        if ":" not in league_name:
            return None
        parts = league_name.split(":", 1)
        c_in = parts[0].strip().upper()
        l_in = parts[1].strip()
        key = f"{c_in}|{l_in}"
        league_stat = self.league_stats.get(key)
        if league_stat:
            return league_stat
        for k in self.league_stats.keys():
            if c_in in k and l_in in k:
                return self.league_stats[k]
        return None
