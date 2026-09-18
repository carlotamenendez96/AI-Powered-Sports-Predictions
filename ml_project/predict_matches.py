import pandas as pd
import numpy as np
import time
import xgboost as xgb
import json
import os
import datetime
from feature_engineering import (FeatureEngineer, FORM_WINDOWS,
                                 H2H_FEATURES, h2h_features)
from entity_resolver import EntityResolver
from data_loader import DataLoader
import glob

from heuristic_adjuster import HeuristicAdjuster
from model_registry import load_1x2_model, load_ou_model

# Per-league Platt calibration (Phase C4). Loaded once at construction.
from ml_project.calibration.apply import (
    apply_platt_1x2, apply_platt_ou, load_calibration_data,
)
from ml_project.calibration.league_aliases import LEAGUE_ALIASES
# National-team competitions are routed to the NT model (predict_nt_batch.py),
# NOT this club predictor — see scripts/national_teams/.
from ml_project.national_teams.nt_competitions import is_international


def _load_league_calibration_flag(default: bool = True) -> bool:
    """Read `use_league_calibration` from `data_sets/betting_config.json`
    (football's sport entry). Falls back to `default` on any error."""
    try:
        with open('data_sets/betting_config.json', 'r') as f:
            cfg = json.load(f)
        return bool(cfg.get('sports', {}).get('football', {}).get(
            'use_league_calibration', default))
    except Exception:
        return default


class MatchPredictor:
    def __init__(self, history_dir="data_sets/MatchHistory", scraper_output="output/output.json"):
        # Load Models — 1X2 head resolves through the registry seam: a
        # model_meta_1x2.json sidecar (if present) picks the family; otherwise
        # it falls back to the legacy XGBoost JSON, so a pre-seam deployment is
        # unchanged.
        self.model_1x2, self.model_1x2_family = load_1x2_model("models")

        # Load O/U Model (Poisson regressor) — also seam-routed via meta with
        # a legacy XGBoost-JSON fallback. predict(X) -> lambda.
        self.model_ou, self.model_ou_family = load_ou_model("models")
        
        # Draw Model (Stage A): no longer used in the prediction loop —
        # the 2-stage average was acting as an implicit base-rate regularizer,
        # made redundant by per-league Platt calibration (Phase C4). The
        # trained model file is left on disk for backward compatibility / a
        # future reactivation; not loaded here.
        self.model_draw = None
        
        # Load Features
        with open("models/features_1x2.json", 'r') as f:
            self.features_1x2 = json.load(f)
        with open("models/features_ou.json", 'r') as f:
            self.features_ou = json.load(f)
        # Draw features kept as empty list — see note on model_draw above.
        self.features_draw = []
            
        self.output_file = scraper_output
        self.fe = FeatureEngineer()
        self.resolver = EntityResolver()
        
        print("Loading historical data for feature calculation...")
        self.loader = DataLoader(history_dir)
        self.history_df = self.loader.load_historical_data()
        self.history_df = self.history_df.sort_values('date')

        # League categories the model actually saw at training time, read
        # straight out of the saved model (authoritative — the live history
        # can contain newer leagues the last-trained model never saw). Newer
        # XGBoost (enable_categorical) hard-errors on an unseen category at
        # inference instead of treating it as missing, so `_as_league_category`
        # rebuilds the inference column against exactly this set and lets any
        # unseen league fall through as missing. See run_predictions failures
        # on "WORLD: World Championship" and "Veikkausliiga " (2026-06-05).
        # league_cat is an XGBoost-only categorical; other families consume a
        # numeric-only feature list. The training categories are identical
        # across heads (same data), so decode them from whichever served head
        # is XGBoost — preferring 1X2, falling back to O/U so a non-XGBoost
        # 1X2 challenger paired with the XGBoost O/U head still gets correct
        # category handling.
        if self.model_1x2_family == 'xgboost':
            self.known_leagues = self._load_model_league_categories("models/xgb_model_1x2.json")
        elif self.model_ou_family == 'xgboost':
            self.known_leagues = self._load_model_league_categories("models/xgb_model_ou.json")
        else:
            self.known_leagues = []

        self.adjuster = HeuristicAdjuster()

        # Per-league Platt calibrators (C4). Empty dict if file missing.
        # Applied to raw probs *before* the heuristic adjuster.
        self.calibration_data = load_calibration_data('data_sets/league_calibration.json')
        self.use_league_calibration = _load_league_calibration_flag()
        if self.calibration_data and self.use_league_calibration:
            n_leagues = len(self.calibration_data.get('leagues', {}))
            print(f"Loaded calibration for {n_leagues} leagues "
                  f"(generated {self.calibration_data.get('generated_at', '?')}).")
        elif not self.use_league_calibration:
            print("League calibration disabled via betting_config.json:use_league_calibration=false")
        else:
            print("No league_calibration.json found — predictions will use raw probs.")

    @staticmethod
    def _load_model_league_categories(model_path):
        """Decode the league_cat training categories stored inside a saved
        XGBoost model. Returns a sorted list (empty on any failure).

        The model JSON keeps per-feature category dictionaries under
        `learner.gradient_booster.model.cats.enc` — one entry per feature, each
        a flat UTF-8 byte `values` array sliced by `offsets`. We pick the entry
        for the single categorical feature (identified via `feature_types == 'c'`).
        """
        try:
            with open(model_path, 'r') as f:
                model = json.load(f)
            learner = model['learner']
            cat_idx = learner['feature_types'].index('c')
            col = learner['gradient_booster']['model']['cats']['enc'][cat_idx]
            offs, vals = col['offsets'], col['values']
            cats = [bytes(vals[offs[i]:offs[i + 1]]).decode('utf-8')
                    for i in range(len(offs) - 1)]
            return sorted(cats)
        except Exception as e:
            print(f"[WARN] Could not read league categories from {model_path}: {e}")
            return []

    def _as_league_category(self, series):
        """Coerce a league_cat column to a pandas Categorical whose categories
        are exactly the leagues seen at training time.

        Leading/trailing whitespace is stripped first ("Veikkausliiga " ->
        "Veikkausliiga") so cosmetic name drift doesn't read as a new league.
        Any value still not in the training set (e.g. "WORLD: World
        Championship") becomes a missing *code* while the column keeps a
        string-typed category dtype. This restores the pre-upgrade behaviour
        where XGBoost treated an unseen category as missing instead of either
        hard-erroring ("category not in the training set") or tripping the
        float-dtype guard a bare NaN would. With no known leagues, fall back to
        a plain categorical so behaviour is unchanged.
        """
        cleaned = series.astype('string').str.strip()
        if not self.known_leagues:
            return cleaned.astype('category')
        return pd.Categorical(cleaned, categories=self.known_leagues)

    def get_team_stats(self, team_name, date_before):
        """
        Calculates the rolling features for a team given the historical database.
        Returns a dictionary of features.
        """
        if not team_name:
            return None
            
        # Filter games played by this team BEFORE the match date
        # Note: We use date_before (string YYYY-MM-DD or datetime)
        mask = (self.history_df['date'] < date_before) & \
               ((self.history_df['home_team'] == team_name) | (self.history_df['away_team'] == team_name))
        
        team_games = self.history_df[mask].tail(50).copy() # Take last 50 to stand a chance of finding 5
        
        if team_games.empty:
            return None
            
        # We need to manually calculate the rolling stats because we can't easily rely on pre-calculated rolling columns
        # if the last match was months ago. But wait, `add_rolling_features` added rolling cols to the WHOLE history?
        # NO, `predict_matches.py` loads raw CSV data via DataLoader. It does NOT have the rolling columns pre-calculated.
        # So we must calculate them on the fly from the last 5 rows.
        
        stats = {}
        
        # Helper to extract stats from a row for the focus team
        def extract_match_stats(row, team):
            is_home = (row['home_team'] == team)
            pts = 3 if row['FTR'] == ('H' if is_home else 'A') else (1 if row['FTR'] == 'D' else 0)
            gf = row['FTHG'] if is_home else row['FTAG']
            ga = row['FTAG'] if is_home else row['FTHG']
            sf = row['HST'] if is_home else row['AST'] # HST = Home Shots on Target
            sa = row['AST'] if is_home else row['HST']
            cf = row['HC'] if is_home else row['AC']
            ca = row['AC'] if is_home else row['HC']

            # OU Logic: 1 if > 2.5 else 0
            ou = 1 if (row['FTHG'] + row['FTAG']) > 2.5 else 0

            return {'pts': pts, 'gf': gf, 'ga': ga, 'sf': sf, 'sa': sa, 'cf': cf, 'ca': ca, 'ou': ou}

        # 1. Overall form, one entry per window in FORM_WINDOWS (L5 + L10 + L15).
        #    Mirrors FeatureEngineer._calculate_rolling_multi: same corpus, same
        #    windows, same "mean over the last n matches before this date", so
        #    the serve-time columns match what training built. The `.tail(50)`
        #    slice above comfortably covers the largest window.
        #    L5 keeps the unsuffixed key names the rest of this file reads.
        for n, sfx in FORM_WINDOWS:
            window = team_games.tail(n)
            sums = {'pts': 0, 'gf': 0, 'ga': 0, 'sf': 0, 'sa': 0, 'cf': 0, 'ca': 0, 'ou': 0}
            count = 0
            for _, row in window.iterrows():
                s = extract_match_stats(row, team_name)
                for k in sums:
                    sums[k] += s.get(k, 0)   # Handle missing stats (NaN -> 0)
                count += 1
            for k in sums:
                # form_ou is the proportion of matches over 2.5; the rest are means.
                stats[f'form_{k}{sfx}'] = (sums[k] / count) if count > 0 else 0

        # 2. Venue Specific Form (Last 5 Home or Away)
        # If we are verifying this team as HOME team, we want last 5 HOME games.
        # But this method is generic. Let's return both or split?
        # Let's return separate dicts for Home and Away specific form.

        return stats

    def get_h2h_stats(self, home_team, away_team, date_before):
        """Head-to-head record between these two teams, oriented to `home_team`.

        Mirrors FeatureEngineer._add_h2h_features: same MatchHistory corpus,
        same H2H_WINDOW, same orientation, same strictly-before filter, and the
        same `h2h_features` helper does the arithmetic — so a column can't mean
        one thing in training and another at serve time.

        Team names must already be canonical (entity_resolver output), as the
        corpus is keyed on the football-data.co.uk spelling, not Flashscore's.
        """
        if not home_team or not away_team:
            return h2h_features([], home_team)

        hist = self.history_df
        pair = (((hist['home_team'] == home_team) & (hist['away_team'] == away_team)) |
                ((hist['home_team'] == away_team) & (hist['away_team'] == home_team)))
        meetings = hist[(hist['date'] < date_before) & pair].sort_values('date')

        return h2h_features(
            [(r.home_team, r.FTHG, r.FTAG) for r in meetings.itertuples()],
            home_team,
        )

    def get_venue_specific_stats(self, team_name, is_home_focus, date_before):
        """
        Get stats for Last 5 HOME games if is_home_focus=True, else Last 5 AWAY games.
        """
        if not team_name: return None
        
        if is_home_focus:
            mask = (self.history_df['date'] < date_before) & (self.history_df['home_team'] == team_name)
        else:
            mask = (self.history_df['date'] < date_before) & (self.history_df['away_team'] == team_name)
            
        specific_games = self.history_df[mask].tail(5)
        
        sums = {'pts': 0, 'gf': 0, 'ga': 0, 'sf': 0, 'sa': 0} # No corners needed for specific as per feature list
        count = 0
        
        for _, row in specific_games.iterrows():
            # Extract stats logic duplicated... simplify
            # Just grab raw columns
            if is_home_focus:
                pts = 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
                gf = row['FTHG']
                ga = row['FTAG']
                sf = row['HST']
                sa = row['AST']
            else:
                pts = 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)
                gf = row['FTAG']
                ga = row['FTHG']
                sf = row['AST']
                sa = row['HST']
            
            sums['pts'] += pts
            sums['gf'] += gf
            sums['ga'] += ga
            sums['sf'] += sf if pd.notna(sf) else 0
            sums['sa'] += sa if pd.notna(sa) else 0
            count += 1
            
        if count > 0:
            return {
                'spec_pts': sums['pts'] / count,
                'spec_gf': sums['gf'] / count,
                'spec_ga': sums['ga'] / count,
                'spec_sf': sums['sf'] / count,
                'spec_sa': sums['sa'] / count
            }
        else:
             return {k: 0 for k in ['spec_pts', 'spec_gf', 'spec_ga', 'spec_sf', 'spec_sa']}



    def predict(self):
        start_time = time.time()
        if not os.path.exists(self.output_file):
            print("No output.json found.")
            return

        with open(self.output_file, 'r') as f:
            try:
                upcoming_matches = json.load(f)
            except json.JSONDecodeError:
                print(f"[-] Error: The file {os.path.basename(self.output_file)} is corrupt or empty.")
                print("[!] Please check 'Force Scrape' in the Dashboard and run Prediction again.")
                return
            
        predictions = []
        match_dates = set()
        
        print(f"Predicting {len(upcoming_matches)} matches...")
        
        for match in upcoming_matches:
            scraper_home = match.get('home_team')
            scraper_away = match.get('away_team')
            league_name = match.get('league', 'Unknown')

            # ROUTING: national-team competitions (World Cup, Euro, Nations
            # League) are handled by the NT model, not this club predictor.
            # The club model has no ELO/standings for national teams, so skip
            # them here — predict_nt_batch.py appends their rows afterwards.
            if is_international(league_name):
                continue

            # Strip playoff / group-stage / phase suffixes so standings
            # lookup, Platt calibration, and the league_cat feature all
            # key off the same canonical base name. Examples:
            #   "ENGLAND: Championship - Play Offs"           -> "ENGLAND: Championship"
            #   "EUROPE: Champions League - League phase"     -> "EUROPE: Champions League"
            #   "SPAIN: LaLiga 2 - Promotion - Play Offs: Semi-finals"
            #                                                  -> "SPAIN: LaLiga 2"
            if ":" in league_name:
                _country, _rest = league_name.split(":", 1)
                _base = _rest.split(" - ", 1)[0].strip()
                league_name = f"{_country.strip()}: {_base}"

            # Translate Flashscore-format name to the training-time
            # league_cat value (e.g. "ENGLAND: Premier League" -> "E0",
            # "BRAZIL: Serie A Betano" -> "Serie A"). Unaliased leagues
            # fall through unchanged and remain an unseen category for
            # the model (same behaviour as before the alias lookup).
            league_cat_value = LEAGUE_ALIASES.get(league_name, league_name)

            # --- LEAGUE FILTERING ---
            SUPPORTED_COUNTRIES = {
                'ENGLAND', 'SPAIN', 'FRANCE', 'GERMANY', 'ITALY', 'NETHERLANDS', 'PORTUGAL', 'SCOTLAND', 
                'TURKEY', 'USA', 'POLAND', 'RUSSIA', 'NORWAY', 'SWEDEN', 'FINLAND', 'ROMANIA', 'GREECE', 
                'IRELAND', 'SWITZERLAND', 'JAPAN', 'MEXICO', 'BELGIUM', 'AUSTRIA', 'DENMARK', 'CZECH REPUBLIC', 'CROATIA',
                'EUROPE', 'WORLD'
            }
            country_prefix = league_name.split(':')[0].upper().strip()
            if country_prefix not in SUPPORTED_COUNTRIES:
                continue

            # Resolve Names
            canon_home = self.resolver.get_canonical_name(scraper_home)
            canon_away = self.resolver.get_canonical_name(scraper_away)
            
            # ELO Lookup
            home_elo = self.resolver.get_elo(scraper_home)
            away_elo = self.resolver.get_elo(scraper_away)
            
            # Defaults
            if not home_elo or pd.isna(home_elo): home_elo = 1500
            if not away_elo or pd.isna(away_elo): away_elo = 1500

            match_date_str = match.get('start_time', 'Unknown')
            match_date_obj = pd.Timestamp.now() # Default
            
            if match_date_str != 'Unknown':
                 try:
                     # Parse the FULL kickoff datetime (e.g. '30.05.2026 19:00'),
                     # keeping the time — it's surfaced in the bet slip downstream.
                     match_date_obj = pd.to_datetime(match_date_str, dayfirst=True)
                     # Standardize to ISO format for output consistency
                     match_date_str = match_date_obj.strftime("%Y-%m-%d %H:%M")
                     match_dates.add(match_date_obj.strftime("%Y-%m-%d"))
                 except:
                     pass

            # Calculate Stats from DB
            try:
                h_stats = self.get_team_stats(canon_home, match_date_obj)
                a_stats = self.get_team_stats(canon_away, match_date_obj)
                
                h_spec = self.get_venue_specific_stats(canon_home, True, match_date_obj)
                a_spec = self.get_venue_specific_stats(canon_away, False, match_date_obj)
                h2h = self.get_h2h_stats(canon_home, canon_away, match_date_obj)
            except Exception as e:
                # print(f"Error calculating stats for {scraper_home} vs {scraper_away}: {e}")
                h_stats = None # Will fallback to zeros
                h2h = h2h_features([], canon_home)   # no-history sentinel, not zeros
            
            # Zero-fill for a team with no corpus history. Must cover every
            # window in FORM_WINDOWS, not just L5, or the input-row build below
            # raises KeyError on the suffixed columns.
            _zero_form = {f'form_{stem}{sfx}': 0
                          for _, sfx in FORM_WINDOWS
                          for stem in ('pts', 'gf', 'ga', 'sf', 'sa', 'cf', 'ca', 'ou')}
            if not h_stats: h_stats = dict(_zero_form)
            if not a_stats: a_stats = dict(_zero_form)
            if not h_spec: h_spec = {k: 0 for k in ['spec_pts', 'spec_gf', 'spec_ga', 'spec_sf', 'spec_sa']}
            if not a_spec: a_spec = {k: 0 for k in ['spec_pts', 'spec_gf', 'spec_ga', 'spec_sf', 'spec_sa']}

            # Odds
            b365_h_str = match.get('interaction_1x2_1', '-')
            b365_d_str = match.get('interaction_1x2_X', '-')
            b365_a_str = match.get('interaction_1x2_2', '-')
            
            if '-' in [b365_h_str, b365_d_str, b365_a_str]: continue
            try:
                b365_h = float(b365_h_str.replace(',', '.'))
                b365_d = float(b365_d_str.replace(',', '.'))
                b365_a = float(b365_a_str.replace(',', '.'))
            except: continue
                
            elo_diff = home_elo - away_elo
            abs_elo_diff = abs(elo_diff)

            # Season-to-date PPG and league-relative strength from current standings.
            # Mirrors FeatureEngineer._add_ppg_strength_features used at training time.
            h_ppg, h_att, h_def = self.adjuster.get_team_strength(league_name, scraper_home)
            a_ppg, a_att, a_def = self.adjuster.get_team_strength(league_name, scraper_away)
            ppg_diff = h_ppg - a_ppg
            abs_ppg_diff = abs(ppg_diff)
            att_def_diff = (h_att - a_att) - (h_def - a_def)

            abs_form_pts_diff = abs(h_stats['form_pts'] - a_stats['form_pts'])

            # Construct Input Vector
            input_row = {
                'B365H': b365_h, 'B365D': b365_d, 'B365A': b365_a,
                'H_form_pts': h_stats['form_pts'], 'H_form_gf': h_stats['form_gf'], 'H_form_ga': h_stats['form_ga'],
                'A_form_pts': a_stats['form_pts'], 'A_form_gf': a_stats['form_gf'], 'A_form_ga': a_stats['form_ga'],
                'H_elo': home_elo, 'A_elo': away_elo,
                'H_home_pts': h_spec['spec_pts'], 'H_home_gf': h_spec['spec_gf'], 'H_home_ga': h_spec['spec_ga'],
                'H_home_sf': h_spec['spec_sf'], 'H_home_sa': h_spec['spec_sa'],
                'A_away_pts': a_spec['spec_pts'], 'A_away_gf': a_spec['spec_gf'], 'A_away_ga': a_spec['spec_ga'],
                'A_away_sf': a_spec['spec_sf'], 'A_away_sa': a_spec['spec_sa'],
                'H_form_sf': h_stats['form_sf'], 'H_form_sa': h_stats['form_sa'],
                'H_form_cf': h_stats['form_cf'], 'H_form_ca': h_stats['form_ca'],
                'A_form_sf': a_stats['form_sf'], 'A_form_sa': a_stats['form_sa'],
                'A_form_cf': a_stats['form_cf'], 'A_form_ca': a_stats['form_ca'],
                'league_cat': league_cat_value,
                'match_id': match.get('match_id', ''),
                'elo_diff': elo_diff,
                'abs_elo_diff': abs_elo_diff,
                'abs_ppg_diff': abs_ppg_diff,
                'abs_form_pts_diff': abs_form_pts_diff,
                # Season-to-date PPG and league-relative strength
                'H_ppg': h_ppg, 'A_ppg': a_ppg, 'ppg_diff': ppg_diff,
                'H_att': h_att, 'A_att': a_att,
                'H_def': h_def, 'A_def': a_def,
                'att_def_diff': att_def_diff,
                # Head-to-head (mirrors FeatureEngineer._add_h2h_features)
                **{c: h2h[c] for c in H2H_FEATURES},
            }

            # Longer-horizon form + trend, mirroring FeatureEngineer exactly:
            # same windows (FORM_WINDOWS), same corpus, same subtraction.
            # Only pts/gf/ga are carried — see the note on common_features in
            # train_model.py for why shots/corners stay L5-only.
            for _n, _sfx in FORM_WINDOWS:
                if not _sfx:
                    continue    # L5 is already in the row above, unsuffixed
                for _stem in ('pts', 'gf', 'ga'):
                    input_row[f'H_form_{_stem}{_sfx}'] = h_stats[f'form_{_stem}{_sfx}']
                    input_row[f'A_form_{_stem}{_sfx}'] = a_stats[f'form_{_stem}{_sfx}']
            input_row['H_form_trend'] = h_stats['form_pts'] - h_stats['form_pts_l15']
            input_row['A_form_trend'] = a_stats['form_pts'] - a_stats['form_pts_l15']
            
            # Generate DF for prediction
            input_df = pd.DataFrame([input_row])
            input_df['league_cat'] = self._as_league_category(input_df['league_cat'])
            
            # --- 1X2 PROBABILITIES (single-stage; the binary draw model
            # used to average in here, but it acted as an implicit
            # regularizer toward the base draw rate — exactly what the
            # downstream per-league Platt calibrator (Phase C4) handles
            # in a statistically grounded way. See FOOTBALL_NEXT_STEPS D0 for
            # the diagnosis and reasoning behind dropping the 2-stage
            # combination.) ---

            for c in self.features_1x2:
                if c not in input_df.columns: input_df[c] = 0

            probs_1x2 = self.model_1x2.predict_proba(input_df[self.features_1x2])[0]
            # [P(H), P(D), P(A)] — Platt calibration is applied below
            # to correct per-league bias.

            # Decision
            pred_1x2_idx = probs_1x2.argmax()
            pred_1x2_label = ['Home', 'Draw', 'Away'][pred_1x2_idx]
            conf_1x2 = probs_1x2[pred_1x2_idx]
            
            # O/U
            # Add OU specific features
            input_row['H_form_ou'] = h_stats['form_ou']
            input_row['A_form_ou'] = a_stats['form_ou']
            input_df_ou = pd.DataFrame([input_row])
            input_df_ou['league_cat'] = self._as_league_category(input_df_ou['league_cat'])
            for c in self.features_ou:
                if c not in input_df_ou.columns: input_df_ou[c] = 0
                
            # --- POISSON PREDICTION ---
            # Output is expected goals (lambda)
            pred_lam = self.model_ou.predict(input_df_ou[self.features_ou])[0]
            
            # Convert lambda to Prob(> 2.5) using Poisson
            # P(X<=2) = e^-lam * (1 + lam + lam^2/2)

            prob_le_2 = np.exp(-pred_lam) * (1 + pred_lam + (pred_lam**2 / 2))
            prob_over = 1.0 - prob_le_2
            prob_under = 1.0 - prob_over
            
            # Construct probs array [Under, Over] to match old interface
            probs_ou = np.array([prob_under, prob_over])

            # --- PER-LEAGUE PLATT CALIBRATION (C4) ---
            # Correct systematic per-league probability bias *before* the
            # heuristic adjuster does its work. The adjuster's domain rules
            # then operate on already-calibrated probs.
            raw_probs_1x2 = probs_1x2.copy()
            raw_probs_ou = probs_ou.copy()
            probs_1x2, applied_1x2, cal_src_1x2 = apply_platt_1x2(
                probs_1x2, league_name, self.calibration_data,
                enabled=self.use_league_calibration)
            probs_ou, applied_ou, cal_src_ou = apply_platt_ou(
                probs_ou, league_name, self.calibration_data,
                enabled=self.use_league_calibration)

            # Extract O/U Odds (Moved Up)
            try:
                ov_str = match.get('over_2_5', '0.0')
                un_str = match.get('under_2_5', '0.0')
                ov_val = float(ov_str.replace(',', '.')) if ov_str and ov_str != '-' else 0.0
                un_val = float(un_str.replace(',', '.')) if un_str and un_str != '-' else 0.0
            except:
                ov_val, un_val = 0.0, 0.0
            
            # --- HEURISTIC ADJUSTMENT ---
            match_info = {
                'League': league_name,
                'Home Team': scraper_home,
                'Away Team': scraper_away,
                'Odds': {
                    '1': b365_h,
                    'X': b365_d,
                    '2': b365_a,
                    'O': ov_val,
                    'U': un_val
                }
            }
            adj_1x2, adj_ou, adj_logs = self.adjuster.adjust_probabilities(match_info, probs_1x2, probs_ou)
            
            # Select Final Predictions based on ADJUSTED probabilities
            # 1X2
            pred_1x2_idx = adj_1x2.index(max(adj_1x2))
            # Use ADJUSTED probs for final decision
            # Use ADJUSTED probs for final decision
            # NEW RULE: Pick Draw only if P(D) > max(P(H), P(A)) + margin
            p_h, p_d, p_a = adj_1x2
            margin = 0.05 # Conservative margin (User suggested 0.03-0.06)
            
            if p_d > (max(p_h, p_a) + margin):
                pred_1x2_label = 'X'
                conf_1x2 = p_d
            else:
                # If Draw is not confident enough, pick the max of Home or Away
                if p_h >= p_a:
                    pred_1x2_label = '1'
                    conf_1x2 = p_h
                else:
                    pred_1x2_label = '2'
                    conf_1x2 = p_a

            
            # OU
            pred_ou_idx = adj_ou.index(max(adj_ou))
            pred_ou_label = "Over 2.5" if pred_ou_idx == 1 else "Under 2.5"
            conf_ou = adj_ou[pred_ou_idx]
            
            
            # Extract O/U Odds (Determine which one matches prediction)
            ou_odd = ov_val if pred_ou_label == "Over 2.5" else un_val
            
            # Odd 1X2
            pred_odd = b365_h if pred_1x2_label == '1' else (b365_d if pred_1x2_label == 'X' else b365_a)
            
            # --- BETTING STRATEGY (EV & KELLY) ---
            def calculate_kelly(odd, prob):
                if odd <= 1.0 or prob <= 0: return 0.0, 0.0
                b = odd - 1.0
                q = 1.0 - prob
                # Full Kelly
                f = (b * prob - q) / b
                f = max(f, 0.0) # No negative bets
                # Quarter Kelly
                return (prob * odd) - 1.0, f * 0.25

            # 1X2 Strategy
            ev_1x2, kelly_1x2 = calculate_kelly(pred_odd, conf_1x2)
            
            # O/U Strategy
            ev_ou, kelly_ou = calculate_kelly(ou_odd, conf_ou)

            predictions.append({
                'Date': match_date_str,
                'League': league_name,
                'Home Team': scraper_home,
                'Away Team': scraper_away,
                'Home ELO': int(home_elo),
                'Away ELO': int(away_elo),
                'Prediction 1X2': pred_1x2_label,
                'Prediction 1X2 Odd': f"{pred_odd:.2f}",
                'Conf 1X2': f"{conf_1x2:.2f}",
                'EV 1X2': f"{ev_1x2:.2f}",
                'Kelly 1X2': f"{kelly_1x2:.2%}",
                'Prediction O/U': pred_ou_label,
                'Prediction O/U Odd': f"{ou_odd:.2f}",
                'Conf O/U': f"{conf_ou:.2f}",
                'EV O/U': f"{ev_ou:.2f}",
                'Kelly O/U': f"{kelly_ou:.2%}",
                'Home Win %': f"{adj_1x2[0]:.2f}",
                'Draw %': f"{adj_1x2[1]:.2f}",
                'Away Win %': f"{adj_1x2[2]:.2f}",
                'Over %': f"{adj_ou[1]:.2f}",
                'Under %': f"{adj_ou[0]:.2f}",
                # Raw probabilities (pre-calibration, pre-heuristic) for audit.
                'Home Win % (raw)': f"{raw_probs_1x2[0]:.2f}",
                'Draw % (raw)': f"{raw_probs_1x2[1]:.2f}",
                'Away Win % (raw)': f"{raw_probs_1x2[2]:.2f}",
                'Over % (raw)': f"{raw_probs_ou[1]:.2f}",
                'Under % (raw)': f"{raw_probs_ou[0]:.2f}",
                'Cal 1X2 Source': cal_src_1x2 or '',
                'Cal O/U Source': cal_src_ou or '',
                'Adj Logs': "; ".join(adj_logs),
                'match_id': match.get('match_id', '')
            })


        # Save Logic (Same as before)
        if match_dates:
            # Use the most common date (mode) to avoid one-off early matches affecting the filename
            # match_dates is a set of strings. We need all dates to find mode.
            # Re-collect all dates
            all_dates = []
            for match in upcoming_matches:
                 ms = match.get('start_time', 'Unknown')
                 if ms != 'Unknown':
                     try:
                         d_str = ms.split(' ')[0]
                         d_obj = pd.to_datetime(d_str, dayfirst=True)
                         all_dates.append(d_obj.strftime("%Y-%m-%d"))
                     except: pass
            
            if all_dates:
                 target_date = pd.Series(all_dates).mode()[0]
            else:
                 target_date = sorted(list(match_dates))[0]

            filename = f"output/predictions_{target_date}.csv"
        else:
            filename = "output/predictions_unknown_date.csv"

        res_df = pd.DataFrame(predictions)
        if not res_df.empty:
             # Reorder cols
             desired = ['Date', 'League', 'Home Team', 'Away Team', 'Home ELO', 'Away ELO',
                        'Prediction 1X2', 'Prediction 1X2 Odd', 'Conf 1X2', 'EV 1X2', 'Kelly 1X2',
                        'Prediction O/U', 'Prediction O/U Odd', 'Conf O/U', 'EV O/U', 'Kelly O/U',
                        'Home Win %', 'Draw %', 'Away Win %', 'Over %', 'Under %',
                        'Home Win % (raw)', 'Draw % (raw)', 'Away Win % (raw)',
                        'Over % (raw)', 'Under % (raw)',
                        'Cal 1X2 Source', 'Cal O/U Source',
                        'Adj Logs', 'match_id']
             existing = [c for c in desired if c in res_df.columns]
             print("\n--- PREDICTIONS ---")
             print(res_df[existing].to_string(index=False))
             res_df[existing].to_csv(filename, index=False)
             print(f"Saved to {filename}")
             
             # --- DIAGNOSTICS ---
             print("\n--- DRAW DIAGNOSTICS ---")
             try:
                 avg_draw_prob = pd.to_numeric(res_df['Draw %']).mean()
                 print(f"Mean Draw Probability: {avg_draw_prob:.4f} (Target < 0.35)")
                 
                 total_preds = len(res_df)
                 draw_preds = len(res_df[res_df['Prediction 1X2'] == 'X'])
                 draw_percentage = (draw_preds / total_preds) * 100
                 print(f"Predicted Draw %: {draw_percentage:.2f}% ({draw_preds}/{total_preds})")
             except Exception as e:
                 print(f"Could not calculate diagnostics: {e}")

        else:
            print("No valid predictions generated.")
            
        elapsed = time.time() - start_time
        print(f"[*] Prediction Finished in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    predictor = MatchPredictor()
    predictor.predict()
