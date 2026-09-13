import pandas as pd
import numpy as np
from elo_engine import EloTracker
import warnings

from rapidfuzz import process, fuzz

# Suppress FutureWarning for GroupBy (Pandas 2.1+ transition)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Rolling-form windows, as (n_matches, column_suffix).
#
# THE single source of truth for both sides of the pipeline: training builds
# these in `_calculate_rolling`, and `predict_matches.get_team_stats` reads
# the same constant to derive serve-time form from the same MatchHistory
# corpus. Adding a window here changes both at once — which is the point,
# since a window present in training but absent at serve time is exactly the
# train/serve skew this constant exists to prevent.
#
# The empty suffix on the 5-match window is deliberate: `H_form_pts` etc. are
# the long-standing column names and every trained model, feature manifest and
# heuristic references them. Keep it.
FORM_WINDOWS = ((5, ''), (10, '_l10'), (15, '_l15'))

class FeatureEngineer:
    def __init__(self):
        pass

    def calculate_features_from_h2h(self, last_matches: list, target_team: str, window: int = 5, venue_filter: str = None) -> dict:
        """
        Calculates rolling features (Pts, GF, GA, O/U rate) from a raw list of H2H matches.
        venue_filter: None (All), 'home' (Only Home games), 'away' (Only Away games)
        """
        if not last_matches:
            return {
                'form_pts': 0, 'form_gf': 0, 'form_ga': 0, 'form_ou': 0
            }
            
        pts = []
        gf = []
        ga = []
        ou = []
        results_list = []
        
        # Sort by date descending? usually scraper gives most recent first.
        # We take top 'window' matches THAT MATCH THE FILTER
        
        count = 0
        for m in last_matches:
            if count >= window:
                break
                
            try:
                # Score format "2-1" or "2 - 1"
                s = m.get('score', '0-0').replace(' ', '')
                if '-' not in s: 
                    continue
                h_score, a_score = map(int, s.split('-')[:2])
                
                # Determine if target is home or away
                # Use fuzzy matching if exact match fails
                # Simple check first
                is_home_game = False
                if m.get('home_team') == target_team:
                    is_home_game = True
                elif m.get('away_team') == target_team:
                    is_home_game = False
                else:
                    # Fuzzy match
                    choices = [m.get('home_team'), m.get('away_team')]
                    best, score, _ = process.extractOne(target_team, choices, scorer=fuzz.ratio)
                    if score > 70:
                        is_home_game = (best == m.get('home_team'))
                    else:
                        continue # Skip if uncertain
                
                # Apply Venue Filter
                if venue_filter == 'home' and not is_home_game:
                    continue
                if venue_filter == 'away' and is_home_game:
                    continue
                        
                # Stats
                my_goals = h_score if is_home_game else a_score
                opp_goals = a_score if is_home_game else h_score
                
                total_goals = h_score + a_score
                
                # Points
                if my_goals > opp_goals: 
                    p = 3
                    res_char = 'W'
                elif my_goals == opp_goals: 
                    p = 1
                    res_char = 'D'
                else: 
                    p = 0
                    res_char = 'L'
                
                pts.append(p)
                gf.append(my_goals)
                ga.append(opp_goals)
                ou.append(1 if total_goals > 2.5 else 0)
                results_list.append(res_char)
                
                count += 1
                
            except Exception as e:
                continue

        if not pts:
            return {'form_pts': 0, 'form_gf': 0, 'form_ga': 0, 'form_ou': 0, 'form_str': ''}

        return {
            'form_pts': np.mean(pts),
            'form_gf': np.mean(gf),
            'form_ga': np.mean(ga),
            'form_ou': np.mean(ou),
            'form_str': ",".join(results_list)
        }


    def add_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Assumes H_elo and A_elo already exist from prepare_data
        if 'H_elo' in df.columns and 'A_elo' in df.columns:
            df['elo_diff'] = df['H_elo'] - df['A_elo']
        return df

    def add_league_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'league' in df.columns:
            df['league_cat'] = df['league'].astype('category')
        return df

    def add_implied_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds 1/Odds features (Implied Probability)."""
        # Ensure numeric (handled in loader, but safety check)
        for col in ['B365H', 'B365D', 'B365A']:
            if col in df.columns:
                # Avoid division by zero
                df[f'IP_{col[-1]}'] = 1.0 / df[col].replace(0, np.nan)
        return df

    def add_rolling_features(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Adds rolling form features (Points, Goals Scored, Goals Conceded)
        for both Home and Away teams based on their last `window` games.
        ALSO adds PPG (season-to-date) and Relative Strength features.
        """
        df = df.sort_values('date').copy()
        
        # 0. Implied Probabilities
        df = self.add_implied_probabilities(df)
        
        # 1. Rolling form over every window in FORM_WINDOWS (L5 + L10 + L15),
        #    computed in a single pass. L5 keeps the unsuffixed column names.
        df = self._calculate_rolling_multi(df, FORM_WINDOWS)

        # 2. League Encoding
        df = self.add_league_encoding(df)
        
        # 3. Elo Diffs
        df = self.add_elo_features(df)
        if 'elo_diff' in df.columns:
            df['abs_elo_diff'] = df['elo_diff'].abs()
            
        # 4. Form Diffs
        if 'H_form_pts' in df.columns and 'A_form_pts' in df.columns:
            df['form_pts_diff'] = df['H_form_pts'] - df['A_form_pts']
            df['abs_form_pts_diff'] = df['form_pts_diff'].abs()

        # 4b. Form trend: recent form minus long-horizon form. Positive means
        #     the team is playing above its own baseline right now. Mirrored
        #     verbatim in predict_matches.get_team_stats.
        for side in ('H', 'A'):
            short_c, long_c = f'{side}_form_pts', f'{side}_form_pts_l15'
            if short_c in df.columns and long_c in df.columns:
                df[f'{side}_form_trend'] = df[short_c] - df[long_c]

        # 5. Specific Home/Away Form
        df = self._calculate_specific_home_away(df)

        # 6. Season-to-date PPG, attack strength, defense weakness
        df = self._add_ppg_strength_features(df)

        return df

    def _calculate_specific_home_away(self, df):
        # Re-implementation of specific form logic
        teams = pd.concat([df['home_team'], df['away_team']]).unique()
        
        h_stats = {'H_home_pts': [], 'H_home_gf': [], 'H_home_ga': [], 'H_home_sf': [], 'H_home_sa': []}
        a_stats = {'A_away_pts': [], 'A_away_gf': [], 'A_away_ga': [], 'A_away_sf': [], 'A_away_sa': []}
        
        team_home_matches = {}
        team_away_matches = {}
        
        for team in teams:
            team_home_matches[team] = df[df['home_team'] == team].sort_values('date')
            team_away_matches[team] = df[df['away_team'] == team].sort_values('date')
            
        for idx, row in df.iterrows():
            date = row['date']
            home = row['home_team']
            away = row['away_team']
            
            # Home Team at Home (Last 5)
            h_hist = team_home_matches[home][team_home_matches[home]['date'] < date].tail(5)
            stats = self._get_stats_from_history(h_hist, home)
            h_stats['H_home_pts'].append(stats['form_pts'])
            h_stats['H_home_gf'].append(stats['form_gf'])
            h_stats['H_home_ga'].append(stats['form_ga'])
            h_stats['H_home_sf'].append(stats.get('form_sf', 0))
            h_stats['H_home_sa'].append(stats.get('form_sa', 0))
            
            # Away Team at Away (Last 5)
            a_hist = team_away_matches[away][team_away_matches[away]['date'] < date].tail(5)
            stats = self._get_stats_from_history(a_hist, away)
            a_stats['A_away_pts'].append(stats['form_pts'])
            a_stats['A_away_gf'].append(stats['form_gf'])
            a_stats['A_away_ga'].append(stats['form_ga'])
            a_stats['A_away_sf'].append(stats.get('form_sf', 0))
            a_stats['A_away_sa'].append(stats.get('form_sa', 0))
            
        for k, v in h_stats.items(): df[k] = v
        for k, v in a_stats.items(): df[k] = v
        
        return df

    # Stat keys `_get_stats_from_history` returns, and the column stem each
    # maps to. Kept here so the train side and `predict_matches` build the
    # same column names from the same list.
    FORM_STAT_KEYS = ('pts', 'gf', 'ga', 'ou', 'str', 'sf', 'sa', 'cf', 'ca')

    def _calculate_rolling(self, df, window, suffix):
        """Rolling stats for a single window. Thin wrapper kept for callers
        that want one window; `_calculate_rolling_multi` does the work."""
        return self._calculate_rolling_multi(df, [(window, suffix)])

    def _calculate_rolling_multi(self, df, windows):
        """Rolling stats for several windows in ONE pass over the history.

        `windows` is [(n_matches, suffix), ...] — normally FORM_WINDOWS.

        Why one pass: the expensive part is the per-row date filter
        `team_matches[team][date < d]`, which scans that team's whole history
        for every match. Slicing it once and taking a different `.tail(n)` per
        window costs a few list appends instead of re-running the scan, so
        three windows cost roughly one window's time rather than three.

        Emits exactly the columns the single-window version did for each
        (window, suffix) pair, so `_calculate_rolling(df, 5, "")` is unchanged.
        """
        teams = pd.concat([df['home_team'], df['away_team']]).unique()

        # {suffix: {column: [values]}} for each side.
        h_stats = {sfx: {f'H_form_{k}{sfx}': [] for k in self.FORM_STAT_KEYS}
                   for _, sfx in windows}
        a_stats = {sfx: {f'A_form_{k}{sfx}': [] for k in self.FORM_STAT_KEYS}
                   for _, sfx in windows}

        # Pre-calculate team match histories for speed
        team_matches = {}
        for team in teams:
            tm = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('date')
            team_matches[team] = tm

        for idx, row in df.iterrows():
            date = row['date']
            for team, side, store in ((row['home_team'], 'H', h_stats),
                                      (row['away_team'], 'A', a_stats)):
                # One date filter per team per row, shared across windows.
                hist = team_matches[team][team_matches[team]['date'] < date]
                for n, sfx in windows:
                    stats = self._get_stats_from_history(hist.tail(n), team)
                    for k in self.FORM_STAT_KEYS:
                        # form_str has no numeric default; the rest fall back to 0
                        # exactly as the single-window version did.
                        default = '' if k == 'str' else 0
                        store[sfx][f'{side}_form_{k}{sfx}'].append(
                            stats.get(f'form_{k}', default))

        # Assign columns
        for store in (h_stats, a_stats):
            for cols in store.values():
                for k, v in cols.items():
                    df[k] = v

        return df

    def _get_stats_from_history(self, history_df: pd.DataFrame, target_team: str) -> dict:
        """
        Calculates rolling features (Pts, GF, GA, O/U rate, Shots, Corners) from a DataFrame of historical matches.
        This is a refactored version of calculate_features_from_h2h to work with DataFrames.
        """
        if history_df.empty:
            return {
                'form_pts': 0, 'form_gf': 0, 'form_ga': 0, 'form_ou': 0, 'form_str': '',
                'form_sf': 0, 'form_sa': 0, 'form_cf': 0, 'form_ca': 0
            }

        pts = []
        gf = []
        ga = []
        ou = []
        results_list = []
        sf = [] # Shots For
        sa = [] # Shots Against
        cf = [] # Corners For
        ca = [] # Corners Against

        for _, m in history_df.iterrows():
            try:
                h_score = m['FTHG']
                a_score = m['FTAG']

                is_home_game = (m['home_team'] == target_team)

                # Stats
                my_goals = h_score if is_home_game else a_score
                opp_goals = a_score if is_home_game else h_score

                total_goals = h_score + a_score

                # Points
                if my_goals > opp_goals:
                    p = 3
                    res_char = 'W'
                elif my_goals == opp_goals:
                    p = 1
                    res_char = 'D'
                else:
                    p = 0
                    res_char = 'L'

                pts.append(p)
                gf.append(my_goals)
                ga.append(opp_goals)
                ou.append(1 if total_goals > 2.5 else 0)
                results_list.append(res_char)

                # Shots and Corners (if available)
                # Shots and Corners (if available)
                sf.append(m.get('HST', 0) if is_home_game else m.get('AST', 0))
                sa.append(m.get('AST', 0) if is_home_game else m.get('HST', 0))
                cf.append(m.get('HC', 0) if is_home_game else m.get('AC', 0))
                ca.append(m.get('AC', 0) if is_home_game else m.get('HC', 0))

            except KeyError:
                # Handle cases where FTHG, FTAG, HST, etc. might be missing
                # For now, just skip this match or use default values
                continue
            except Exception as e:
                # Catch other potential errors during processing
                continue

        if not pts:
            return {
                'form_pts': 0, 'form_gf': 0, 'form_ga': 0, 'form_ou': 0, 'form_str': '',
                'form_sf': 0, 'form_sa': 0, 'form_cf': 0, 'form_ca': 0
            }

        return {
            'form_pts': np.mean(pts),
            'form_gf': np.mean(gf),
            'form_ga': np.mean(ga),
            'form_ou': np.mean(ou),
            'form_str': ",".join(results_list),
            'form_sf': np.mean(sf) if sf else 0,
            'form_sa': np.mean(sa) if sa else 0,
            'form_cf': np.mean(cf) if cf else 0,
            'form_ca': np.mean(ca) if ca else 0
        }

    def _add_ppg_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds season-to-date PPG, attack strength, defense weakness, and
        derived diffs. All cumulative stats are computed from games STRICTLY
        BEFORE each match (within the same Aug→Jul season) to avoid leakage.

        Strength is normalized against the league's running average goals
        scored per team-game in the same season.
        """
        required = {'date', 'home_team', 'away_team', 'FTHG', 'FTAG', 'FTR', 'league'}
        if not required.issubset(df.columns):
            return df

        home_view = pd.DataFrame({
            'date': df['date'],
            'team': df['home_team'],
            'league': df['league'],
            'points': df['FTR'].map({'H': 3, 'D': 1, 'A': 0}),
            'goals_for': df['FTHG'],
            'goals_against': df['FTAG'],
            'is_home': 1,
        })
        away_view = pd.DataFrame({
            'date': df['date'],
            'team': df['away_team'],
            'league': df['league'],
            'points': df['FTR'].map({'A': 3, 'D': 1, 'H': 0}),
            'goals_for': df['FTAG'],
            'goals_against': df['FTHG'],
            'is_home': 0,
        })

        team_stats = pd.concat([home_view, away_view], ignore_index=True)
        team_stats = team_stats.dropna(subset=['points'])
        team_stats = team_stats.sort_values('date').reset_index(drop=True)

        # Football season key: Aug–Dec → year; Jan–Jul → year-1.
        months = team_stats['date'].dt.month
        years = team_stats['date'].dt.year
        team_stats['season_year'] = years.where(months >= 8, years - 1)

        def _running_team(group):
            past = group[['points', 'goals_for', 'goals_against']].shift(1)
            cum_games = past['points'].notna().cumsum().replace(0, np.nan)
            return pd.DataFrame({
                'ppg': (past['points'].cumsum() / cum_games).fillna(0),
                'avg_gf': (past['goals_for'].cumsum() / cum_games).fillna(0),
                'avg_ga': (past['goals_against'].cumsum() / cum_games).fillna(0),
            }, index=group.index)

        running_team = team_stats.groupby(
            ['team', 'season_year'], group_keys=False
        ).apply(_running_team)
        team_stats[['ppg', 'avg_gf', 'avg_ga']] = running_team

        def _running_league(group):
            past = group['goals_for'].shift(1)
            cum_games = past.notna().cumsum().replace(0, np.nan)
            return (past.cumsum() / cum_games).fillna(0)

        team_stats['league_avg_gf'] = team_stats.groupby(
            ['league', 'season_year'], group_keys=False
        ).apply(_running_league)

        baseline = team_stats['league_avg_gf'].replace(0, np.nan)
        team_stats['att_strength'] = (team_stats['avg_gf'] / baseline).fillna(1.0)
        team_stats['def_weakness'] = (team_stats['avg_ga'] / baseline).fillna(1.0)

        keep = ['date', 'team', 'ppg', 'att_strength', 'def_weakness']
        home_feats = team_stats.loc[team_stats['is_home'] == 1, keep].rename(columns={
            'team': 'home_team',
            'ppg': 'H_ppg', 'att_strength': 'H_att', 'def_weakness': 'H_def',
        })
        away_feats = team_stats.loc[team_stats['is_home'] == 0, keep].rename(columns={
            'team': 'away_team',
            'ppg': 'A_ppg', 'att_strength': 'A_att', 'def_weakness': 'A_def',
        })

        df = df.merge(home_feats, on=['date', 'home_team'], how='left')
        df = df.merge(away_feats, on=['date', 'away_team'], how='left')

        df['ppg_diff'] = df['H_ppg'] - df['A_ppg']
        df['abs_ppg_diff'] = df['ppg_diff'].abs()
        df['att_def_diff'] = (df['H_att'] - df['A_att']) - (df['H_def'] - df['A_def'])

        return df


if __name__ == "__main__":
    from data_loader import DataLoader
    loader = DataLoader("data_sets/MatchHistory")
    df = loader.load_historical_data()
    
    fe = FeatureEngineer()
    df_features = fe.add_rolling_features(df)
    
    print(f"Enriched Data Shape: {df_features.shape}")
    print(df_features[['date', 'home_team', 'away_team', 'H_form_pts', 'FTR']].tail(10))
