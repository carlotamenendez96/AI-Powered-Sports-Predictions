import pandas as pd
import numpy as np
import glob
import os
from typing import List, Optional

# Preference chain for the reference 1X2 price that becomes B365H/D/A and so
# feeds IP_H/IP_D/IP_A — the model's strongest features.
#
# THE ORDER IS LOAD-BEARING: every OPENING price is preferred over every
# CLOSING price. Predictions are produced the night before off a Flashscore
# scrape, so a closing price is something the model can never have at
# inference; training on one teaches a relationship that does not hold at
# serve time.
#
# Before 2026-09-18 the fallback jumped straight from "no B365 column" to
# AvgC*/MaxC* — both CLOSING — and overwrote B365H in place, erasing the
# provenance. Measured: 63,218 of 186,557 corpus rows (33.9%) were training on
# a closing price presented as an opening one, concentrated on exactly the
# extra leagues with the least other data. The fallback still exists (dropping
# those rows would cost a third of the corpus) but it is now last-resort and
# always recorded in `odds_source` / `odds_is_closing`.
#
# Within the closing block the order is AvgC -> MaxC first, which REPRODUCES
# the pre-2026-09-18 fallback exactly, so this change alters no existing
# feature value. That is deliberate. Pinnacle closing is the sharper price
# (corpus Brier 0.59930 vs AvgC's 0.59957) and is tempting here, but IP_* is
# raw 1/odds and is never devigged, so swapping sources also swaps the
# BOOKMAKER MARGIN baked into the feature: measured mean overround is B365
# 1.0609, AvgC 1.0863, PSC 1.0314 — a 5.5pp scale shift across 31% of the
# corpus. Changing the source is a model change, not a provenance fix, and it
# belongs behind its own A/B. See FOOTBALL_NEXT_STEPS "E — Market-edge
# validation" for that decision (and the related question of whether IP_*
# should be devigged at all, which would remove the margin confound entirely).
#
# (name, (home, draw, away), is_closing)
ODDS_PREFERENCE = [
    ('B365',  ('B365H',  'B365D',  'B365A'),  False),
    ('PS',    ('PSH',    'PSD',    'PSA'),    False),
    ('Avg',   ('AvgH',   'AvgD',   'AvgA'),   False),
    ('Max',   ('MaxH',   'MaxD',   'MaxA'),   False),
    ('AvgC',  ('AvgCH',  'AvgCD',  'AvgCA'),  True),
    ('MaxC',  ('MaxCH',  'MaxCD',  'MaxCA'),  True),
    ('B365C', ('B365CH', 'B365CD', 'B365CA'), True),
    ('PSC',   ('PSCH',   'PSCD',   'PSCA'),   True),
]

# Separate chain for the CLOSING price, exposed as close_H/close_D/close_A.
# This is the benchmark the model has to beat, not a model input — Pinnacle
# closing is the sharpest line in the corpus (Brier 0.59811 vs B365 opening's
# 0.60070 on the same 101,319 rows). Needed by the CLV work (E0) and the
# line-movement test (E2); see FOOTBALL_NEXT_STEPS "E — Market-edge validation".
# NEVER add these to a model feature list: at serve time they do not exist.
CLOSING_PREFERENCE = [
    ('PSC',   ('PSCH',   'PSCD',   'PSCA')),
    ('B365C', ('B365CH', 'B365CD', 'B365CA')),
    ('AvgC',  ('AvgCH',  'AvgCD',  'AvgCA')),
    ('MaxC',  ('MaxCH',  'MaxCD',  'MaxCA')),
]


def _pick_odds(df, chain):
    """First valid (home, draw, away) triple per row from `chain`.

    Row-wise, not file-wise: a file can carry a B365 column that is blank for
    some rows, and those rows should fall through to the next source rather
    than be discarded. Returns (home, draw, away, source, is_closing) as
    Series aligned to df.index; `is_closing` is None for chains that do not
    carry the flag.
    """
    n = len(df)
    h = pd.Series(np.nan, index=df.index, dtype='float64')
    d = pd.Series(np.nan, index=df.index, dtype='float64')
    a = pd.Series(np.nan, index=df.index, dtype='float64')
    src = pd.Series(pd.NA, index=df.index, dtype='object')
    closing = pd.Series(pd.NA, index=df.index, dtype='object')

    for entry in chain:
        name, cols = entry[0], entry[1]
        is_closing = entry[2] if len(entry) > 2 else None
        if not all(c in df.columns for c in cols):
            continue
        ch, cd, ca = (pd.to_numeric(df[c], errors='coerce') for c in cols)
        # Odds must exceed 1.0 — a decimal price at or below evens implies a
        # probability >= 1 and poisons IP_*.
        ok = (ch.notna() & cd.notna() & ca.notna()
              & (ch > 1.0) & (cd > 1.0) & (ca > 1.0))
        fill = ok & h.isna()
        if not fill.any():
            continue
        h[fill], d[fill], a[fill] = ch[fill], cd[fill], ca[fill]
        src[fill] = name
        if is_closing is not None:
            closing[fill] = is_closing
    return h, d, a, src, closing

class DataLoader:
    def __init__(self, history_dir: str):
        self.history_dir = history_dir
        self.required_columns = [
            'date', 'home_team', 'away_team', 'FTHG', 'FTAG', 'FTR',
            'B365H', 'B365D', 'B365A'
        ]

    def load_historical_data(self) -> pd.DataFrame:
        """
        Loads and concatenates all CSV files from the history directory.
        """
        all_files = glob.glob(os.path.join(self.history_dir, "*.csv"))
        df_list = []
        
        print(f"Found {len(all_files)} historical files.")

        for filename in all_files:
            try:
                # Read CSV
                df = pd.read_csv(filename)
                
                # Normalize Columns
                # Standard map: Date->date, HomeTeam->home_team, AwayTeam->away_team
                # "New" Format map: Home->home_team, Away->away_team, HG->FTHG, AG->FTAG, Res->FTR
                col_map = {
                    'Date': 'date', 
                    'HomeTeam': 'home_team', 'AwayTeam': 'away_team',
                    'FTHG': 'FTHG', 'FTAG': 'FTAG', 'FTR': 'FTR', 
                    'Div': 'league',
                    'League': 'league', # "New" format has 'League' instead of 'Div'
                    # "New" Format extensions
                    'Home': 'home_team', 'Away': 'away_team',
                    'HG': 'FTHG', 'AG': 'FTAG', 'Res': 'FTR'
                }
                # Rename if exists
                df = df.rename(columns=col_map)
                
                # Lowercase other standard columns if needed, but strict mapping is safer for required ones
                
                # Ensure date is datetime
                if 'date' in df.columns:
                    # Football-data often uses dd/mm/yy or dd/mm/yyyy
                    # We utilize dayfirst=True for efficiency if standardized, 
                    # but if formats mix, 'mixed' is safer though slower.
                    # The warning suggests specifying format or avoiding dayfirst=True if iso.
                    # Given football-data is consistently DD/MM/YY(YY), we keep dayfirst but suppress warning
                    # or better: we use format='mixed' if available (pd 2.0+) or just ignore errors.
                    
                    # Fix: Use format='mixed' to silence warning about mixed iso/dayfirst
                    try:
                        df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True, errors='coerce')
                    except:
                         # Fallback for older pandas versions
                        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
                
                # Reference 1X2 price -> B365H/D/A (historical column names; the
                # values are not necessarily Bet365, see ODDS_PREFERENCE). The
                # provenance is recorded rather than erased.
                h, d, a, src, is_closing = _pick_odds(df, ODDS_PREFERENCE)
                if not h.notna().any():
                    print(f"Warning: File {filename} has no usable 1X2 odds - skipped.")
                    continue
                df['B365H'], df['B365D'], df['B365A'] = h, d, a
                df['odds_source'] = src
                df['odds_is_closing'] = is_closing
                # Bookmaker margin actually present in IP_*. IP_* is raw
                # 1/odds, so this rides along in the features; recording it
                # makes the per-source margin mix measurable.
                df['odds_overround'] = 1.0 / h + 1.0 / d + 1.0 / a

                # Closing benchmark -> close_H/D/A. Evaluation only, never a feature.
                ch, cd, ca, csrc, _ = _pick_odds(df, CLOSING_PREFERENCE)
                df['close_H'], df['close_D'], df['close_A'] = ch, cd, ca
                df['close_source'] = csrc

                # ENFORCE NUMERIC TYPES for Key Columns
                # This prevents 'object' type errors in XGBoost if CSV contains strings/empty values
                for col in ['FTHG', 'FTAG']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Check for required columns (warn if missing but don't crash)
                missing_cols = [c for c in self.required_columns if c not in df.columns]
                if missing_cols:
                    print(f"Warning: File {filename} is missing columns: {missing_cols}")
                    continue 

                df_list.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

        if not df_list:
            raise ValueError("No valid historical data found.")

        combined_df = pd.concat(df_list, ignore_index=True)
        
        # Sort by date
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        
        return combined_df

    def get_team_names(self, df: pd.DataFrame) -> List[str]:
        """Returns unique team names from the historical dataset."""
        unique_teams = pd.concat([df['home_team'], df['away_team']]).unique()
        return sorted(unique_teams.tolist())

if __name__ == "__main__":
    # Test execution
    loader = DataLoader("data_sets/MatchHistory")
    try:
        df = loader.load_historical_data()
        print(f"Total Matches Loaded: {len(df)}")
        print(df.head())
    except Exception as e:
        print(f"Loader failed: {e}")
