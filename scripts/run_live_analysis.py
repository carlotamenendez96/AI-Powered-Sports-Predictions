import pandas as pd
import json
import os
import datetime
import subprocess
import sys
from thefuzz import process, fuzz
from ml_project.live_adjuster import LiveAdjuster

# Config
OUTPUT_DIR = "output"
TODAY = datetime.datetime.now().strftime('%d.%m.%Y') # 10.12.2025 match format
TODAY_FILE_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, f"predictions_{TODAY_FILE_DATE}.csv")
LIVE_OUTPUT = os.path.join(OUTPUT_DIR, "live_data.json")
# Append-only per-tick snapshots. One JSON object per match per refresh.
# Feeds the cashout backtest harness (scripts/run_backtest.py). Kept in a
# dedicated subdir so it's clearly separate from the throwaway live_data.json
# the UI Clear button zeroes — this is accruing data the weekly backtest needs.
LIVE_HISTORY_DIR = os.path.join(OUTPUT_DIR, "live_history")
LIVE_HISTORY = os.path.join(LIVE_HISTORY_DIR, f"live_history_{TODAY_FILE_DATE}.jsonl")

def _open_bet_priors_by_team():
    """Build minimal prediction-like rows for matches with open bets but
    no entry in today's prediction CSV. Lets the live snapshot cover
    bets that don't trace back to today's predictions — e.g. a bet
    placed weeks ago for today's match, where the prediction CSV from
    that date is no longer the current `predictions_<TODAY>.csv`.

    Each synthesized row carries `_synthesized=True` so downstream
    records can flag the priors as weak (odds-derived rather than
    full-model probabilities).

    Returns {home_team_name: row_dict}.
    """
    bets_path = os.path.join(OUTPUT_DIR, f"bets_{TODAY_FILE_DATE}.json")
    out = {}
    if not os.path.exists(bets_path):
        return out
    try:
        with open(bets_path) as f:
            slip = json.load(f)
    except (OSError, ValueError):
        return out

    for b in slip.get('bets', []):
        if b.get('status') != 'OPEN':
            continue
        match_str = b.get('match', '')
        home = b.get('home') or (match_str.split(' vs ')[0].strip()
                                  if ' vs ' in match_str else '')
        away = b.get('away') or (match_str.split(' vs ')[1].strip()
                                  if ' vs ' in match_str else '')
        if not home or not away:
            continue
        try:
            odds = float(b.get('odds', 0) or 0)
        except (TypeError, ValueError):
            odds = 0
        if odds <= 1.0:
            continue
        impl = 1.0 / odds
        bet_type = b.get('type', '1X2')
        selection = str(b.get('selection', ''))

        row = out.setdefault(home, {
            'Home Team': home,
            'Away Team': away,
            # Uniform priors as a baseline. Specific bets override below.
            'Home Win %': 0.33, 'Draw %': 0.33, 'Away Win %': 0.34,
            'Over %': 0.50, 'Under %': 0.50,
            'League': b.get('league', ''),
            '_synthesized': True,
        })

        if bet_type == '1X2':
            other = max(0.0, (1.0 - impl) / 2.0)
            if selection in ('1', 'Home', 'home'):
                row['Home Win %'] = impl
                row['Draw %'] = other
                row['Away Win %'] = other
            elif selection in ('X', 'x', 'Draw', 'draw'):
                row['Draw %'] = impl
                row['Home Win %'] = other
                row['Away Win %'] = other
            elif selection in ('2', 'Away', 'away'):
                row['Away Win %'] = impl
                row['Home Win %'] = other
                row['Draw %'] = other
        elif bet_type == 'O/U':
            if 'Over' in selection:
                row['Over %'] = impl
                row['Under %'] = max(0.0, 1.0 - impl)
            elif 'Under' in selection:
                row['Under %'] = impl
                row['Over %'] = max(0.0, 1.0 - impl)
    return out


def main():
    # `--with-bookmaker`: after the Flashscore scrape, conditionally
    # trigger the (slow, headed) Pamestoixima open-bets scrape — but
    # ONLY if a live match actually has an open bet on it (the only case
    # where a cashout offer would be shown). Set by the manual "Refresh
    # Live Snapshot" button; avoids a pointless ~25s Chromium scrape
    # when nothing relevant is live.
    with_bookmaker = '--with-bookmaker' in sys.argv

    # Open-bet priors built early so we know whether the run is
    # worth doing at all when there's no predictions file.
    synthesized_rows = _open_bet_priors_by_team()

    if not os.path.exists(PREDICTIONS_FILE):
        if not synthesized_rows:
            print(f"No predictions file found for {TODAY_FILE_DATE} "
                  f"AND no open bets to drive a snapshot. Nothing to do.")
            with open(LIVE_OUTPUT, 'w') as f:
                json.dump([], f)
            return
        print(f"No predictions file for {TODAY_FILE_DATE}, but found "
              f"{len(synthesized_rows)} open-bet team(s) — running snapshot "
              f"with synthesized priors only.")
        df = pd.DataFrame(columns=['Home Team', 'Away Team',
                                   'Home Win %', 'Draw %', 'Away Win %',
                                   'Over %', 'Under %', 'League'])
    else:
        print("Loading predictions...")
        try:
            df = pd.read_csv(PREDICTIONS_FILE)
        except Exception as e:
            print(f"Error reading predictions: {e}")
            return

    # Step 1: Scrape List of Currently Live Matches
    print("Scraping list of LIVE matches from Flashscore...")
    cmd_list = [
        "venv/bin/python", "-m", "scrapy", "crawl", "flashscore",
        "-a", "live_list=true",
        "-O", "output/live_list.json",
        "--nolog"
    ]
    try:
        subprocess.run(cmd_list, check=True)
    except Exception as e:
        print(f"Error scraping live list: {e}")
        return
        
    if not os.path.exists("output/live_list.json"):
        print("No live list scraped.")
        return
        
    with open("output/live_list.json", 'r') as f:
        try:
            live_matches_raw = json.load(f)
        except:
            live_matches_raw = []
            
    if not live_matches_raw:
        msg = "No live matches found on Flashscore." # Or use the requested message if preferred, but strictly this means NONE are live.
        # User requested: "No live matches we have predictions for found on Flashscore"
        # If there are NO live matches, then obviously we have no predictions for them.
        # So I will use the requested message for consistency, or a clear variant.
        # Let's use the requested one to satisfy the user strictly.
        msg = "No live matches we have predictions for found on Flashscore" 
        print(msg)
        with open(LIVE_OUTPUT, 'w') as f:
             json.dump([{'message': msg}], f)
        return

    print(f"Found {len(live_matches_raw)} live matches on Flashscore.")

    # Step 2: Crosscheck with Predictions
    # We want to find matches in 'df' that correspond to 'live_matches_raw'
    # Use Home Team Name for matching
    
    predicted_teams = df['Home Team'].tolist()
    live_pairs = []
    matched_live_ids = set()

    # Prediction-driven matching (skipped cleanly if df is empty,
    # which happens when there's no predictions file but we still have
    # open bets to anchor the snapshot).
    if predicted_teams:
        for m in live_matches_raw:
            h_team = m['home_team']
            # Fuzzy match
            match, score = process.extractOne(h_team, predicted_teams, scorer=fuzz.token_sort_ratio)
            if score > 80: # Threshold
                # Found a candidate
                # Verify Away team too?
                row = df[df['Home Team'] == match].iloc[0]
                a_team_pred = row['Away Team']

                # Simple check on away team
                if fuzz.token_sort_ratio(m['away_team'], a_team_pred) > 70:
                    print(f"MATCH FOUND: {h_team} vs {m['away_team']} (ID: {m['match_id']})")
                    live_pairs.append((m, row))
                    matched_live_ids.add(m['match_id'])

    # Phase 7 — also include live matches with open bets but no entry in
    # today's predictions CSV. Uses synthesized priors derived from the
    # bet's odds. Downstream code reads the same fields (Home Win %,
    # Draw %, Away Win %, Over %, Under %).
    # (synthesized_rows was already computed at the top of main().)
    if synthesized_rows:
        bet_teams = list(synthesized_rows.keys())
        for m in live_matches_raw:
            if m['match_id'] in matched_live_ids:
                continue
            h_team = m['home_team']
            bm, score = process.extractOne(h_team, bet_teams, scorer=fuzz.token_sort_ratio)
            if score <= 80:
                continue
            row_dict = synthesized_rows[bm]
            if fuzz.token_sort_ratio(m['away_team'], row_dict['Away Team']) <= 70:
                continue
            print(f"OPEN-BET MATCH FOUND: {h_team} vs {m['away_team']} "
                  f"(ID: {m['match_id']}) [synthesized priors from bet odds]")
            # Wrap as a pandas Series so it behaves like a prediction-CSV row.
            row = pd.Series(row_dict)
            live_pairs.append((m, row))
            matched_live_ids.add(m['match_id'])

    if not live_pairs:
        msg = "No live matches we have predictions for found on Flashscore"
        print(msg)
        with open(LIVE_OUTPUT, 'w') as f:
            json.dump([{'message': msg}], f)
        return
        
    print(f"Processing {len(live_pairs)} active matches...")
    
    # Optimize: Batch Scrape ALL IDs at once
    active_ids = [m['match_id'] for m, _ in live_pairs]
    ids_str = ",".join(active_ids)
    
    # Store minimal lookup map for stats association
    match_lookup = {m['match_id']: (m, row) for m, row in live_pairs}

    print(f"Fetching stats for {len(active_ids)} matches in BATCH mode...")
    
    # Single Scrapy Call
    cmd_batch = [
        "venv/bin/python", "-m", "scrapy", "crawl", "flashscore",
        "-a", f"live_ids={ids_str}",
        "-O", "output/live_stats_batch.json",
        "--nolog"
    ]
    
    try:
        subprocess.run(cmd_batch, check=True)
    except Exception as e:
        print(f"Error in batch scrape: {e}")
        # Continue to process whatever we have (or empty)

    # Process Batch Output
    final_results = []
    adjuster = LiveAdjuster()
    
    if os.path.exists("output/live_stats_batch.json"):
        try:
            with open("output/live_stats_batch.json", 'r') as f:
                batch_data = json.load(f)
        except:
             batch_data = []
    else:
        batch_data = []

    # Map results back to predictions
    # Note: batch_data contains {'match_id': ..., 'stats': ...}
    
    # Create a set of processed IDs to track misses if needed
    processed_ids = set()

    snapshot_ts = datetime.datetime.now().isoformat(timespec='seconds')
    history_lines = []

    for item in batch_data:
        m_id = item.get('match_id')
        if m_id not in match_lookup: continue

        processed_ids.add(m_id)
        live_meta, pred_row = match_lookup[m_id]

        try:
              pre_probs = {
                'home': float(pred_row['Home Win %']),
                'draw': float(pred_row['Draw %']),
                'away': float(pred_row['Away Win %'])
              }
        except:
             pre_probs = {'home':0.33, 'draw':0.33, 'away':0.33}

        try:
            pre_ou_probs = {
                'over':  float(pred_row['Over %']),
                'under': float(pred_row['Under %']),
            }
        except (KeyError, ValueError, TypeError):
            pre_ou_probs = {'over': 0.5, 'under': 0.5}

        # Traces record WHICH adjustment stage moved each probability and by
        # how much. They're persisted to live_history (not surfaced in the UI
        # record) — without them a stored snapshot shows only the before/after
        # and the per-stage contributions can't be recovered later.
        trace_1x2, trace_ou = {}, {}

        adjusted = adjuster.adjust_probabilities(
            pre_probs,
            item.get('stats', {}),
            item.get('minute', 0),
            item.get('score', '0-0'),
            trace=trace_1x2
        )

        adjusted_ou = adjuster.adjust_ou_probabilities(
            pre_ou_probs,
            item.get('stats', {}),
            item.get('minute', 0),
            item.get('score', '0-0'),
            trace=trace_ou
        )

        record = {
            'match': f"{live_meta['home_team']} vs {live_meta['away_team']}",
            'match_id': m_id,
            'score': item.get('score', '0-0'),
            'minute': item.get('minute', 0),
            'stats': item.get('stats', {}),
            'pre_probs': pre_probs,
            'adj_probs': adjusted,
            'pre_ou_probs': pre_ou_probs,
            'adj_ou_probs': adjusted_ou,
            # Phase 7: surface whether this match's priors came from a
            # real prediction row or were synthesized from open-bet odds
            # (open-bet matches without a prediction entry). UI uses this
            # to label weak-prior records distinctly.
            'priors_synthesized': bool(pred_row.get('_synthesized', False)),
        }
        final_results.append(record)

        history_lines.append({
            'ts': snapshot_ts,
            'date': TODAY_FILE_DATE,
            'match_id': m_id,
            'home_team': live_meta['home_team'],
            'away_team': live_meta['away_team'],
            'league': pred_row.get('League', ''),
            'minute': item.get('minute', 0),
            'score': item.get('score', '0-0'),
            'stats': item.get('stats', {}),
            'pre_probs': pre_probs,
            'adj_probs': adjusted,
            'pre_ou_probs': pre_ou_probs,
            'adj_ou_probs': adjusted_ou,
            # Per-stage attribution for the two adjustments above.
            'trace_1x2': trace_1x2,
            'trace_ou': trace_ou,
            # Marks snapshots whose priors were synthesized from open-bet odds
            # rather than a real prediction row — lower-quality records that
            # later analysis must be able to separate out.
            'priors_synthesized': bool(pred_row.get('_synthesized', False)),
        })

    if history_lines:
        os.makedirs(LIVE_HISTORY_DIR, exist_ok=True)
        with open(LIVE_HISTORY, 'a') as hf:
            for line in history_lines:
                hf.write(json.dumps(line) + '\n')
        print(f"Appended {len(history_lines)} snapshot(s) to {LIVE_HISTORY}")

    # Add matches that failed to scrape (keep them in list with old data or error?)
    # For now, only show successfully scraped ones to avoid stale data confusion.
    # Or show "Waiting for data..."? 
    # Let's keep it clean: only update live_data.json with fresh results.
        
    with open(LIVE_OUTPUT, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"Updated live data for {len(final_results)} matches.")

    # Conditionally chain the Pamestoixima open-bets scrape. Gated on a
    # live match having an OPEN bet — that's the only case where a real
    # cashout offer would be attached + shown. Skips the slow headed
    # scrape entirely when nothing relevant is live. (The early-return
    # "no live matches" paths above never reach here, so they also skip.)
    if with_bookmaker:
        open_bet_matches = set()
        bets_path = os.path.join(OUTPUT_DIR, f"bets_{TODAY_FILE_DATE}.json")
        if os.path.exists(bets_path):
            try:
                with open(bets_path) as f:
                    slip = json.load(f)
                open_bet_matches = {
                    (b.get('match') or '').strip()
                    for b in slip.get('bets', [])
                    if b.get('status') == 'OPEN' and b.get('match')
                }
            except (OSError, ValueError):
                pass
        live_matches = {(r.get('match') or '').strip()
                        for r in final_results if not r.get('message')}
        relevant = open_bet_matches & live_matches
        if relevant:
            print(f"[live] {len(relevant)} live match(es) with open bets — "
                  f"triggering Pamestoixima open-bets scrape.")
            try:
                bm_log = open(os.path.join("logs", "bookmaker.log"), 'w')
                subprocess.Popen(
                    ["venv/bin/python", "-m", "real_betting", "read-open-bets"],
                    stdout=bm_log, stderr=subprocess.STDOUT,
                )
            except Exception as e:
                print(f"[live] Could not start Pamestoixima scrape: {e}")
        else:
            print("[live] No live match has an open bet — skipping "
                  "Pamestoixima scrape (nothing for offers to attach to).")

if __name__ == "__main__":
    main()
