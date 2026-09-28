# Euroleague Next Steps

Euroleague-specific roadmap. Sibling to `FOOTBALL_NEXT_STEPS.md` (football / cashout / cross-cutting) and `NBA_NEXT_STEPS.md` (NBA).

For project-wide context (cashout phases, real-betting integration, cross-cutting model experiments) see `FOOTBALL_NEXT_STEPS.md`. NBA work — the closest architectural prior, since both are basketball + same UI/betting shape — lives in `NBA_NEXT_STEPS.md`. Mirror NBA's three-phase shape unless a Euroleague-specific reason forces a divergence.

## Scope (lock down before Phase 1)

Open questions to answer before any code:

1. **Competition scope.** Default: men's top-tier **ULEB Euroleague** only (no EuroCup, no Basketball Champions League, no domestic leagues). If we want EuroCup too, decide now — `euroleague-api` covers both for free, so the marginal cost is in the model (one combined model or two?) not the data fetch.
2. **Markets.** Default: **moneyline + total points** (mirrors NBA Phase 3 v1). Point spread is the dominant Euroleague market in some books — add as v1.5 once moneyline ships.
3. **Re-use vs. duplicate of the NBA pipeline.** Two basketball pipelines side-by-side is fine; the feature engineering (rest_days, b2b, L5/L10 rolling, venue-matched L5, ELO) transfers wholesale. Decision: copy `ml_project/nba/` → `ml_project/euroleague/` and diverge where needed, rather than parameterising a shared module before we know what differs. Same call as NBA-vs-football.
4. **Cadence.** Euroleague plays ~2× / week per team during the regular season (Oct–Apr) + playoffs through May/early-June. Daily predict/verify is overkill on off-days; same `bin/run_*.sh` cadence is fine, those just no-op on empty slates.

## Phase status

| # | Phase | Status | Notes |
| --- | --- | --- | --- |
| 0 | Data-source evaluation (this doc) | ✅ DONE (2026-05-28, mod. Flashscore probe deferred) | `euroleague_odds` evaluated → demoted (stale OddsPortal scraper). `euroleague-api` probed → GREEN. EuroCup decided IN, combined-model architecture decided. Multi-season historical fetch in progress (E + U, 2016-17 → 2024-25, builder at `scripts/euroleague_probe/fetch_seasons.py`). Only Flashscore Euroleague coverage probe remains; deferred to next season (off-season now). |
| 1 | Data layer | ✅ DONE (2026-05-29) | **Corpus + daily fetcher + bin wrappers all built & validated.** `ml_project/euroleague/build_corpus.py` ETLs the 18 raw season CSVs → `data_sets/Euroleague/team_game_stats.csv` (**4,483 games**: E 2,790 / U 1,693; 9 seasons; 90 teams; home-win 0.615; avg 81.6 pts), in the exact NBA-feature-engineering column contract + a `competition` column. `ml_project/euroleague/fetch_euroleague_daily.py` has `append-results` (per-game box-score fetch for only the day's games — idempotent, verified +0 on a known date) and `fixtures` modes. `euroleague_utils.py` holds a stable per-competition team-id registry (`team_ids.json`). See "Phase 1 — where we left off" below for the remaining bin scripts + a key gotcha. Branch: `feat/euroleague-phase0-multi-sport-rollout`. |
| 2 | Model + calibration | ✅ DONE (2026-05-29) | `euroleague_feature_engineering.py` (ELO + L5/L10 + venue-matched L5 + rest/b2b; teamId namespaced per competition → ladders auto-separate; carries `competition`) → `training_data.csv` (4,352 games). `train_euroleague_models.py`: **combined** winner `XGBClassifier` + total `XGBRegressor` with an `is_eurocup` game-level feature; 5-fold TimeSeriesSplit. **Winner acc 62.3% / Brier 0.240** (per-competition near-identical E 0.241 / U 0.237 → combined model healthy, no split needed); **total MAE 14.05**. `euroleague_calibration.py`: **per-competition Platt** (E + U), both improve Brier (E 0.241→0.226, U 0.237→0.225); `apply_home_win_platt(prob, cal, competition=...)` returns the canonical `(prob, applied, source)` 3-tuple. `predict_euroleague.py`: serve-time-from-corpus, validated on synthetic E+U fixtures. Models → `models/euroleague/{winner,total}_model.pkl` (+ feature manifests). Bin wrappers (`run_euroleague_predictions.sh` / `run_euroleague_verification.sh` / `retrain_euroleague_pipeline.sh`) built; full retrain runs end-to-end. |
| 3 | UI + betting integration | ✅ DONE (2026-05-29) | Ported the NBA blueprint, fully additive (football/NBA byte-unchanged). `EuroleagueBettingBackend(VirtualBettingBackend)` (`SPORT='euroleague'`, one-line subclass); additive `sports.euroleague` block in `betting_config.json`; `web_ui/euroleague/{__init__,routes}.py` (dashboard `/`, `auto_wager`, `place_bets`, `predict`/`verify`/`retrain` triggers — moneyline v1, slug-separated `output_euroleague/`); `web_ui/templates/euroleague/index.html` (predictions table w/ competition badge, bankrolls, recent slips, inline slip preview, **date inputs on predict/verify from day one**). Registered in `app.py`; `SPORTS` euroleague entry flipped `active:True` + `bets_dir:output_euroleague`; `EUROLEAGUE_TASKS` wired into `/status`. **Verified**: all routes 200, landing card active + portfolio aggregation picks it up, full place_bets path (debit €5 → slip write → reset) works, football/NBA untouched. **Betting flow is functional-but-empty until odds**: `auto_wager` joins `euroleague_odds_<date>.json` (season-start Flashscore probe) — no odds → empty slip, by design (same as NBA waited on ESPN). |
| 3.5 | Betting dashboard tab — placeholder | ✅ DONE (2026-05-28) | Fourth tab added to `templates/betting_tabbed.html` in canonical landing order (All / Football / 🏆 Euroleague / 🏀 NBA). Dormant placeholder card pointing at `EUROLEAGUE_NEXT_STEPS.md`. `/betting` route allowlist updated, docstring updated. Real per-bet panel = Phase B, lands with Phase 3. |
| 3.5b-PhaseB | Betting tab — real per-bet panel | ✅ DONE (2026-05-30) | Placeholder pane replaced with the shared **`_betting_basketball_panel.html`** (lane comparison + collapsible slip history, ML + O/U markets, read-only — no football-only void/cashout since basketball has no live feed in v1). Shared verbatim with NBA's Phase B. `betting_tabbed()` route now computes the full `compute_sport_summary` + per-sport `lane_defaults` for euroleague and nba (helper `_lane_defaults_for(slug)`); the tab `{% with %}`-scopes them into the partial. Renders empty-state cleanly (no slips yet, off-season). Smoke-tested: `/betting?tab=euroleague` 200, panel present. |
| 3.5b | Landing-page card + navbar entry | ✅ DONE (2026-05-28) | `SPORTS` list in `web_ui/app.py` gets a third entry (`slug: euroleague`, `active: False`, no `bets_dir` so it's skipped by cross-sport summary aggregation). Landing page reshaped from 2-per-row (`col-md-6`) to 3-per-row (`col-md-4`) + tighter card padding so the three cards fit on one row at md+ widths. Navbar Sport ▾ dropdown auto-inherits. |

## Alignment checklist — what "aligned with the other sports" means

The surface a new sport has to cover for it to look and act like football / NBA across the whole codebase. ✅ items are done (UI scaffold from 2026-05-28 + cross-cutting helpers that need zero per-sport work). ⬜ items remain.

### Already in place (no work required)

These are **sport-agnostic helpers that auto-inherit from `SPORTS`** — adding the `euroleague` entry was enough, they need nothing further:

- ✅ **Landing-page card** + **Portfolio Summary row** (`web_ui/templates/landing.html` iterates `SPORTS`; rows with no `bets_dir` are simply skipped from totals).
- ✅ **Navbar Sport ▾ dropdown** (`layout.html` via `inject_sports` context processor).
- ✅ **Betting tab on `/betting`** — placeholder pane shipped; cross-sport summary table already iterates `SPORTS`.
- ✅ **`sports_config.py` API** — `get_sport_config('euroleague')` / `lane_bankrolls('euroleague')` / `update_bankroll('euroleague', …)` / `all_bankrolls()` / `total_bankroll()` already work the moment a `euroleague` block exists in `betting_config.json`. Do **not** read or mutate the JSON directly; everything goes through this module.
- ✅ **`compute_sport_summary(bets_dir)`** — sport-agnostic, ingests any sport's `bets_<date>.json` directory and returns the same `{history, lane_stats, totals}` shape the football and NBA panels consume.
- ✅ **Sport-agnostic routes** — `/status`, `/stop/<task>`, `/server/<action>` are registered on `app` (not on any blueprint); no per-sport work.

### Per-sport surface still required

Grouped by area, with parity references to the football and NBA paths so the shape is unambiguous.

#### Web UI

- ⬜ **Blueprint** `web_ui/euroleague/routes.py` + `web_ui/euroleague/__init__.py`. Register in `app.py` next to NBA: `from euroleague.routes import euroleague_bp` + `app.register_blueprint(euroleague_bp, url_prefix='/euroleague')`. Mirror `web_ui/nba/routes.py` — same route names (`/`, `/predict`, `/verify`, `/retrain`, `/auto_wager`, `/place_bets`, `/cashout/<bet_id>`, `/void_bet/<bet_id>`, `/view/<filename>`, `/delete_file/<filename>`) with `g.backend = EuroleagueBettingBackend(...)` injected on `before_request`.
- ⬜ **Dashboard template** `web_ui/templates/euroleague/index.html` — mirror `web_ui/templates/nba/index.html`. Date inputs on Predict / Verify (parity item the NBA roadmap still has open — Euroleague should ship with them from day one, not inherit NBA's gap).
- ⬜ **Rich betting panel partial** `web_ui/templates/_betting_euroleague_panel.html` + swap into `betting_tabbed.html`'s Euroleague tab. Mirrors `_betting_football_panel.html` shape. Lands with Phase 3 / Phase B; placeholder stays until then.
- ⬜ **Sport entry's `bets_dir`** — flip `SPORTS` entry's `'active': False` → `True` and add `'bets_dir': 'output_euroleague'` once the bets-writing flow exists. (Until then the entry is dormant on purpose — `compute_sport_summary` is skipped, landing card greyscale.)

#### Betting backend

- ⬜ **`EuroleagueBettingBackend(VirtualBettingBackend)`** in `web_ui/betting_backend.py`, with `SPORT = 'euroleague'`. Mirror `NbaBettingBackend` (`web_ui/betting_backend.py:492-517`) — same subclass shape, one-line SPORT override; the parent class already routes bankroll updates to the right `sports.<slug>` block via `sports_config.py`.

#### Config

- ⬜ **`data_sets/betting_config.json` `sports.euroleague` block** — additive (zero edits to existing football / nba blocks). Required keys mirror NBA's block:
  ```
  bankrolls: {value, conviction, model} each with {current, initial}
  min_confidence, stake_multiplier, min_stake_eur, max_stake_pct, ev_cap_value,
  use_league_calibration (optional, default true),
  conviction_min_confidence, conviction_min_odds, conviction_stake_pct,
  model_base_pct, model_max_stake_pct, model_min_stake_eur,
  model_ev_factor_min, model_ev_factor_max,
  value_max_daily_exposure_pct, conviction_max_daily_exposure_pct, model_max_daily_exposure_pct
  ```
  Start with zero bankrolls + the same tunables as NBA; tune later.

#### ML pipeline (`ml_project/euroleague/`)

Mirror `ml_project/nba/` file-for-file:

- ⬜ **`__init__.py`**, **`euroleague_utils.py`** (shared helpers, team-name canonicalisation).
- ⬜ **Corpus builder** — `build_corpus.py` or `process_archive.py` (one-shot history pull via `euroleague-api`). Writes `data_sets/Euroleague/team_game_stats.csv` (one row per team per game) — same long-format the NBA feature engineering consumes. Merge-safe rebuild.
- ⬜ **Daily fetcher** — `fetch_euroleague_daily.py` with `append-results` + `fixtures` modes. Idempotent dedup. `time.sleep(1)` between API calls.
- ⬜ **Feature engineering** — `euroleague_feature_engineering.py` mirroring `nba_feature_engineering.py` (ELO_pre + L5/L10 rolling + venue-matched L5 + rest_days / b2b, all shift-leak-safe). Writes `data_sets/Euroleague/training_data.csv`.
- ⬜ **Trainer** — `train_euroleague_models.py`. Two models: winner `XGBClassifier`, total `XGBRegressor`. 5-fold `TimeSeriesSplit`. Saves to `models/euroleague/{winner,total}.{json|pkl}` + feature manifests `models/euroleague/features_{winner,total}.json` (same predictor-reads-manifest pattern as NBA).
- ⬜ **Calibration** — `euroleague_calibration.py` exposing `apply_home_win_platt` with the canonical **`(prob, applied, source)` 3-tuple** contract. Same as `ml_project/nba/nba_calibration.py:apply_home_win_platt` and `ml_project/calibration/apply.py:apply_platt_1x2` — UI / betting layer stays sport-agnostic. Output: `data_sets/Euroleague/euroleague_calibration.json`.
- ⬜ **Predictor** — `predict_euroleague.py`. Mirrors `predict_nba.py`'s **serve-time-from-corpus** pattern (the train/serve-skew fix). Writes `output_euroleague/predictions_euroleague_<date>.csv` with the column shape that downstream `auto_wager` understands:
  - Required: `gameId, Home Team, Away Team, Home Win Prob, Home Win Prob (raw), Cal Source, Predicted Winner, Predicted Total`.
  - Plus odds + EV + Kelly once odds are joined in.
- ⬜ **Evaluator** — `evaluate_euroleague_predictions.py` (can ship later; flag non-fatal in the bin script like NBA does).
- ⬜ **Tuner** — `tune_euroleague_models.py` (deferred until the feature set has stabilised — same posture as NBA's deferred tuner).

#### Data layout

- ⬜ **`data_sets/Euroleague/`** — corpus CSV, ELO cache JSON, calibration JSON, training data CSV, optional raw API cache. Gitignored if any of it is large/regenerable (mirrors NBA's gitignored archive).
- ⬜ **`output_euroleague/`** — sport's bets-and-artifacts directory. `predictions_<date>.csv`, `verification_<date>.csv`, `bets_<date>.json`, soft-delete subdir `output_euroleague/history/`. This is the path that becomes the `SPORTS` entry's `bets_dir`.
- ⬜ **`models/euroleague/`** — trained artifacts (winner + total + features manifests).
- ⬜ **`logs/euroleague_predict.log`**, **`logs/euroleague_verify.log`**, **`logs/euroleague_retrain.log`** — created on first run by the bin scripts.

#### Scrapers

- ⬜ **Flashscore Euroleague spider** — basketball variant of `flashscore_scraper/spiders/flashscore_spider.py` (or extension of `nba_spider.py`). Forward-only odds + live data per the free-only strategy in Data sources §B. Same Playwright infra, same lockfile / delay discipline.
- ⬜ **`data_sets/team_mappings.json` Euroleague entries** — reconcile `euroleague-api` team names ↔ Flashscore display names ↔ any backup data source via `entity_resolver.py` / fuzzy match.

#### Bin scripts

Mirror NBA's wrappers in `bin/`:

- ⬜ **`bin/run_euroleague_predictions.sh`** (takes optional `YYYY-MM-DD`; defaults to tomorrow).
- ⬜ **`bin/run_euroleague_verification.sh`** (takes optional `YYYY-MM-DD`; defaults to yesterday).
- ⬜ **`bin/retrain_euroleague_pipeline.sh`** — chains corpus refresh → feature build → train → fit calibrator → validate calibrator (last two non-fatal, same posture as football's pipeline).

All three must:
- `source venv/bin/activate`
- `export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ml_project:$(pwd)/ml_project/euroleague`
- Be `set -u` safe.
- Use the MacOS `date -v` + Linux `date -d` portable-fallback pattern from NBA's wrappers.

#### Contracts / schema (the parity that actually matters)

Surface alignment isn't enough; the **shapes** other code consumes have to match:

- ⬜ **Calibration return shape**: `(prob, applied, source)` 3-tuple (matches `apply_home_win_platt` / `apply_platt_1x2`). Non-negotiable — `MatchPredictor`-style consumers branch on it.
- ⬜ **Predictions CSV columns**: at minimum the NBA shape (`gameId, Home Team, Away Team, Home Win Prob, Home Win Prob (raw), Cal Source, Predicted Winner, Predicted Total`) plus odds / EV / Kelly columns once odds join.
- ⬜ **`bets_<date>.json` per-bet schema** — `bet_id, lane, status, stake, odds, pnl, ...` exactly as `VirtualBettingBackend` writes for football and NBA. `EuroleagueBettingBackend` inherits this for free if the subclass only overrides `SPORT`.
- ⬜ **Status taxonomy** — `OPEN / WON / LOST / VOID / CASHED_OUT` (per CLAUDE.md). Use the inherited resolver; don't re-invent.

#### Docs

- ⬜ **`CLAUDE.md` Euroleague section** — slot after the NBA pipeline description with the same shape (data layer → model → predictor → reactivation notes). Reference this doc for the roadmap.
- ✅ **This document (`EUROLEAGUE_NEXT_STEPS.md`)** — created 2026-05-28.

### What's intentionally NOT on this list (out of scope for v1)

To avoid scope creep, the following sport-level surfaces exist for football but are deliberately **not** part of v1 Euroleague parity:

- **Auto-cashout scheduler** (football-only daemon; NBA doesn't have it either).
- **Live in-play snapshots + live history JSONL + LiveAdjuster** (football's cashout-decision foundation; basketball cashout is its own multi-week project).
- **Real-betting / Pamestoixima integration** (dormant for football, no plans for basketball).
- **Bookmaker cashout offer integration** (football-only via Pamestoixima scrape).

## Data sources

Three concerns, three answers:

### A. Historical results + box scores (training corpus)

**Recommendation: `euroleague-api`** — official Euroleague Stats API wrapper. Free, sustainably maintained (recent PyPI release), covers Euroleague + EuroCup, includes box scores, game stats, player stats, team stats, shot data. Plain HTTP, no browser. Behaves the way NBA's local Kaggle archive does: deep history + structured schema + no rate-limit headaches.
- PyPI: <https://pypi.org/project/euroleague-api/>
- GitHub: <https://github.com/giasemidis/euroleague_api>
- A Kaggle mirror exists (<https://www.kaggle.com/datasets/babissamothrakis/euroleague-datasets>) — could be the corpus seed, with `euroleague-api` doing the daily-refresh role (matching NBA's archive-seed + `fetch_nba_daily` pattern). **This is the suggested split.**

Open: confirm history depth and per-game richness (advanced metrics? play-by-play?) once we install it. Probe script analogous to NBA's `process_archive.py` will tell us in 30 minutes.

Also worth a look as a reference / cross-check: `sakisbl/euroleague-scraper` (raw Euroleague.net scrape) and `flavioleccese92/euroleaguer` (R wrapper, but useful for endpoint discovery). Not recommended as primary; `euroleague-api` is the cleaner Python entry point.

### B. Odds (the bit you asked about)

**Constraint (operator, 2026-05-28): paid services are off the table.** That rules out The Odds API's historical endpoint (~$25–$100/mo paid tier) and SportsDataIO / OpticOdds / OddsJam. Free-only from here on.

**`euroleague_odds` (`giasemidis/euroleague_odds`) — probed 2026-05-28, verdict: usable as backfill-only, with caveats.** Source readout:
- **Target site: OddsPortal** (`oddsportal.com/basketball/europe/euroleague/...`). Hardcoded in `settings.json`. ⇒ same ToS-hostile + behavioral-detection risk class as the other OddsPortal scrapers, with all the implications (we already ate an Akamai block on Pamestoixima — see `real_betting/PAMESTOIXIMA_NOTES.md`).
- **Stack: Selenium + ChromeDriver + BeautifulSoup**, headless. Predates Playwright; would need modernisation to align with our `flashscore_scraper/` stack (Playwright everywhere else).
- **Markets: moneyline ONLY** — output schema is exactly `home_team, away_team, home_team_points, away_team_points, home_team_odds, away_team_odds` per game. **No totals, no spread, no date column, no bookmaker column.** OddsPortal shows aggregate/average odds by default, so the values are blended across books, not from any one bookmaker.
- **Stale**: `current_season: 2021` in `settings.json`; no commits beyond the 2020–21 season setup. URL pattern still works (OddsPortal's structure hasn't changed) but the package itself isn't maintained.
- **No license file** in the repo — copy-paste / fork rather than dependency.
- **Sibling-author with `euroleague_api`** (same maintainer, `giasemidis`) — context for why you found them together, not a quality signal for the odds package.

**The realistic free-only matrix:**

| Source | Markets | History | Risk | Verdict |
| --- | --- | --- | --- | --- |
| **`euroleague_odds` (OddsPortal scrape, ML only)** | Moneyline only | Variable depth (OddsPortal goes back ~15 yrs for ULEB Euroleague); needs modernisation (Selenium → Playwright) and reactivation of newer seasons | **HIGH** (OddsPortal anti-bot) | 🟡 **Optional historical backfill**, fork-and-modernise. Not a Phase-1 dependency. Build only if we want odds-aware training features after the odds-free baseline ships. |
| **`OddsHarvester` / `gingeleski/odds-portal-scraper`** | h2h + totals + spreads across multiple sports | Same OddsPortal depth | **HIGH** (same anti-bot) | Same risk class as `euroleague_odds` but actively maintained and multi-market. If we do go down the OddsPortal path, prefer one of these forks over the stale `euroleague_odds` package. |
| **Flashscore Euroleague spider** (variant of existing `flashscore_scraper/`) | Whatever Flashscore exposes for basketball (typically moneyline + total) | Forward only — accumulates from the day we turn it on | LOW (already in stack, Playwright-based) | ✅ **Use for current/live odds.** No history, but cheapest and lowest-risk. Parity with how football/NBA scrape their daily odds today. |
| **The Odds API free tier** (`the-odds-api.com`) | h2h / spreads / totals; `basketball_euroleague` is a supported sport key | **Current only on free tier** (500 req/mo). Historical = paid (ruled out). | LOW (compliant API) | Secondary current-odds source if Flashscore's basketball odds turn out thin. Useful as a cross-check, not a primary. |
| **API-Sports basketball** (`api-sports.io`) | Multiple books | 7-day rolling window only | LOW | ❌ Useless for training (no archive). Free tier is fine for current odds, but Flashscore covers that already. |

**Suggested odds strategy (free-only):**

1. **Forward / daily / live**: extend the Flashscore spider to a `euroleague` (basketball) variant — same shape, same Playwright infra. Free, in-stack. **This is the Phase-1 odds source.**
2. **Training the v1 model: go odds-free.** NBA already proves this pattern — `predict_nba.py` doesn't depend on odds; odds are joined *at inference* (Phase 3) for EV/Kelly grading. Same pattern for Euroleague. Removes the entire historical-odds dependency from Phases 1–3.
3. **Historical odds backfill = optional enrichment, deferred.** Only worth doing if a forward-only A/B (odds-free baseline vs. odds-aware variant) on accumulated forward data suggests odds-aware features would lift Brier — same OOF gate the football D2 features go through. Even then, weigh the scraping-maintenance burden against the lift; the football pipeline gets odds for free from `football-data.co.uk` CSVs, but no equivalent exists for Euroleague, so this would be a permanent maintenance line item.
4. **If we do the backfill**: don't depend on `euroleague_odds` as-is — it's stale and Selenium-based. Two options, in preference order:
   - **(a) Fork + modernise**: port the OddsPortal selectors to Playwright, generalise to totals/spreads, drive via our existing scraper infra. ~1 week of work; brittle to OddsPortal DOM changes.
   - **(b) Use `OddsHarvester`**: actively maintained, Playwright-based, multi-market. Less custom code, same anti-bot risk. Probably the right call if we commit.
   Either way: rate-limit aggressively, cache the full pull, treat the OddsPortal access as a one-time-per-season operation (not daily).

### C. Daily live data (in-play snapshots, the cashout/live-adjuster path)

Mirror the football live-history pattern: extend `flashscore_scraper/` with a Euroleague spider that emits the basketball equivalent of the football live JSONL (score, period/minute, basic per-team stats). Defer; not needed for Phases 1–3. Revisit when cashout / live-betting work gets prioritised for basketball — currently NBA hasn't built this either.

## Active queue (ordered)

### Phase 0 (current) — Source evaluation

1. **`euroleague_odds` evaluated 2026-05-28** ✅ — verdict: stale OddsPortal scraper (Selenium, moneyline-only, last touched for the 2020–21 season, no license). Demoted to "optional historical backfill" (see Data Sources §B). Not a Phase-1 dependency.
2. **`euroleague-api` probed 2026-05-28** ✅ — verdict: **GREEN, use as Phase-1 corpus source.** `pip install euroleague-api` (version 0.1.1 picked up); pulled 2023-24 season (`competition='E'`, `season=2024` — convention is ending year).
    - **`GameStats.get_game_report_single_season(2024)` → 330 rows × 47 cols.** Per-game schedule + result: `Season`, `Gamecode`, `Round`, `Phase` (`RS`/`PO`/`FF`), `date`/`localDate`/`utcDate`, `local.club.name|code` / `road.club.name|code`, `local.score` / `road.score`, plus pre-computed `localLast5Form` / `roadLast5Form`. Clean date strings, no missing scores on played games (`played: True` flag).
    - **`GameStats.get_game_stats_single_season(2024)` → 330 rows × 106 cols.** Per-game team box scores: `local.team.*` and `road.team.*` for points / FG2 / FG3 / FT made+attempted / rebounds (total / def / off) / assists / steals / turnovers / blocks (favour/against) / fouls (committed/received) / plus-minus / valuation / time played. `local.total.*` / `road.total.*` give the team+bench aggregate splits. **Caveat:** `local.players` / `road.players` columns carry serialized JSON player lists per game (~16 MB / season). Drop for the v1 long-format corpus; preserve in `raw/` for future D4-style availability work.
    - **`BoxScoreData.get_teams_boxscore_quarter_scores_single_season(2024)`** running in the background — quarter-level scores are a nice-to-have (pace / blowout features) but not required for v1 (NBA corpus doesn't use them either; per-game totals come from `game_stats`).
    - **Files promoted** to `data_sets/Euroleague/raw/` (gitignored via `data_sets/*`). Probe is re-runnable via `scripts/euroleague_probe/probe_euroleague_api.py [SEASON]`. Schema notes + layout documented in `data_sets/Euroleague/README.md`.
    - **NBA-corpus shape compatibility**: every column needed to derive NBA's ~40-feature set (rest_days / b2b / ELO_pre / L5/L10 rolling / venue-matched L5) is present after a wide-format → long-format ETL (one row per team per game). That ETL is the first job of Phase 1.
3. **Probe Flashscore Euroleague coverage** — visit one live Euroleague match page in the existing Playwright spider, dump the available markets and stats fields, confirm we get moneyline + total. Validates the Phase-1 odds source. **DEFERRED** — Euroleague is currently off-season (Final Four wrapped late May; 2026-27 starts ~October). Pick up at season start, or use an archived match URL sooner if Phase 1 is ready to move.
4. **EuroCup inclusion — DECIDED 2026-05-28 ✅** Include EuroCup from day one. Same `euroleague-api`, swap `competition='E'` → `'U'`. Tagged with `competition` column in the canonical corpus.
    - **Model architecture decided alongside**: **one combined model with `competition` as a categorical feature** (mirrors football's "one XGBoost across all leagues with `league_cat`" pattern, validated through C4 Platt). Only revisit if Phase 2's OOF Brier shows the combined model systematically under-performs on one competition vs. a separate-model baseline. The Phase-2 calibration note in the table above ("Single global Platt on `P(home_win)`") becomes **two-calibrator Platt (per competition)**, same shape as football's per-league fit.
    - **ELO**: computed per `(team, competition)` pair, separate caches (`euroleague_elo_E.json` / `euroleague_elo_U.json`) — same pattern football uses to separate per-league ELO. A team's Euroleague ELO ≠ EuroCup ELO; the rating ladders are independent because the opponent pools are different competitive levels.

### Multi-season historical fetch — ✅ DONE (2026-05-28, 53.4 min, zero failures)

Builder: `scripts/euroleague_probe/fetch_seasons.py`. Fetched both competitions × 9 seasons (2017 = 2016-17 → 2025 = 2024-25) × 2 endpoints (`game_report` + `game_stats`). 36 CSVs in `data_sets/Euroleague/raw/` named `{COMP}_{SEASON}_{endpoint}.csv` (plus the legacy `E_2024_quarter_scores.csv` kept from the original Phase-0 probe). 204 MB total on disk; gitignored via `data_sets/*`.

Seasonal coverage check (rows per `game_report` ≈ games per season):
- Euroleague: 2016-17 = 260 games (pre-expansion); steady ~310–330 through 2024-25 (post-expansion + Final Four).
- EuroCup: 2016-17 → 2024-25 = 195–200 games per season, consistent.

Script is idempotent — re-runs skip files already on disk; safe to extend by adding a new season (e.g. `--start 2026 --end 2026` next May) once 2025-26 finishes. **Phase 1 (canonical long-format corpus from these raw CSVs + ETL into `team_game_stats.csv`) is now unblocked.**

### Phase 1 — where we left off (2026-05-29)

**✅ DONE (built, run, validated):**
- **`ml_project/euroleague/build_corpus.py`** — ETLs the 18 raw season CSVs (`data_sets/Euroleague/raw/{E,U}_<season>_game_{report,stats}.csv`) → `data_sets/Euroleague/team_game_stats.csv`. **4,483 games** (E 2,790 / U 1,693), 9 seasons (2017→2025), 90 teams, home-win 0.615, avg 81.6 pts. One row per team per game, in the **exact column contract `nba_feature_engineering` consumes** (`gameId, date, season, postseason, teamId, home, win, teamScore, opponentScore, fieldGoalsPercentage, threePointersPercentage, freeThrowsPercentage, reboundsTotal, assists, turnovers, plusMinusPoints`) **+ a `competition` column** + extras (FG2/3 split, FT, off/def reb, steals, blocks, valuation=PIR, codes/names). Merge-safe rebuild (preserves daily-appended rows). Exposes `finalize_rows()` / `rows_for_merged()` reused by the daily fetcher.
- **`ml_project/euroleague/euroleague_utils.py`** — paths + `TeamIdRegistry` (stable append-only integer id per `(competition, club_code)`, persisted to `data_sets/Euroleague/team_ids.json`; ELO ladders kept separate per competition).
- **`ml_project/euroleague/fetch_euroleague_daily.py`** — `append-results` (default yesterday) + `fixtures` (default tomorrow), both competitions, `time.sleep(1)` per call. `append-results` fetches the cheap season **report** then per-game `get_game_stats(season, gamecode)` for **only the day's games** (not the 402-game season scan) → appends idempotently (dedup on `(gameId, teamId)`). Verified: 2024-10-03 found the 6 Euroleague opening-night games and re-added **+0** rows (idempotent). `fixtures` writes `data_sets/Euroleague/fixtures_<date>.json` in the NBA predictor's shape.

**⚠️ Key gotcha corrected:** euroleague-api's **season code = STARTING year** (2024 = the 2024-25 season; verified: `E_2024_game_report` is dated Oct 2024). The Phase-0 note that called it "ending year" was wrong. `_season_code()` in the fetcher uses `year if month>=8 else year-1`. Box-score totals are in `{side}.total.*` (`.team.*` is zeroed); score from the report's `{side}.score`; `plusMinusPoints` computed as `teamScore − opponentScore` (the API's `total.plusMinus` is summed-player noise).

**✅ Bin wrappers now built (Phase 2 done):** `bin/run_euroleague_predictions.sh` (fixtures → predict), `bin/run_euroleague_verification.sh` (append-results; a predictions-vs-results evaluator + bet settlement are a Phase 3 / later-evaluator leftover), `bin/retrain_euroleague_pipeline.sh` (build_corpus → features → train → calibrate, last step non-fatal; runs end-to-end). All `set -u`, portable `date -v`/`date -d`, `PYTHONPATH += ml_project/euroleague`.

**▶️ Phases 1–3 all DONE (2026-05-29).** Euroleague is a live, clickable paper-betting sport in the UI (dashboard + predictions + moneyline slip flow), mirroring NBA. **All remaining work is season-gated (handle at season start, ~Oct):**
- **Flashscore Euroleague odds probe** → `output_euroleague/euroleague_odds_<date>.json` (`{home_team, away_team, home_ml_decimal, away_ml_decimal}`). This is the one thing that turns the betting flow from empty → live (EV/Kelly). The `auto_wager` join is already coded for it.
- **Live fixtures/results validation** against a real slate (confirm `fetch_euroleague_daily` fixtures/append on in-season data; team-name join on live names).
- **Predictions-vs-results evaluator** (Brier/acc on settled games) — wire into `run_euroleague_verification.sh` (currently append-results only).
- **Totals (Over/Under) market — ✅ model side DONE (2026-05-29; betting still odds-gated).** `euroleague_calibration.py` estimates per-competition residual **σ** (E 17.5 / U 18.2 / pooled 17.8) and validates the normal-approx (over-rate@μ=0.501, ±1σ=0.693 / ±2σ=0.954 — Gaussian). `prob_over` + per-competition `total_sigma` helpers added; `predict_euroleague.py` emits `Total Sigma` / `Over Line` / `P(Over)` / `P(Under)`; `/euroleague/auto_wager` builds an **O/U market** alongside moneyline. Like moneyline, it only populates once the Flashscore odds file supplies a total line — so it's built + tested (synthetic) now, live at season start.
- **Betting tab Phase B — ✅ DONE (2026-05-30).** `/betting` Euroleague placeholder replaced with the shared `_betting_basketball_panel.html` (lane comparison + slip history, ML + O/U). Shared with NBA's Phase B (same partial). Renders empty until a live slate produces slips.
- Optional: retune `tune_euroleague_models.py`, historical odds backfill.

### Phase 2 — Model + calibration

Copy `ml_project/nba/` → `ml_project/euroleague/` and diverge:
- Same ~40-feature set, same time-series CV, same Platt-on-P(home_win), same `(prob, applied, source)` 3-tuple contract.
- **Euroleague-specific watch-outs**: shorter season → smaller sample (a couple of thousand games over 20 years vs. NBA's 30k); higher roster volatility between seasons; neutral-court playoff games (Final Four — handle like national-team finals in D7, predict orientation-averaged); 40-min games and trapezoid key vs. NBA's 48-min — won't affect the modelling but will affect any feature inherited from "minutes" or "possessions" assumptions.
- Manifest models to `models/euroleague/{winner,total}.json` + `features_{winner,total}.json`.

### Phase 3 — UI + betting integration

Copy the NBA blueprint and rebrand. Same football-isolation rules. Add a `euroleague` entry to `SPORTS` in `web_ui/app.py`; new `EuroleagueBettingBackend(VirtualBettingBackend)`; new `euroleague` bankroll/config block in `betting_config.json`. Reuse `sports_config` / `compute_sport_summary(bets_dir)`. Bare moneyline-only panel in v1; total when P(Over) is plumbed.

### Phase 3.5 — Betting dashboard fourth tab

Placeholder card first (mirrors NBA Phase A), real panel once the moneyline flow stabilises (mirrors NBA Phase B).

## Open / deferred

- **Player availability + PIR-impact adjuster (re-homed from football D4). Importance half ✅ BUILT (2026-05-30); adjuster still gated on a forward feed.** Same idea NBA is scoping (`NBA_NEXT_STEPS.md` → "Player availability + on/off impact"), and **cheapest of all three sports here**: the `game_stats` raw we already fetched carry per-player box scores with the **official PIR (`valuation`) + `plusMinus`** — a ready-made efficiency / value-above-average metric, *no extra scraping and no model to build*. The three pieces and their status:
  - **(b) per-player PIR importance — ✅ DONE.** `ml_project/euroleague/euroleague_player_importance.py` parses the per-player JSON in every raw `{E,U}_<season>_game_stats.csv` → `data_sets/Euroleague/player_importance.json`: per (competition, season, player) `pir_pg` / `mpg` / `plus_minus_pg` and **`importance = max(0, pir_pg − team roster-median pir_pg)`** (the squad-median replacement-level proxy, mirroring football N2). Built over **95,283 player-game rows → 5,556 player-season records** (E+U, 2017–2025); top-importance names validate (TJ Shorts, Vezenkov, Mirotić, Maledon). Has a `--self-test`. This is the data-backed, feed-independent half — analogous to the football N2 that *was* built before stopping at the adjuster.
  - **(a) forward "who's out" feed — ❌ the make-or-break gap, NOT available.** No Euroleague injury/roster news source is wired, and it's off-season (no live slate to consume or validate against). Same gate as NBA NA0.
  - **(c) depletion → margin → win-prob shift** — the basketball margin↔prob mapping NBA will use; trivial once (a) exists.
  - **Why the adjuster is NOT built now:** per the operator's football-D4 call, a feed-less / importance-blind adjuster is "the useless case" — building it now would be dead code with no input and no way to validate. Resume at (a) when an availability feed is found and the season is live; (c) + wiring into `predict_euroleague.py` (behind `use_availability_adjustment`, default off) is then a small add reading `player_importance.json`.
- **Per-sport tuning of the betting strategy tunables — check if it has value (NEW 2026-05-30).** The `sports.euroleague` block in `betting_config.json` is per-sport *structurally* but currently carries football's *values* (`min_confidence`, `stake_multiplier`, `max_stake_pct`, `ev_cap_value`, conviction `min_odds`/`min_confidence`/`stake_pct`, `model_*` factors, the `*_max_daily_exposure_pct` caps). Those were tuned for football's 1X2 + O/U markets; Euroleague/EuroCup is **moneyline + totals** with a different odds shape (favourite-heavy ML), so the inherited gates may mis-size (e.g. conviction `min_odds 1.40` rarely fires on big favourites; EV cap calibrated to football variance). **Action**: once a live slate accrues settled bets, run the football-style lane backtest on Euroleague odds to check whether per-sport tuned values beat the football defaults, and set them if so. Gated on settled-bet volume. Same item is open for NBA (`NBA_NEXT_STEPS.md`) — basketball-symmetric; consider tuning both together once either has volume.
- **Centralise basketball odds source?** Both NBA (ESPN, forward-only) and Euroleague (Flashscore, forward-only) are now forward-only by choice. If the odds-free baseline proves out for Euroleague, the same odds-free pattern is worth re-validating for NBA training too. Symmetry > shared API.
- **Live / cashout for basketball.** Not in scope for v1. When prioritised, Euroleague + NBA can probably share a `BasketballLiveAdjuster` (Poisson goal model becomes a Poisson points model; xG → eFG% or pace-adjusted scoring; possession proxies via assists/turnovers).
- **EuroCup as a second competition.** Cheap to fetch (same API), the model question is "one model with `competition` as a categorical feature" vs. "two models." Default to one model + categorical feature unless OOF Brier says otherwise.
- **National-team competitions overlap.** Euroleague-Bs and the FIBA national-team calendars overlap; `target_leagues.json` and `is_international()` already handle the football side — basketball won't share that infra but should follow the same exact-match approach for competition filtering.
- **Optional historical odds backfill** — if forward-only A/B testing shows odds-aware features would lift Brier (Phase 4-ish), do a one-time OddsPortal pull via a forked/modernised `euroleague_odds` or `OddsHarvester`, cache to `data_sets/Euroleague/odds/`, treat as static training data. Aggressive rate-limiting, single-shot per season, accept the brittleness (OddsPortal DOM changes break it). Defer until the lift is demonstrated, not before.

## Settlement

**Basketball resolver — ✅ BUILT (2026-09-28)**, `ml_project/resolve_basketball_bets.py`, wired into `bin/run_euroleague_verification.sh` after the corpus append (non-fatal: the append is irreplaceable since the API only serves the current season, settlement just re-runs).

Before this, Euroleague bets debited a bankroll and stayed `OPEN` forever — placing them was spending, not betting. The football resolver could not be reused: `resolve_daily_bets.py` hardcodes `res_ou = (h+a) > 2.5`, so every basketball total is "over" and **every** O/U bet would settle WON, and its 1X2 branch assumes a draw exists and expects `1`/`X`/`2` rather than our team-name moneyline.

Four things it does differently, each an improvement rather than an adaptation:

- **Joins on `gameId`, not fuzzy names.** Bets carry the fixture's `gameId` (`E2026_13`) in `match_id`; the corpus is keyed by the same string. Football needs rapidfuzz because its sources disagree on club names; here they cannot.
- **Each totals bet settles against its OWN line**, read from the bet (`line`/`side`), because the staked row is whichever the ladder selection chose — with a fallback that parses `"Over 168.5"` for slips written before those fields existed.
- **PUSH is a real state** (integer line, total == line → stake refunded). Football's 2.5-style lines can never push.
- **Scores the counterfactual without paying it**, into `counterfactual_result`, so ladder-shopping vs the consensus line is measurable on identical games.

Idempotent (only `OPEN` bets are touched) and partial-friendly (a game absent from the corpus leaves its bet `OPEN`). `--sport nba` works off the same code path — the two corpora share a column contract — and `--dry-run` reports without writing or crediting.

Verified on a synthetic slip built from the real settled game `E2026_1` (Crvena Zvezda 77-83 Zalgiris, total 160): ML home LOST / ML away WON +14, Over 154.5 WON +9 while its counterfactual Over 165.5 LOST −10 (the A/B working on one game), unknown game left OPEN, an already-settled bet untouched and not re-credited, lane credits value 24.00 / model 19.00, and a second run settling 0 with no credits. 13 unit cases cover every branch including PUSH and undecidable input.

## Bugs / fixes queue

**Season-start reactivation (2026-09-27) — four defects, all fixed.** The pipeline was built off-season in May and had never run against a live season. First contact with the 2026-27 season (EuroLeague tipped off 24 Sep) found:

1. **`euroleague-api` was never in `requirements.txt`** and was not installed. It is the *only* data source for both competitions. Built in May on the phase branch with an ad-hoc install, so every checkout since had a Euroleague pipeline that could not fetch anything. Now pinned at `0.1.1`.
2. **…and it failed silently.** `fixtures()` imported the package *inside* the per-competition `try`, so `ModuleNotFoundError` was caught twice, an **empty** `fixtures_<date>.json` was written, and the process exited **0** — indistinguishable from a day with no games. Same silent-failure class as football's `update_leagues_data.sh` / `update_football_data.py`. The import now sits outside the loop and returns 1 with an install hint; a per-competition failure is still tolerated (EuroCup legitimately raises `KeyError('game')` on a season with no games yet, and hard-failing would block EuroLeague predictions all autumn), but **all** competitions failing returns 1 and writes no file, so an outage can never masquerade as an empty slate or clobber a good file.
3. **`fixtures()` queried a results-only endpoint.** It used `GameStats.get_game_report_single_season`, which on 2026-09-27 returned 10 rows for season 2026 — every one `played=True` — so the `played != True` filter could never match and the mode returned **0 fixtures for every date**. No fixtures, no predictions, all season. The right source is `Schedule.get_schedule(season)`: the full 380-game calendar including unplayed games, and its `gamecode` (`"E2026_7"`) is already exactly the `gameId` that `build_corpus` produces. Two traps in that feed: `played` is the **string** `"true"`/`"false"` (a real bool on the other endpoint, so `!= True` matches everything), and `confirmedtime` is a boolean flag despite the name — the tip-off is `startime`.
4. **Corpus was 4 months stale.** Backfilled 2026-09-24 and 2026-09-25 via `append-results` (+20 team-rows, 10 games, idempotent — a re-run adds 0), then re-ran `euroleague_feature_engineering.py`, because `predict_euroleague.py` reads the **cached** `euroleague_elo.json` rather than recomputing: without that step the backfill would have reached rolling-form features but not ELO. 20 ratings moved (exactly the 20 team-rows) plus one new ladder (91 vs 90 — a club new to the competition); `training_data.csv` 4,352 → 4,361 games.

Verified end to end afterwards: `./bin/run_euroleague_predictions.sh 2026-09-29` → 13 fixtures (8 E + 5 U), 13 predictions, per-competition Platt applied, exit 0. **EuroCup fixtures work through the schedule endpoint** even though its game-report call still raises, so the tolerate-one-competition rule is what keeps it usable. Calibration is compressing hard on this slate (raw 0.23–0.80 → calibrated 0.46–0.70) but is **order-preserving within each competition** — not the ordering inversion that got football's calibration disabled. Worth re-checking once the season accrues settled games.

**Odds probe — ✅ BUILT (2026-09-27)**, `ml_project/euroleague/fetch_euroleague_odds.py`, wired non-fatally into `bin/run_euroleague_predictions.sh` between fixtures and predict. Flashscore, both competitions, moneyline + totals. This closes the Phase 0 "Flashscore Euroleague coverage probe" deferred to season start.

**It is a ladder, not a line — the central design point.** Football's O/U is a fixed 2.5; basketball totals are not. Each book posts its own line *and* its own prices, and Flashscore lists all of them: 17–19 rows spanning ~162.5–179.5 on a EuroLeague game, sometimes three books on the same 169.5 at different prices. **Line and price must come from the same row** — mixing one book's line with another's price records a bet nobody offered. Every row carries its own `book`, and the full ladder is persisted (operator's call) so the selection policy can change, and be *measured*, without re-scraping.

Schema per match: `moneyline[] {book, home, away}`, `totals[] {book, line, over, under}`, plus back-compatible scalars (`home_ml_decimal`/`away_ml_decimal`/`total`/`over_ml_decimal`/`under_ml_decimal` + `ml_book`/`total_book`) so the existing predictor and `auto_wager` work unchanged. Scalars use book preference **bwin → bet365 → Stoiximan → PameStoixima → Novibet** for the moneyline, and for totals the **main line** = the row with the most balanced over/under prices (book preference only as tie-break).

Four things learned, each now guarded in code:

- **Market coverage differs by book.** Moneyline carries 5 books *including PameStoixima* — the book `real_betting/` drives — but totals carry only 3 (bet365, Stoiximan, Bwin) and **PameStoixima is absent**. So our own book can price the ML but not the total.
- **`ft-including-ot`, never `full-time`.** Both tabs exist; basketball settles including overtime, so the wrong tab mis-settles every OT game.
- **A failed odds tab silently serves the STANDINGS table.** `.ui-table__row` matches it just as happily: `Lietkabelis v KK Bosna` (2026-09-30) produced 8 "totals" whose book was a team name, whose line was the league rank (1,2,3…) and whose prices were 0.0. Plausibility bounds (`MIN/MAX_DECIMAL_ODDS`, `MIN/MAX_TOTAL_LINE`) now discard those and report the count; that match correctly reports 0 ML / 0 totals.
- **The fixture join needs a margin, not a threshold.** Flashscore short names vs the API's sponsor-laden ones: the worst true pair was `Venezia v Frankfurt` → `UMANA REYER VENICE v SKYLINERS FRANKFURT` at just **66**. But across all 13 pairs the correct fixture won by **37–72 points** and no runner-up exceeded 46, so acceptance is a modest floor (60) *plus* a margin over the runner-up (20). 13/13 joined.
- A partial run (`--competition U`) used to overwrite the file and drop the other competition's records (13 → 5). Writes now **merge** on `flashscore_match_id`.

**EuroCup totals bias — ✅ DIAGNOSED AND FIXED (2026-09-28).** It was a train/serve contract violation, not a calibration problem.

*Measurement.* Pooling two slates, `Predicted Total − market line` ran **+8.63 for EuroCup (n=15, t=+4.13, 95% CI [+4.54, +12.73], 13/15 overs)** against **+0.62 for EuroLeague (n=10, t=+0.37)** — real, not noise. But the stored OOF diagnostics say the total model is *unbiased against outcomes* on both competitions (`resid_mean` E +0.237 / U +0.735; `over_rate_at_mu` 0.4985 / 0.5046). Model right vs market right cannot be settled by those two facts alone, so the question became: why does the model behave differently on EuroCup *now*?

*Root cause.* `euroleague_feature_engineering` ends with `dropna(subset=required)` over the L10 and venue-L5 columns — so **the model is never trained on a game with incomplete history** — while `predict_euroleague` happily served such fixtures with NaN features. EuroCup churns its field every season: **13 of 32 teams on these slates have ZERO corpus rows** (median 19) against 0 of 20 for EuroLeague (median 306). Measured directly: **75% of EuroCup fixtures violate the training contract vs 10% of EuroLeague**, and on them the model predicts a mean total of **178.6 against its own training mean of 164.1**. That is extrapolation outside the fitted shape, not a bias worth calibrating away — and note fitting a correction to the *market* would just be regressing the model onto the price, which this project has already shown is worthless.

*Fix.* `predict_euroleague` now imports `L10_METRICS` / `VENUE_METRICS` from the trainer (shared constant, cannot drift), counts missing required features per fixture, and emits `Feature Gaps` + `Totals Eligible`. Ineligible fixtures still get a row and a `Predicted Total` for visibility, but no `Bet *` / `Cf *` pick. `/euroleague/auto_wager` re-checks the flag itself, since it can be pointed at any CSV on disk (a CSV predating the column is allowed through, as before). **Result: EuroCup +8.63 (t=4.13) → +4.13 (n=4, t=+1.11)** — no longer distinguishable from zero — and EuroLeague is unchanged at −0.29. The cost is coverage: EuroCup goes from 16 priced fixtures to 4.

*Still open from this.* The **winner model shares the same training matrix and the same `dropna`**, so its EuroCup predictions are equally out-of-contract — but the harm is unmeasured (no market comparison was run for moneyline), so ML is deliberately left ungated. Either gate it too, or measure it first. Coverage will also repair itself as the season accrues games: every EuroCup team crosses the L10 threshold after ~10 games.

**(superseded) original observation —** On the 2026-09-29 slate, mean `Predicted Total − line` was **−0.29 for EuroLeague (n=8, 4/8 overs)** but **+7.77 for EuroCup (n=5, range −5.7…+17.1, 4/5 overs)**. Small n, but consistent with the total regressor never getting a Platt fit ("diagnostic only, well-calibrated") — an assessment made on pooled E+U OOF, which would hide a per-competition bias. Model-driven line selection would bet Over on nearly every EuroCup game at an inflated `P(Over)`.

**Still open**:
- **Ladder selection — ✅ BUILT (2026-09-28)**, with the A/B baked in. `ml_project/euroleague/euroleague_totals.py` is the single source of truth for both sides (same pattern as football's `FORM_WINDOWS`): `best_ev` evaluates P(Over) at **every** offered line and takes the highest-EV side — the operator's chosen strategy — while `counterfactual` records the **main-line** bet that was *not* placed. `predict_euroleague.py` emits both as `Bet *` / `Cf *` columns; `/euroleague/auto_wager` stakes the `best_ev` pick and stamps `counterfactual` on every O/U bet (verified: 23/23 on the 2026-09-29 slate). The scraper imports `main_line` from the same module, so scrape, display and stake cannot drift.

  **First slate already shows the risk the A/B exists to measure.** Where a book posts a deep ladder (bwin, 17-19 rows) the selection lands on a *tail* row every time — 6-9 points off the consensus line at 2.45-3.25 odds — and claims 2-20× the main line's EV: `Under 159.5 @2.75 EV +0.419` against main `Under 166.5 @1.82 EV +0.219`; `Over 169.5 @2.75 EV +0.589` against `Over 162.5 @1.82 EV +0.318`. EVs of +40-60% in a liquid market are not edge, they are the model's own distribution being wider than the market's, which is exactly the shape CLAUDE.md's "No measured edge" section documents (claimed EV > 0.50 → −25% realised). On EuroCup, where only 1-2 rows exist, pick == counterfactual and the two policies coincide. **Settle ~50-100 O/U bets, then compare `Bet *` against `Cf *` before trusting ladder-shopping.**
- Models are still the May fit — no retrain run, none needed until the season accrues games.

## How to update this doc

Same rules as `FOOTBALL_NEXT_STEPS.md` / `NBA_NEXT_STEPS.md`:
1. Flip rows to ✅ with a one-line summary when phases complete.
2. New work goes under "Open / deferred" unless it's actively queued.
3. Don't grow into a changelog — git history is the changelog. This file is the *forward-looking* Euroleague roadmap.

## Sources / references

**Selected:**
- [euroleague-api · PyPI](https://pypi.org/project/euroleague-api/) — Phase-1 history corpus
- [giasemidis/euroleague_api · GitHub](https://github.com/giasemidis/euroleague_api)
- [Kaggle: Euroleague & Eurocup Datasets](https://www.kaggle.com/datasets/babissamothrakis/euroleague-datasets) — optional corpus seed

**Probed and demoted:**
- [giasemidis/euroleague_odds · GitHub](https://github.com/giasemidis/euroleague_odds) — OddsPortal scraper, Selenium, ML-only, stale (2020–21). Optional backfill only, prefer `OddsHarvester` if pursued.

**Optional / backfill (if odds-aware features prove out):**
- [jordantete/OddsHarvester · GitHub](https://github.com/jordantete/OddsHarvester) — Playwright-based OddsPortal scraper, multi-market, maintained
- [gingeleski/odds-portal-scraper · GitHub](https://github.com/gingeleski/odds-portal-scraper)

**Reference / cross-check only:**
- [sakisbl/euroleague-scraper · GitHub](https://github.com/sakisbl/euroleague-scraper)
- [flavioleccese92/euroleaguer (R)](https://flavioleccese92.github.io/euroleaguer/) — endpoint discovery

**Rejected (paid or unsuitable):**
- [The Odds API — Historical](https://the-odds-api.com/historical-odds-data/) — paid tier; ruled out per operator constraint
- [API-Sports — Basketball](https://api-sports.io/documentation/basketball/v1) — 7-day rolling window only; useless for training
