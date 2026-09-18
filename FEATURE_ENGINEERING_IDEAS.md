# D2 — Feature Engineering Brainstorm

Living scratchpad for the D2 feature-engineering pass referenced in `FOOTBALL_NEXT_STEPS.md`.
Goal: get a candidate list down on paper, then pick 2–3 to implement and measure against the current model's OOF Brier per league.

Status: brainstorm. Nothing here is committed work.

## Ground rules

- Every candidate must be derivable from data we already have, OR have a concrete plan for where the new data comes from. No "we'd need a Premier League injury feed" hand-waves without an answer.
- Every candidate has to respect the no-leakage rule (use only data strictly before the match's kickoff). The existing `_add_ppg_strength_features` is the template — copy that shape.
- Acceptance per feature: ≥1% Brier improvement on TimeSeriesSplit OOF (the same harness `train_model.py` uses), measured per market (1X2 + O/U). Anything under 1% is noise on our sample sizes.
- Order of attack: cheapest data + biggest expected mechanism first.

## Current feature inventory (for reference)

So we don't re-invent what's already there. From `ml_project/feature_engineering.py`:

- **Market**: implied probs `IP_H/D/A` from Bet365 (falls back to AvgC/MaxC).
- **Strength**: `H_elo`, `A_elo`, `elo_diff`, `abs_elo_diff`.
- **Rolling form (last-5, all venues)**: pts, gf, ga, O/U rate, form-string, shots-for/against, corners-for/against — both sides.
- **Venue-specific form (last-5)**: home team's last-5 *at home*, away team's last-5 *away*.
- **Season-to-date**: `H_ppg`, `A_ppg`, `ppg_diff`, `abs_ppg_diff`, `H_att`, `A_att`, `H_def`, `A_def`, `att_def_diff` (all normalised vs league running average).
- **League**: `league_cat` categorical.

Everything below is additive to that.

## Tier 1 — Cheap (data already in `data_sets/MatchHistory/` CSVs)

Pure compute work on data we already ingest. Each is 1–2 days end-to-end including the OOF eval.

### 1.1 Variable rolling windows per league
**What**: replace the fixed `window=5` with a per-league window (e.g. EPL=5, Brazilian Serie A=8, Veikkausliiga=3).
**Why**: leagues with shorter seasons and higher variance need narrower windows to stay reactive; stable leagues benefit from more history. Currently we use the same window everywhere, which is a compromise that's optimal nowhere.
**Cost**: small. Add a league→window map (start by tuning empirically — fit Brier on a held-out split for window ∈ {3,5,7,10} per league, pick the argmin).
**Expected lift**: 1–3% Brier on small/volatile leagues; rounding error on EPL/Bundesliga.
**Risk**: if we tune the window on the same data we evaluate on, we'll over-fit. Need an outer CV layer.

### 1.2 Recency-weighted form (exponential decay) — ❌ TESTED & ROLLED BACK (2026-05-25)
**What**: replace `np.mean(last_5)` with an exponentially weighted mean — most recent game weighted heaviest. Half-life as a hyperparam (e.g. 3 games).
**Why**: a 5-0 win last week tells you more than a 5-0 win two months ago. The flat mean treats them identically.
**Cost**: trivial — one-line change in `_get_stats_from_history`.
**Expected lift (doc, pre-test)**: 1–2% Brier broadly; bigger on leagues where form fluctuates fast.
**Empirical result (post-test)**: **no measurable lift, globally or per-league**. Rolled back.

**Test setup (2026-05-25)**: built `_weighted_mean(values, half_life=3.0)` helper + 8 new training features (`H_form_{pts,gf,ga,ou}_w`, `A_form_*_w`). Half-life 3 games — most recent match weight 1.0, 3 matches back ½, 6 matches back ¼. Inference path re-used the same helper so train/serve produced identical values. Retrained the full pipeline; calibrators refit; calibration validate passed at 85.7% (full) / 89.2% (minimal) — same band as the pre-D2.1.2 baseline.

**Empirical findings**:
- **Global CV Brier (1X2)**: 5-fold mean 0.6011 (pre) → 0.6012 (post). Δ = +0.00008. Well within fold-to-fold noise (±0.003).
- **Per-league ablation** (`scripts/sweep_d212_per_league.py`, 21 leagues with n≥100, 11 685 matches):

| | count |
|---|---:|
| Leagues helped (Δ < −0.001) | **0** |
| Leagues hurt (Δ > +0.001) | **0** |
| Best per-league Δ | D1: −0.0006 (−0.10%) |
| Worst per-league Δ | F2: +0.0005 (+0.07%) |
| Weighted-mean Δ | **−0.0000 (−0.01%)** |

Every per-league delta is within ±0.0006 — noise. Even Serie B (I2, supposedly volatile) only showed Δ=−0.0003.

**Why D2.1.2 didn't help (best hypothesis)**: the model already has plenty of recency signal — ELO updates by goal margin (naturally recency-weighted); season-to-date PPG / attack / defense strength features carry recent trend; the flat L5 mean is already "recent" by construction. The weighted variants were collinear enough that XGBoost found no orthogonal signal to split on.

**Rollback (2026-05-25)**: removed `_weighted_mean`, removed `form_*_w` keys from `_get_stats_from_history`, removed 8 `_w` features from `train_model.common_features`, stripped `_w` wiring from `predict_matches.py`. The currently-deployed model (trained earlier the same day) still has the 8 `_w` columns in its `features_*.json`; predict_matches' existing defensive `if c not in input_df.columns: input_df[c] = 0` block handles them as zero-defaults until the next retrain produces a model trained without them. The ablation script `scripts/sweep_d212_per_league.py` is kept — it's a reusable tool for future feature-engineering decisions.

### 1.3 Opponent-adjusted form — ❌ TESTED & ROLLED BACK (2026-05-20)

**Verdict**: Implemented Flavours A (weighted form) + B (xPts residual) as 6 new features (3 per flavour incl. diffs). Trained + ablated with TimeSeriesSplit 5-fold CV against the production hyperparams. **Global Brier deltas were all within ±0.03%** for both 1X2 and O/U markets. **Per-league breakdown across 21 leagues** (n ≥ 100 OOF samples) found **zero leagues improving by ≥1% in any of the 3 variants × 2 markets = 6 column combinations**. Worst per-league delta was D2 OU at +0.26%; best was T1 1X2 at -0.15%. No consistent pattern across variants — clear noise.

**Why the null**: XGBoost already has `H_elo`, `A_elo`, `elo_diff`, `H_att`, `A_def`, and the form features as direct inputs. The "weight points by opponent strength" and "actual minus expected points" transformations are exactly the kind of interaction XGBoost discovers natively through tree splits. We hand-crafted features the model was already computing internally.

**Rolled back**: features removed from `feature_engineering.py`, `train_model.py`, `predict_matches.py`. Model retrained on the 42-feature baseline. **Kept**: the HFA fix in `elo_engine.py` (foundational correctness, not part of §1.3).

**Don't re-attempt** without a different mechanism. If we want to revisit opponent-adjustment, the next angle is **Flavour C (xGD-like)** which uses `H_att × A_def × league_avg_gf` — the goal-magnitude residual rather than the points residual — but only if there's a reason to believe XGBoost can't already discover this interaction either.

Original brainstorm preserved below for context.

**What**: each "form point" gets weighted by opponent strength. A win vs a top-3 team contributes more than a win vs a relegation candidate.
**Why**: "5 in a row vs bottom half" and "5 in a row vs top half" look identical to the model today. The H_att / A_def features capture *who's playing now* but not *who they beat to look this good*.
**Cost**: medium. Need to look up each opponent's running ELO (or PPG) as of the match date, then weight `form_pts` by `opp_elo / league_avg_elo`. Be careful with the no-leakage requirement — use the opponent's ELO *as of that historical match*, not their current ELO.
**Expected lift**: 2–4% Brier on 1X2. This is the most likely big winner of Tier 1.
**Risk**: complexity in the rolling computation. Easy to introduce subtle leakage.

#### 1.3.x Flavour breakdown
"Opponent-adjusted form" is a family of approaches, not one. Worth picking the flavour before writing code. All four use the same fact: `H_elo` and `A_elo` are already snapshotted on every historical row at pre-match time by `elo_engine.process_history`, so opponent strength at the time of each historical match is a free lookup with no leakage.

**Flavour A — Weighted rolling mean**
For each of the last-N matches, weight the contribution by opponent strength:
```
form_pts_w = sum(pts_i × w_i) / sum(w_i)
where w_i = opp_elo_i / league_avg_elo   (clipped to e.g. [0.5, 2.0])
```
- Simplest. Same input data, different aggregation.
- Pro: trivially understandable, easy to ablate.
- Con: a draw vs a top side still gets only 1 point of credit × a weight — doesn't capture "you held a much stronger team to a draw" properly.

**Flavour B — Expected-points residual (xPts)**
For each historical match, compute what was *expected* from the ELO diff at that point (with home-field bump), then store the residual:
```
exp_win = 1 / (1 + 10^(-(own_elo - opp_elo + HFA)/400))
exp_pts ≈ 3·exp_win + 1·(1 − exp_win)·draw_rate
form_xpts = mean(actual_pts − exp_pts)
```
A team with `form_xpts = +0.4` is consistently over-performing; much stronger signal than raw form because it controls for schedule.
- Pro: cleaner signal, matches how analysts already think ("xPts above expectation").
- Con: more moving parts (draw-rate assumption, HFA constant), one extra failure mode if ELO is wrong at that snapshot — see ELO audit below.

**Flavour C — Goal-based version (xGD-like)**
Same idea as B but for goals:
```
exp_gf = H_att × A_def × league_avg_gf   (we already compute these!)
form_gf_above_expectation = mean(actual_gf − exp_gf)
```
Targets the O/U model specifically. Pairs naturally with existing `H_att` / `A_def` features.
- Pro: feeds the O/U market, which is the weaker of our two markets.
- Con: noisier than xPts because goal counts have more variance per match.

**Flavour D — All three, as additive features**
Keep existing unweighted features unchanged, add A + B + C as *new* columns. Let XGBoost decide which signal to lean on.
- Pro: zero regression risk, full attribution at ablation time.
- Con: ~6–8 new features, mild collinearity, more retrain time.

#### 1.3.y Watch-outs that apply to all flavours
1. **Use opponent ELO at the time of that match** — `H_elo`/`A_elo` snapshotted on every row by the ELO engine. No leakage, no extra lookup.
2. **Cross-league matches** (e.g. Champions League games mixed into Premier League form). Decide whether to filter to same-league only or trust absolute ELO scale. Current `_calculate_rolling` doesn't filter, so we inherit this either way.
3. **Cold-start at season-open**: first 2–3 matches have unstable opponent-ELO snapshots. Cap or fall back to unweighted when N < 3.
4. **Normalisation midpoint**: 1500 is arbitrary. Better to use a per-league running mean ELO — but we don't compute one yet. Easy to add (same shape as `league_avg_gf`).
5. **ELO quality** — these features inherit any biases in the ELO computation itself. Audit findings below.

#### 1.3.z Recommended path
**Flavour D first, ablate after** — marginal cost of computing all three at once is small (one extra pass over the same rows in `_get_stats_from_history`), and we get clean attribution at ablation time. Drop the dead-weight features afterward. Alternative if we want maximum simplicity: **Flavour B alone** — xPts residual is the highest-signal single feature in the family.

### 1.4 Promoted-team indicator
**What**: binary flag `is_promoted` for teams in their first season of a league.
**Why**: promoted teams are systematically mispriced — both by bookmakers and by ELO (which carries no league-rank prior). The model usually treats them as "weak team in current league" once ELO catches up, but the first ~10 matches of the season are noisy.
**Cost**: small. Derive from history: a team is promoted into league X in season Y if its previous season was in a different league. Add the inference alongside data loading.
**Expected lift**: 2–4% Brier on the affected matches (a fraction of the dataset), so the global lift is more modest — maybe 0.5–1%.
**Risk**: same-name teams across divisions / promotions across confederations need careful handling. `entity_resolver.py` covers most.

### 1.5 Rest days / fixture congestion — ❌ TESTED & ROLLED BACK (2026-05-26)
**What**: `H_rest_days`, `A_rest_days` = days since each team's previous match. Also `rest_diff = H - A`.
**Why**: well-documented effect — three games in a week vs a fully-rested opponent matters. Currently invisible to the model.
**Cost**: small. Pure date arithmetic on existing match history.
**Expected lift (doc, pre-test)**: 1–2% Brier, concentrated on midweek fixtures and post-cup matchdays.
**Empirical result (post-test)**: **no measurable lift, anywhere** — including on the congested subsets the doc predicted it would help. Rolled back.

**Test setup**: `H_rest_days`/`A_rest_days`/`rest_diff` built in `_add_rest_days` (cap/sentinel 14), train + serve, added to `common_features`. Same per-league + subset OOF 1X2 Brier ablation used to kill §1.2.

**Findings** (5-fold OOF, 11 685 matches):
- Global weighted-mean Brier delta (with rest − without): **+0.0001 (+0.01%)** — flat, within noise. 1 league helped, 1 hurt, 19 neutral.
- **Subset test (the concentrated-effect hypothesis)** — even where rest *should* matter most:
  - `|rest_diff| ≥ 3` (n=1049): +0.0002
  - `|rest_diff| ≥ 4` (n=522): +0.0001
  - either team rest ≤ 3 / congested (n=1482): −0.0002
  - both teams rested (n=8180): +0.0001
  All within noise. The concentrated effect didn't show.

**Why it didn't help (strategic insight — applies to §1.2 too)**: the model's strongest features are the **bookmaker implied probabilities** (`IP_H/D/A` from B365 odds). The betting market already prices rest / congestion (it knows the fixture calendar), so feeding the model a raw version of a signal the odds already encode gives XGBoost nothing orthogonal. Two cheap features now rolled back (§1.2 recency form, §1.5 rest) both plausibly for this reason.
**Implication for the rest of D2**: candidates the market *also* efficiently prices (promoted-team, H2H, manager-change) are likely to come up similarly flat. The real headroom is either (a) signals the market prices *inefficiently* (rare/hard-to-quantify — injuries to specific key players, lineup leaks), or (b) dropping the odds-dependence so the model must learn fundamentals (a bigger architectural choice, overlaps with D3). Worth weighing before spending more days on Tier-1 date/form features.
**Risk**: low. Just be sure to skip the first matchday of each season cleanly (no prior match → use a sentinel like 14 days).

### 1.6 Head-to-head specifics (last 2–3 H2H) — ❌ TESTED, NO EDGE (2026-09-18)
**What**: features from the last meetings between these specific teams: H2H goals avg, H2H result, H2H total goals.
**Why**: rivalries / stylistic matchups produce repeatable patterns the global model misses ("these two always draw," "this fixture is always 3+ goals").
**Cost**: medium. Need to build a lookup of past meetings per (home_team, away_team) ordered pair; respect the date filter.
**Expected lift (doc, pre-test)**: 1–2% Brier. Smaller than opponent-adjusted form, but a free additive on top.
**Empirical result (post-test)**: **no edge over the market on either head.** The columns are built, mirrored at serve time and left in the pipeline, but they do not move the decisive metric.

**Sparsity was not the problem.** The doc feared "~1–2 H2H matches per team-pair per season". With `MatchHistory` running from 2010, **89.4%** of corpus rows have at least one prior meeting and **56.3%** have five, so `H2H_WINDOW = 5` was used rather than 2–3, and no league-wide fallback was needed. Averages are NaN (not 0) for a first-ever meeting, and `h2h_n` / `h2h_venue_n` carry the sample size so the model can discount a thin one.

**There is real univariate signal.** On rows with a full 5-meeting sample (n=105,119), sorting by the H2H rate is monotone in the outcome:

| `h2h_ou_rate` band | n | actual over 2.5 |
| --- | ---: | ---: |
| 0.0–0.2 | 4,131 | 45.0% |
| 0.2–0.4 | 17,247 | 47.1% |
| 0.4–0.6 | 31,025 | 49.5% |
| 0.6–0.8 | 30,827 | 52.1% |
| 0.8–1.0 | 21,889 | 55.1% |

`h2h_draw_rate` is monotone too, but much weaker (24.6% → 27.3% against a 25.9% base rate). So "this fixture is always 3+ goals" is a real effect; "these two always draw" barely registers.

**It is not incremental.** `scripts/experiment_h2h.py` runs three arms on one engineered frame (11,720 OOF rows on the picked side) and scores nats added on top of the devigged market price:

| head | arm | nats added | OOF Brier | vs base |
| --- | --- | ---: | ---: | ---: |
| 1X2 | base (59 feats) | +0.00002 | 0.60124 | — |
| 1X2 | **placebo** (68) | +0.00003 | 0.60141 | −0.029% |
| 1X2 | h2h (68) | +0.00000 | 0.60124 | −0.001% |
| O/U | base (61 feats) | −0.00000 | 0.24565 | — |
| O/U | **placebo** (70) | +0.00000 | 0.24573 | −0.036% |
| O/U | h2h (70) | +0.00000 | 0.24551 | +0.054% |

Against the doc's own ≥1% Brier bar: 1X2 came in at **−0.001%** and O/U at **+0.054%**. Both are noise.

**The placebo arm is the methodological point, and it should be reused.** The tuned booster runs `colsample_bytree=0.6`, so widening 59 → 68 columns changes which features every tree may choose from. That perturbation alone moves Brier by ~0.03% — the same order as the effect being measured. The placebo carries the nine columns with their values **permuted as a block across rows** (same width, same marginals, same NaN pattern, no link to the fixture). On 1X2 the placebo scored **+0.00003 nats against the real block's +0.00000** — the shuffled columns did better. Without that control, "h2h − base" would have been read as a small positive on O/U.

**Gain share is not evidence of information.** The model leans on these columns hard — 8.2% of total 1X2 gain (8 of 9 used, `h2h_pts` at rank #12) and 11.9% of O/U gain (`h2h_goals` at rank #6 of 70). But the **shuffled** block takes **8.9%** of 1X2 gain and **10.1%** of O/U gain, with `h2h_n` ranked #11. XGBoost spends splits on a high-cardinality numeric column whether or not it knows anything. **This retroactively weakens the gain-share evidence quoted for the L10/L15 form features** ("13 of 14 in the booster at 15.9% of gain", CLAUDE.md) — that number was never compared against a shuffled control, and this experiment suggests a control would have absorbed much of it.

**Matched retrain comparison (the "what would actually change" read)**: two models trained by the real `ModelTrainer` on the identical frame, differing only in the nine columns, then both run through `predict_matches` on the 2026-09-17 slate (12 matches):

| | base | with h2h |
| --- | ---: | ---: |
| CV accuracy, 1X2 | **0.5053** | 0.5040 |
| CV accuracy, O/U | 0.5568 | 0.5568 |
| 1X2 picks changed | — | **0 / 12** |
| O/U picks changed | — | **0 / 12** |
| mean \|Δ Home Win %\| | — | 0.0125 |
| mean \|Δ Over %\| | — | 0.0158 |

Not one pick moved. Note the first attempt at this comparison ran the h2h model against the **currently deployed** model and showed 1 pick changing on each market — that delta was entirely the deployed model's age, not the feature. Compare like with like.

**Why it didn't help — same mechanism as §1.2 and §1.5**: the price already contains it. A recurring 3+ goal fixture is visible to the market too, and the O/U line moves accordingly, so the model gains nothing orthogonal by re-deriving it from the same results. This is the third Tier-1 feature to die this way, and the pattern in the §1.5 note ("candidates the market *also* efficiently prices … are likely to come up similarly flat") now has H2H as a confirmed instance rather than a prediction.

**Also tested as a post-prediction calibrator, not just as a feature (2026-09-18).** Worth asking separately: the booster sees 65 columns through `colsample_bytree=0.6`, `max_depth=3` and `gamma=5`, so a weak-but-real signal could be drowned there and still be recoverable by a direct one-parameter fit on the finished probability. `scripts/experiment_h2h_calibrator.py` fits exactly that — unregularised logistic, coefficient per 1 SD of the h2h term, 5-fold cross-fitted so a calibrator is never graded on its own fitting rows:

| market | h2h term | on top of PRICE | on top of MODEL | on top of MODEL+PRICE |
| --- | --- | ---: | ---: | ---: |
| O/U 2.5 | `h2h_ou_rate` | +0.016 (0.020) **z +0.8** | +0.050 (0.020) **z +2.6** | +0.016 (0.020) **z +0.8** |
| O/U 2.5 | `h2h_goals` | +0.011 (0.020) z +0.5 | +0.054 (0.020) **z +2.7** | +0.011 (0.020) z +0.5 |
| 1X2 home | `h2h_pts` | +0.019 (0.023) z +0.8 | +0.034 (0.023) z +1.5 | +0.019 (0.023) z +0.8 |
| 1X2 draw | `h2h_draw_rate` | −0.024 (0.022) z −1.1 | −0.018 (0.022) z −0.8 | −0.024 (0.022) z −1.1 |

**The answer is a precise "yes, and it still doesn't help".** A post-hoc H2H correction genuinely does improve the model's own O/U probabilities — `h2h_ou_rate` at **z = +2.6** and `h2h_goals` at **z = +2.7** are real, not noise, which is the one place anything in this whole investigation cleared a significance bar. But on top of the *price* the same terms are flat (|z| ≤ 1.1), and the moment the price is in the equation the term collapses from z +2.6 to **z +0.8**. So the calibrator's entire contribution is pushing the model back toward the price it already had. Scale: the best case is **+0.00020 nats**, against vig worth roughly 0.02–0.05 — two to three orders of magnitude short.

This also rules out the more elaborate versions before they get built (per-league H2H calibrator, shrinkage toward a league base rate, an H7 rule in `heuristic_adjuster.py`). They are all restricted forms of the fit above, so none can beat a coefficient that is already indistinguishable from zero against the price. It is the same conclusion as the feature test, reached independently and by a route that could have disagreed.

**Rolled back from `common_features` (2026-09-18).** Unlike the L10/L15 precedent, the nine columns are *not* left in the trained feature list — they measured flat, they cost nine columns of `colsample_bytree` perturbation, and the placebo showed that perturbation is not free. `common_features` is back to **56**. What stays: `h2h_features()` + `_add_h2h_features` (still run by `prepare_data`, so the frame carries the columns), the serve-time mirror `get_h2h_stats`, `ModelTrainer._training_rows`, and both experiment scripts. Re-adding is one line in `train_model.common_features`, and both experiments read the columns off the frame whether or not they are trained on — so this stays re-testable at any time without rebuilding anything.

**Implementation notes**:
- `feature_engineering.h2h_features()` is the single source of truth, called by `FeatureEngineer._add_h2h_features` (training, one date-ordered pass accumulating per-pair history) and `predict_matches.MatchPredictor.get_h2h_stats` (serve time, per-fixture filter). Verified: 1,080 (fixture, column) values compared across the two paths, **0 mismatches**; and 400 rows × 9 columns against a brute-force recompute, **0 mismatches**.
- Rows sharing a date are scored as a block before any of them is folded into history, so same-day fixtures cannot see each other. Every first-ever meeting has `h2h_n == 0` and NaN averages (verified over all 19,777 of them).
- **`ModelTrainer._training_rows` was added for this.** All three heads used to `dropna` on the full feature list; with NaN-bearing H2H columns in it, that silently dropped **2,071 rows — 14.7% of the training set** — as the price of adding a feature, and biased what remained toward long-established pairings. The helper exempts the H2H block (XGBoost splits on NaN natively); `oof_predictions(dropna_on=...)` mirrors it so the experiment scores the rows production would actually train on.

**Artifacts**: `output/experiments/h2h_20260918_093710.{json,txt}` (feature test) and `output/experiments/h2h_calibrator_20260918_094842.{json,txt}` (calibrator test). Re-run with `python3 scripts/experiment_h2h.py` and `python3 scripts/experiment_h2h_calibrator.py`. The latter caches base-arm OOF predictions at `output/experiments/_oof_base.pkl`, which is a reusable starting point for any other post-hoc-adjustment question.

### 1.7 Goal-difference / scoring-trend features
**What**: rate of change of `form_gf` and `form_ga` — i.e. is the team trending up or down? Compute as `recent_3_avg − previous_5_avg`.
**Why**: captures a "heating up" / "cooling off" signal that the current heuristic adjuster has, but in feature space where the model can use it natively rather than as a post-hoc nudge.
**Cost**: small. Two extra rolling computations.
**Expected lift**: probably 0.5–1%. The heuristic already does some of this in `heuristic_adjuster.py`.
**Risk**: overlaps with the heuristic. If this works, the H4/H5 heuristic deltas should be reduced or removed (cleaner factoring).

## Tier 2 — Medium (we scrape it but don't store / use it for training)

Each is 2–5 days because we have to add a data column to the historical pipeline first.

### 2.1 Matchday number / season progress + stakes-awareness — 🟢 NEXT UP (2026-05-28)

**Why this is next**: §1.2 / §1.3 / §1.5 all tested flat — the consistent finding was *cheap features the market already prices don't move Brier*. §2.1 is a different category. **Motivational state** (relegation fight, European-qualification battle, dead-rubber coasting) is the kind of signal the market is known to price by *mean effect* but probably under-prices by *per-team variance*, especially in lower-tier leagues where bet volume is thinner. This is the "signals priced inefficiently" headroom the D2 verdict explicitly called out as the real lever (along with D4 player availability). Crucially, it does NOT overlap the B365 implied-prob features that defeated §1.2 / §1.5 — stakes are about future match urgency, not past form.

**Features — two layers:**

1. **Temporal context** (the original 2.1):
   - `matchday` (integer, 1..N for the league's season length).
   - `season_progress` = `matchday / total_matches_in_season` (0..1).
   - `is_endgame` (boolean — `season_progress > 0.80`). Start at a fraction, not a fixed number of games (Premier League last 5 of 38 ≈ 13%; Belgian Jupiler last 5 of 30 ≈ 17% — fraction normalises this).

2. **Stakes-awareness** (the new layer):
   - `H_relegation_threatened` / `A_relegation_threatened` — team currently in or within ≤K points of the relegation zone, gated by `is_endgame`. K starts at ~6 (≈2 wins).
   - `H_european_chasing` / `A_european_chasing` — team currently in or within ≤K points of the league's "European-spot cluster" (Champions League / Europa League / Conference League spots combined). Single flag in v1; could split into per-competition flags (`H_cl_chasing` / `H_el_chasing` / `H_ecl_chasing`) once we have league-by-season metadata for which positions earn which competition.
   - `H_dead_rubber` / `A_dead_rubber` — team mathematically out of relegation, out of European reach, AND no title hope. "Mid-table coasting" signal. Late-season only.
   - Optional quantitative companions: `H_pts_from_safety`, `H_pts_from_european_cluster` — let the model decide whether the boolean or the continuous form is sharper.

**Why stakes-awareness specifically (the directional argument)**: Relegation teams play with desperation in the last 5–8 matchdays — empirically more chances, more cards, more *variance*; not always more wins, but predictable *direction* of effort. Same for Champions-League-chasing teams in April/May. The market knows this and prices it on average, but the magnitudes are noisy (one team fights, another collapses) and *that noise is probably mispriced in lower leagues*. Lower leagues are also where the model lane currently has its best ROI — alignment of opportunity and edge.

**Cost — ~3–4 days, plumbing-heavy:**

1. **Historical standings reconstruction**. football-data.co.uk CSVs are per-match results only. Need a one-off script that derives `standings_at(team, league, date)` by cumulating prior results within each season — points, goal diff, position. Doable but careful with promoted/relegated rosters changing per season.
   - **This is the same plumbing §2.2 needs** — build it once, fold both features in at the same training cycle. Treat §2.1 and §2.2 as one D2 sub-pass, not two.
2. **League metadata table**: for each league × season, `top_N_european_spots` (how many positions earn European football) and `bottom_N_relegated` (how many drop). Small JSON. Some leagues' rules evolved (Conference League didn't exist before 2021); v1 can hardcode the current rules and bear the small training-time error on pre-2021 seasons.
3. **Feature derivation** in `feature_engineering.py`: combine reconstructed standings + matchday + league metadata into the boolean / quantitative flags.
4. **Train/serve parity check**: training-time standings come from the reconstruction; serving-time standings come from the live `standings_spider`. Must verify they produce the same features for the same `(team, league, matchday)` — same posture as the NBA Phase-2 serve-time-from-corpus parity check that killed the train/serve skew there.
5. **OOF Brier evaluation**: standard D2 gate. Two metrics matter: global Brier delta (probably small) AND `is_endgame`-conditional Brier delta (where the real lift should land). If the conditional delta is positive but the global is flat, ship it — the feature is doing its job exactly where stakes apply.

**Expected lift**: 1–3% Brier on `is_endgame` matches (~last 20% of each season per league). Modest globally because most matches are mid-season. Heavily concentrated in lower/mid-tier leagues where the market underprices the magnitude.

**Risk / watch-outs**:
- **Historical standings reconstruction has edge cases**: mid-season point deductions (rare but happens — Juventus 2006, Everton 2023), team withdrawals (Bury 2019), expansion seasons. Pre-validate the reconstruction against current standings JSON for the most recent season — if last-matchday reconstructed standings match the live standings, the cumulative logic is sound.
- **Train/serve skew is the single biggest risk**. Standings reconstruction and live `standings_spider` use different code paths; even with identical inputs they can disagree on edge cases. Add a one-shot diff check at deploy time.
- **League-spot rules drift**: top-4 CL became top-5 CL for England 2024-25 (because of coefficient rank); Conference League didn't exist before 2021. Either parameterise by season-range or live with a small training-time error on pre-2021 European-spot flags — both acceptable.
- **`is_endgame` threshold sensitivity**: 0.80 vs 0.85 vs 0.90 all defensible. Add it as a tunable, default 0.80, sweep on the OOF backtest before deploying.

### 2.2 Standings-derived features (table position, points gap)
**What**: `H_position`, `A_position`, `position_diff`, `points_gap_to_top`, `points_gap_to_safety`.
**Why**: ELO and PPG capture strength but not *standings context*. A 7th-vs-8th match has different stakes than 1st-vs-2nd even with similar PPG diffs.
**Cost**: medium. Standings JSON is current-only — we'd need to either reconstruct historical standings from results (doable but tedious) or limit this feature to inference-time only (asymmetric — train without it, predict with it: doesn't work).
**Expected lift**: 1–3% Brier. The "stakes" signal is real but hard to extract without genuine motivational state (which we don't have).
**Risk**: data plumbing is the bottleneck. The asymmetry above means we'd need to compute *historical* standings from `MatchHistory/` CSVs at training time — non-trivial, but a one-off script.

### 2.3 Cup-involvement / fixture-density indicator
**What**: flag for "team played a cup match midweek" — proxy for tired legs, rotated lineup.
**Why**: explains a lot of upset results that look random to a league-only model.
**Cost**: medium-high. football-data.co.uk CSVs are league-only; we'd need an extra data source (or cross-reference Flashscore for cup matches). May not be worth it pre-D3.
**Expected lift**: 1–2% Brier on the ~20% of matches with a midweek cup tie nearby.

## Tier 3 — Expensive (new data acquisition required)

These are 1–4 week projects on the *data* side before any modelling starts. Listed for completeness; not recommended until Tier 1 is exhausted.

- **Manager-change indicator**: well-documented signal but we have no manager-tenure feed. Would need to either scrape Wikipedia/Transfermarkt or hand-curate.
- **Lineup / availability**: key player out is highly predictive; lineup data is the single biggest "easy" feature we don't have. Sources exist (e.g. Sofascore, official club sites) but scraping is brittle.
- **Bookmaker line movement**: line drift in the 24h before kickoff carries signal. Requires us to start logging odds snapshots over time — a 6-month data wait before useful.
- **Public expected lineup sentiment**: media / news features. Adds noise as often as signal. Skip.

## Cross-cutting considerations

- **Interactions**: once we have 5+ new base features, consider explicit interaction terms (e.g. `elo_diff × season_progress`, `rest_diff × is_endgame`). XGBoost discovers interactions natively but engineered ones can help on small-data leagues.
- **Re-tune after**: any meaningful feature addition invalidates the D1 hyperparameter set somewhat. Plan to re-run `tune_model.py` after Tier 1 completes, not after every individual feature.
- **Re-calibrate after**: same applies to the per-league Platt calibrators. The C5 pipeline already chains fit + validate after retrain, so this is automatic — but the *first* run with new features should be eyeballed.
- **Backward compatibility**: existing models on disk use the current feature list (`models/features_*.json`). Adding features is a hard break — old models will fail to predict. Bump model file naming or accept that retrain is required.

## Proposed first cut

**Updated 2026-05-20** after §1.3 ablation came back null. Key lesson from that exercise: XGBoost is good at discovering interactions between features it already has. Hand-engineered transformations of existing inputs (weighting one feature by another, computing residuals against another) are unlikely to add signal. Future candidates should add information the model **cannot derive from current features**, not just re-express it.

Updated ranking:

1. **1.5 Rest days** — adds information the model genuinely doesn't have (fixture timing). Trivial cost.
2. **1.4 Promoted-team indicator** — also adds new information (squad-discontinuity signal). Cheap to derive.
3. **1.2 Recency-weighted form** — replaces an aggregation, doesn't add information. Lower expected value but trivial cost. Test second.
4. **1.1 Per-league window size** — same caveat as 1.2, plus a tuning loop. Lower priority.

**Skip permanently**:
- 1.3 (opponent-adjusted form) — tested, null. See section above.
- 1.6 (H2H) and 1.7 (trends) — overlap with existing features and heuristics. Same XGBoost-already-knows risk as 1.3.

## ELO audit (prerequisite for Flavours B / C of §1.3)

Triggered by the opponent-adjusted-form work: every flavour above leans on `H_elo`/`A_elo` snapshots being accurate. Auditing `ml_project/elo_engine.py` + the `train_model.py` integration turned up a mix of confirmed-correct and questionable-correct behaviour.

### Verified correct

- **Per-match snapshot is pre-match.** `process_history` records `H_elo`/`A_elo` *before* calling `update_ratings`. No leakage — when we look up opponent ELO in `_get_stats_from_history`, we get their rating as of that match. ✅
- **Global date sort.** Matches are processed in chronological order across all leagues. ✅
- **Predict-time ELO matches train-time semantics.** `data_sets/elo_ratings.json` holds the final state and is loaded at predict time, which is the correct "current pre-match" rating for tomorrow's fixtures. ✅
- **Goal-difference K-multiplier follows World Football Elo Ratings conventions** (G=1 / 1.5 / (11+N)/8). ✅
- **Buffer dating mitigates cold-start.** ELO is computed on full history from ~2010; training filter starts 2020-01-01. By then teams have ~30–50 matches of history, so initial-1500-noise has washed out for most teams. ✅

### Fixed (2026-05-20)

- **✅ Home-field advantage in `expected_result`.** `EloTracker.__init__` now takes `hfa=100`; `update_ratings` applies HFA to the home team's *effective* rating for the expected-result calculation (`expected_result(r_home + hfa, r_away)`). Stored `H_elo`/`A_elo` remain the intrinsic team ratings — no HFA baked in — so `elo_diff` semantics are unchanged downstream. Verified: home expected vs equal opponent now = 0.6401 (matches World Football Elo Ratings reference). Deployment requires a retrain (`./bin/retrain_pipeline.sh`) — current `H_elo`/`A_elo` snapshots in the trained model were computed without HFA.
- **✅ NaN match handling.** `process_history` now skips rows where `FTHG` or `FTAG` is NaN instead of silently treating them as home losses.

### Still open

- **No seasonal regression to the mean.** ELO carries forward unchanged across season boundaries. Standard practice is to pull ratings ~1/3 toward 1500 (or toward the league mean) between seasons because roster changes invalidate prior signal — promoted teams especially. Impact: end-of-season ELO is over-confident going into a new season; ~10–20 matches into the new season the system catches up.

- **Cross-league mixing without baseline.** All leagues share one pool. A 1500 Premier League team is rated identically to a 1500 Greek SL team. Mostly self-corrects via European competition results, but the lower-tier baseline is biased upward (their cross-league losses haven't propagated far). Probably not worth fixing — adding per-league offsets is a real project.

- **K-factor not validated.** K=20 is defensible (World Football Elo uses 20 for non-tournament league play) but never tuned for our data mix.

### Recommendation

HFA + NaN guard are in. Before §1.3 starts, run `./bin/retrain_pipeline.sh` so the model and calibrators are re-fit against the corrected ELO. Seasonal regression-to-mean and per-league offsets stay deferred — neither blocks D2.

## Open questions before starting

- Do we want each feature merged independently (so we can attribute Brier delta to each), or all three at once (faster, less attribution)? Independent is more rigorous but ~3× the dev time.
- Where does the OOF eval live? `train_model.py` has CV but no per-feature ablation harness. Might be worth writing a tiny `scripts/ablate_features.py` that loops a list of feature sets and prints Brier per league per market.
- What's the bar for "this feature stays"? Suggest: ≥1% global Brier improvement AND ≥0% on every major league (no league regresses meaningfully). Stricter than just "global wins" — avoids globally-useful-but-locally-harmful features.
