# Telegram daily football notifications

Scheduled GitHub Actions job that refreshes football data, runs tomorrow's
predictions and yesterday's verification, appends a git-tracked log, and
sends one Telegram message summarising both.

This path does **not** need the Flask dashboard, NBA/Euroleague, or
`real_betting/`. It is football-only batch automation.

## Repository secrets

Create these under **Settings → Secrets and variables → Actions**:

| Secret | Purpose |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | Bot API token from [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | Numeric chat ID of the user/group that should receive messages |

`GITHUB_TOKEN` is provided automatically by Actions and needs no setup. The
workflow uses it only to push the updated prediction log back to the repo.

### Getting a bot token

1. Open Telegram and message [@BotFather](https://t.me/BotFather).
2. Send `/newbot`, follow the prompts, and copy the token
   (`123456:ABC-DEF...`).
3. Paste it into the `TELEGRAM_BOT_TOKEN` repository secret.

### Getting a chat ID

1. Start a chat with your new bot (press Start), or add it to a group.
2. Send any message to the bot.
3. Open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser
   (replace `<TOKEN>` with your bot token).
4. Find `"chat":{"id": ...}` in the JSON — that number is your
   `TELEGRAM_CHAT_ID`. For groups it is often negative.

## Prediction / verification log

| Path | Format |
| --- | --- |
| `history/football/daily.jsonl` | One JSON object per match, UTF-8 JSONL |

Why here: `data_sets/`, `output/`, `models/`, and `logs/` are all gitignored.
The daily log must be committed (it is not re-derivable once a day has
passed), so it lives under a tracked top-level `history/football/` directory.

Each row carries prediction fields (`pred_1x2`, `conf_1x2`, `odds_1x2`,
`ev_1x2`, plus O/U equivalents) when written after `run_predictions.sh`, and
verification fields (`score`, `actual_1x2`, `hit_1x2`, …) once
`run_verification.sh` has produced `output/verification_<date>.csv`. Rows are
upserted by `(date, match_id)` so re-runs update in place.

Writers:

```bash
python3 scripts/append_prediction_log.py
python3 scripts/append_prediction_log.py --pred-date YYYY-MM-DD --verify-date YYYY-MM-DD
```

## Notifier

```bash
TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... python3 scripts/notify_telegram.py
python3 scripts/notify_telegram.py --dry-run   # print only, no API call
```

Uses `requests` (already in `requirements.txt`) against
`https://api.telegram.org/bot<TOKEN>/sendMessage`. Exits non-zero on API
failure. A day with zero fixtures sends a short "no matches" message instead
of crashing.

## Workflow

File: `.github/workflows/daily_football.yml`

- **Cron**: `0 6 * * *` (06:00 UTC) — after late European results settle,
  before the next evening slate matters.
- **Manual**: Actions → Daily Football → Run workflow (`workflow_dispatch`).

Order of steps: checkout → Python → cache pip / Playwright / `data_sets/` /
`models/` → venv + install → `playwright install` →
`./bin/setup_data.sh --sport football <season>` → train models if missing →
`./bin/run_predictions.sh` → `./bin/run_verification.sh` → append log →
Telegram → commit+push log (runs even if Telegram failed) → fail the job if
predict/notify/log failed.

### Caching strategy

| Cache | Key cadence | Why |
| --- | --- | --- |
| pip | `hashFiles('requirements.txt')` | Avoid re-downloading wheels |
| Playwright browsers | same hash | Chromium is large and independent of pip |
| `data_sets/` | ISO week (`2026-W38`) + `restore-keys` | Avoid a full cold scrape every day |
| `models/` | same week key | Models are gitignored; predictions need them |

**First run (empty cache):** `setup_data.sh` downloads the current season from
football-data.co.uk and scrapes Flashscore standings; if `models/` is empty
the job trains once. Slow, but populates both caches.

**Subsequent runs:** cache restore brings back last week's MatchHistory /
standings / models. `setup_data.sh --sport football <season>` still runs, but
it only re-downloads the current-season CSVs and re-runs the standings spider
— an incremental top-up, not a multi-season historical rebuild. Pass the
season code explicitly: without it the script prompts interactively and would
hang in CI.

`data_sets/` and `models/` stay gitignored — the cache is the only persistence
for them on Actions. Only `history/football/daily.jsonl` is committed.
