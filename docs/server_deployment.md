# Subir el proyecto a un servidor

Guía para cuando quieras dejar de depender del Mac local. Hoy el historial
de entrenamiento se puede volver a descargar; lo frágil son predicciones,
verificaciones, apuestas y bankrolls si solo viven en un disco.

Hay **dos caminos**. Empieza por el más barato si solo quieres batch +
Telegram; monta un VPS si además quieres la UI Flask 24/7 o scrapes
largos sin límites de Actions.

## Qué hay que preservar (checklist)

| Ruta | ¿En git? | En el servidor |
| --- | --- | --- |
| Código (`bin/`, `ml_project/`, …) | Sí | `git clone` / `git pull` |
| `history/football/daily.jsonl` | Sí (commiteado) | Viene del repo; el job diario lo amplía |
| `data_sets/MatchHistory/`, standings, elo… | No | Regenerar con `setup_data` o copiar desde local |
| `models/` | No | Copiar desde local **o** reentrenar una vez |
| `output/` (predictions, verification, bets) | No | Copiar si quieres continuidad de slips |
| `data_sets/betting_config.json` | No (template sí) | Copiar bankrolls vivos, o partir del template |
| `data_sets/team_mappings.json` | No (template sí) | Copiar o seed desde template |
| Secrets Telegram | No | Env vars o secrets de Actions |

El corpus de entrenamiento **no es irrecuperable**: football-data.co.uk +
`./bin/setup_data.sh --sport football <season>`. Lo que sí pierdes sin
backup es tu rastro de apuestas/P&L y los CSV diarios de `output/`.

## Opción A — Sin servidor: GitHub Actions (recomendado primero)

Ya existe `.github/workflows/daily_football.yml`: cron 06:00 UTC,
predict + verify + log + Telegram, y commit del log a git.

1. Push del repo a GitHub.
2. Secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (ver
   [`telegram_notifications.md`](telegram_notifications.md)).
3. Actions → **Daily Football** → Run workflow (prueba manual).
4. La primera ejecución (cache vacío) es lenta: descarga temporada y, si
   no hay modelos en cache, entrena. Después `data_sets/` y `models/`
   se restauran por semana vía `actions/cache`.

**Límites:** ~6 h de job, runners efímeros, Flashscore a veces caprichoso
desde IPs de datacenter. No sirve para el dashboard Flask ni para
auto-cashout en vivo.

Detalle: [`telegram_notifications.md`](telegram_notifications.md).

## Opción B — VPS Linux (Ubuntu 22.04/24.04)

Máquina pequeña (1–2 vCPU, 2–4 GB RAM) basta para fútbol batch + UI
opcional. Playwright/Chromium necesita algo de RAM; 2 GB es el mínimo
cómodo.

### 1. Dependencias del sistema

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git cron
# Chromium de Playwright trae libs; el instalador las pide:
#   playwright install --with-deps chromium
```

### 2. Clonar fuera de rutas “especiales”

En Linux no aplica el problema TCC de macOS (`~/Documents`), pero evita
directorios que el proceso limpie. Ejemplo:

```bash
mkdir -p ~/projects
cd ~/projects
git clone <URL-del-repo> sports_predictor
cd sports_predictor
```

### 3. Entorno Python

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install --with-deps chromium
```

### 4. Sembrar datos y modelos

**Desde cero (lento):**

```bash
./bin/setup_data.sh --sport football 2526   # ajusta el código de temporada
python3 ml_project/train_model.py           # si no copias models/
```

**Desde el Mac (más rápido, conserva bankrolls/apuestas):**

En el Mac (con el proyecto parado / UI parada):

```bash
# Empaquetar lo gitignored que importa
tar czf sports_state.tgz \
  data_sets/MatchHistory \
  data_sets/standings \
  data_sets/elo_ratings.json \
  data_sets/league_analytics.json \
  data_sets/betting_config.json \
  data_sets/team_mappings.json \
  data_sets/league_calibration.json \
  models \
  output
scp sports_state.tgz user@SERVIDOR:~/projects/sports_predictor/
```

En el servidor:

```bash
cd ~/projects/sports_predictor
tar xzf sports_state.tgz
# Si faltan templates → live:
python3 -m ml_project.config_bootstrap
```

Actualiza el código de temporada en `setup_data` cuando cambie el curso
(julio → nuevo código, p.ej. `2627`).

### 5. Cron (fútbol)

```bash
crontab -e
```

Ejemplo (horas en **hora del servidor**; preferible UTC):

```cron
SHELL=/bin/bash
PATH=/usr/bin:/bin
PROJECT=/home/USER/projects/sports_predictor

# Standings + predicciones de mañana — ~18:00 UTC (noche europea)
0 18 * * * cd $PROJECT && source venv/bin/activate && export PYTHONPATH=$PROJECT:$PROJECT/ml_project && ./bin/update_leagues_data.sh >> logs/cron_standings.log 2>&1 && ./bin/run_predictions.sh >> logs/cron_predict.log 2>&1

# Verificación de ayer — ~06:00 UTC (tras partidos europeos tardíos)
0 6 * * * cd $PROJECT && source venv/bin/activate && export PYTHONPATH=$PROJECT:$PROJECT/ml_project && ./bin/run_verification.sh >> logs/cron_verify.log 2>&1

# Retrain semanal — lunes 07:00 UTC
0 7 * * 1 cd $PROJECT && source venv/bin/activate && export PYTHONPATH=$PROJECT:$PROJECT/ml_project && ./bin/retrain_pipeline.sh >> logs/cron_retrain.log 2>&1

# Opcional: append log + Telegram (si no usas Actions)
# 30 6 * * * cd $PROJECT && source venv/bin/activate && export PYTHONPATH=... && \
#   export TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... && \
#   python3 scripts/append_prediction_log.py && \
#   python3 scripts/notify_telegram.py && \
#   git add history/football && git diff --staged --quiet || \
#   (git -c user.name='cron' -c user.email='cron@localhost' commit -m 'chore: daily log' && git push)
```

Crea `logs/` si no existe (`mkdir -p logs`). Los wrappers ya activan
`venv` cuando existe `venv/bin/activate`.

### 6. UI Flask (opcional)

```bash
./bin/manage_server.sh start   # puerto 5001
```

En un VPS expón solo con firewall + SSH tunnel o reverse proxy (nginx) +
HTTPS. **No** abras `5001` a internet sin autenticación: mueve bankrolls
virtuales y puede disparar flujos sensibles.

Para arranque al boot en Linux, un unit systemd simple que ejecute
`manage_server.sh start` basta; en macOS el proyecto usaba LaunchAgent
(ver nota abajo).

### 7. Telegram en el servidor

```bash
# En ~/.bashrc o un archivo env que cargue el cron:
export TELEGRAM_BOT_TOKEN='...'
export TELEGRAM_CHAT_ID='...'
```

Nunca commitees el token. Misma guía que
[`telegram_notifications.md`](telegram_notifications.md).

### 8. Actualizar código sin pisar estado

```bash
cd ~/projects/sports_predictor
git pull
source venv/bin/activate
pip install -r requirements.txt   # si cambió requirements
# No borres data_sets/, models/, output/
```

Si mueves el proyecto de ruta, **recrea el venv** (los shebangs de
`venv/bin/*` llevan rutas absolutas).

## macOS como “servidor” en casa (referencia)

Si en vez de un VPS dejas el Mac encendido:

- **No** pongas el repo bajo `~/Documents`, `~/Desktop` ni `~/Downloads`
  (TCC: launchd no puede leer archivos ahí sin Full Disk Access).
  Preferible `~/projects/sports_predictor`.
- Había un LaunchAgent
  (`~/Library/LaunchAgents/com.sportspredictor.webui.plist`) para la UI
  al login (`RunAtLoad`, sin `KeepAlive` agresivo).
- Tras mover el proyecto: recrear venv y actualizar rutas absolutas del
  plist.

Para batch diario en Mac, `cron` o `launchd` con los mismos comandos que
en la opción B sirven; Actions sigue siendo más simple si el Mac se
apaga.

## Qué NO hace falta en un servidor solo-batch

- Dashboard Flask
- NBA / Euroleague / `real_betting/`
- Auto-cashout / live snapshot (salvo que quieras ese flujo en la misma
  máquina)

## Resumen rápido

| Objetivo | Camino |
| --- | --- |
| Notificación diaria + log en git, sin máquina propia | **Opción A** (Actions) |
| Mismos scripts que en local, 24/7, UI opcional | **Opción B** (VPS + cron) |
| No perder apuestas/bankroll al migrar | Copiar `output/` + `betting_config.json` (+ mappings) |
| No perder capacidad de predecir | Copiar `models/` **o** reentrenar tras `setup_data` |

Cadencia una vez desplegado: la misma que en el README
(**Cadencia operativa**): diario standings → predict → verify; semanal
`retrain_pipeline.sh`.
