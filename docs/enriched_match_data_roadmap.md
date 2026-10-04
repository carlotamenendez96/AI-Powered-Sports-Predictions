# Roadmap: datos enriquecidos de partido (alineaciones, árbitro, tarjetas, córners)

Documento operativo para saber **qué hacer a continuación y por qué**.
Complementa [`FOOTBALL_NEXT_STEPS.md`](../FOOTBALL_NEXT_STEPS.md) (D4) y la
cadencia diaria del [README](../README.md). No sustituye el pipeline actual
1X2 / O/U: lo **amplía** con una capa de datos de partido que hoy no existe.

Idioma: español (guía de trabajo). Estado inicial: **2026-09-21**.

---

## 1. Objetivo

Conseguir que las predicciones (y más adelante mercados como **tarjetas** y
**córners**) se basen en información de partido real:

* quién juega / quién no (alineación prevista + bajas),
* qué árbitro pita y cómo pita (historico de tarjetas, expulsiones, ritmo),
* más adelante: corners, faltas, etc.

Sin esa capa, el modelo de equipo+cuotas **no puede** volverse “más preciso”
en el sentido tipster: le faltan las variables. Un LLM inventando narrativa
tampoco sirve: necesitamos **hechos estructurados** primero, prosa después.

### Principio de diseño

```
DATOS (scrapes + tablas)  →  FEATURES / AJUSTES  →  MERCADOS NUEVOS  →  TEXTO
     primero                    segundo              tercero           último
```

No entrenar XGBoost con lesiones hasta tener histórico fiable. Preferir
ajustes post-modelo o modelos **por mercado** cuando el target sea otro
(tarjetas ≠ 1X2).

---

## 2. ¿Tiene sentido un listado de árbitros + histórico?

**Sí.** Es una de las piezas más útiles para:

| Uso futuro | Por qué el árbitro importa |
| --- | --- |
| Amarillas / rojas | Ritmo de cartulina muy personal (unos pitan 2/partido, otros 6) |
| Justificaciones | “Sam Barrott: 6 amarillas en 2 Premier…” es un hecho, no opinión |
| Córners (secundario) | Menos directo; a veces correlaciona con ritmo/faltas, no es el driver principal |
| 1X2 | Efecto débil; no priorizar aquí el edge de ganador |

### Qué guardar (mínimo viable de catálogo)

Por **árbitro** (id estable + nombre + país/liga habitual):

* partidos pitados (fecha, liga, home, away, score),
* amarillas home/away / total,
* rojas,
* penaltis (si está en la fuente),
* opcional: faltas, córners del partido (si la fuente los trae en el mismo
  box-score).

Por **partido futuro** (slate diario):

* `match_id` → `referee_id` / nombre (asignación del día).

Almacenamiento sugerido (gitignored bajo `data_sets/` o `output/`, según si
es corpus o snapshot diario):

```
data_sets/referees/
  referees.json                 # catálogo id → nombre
  referee_matches.csv           # una fila por partido pitado (histórico)
output/referees_<date>.json     # asignación árbitro por match_id del día
```

Fuente candidata: Flashscore (página del partido / “match facts” suele
mostrar árbitro; el histórico hay que **acumular** partido a partido o
buscar un dump externo). No asumir un CSV mágico el día 1: el histórico se
**construye** scrapeando resultados pasados o guardando cada verificación.

---

## 3. Qué hay hoy vs qué falta

| Pieza | Estado | Notas |
| --- | --- | --- |
| Predicción 1X2 + O/U 2.5 | Producción | `run_predictions.sh` |
| Standings / form | Producción | `update_leagues_data.sh` |
| Justificación nivel 1 (probs/ELO/EV + bajas + árbitro) | Hecho (Pasos 2+5) | `justify_predictions.py` lee `availability_*.json` + `referees_*.json` + catálogo; contexto tipster, no input del pick |
| Bajas / “Will not play” Flashscore | **Cableado en `run_predictions.sh` (paso no fatal) — validando 1ª semana** | `scripts/d4_injuries/extract_availability.py` → `output/availability_<date>.json` |
| Importancia jugador (SoFIFA OVR) | Hecho pero D4 aparcado (OVR ≠ impacto) | `ml_project/availability/sofifa_importance.py` |
| Adjuster 1X2 por bajas | No hecho (shelved N3) | No priorizar hasta medir |
| Árbitro del partido | **Cableado en `run_predictions.sh` (paso no fatal)** | `scripts/d4_referees/extract_referees.py` → `output/referees_<date>.json` |
| Histórico árbitros (tarjetas…) | **Hecho (2026-09-21) — capa de datos** | `scripts/referees/build_referee_history.py` → `data_sets/referees/{referee_matches.csv,referees.json}`. Ver Paso 4. |
| Modelo / mercado tarjetas | **Hecho (2026-09-21) — pipeline aislado, serve ON** | Fase E / Paso 6. Head binario `P(HY+AY > 3.5)` en `ml_project/cards/`. Gate OOF pasó. Serve activo. Cuotas reales vía **Winamax** (`winamax_odds`, no Flashscore). **Siguiente precisión:** Paso 6b — bajas/sancionados (post §4.1), derbis, presión clasificatoria (info que el prior no ve); no más ventanas L15 de forma. |
| Señales contextuales Cards (6b) | **Plan (2026-09-21) — no implementado** | Bajas pivotes duros / sancionados; derbis; presión de tabla. Flags + experiment con placebo. |
| Modelo / mercado córners | No existe | Fase F — preferible tras al menos un experiment 6b PASS/FAIL |

Las bajas se **extraen** a diario (Paso 1) y se **citan** en la justificación
cuando son relevantes (Paso 2). El modelo 1X2/O/U **aún no** las usa en el pick
(adjuster N3 aparcado).

---

## 4. Pasos a seguir (orden obligatorio)

Cada paso tiene: **qué**, **por qué**, **entregable**, **criterio de listo**,
**siguiente**.

### Paso 0 — Congelar expectativas (ahora)

* **Qué:** Asumir que el 1X2 actual no se volverá mágicamente mejor solo con
  retrain semanal.
* **Por qué:** Ya medido: sin info nueva, el modelo reproduce la cuota.
* **Listo cuando:** Este documento leído; cadencia diaria+semanal del README
  en marcha.

### Paso 1 — Cablear bajas al flujo diario (Fase A)

* **Estado (2026-09-21):** cableado. Tras predicción (y NT),
  `bin/run_predictions.sh` llama al extractor con la misma `$DATE` (paso no
  fatal). Manual: `python3 scripts/d4_injuries/extract_availability.py YYYY-MM-DD`.
* **Qué:** Tras scrapear `matches_<date>.json` (lo hace `run_predictions` o
  el scrape previo), ejecutar:

  ```bash
  python3 scripts/d4_injuries/extract_availability.py YYYY-MM-DD
  ```

  Idealmente wrapper `bin/` o paso al final de `run_predictions.sh`
  (no fatal si falla).
* **Por qué:** Es el dato de disponibilidad que ya sabemos extraer; alimenta
  justificaciones y futuros adjusters/mercados.
* **Entregable:** `output/availability_<date>.json` cada día con partidos.
* **Listo cuando:** ≥1 semana de JSONs sin romper el pipeline; spot-check
  2–3 partidos a mano vs Flashscore. (Cableado hecho; falta acumular la semana.)
* **Siguiente:** Paso 2 (hecho) + **validar 1ª semana** (§4.1 abajo) antes de
  cualquier adjuster/modelo (Paso 8).

### Paso 2 — Meter bajas en la justificación (texto)

* **Estado (2026-09-21):** hecho en `scripts/justify_predictions.py`
  (`level1+availability-v1`). Lee `availability_<date>.json` por `match_id`;
  menciona lesión/sanción/dudoso (omite `inactive`). Deja claro que es
  **dato Flashscore**, no input del modelo.
* **Qué:** Extender `justify_predictions.py` para leer
  `availability_*.json` y añadir frases del tipo: “Bajas locales: …;
  visitantes: … (motivo)”.
* **Por qué:** Valor inmediato en UI/Telegram sin tocar el modelo.
* **Entregable:** Justificaciones que citen bajas reales cuando existan.
* **Listo cuando:** Columna Justification en la UI muestra bajas en partidos
  con availability. (Regenerar con `python3 scripts/justify_predictions.py --date …`.)
* **Siguiente:** Paso 3 (árbitro). En paralelo: validación §4.1.

### Paso 3 — Árbitro del día (Fase B, asignación)

* **Estado (2026-09-21):** cableado. Tras availability, `bin/run_predictions.sh`
  llama a `scripts/d4_referees/extract_referees.py` con la misma `$DATE` (paso
  no fatal). Selector Flashscore: `data-testid="wcl-summaryMatchInformation"`
  (fila Referee). Smoke: 14/14 (2026-09-20) y 7/7 (2026-09-21) con
  `referee_name` — 100% cobertura en esas slates. `referee_id` suele ser
  `null` (Flashscore no enlaza al árbitro hoy). Manual:
  `python3 scripts/d4_referees/extract_referees.py YYYY-MM-DD`.
* **Qué:** Scrape del árbitro asignado por `match_id` →
  `output/referees_<date>.json`.
* **Por qué:** Sin nombre/id del colegiado del partido no hay histórico que
  cruzar.
* **Entregable:** JSON `{match_id: {referee_name, referee_id?, source_url}}`
  (también `home_team`/`away_team`/`league`/`ts`/`referee_country`).
* **Listo cuando:** Cobertura alta en ligas objetivo (p.ej. >80% de la slate
  con árbitro). (Cumplido en smoke 2026-09-20/21; puede bajar en partidos
  futuros aún sin asignación publicada.)
* **Siguiente:** Paso 4.

### Paso 4 — Histórico de árbitros (Fase B–C, catálogo)

* **Estado (2026-09-21): hecho — capa de datos, sin modelo.**
  `scripts/referees/build_referee_history.py` (self-contained, no toca
  `ml_project` ni el modelo). Dos fuentes que conviven en el mismo CSV vía
  la columna `source`:

  **(A) Seed / backfill** — `--seed-from-matchhistory [--since YYYY-MM-DD]
  [--seasons N]`, desde `data_sets/MatchHistory/*.csv` (columnas
  football-data.co.uk: `Referee, HY, AY, HR, AR, HC, AC`). Idempotente
  (dedup por `match_key = mh:<liga>:<fecha>:<home>:<away>`, upsert
  last-write-wins) — re-ejecutar no duplica filas.

  **(B) Forward / append** — `--append-verification YYYY-MM-DD`, cableado
  como **paso 7, no fatal**, en `bin/run_verification.sh` (justo después de
  resolver apuestas). Cruza tres ficheros del mismo día:
  `output/referees_<date>.json` (D4 Paso 3 — quién pitó), `output/results_<date>.json`
  (marcador) y `output/predictions_<date>.csv` (liga/equipos canónicos, porque
  `results_<date>.json` en modo verificación por ID trae `home_team`/`away_team`
  poco fiables — a veces ambos iguales al equipo local). Idempotente
  (`match_key = fs:<match_id>`). **Si Paso 3 no corrió ese día, o el
  árbitro salió `null`, o no hay `results_<date>.json` todavía, se salta y
  lo dice — nunca rompe la verificación** (`|| true` en el wrapper + manejo
  interno que retorna 0).

* **Limitación honesta — las filas forward NO llevan tarjetas.** Flashscore
  en modo verificación (`mode=verification`, el que usa `run_verification.sh`)
  solo expone el marcador final, no el box score. Las filas `source=verification`
  tienen `hy/ay/hr/ar/hc/ac` vacíos; solo sirven para saber **quién pitó
  qué** (útil para "¿cuántos partidos lleva pitados X esta temporada?"),
  no para tasas de tarjetas. Las tasas de tarjetas solo existen donde hay
  backfill de MatchHistory (ver cobertura abajo).

* **Cobertura real de MatchHistory (`--coverage`, 2026-09-21):** de 38
  ficheros, **9/38 tienen columna `Referee`** — únicamente ENG (Premier
  League, Championship, League 1, League 2, Conference) y SCO (Premier
  League, Division 1-3). El resto de "main leagues" con sufijo de temporada
  (ESP, FRA, GER, ITA, NED, BEL, POR, TUR, GR) **no** traen `Referee` en la
  fuente — no es un bug del downloader, football-data.co.uk simplemente no
  la publica para esas ligas. Las "extra leagues" consolidadas desde `/new/`
  (ARG, AUT, BRA, CHN, DEN, FIN, IRL, JPN, MEX, NOR, POL, ROU, RUS, SUI,
  SWE, USA) tampoco. Para esas ligas **no hay backfill posible**: el
  histórico de esos árbitros solo puede crecer hacia delante, y sin
  tarjetas (ver limitación arriba).

  Solo hay **tres temporadas (23-24 / 24-25 / 25-26)** de los ficheros con
  `Referee` en disco tras el seed 2026-09-21: **10.085 filas**, **245**
  árbitros. Las tasas de tarjetas siguen limitadas a ENG/SCO.

* **Investigación — cómo cubrir ESP / FRA / ITA / GER / etc. (2026-09-21):**

  | Fuente | Árbitro | HY/AY por partido | ¿Backfill Big-5? | Coste / riesgo |
  | --- | --- | --- | --- | --- |
  | football-data.co.uk CSV | Solo ENG(+SCO) | Sí en Big-5 | **No** (columna `Referee` ausente en SP1/I1/F1/D1 incluso 15/16→25/26; verificado en el zip crudo) | Ya integrado |
  | Flashscore (nuestro spider) | Sí (Paso 3) | Sí en live/stats; **no** en `mode=verification` hoy | Solo forward | Playwright; ampliar verification a stats |
  | **Sofascore API no oficial** | **Sí** (`event.referee`) | **Sí** (`/event/{id}/statistics` → Yellow/Red cards) | **Sí** (paginar `/unique-tournament/{id}/season/{sid}/events/last/{page}`) | `curl_cffi` (TLS); ToS/WAF; frágil |
  | API-Football (api-sports.io) | Sí en fixture | Sí vía `/fixtures/events` o statistics | Sí (pagado) | Cuota $; estable; lo usa p.ej. RefOdds |

  **Probe Sofascore PASS (2026-09-21, `curl_cffi` impersonate chrome120):**
  - LaLiga `uniqueTournament=8`, Serie A `23`, Ligue 1 `34`, Bundesliga `35`
  - Ejemplo LaLiga Elche–Real Sociedad: árbitro *Miguel Angel Ortiz Arias*
    (id 786436, career 700 amarillas / 156 partidos) + stats partido HY=4 AY=1
  - Perfil `/referee/{id}` trae agregados de carrera (útil como prior rápido;
    para el modelo preferimos **filas partido a partido** con date-cut, igual
    que ENG/SCO)

  **Vía recomendada (orden):**

  1. **Backfill Sofascore → `referee_matches.csv`** (`source=sofascore`) —
     **implementado** `scripts/referees/seed_sofascore.py` (2026-09-21).
     Big-5: `--all-big5 --years 23/24,24/25,25/26`. Schema actual; idempotente
     `match_key=ss:<event_id>`. Requiere `curl_cffi` (en `requirements.txt`).
     Nombres Sofascore → slug estilo Flashscore (`Surname I.`) best-effort.
  2. **Alternativa limpia:** API-Football de pago si Sofascore se pone
     hostil — mismo schema, `source=api_football`.
  3. **Forward gratis (complemento):** en `mode=verification`, además del
     marcador, scrapear panel stats (el spider live ya sabe
     `extractStat('Yellow cards')`) y rellenar `hy/ay` en
     `--append-verification`. Cierra el agujero “forward sin tarjetas” para
     **todas** las ligas del slate diario, no solo Big-5.

  **Join alternativo (si solo quisiéramos árbitro):** MatchHistory ya tiene
  HY/AY en ESP/FRA/ITA — bastaría asignar `Referee` por fuzzy
  fecha+equipos desde Sofascore. Preferible scrapear ambos del mismo evento
  Sofascore para no desalinear conteos de tarjetas.

  **Fuera de alcance inmediato:** Transfermarkt (ToS-hostile), FBref (sin
  feed de árbitro fiable), depender del marketing de football-data (“referees
  for major leagues”) — **no se cumple en los CSV**.

* **Cómo sembrar 3 temporadas (ENG/SCO):**

  ```bash
  ./bin/setup_data.sh 2324   # descarga la temporada 23-24 (solo main leagues)
  ./bin/setup_data.sh 2425   # 24-25
  ./bin/setup_data.sh 2526   # 25-26 (probablemente ya en disco)
  python3 scripts/referees/build_referee_history.py --seed-from-matchhistory
  ```

  Cada temporada llega en un fichero `<Liga>_AA-BB.csv` distinto (no se
  pisan), así que el seed las recoge todas automáticamente. `--seasons N`
  permite acotar a las N temporadas más recientes **por liga** si en algún
  momento hay más de 3 en disco; `--since YYYY-MM-DD` filtra por fecha en
  vez de por fichero.

* **Esquema** (`data_sets/referees/`, gitignored — templates
  `referee_matches.template.csv` / `referees.template.json` trackeados
  para el esquema, mismo patrón que `betting_config.template.json`):
  - `referee_matches.csv` — una fila por partido pitado: `match_key, date,
    league, home, away, score_home, score_away, referee_name, referee_id,
    hy, ay, hr, ar, hc, ac, source, ingested_at`.
  - `referees.json` — índice derivado (se regenera entero desde el CSV en
    cada run, no es una segunda fuente de verdad): `id/slug → {canonical_name,
    aliases, n_matches, sources}`. El slug normaliza "Initial Surname"
    (MatchHistory, `"A Taylor"`) y "Surname Initial." (Flashscore,
    `"Letexier F."`) al mismo id cuando coinciden — best-effort, no es
    resolución de entidades completa (dos árbitros con mismo apellido +
    inicial colisionan).
* **Por qué:** Un listado de nombres sin números no sirve; el valor está en
  **tasas** (amarillas/partido, % partidos con roja, córners/partido).
* **Entregable:** Catálogo + consulta local (`--stats NOMBRE`). El gate de
  "≥ N partidos/árbitro" (p.ej. 20) antes de usar en tipster/modelo sigue
  pendiente para Paso 5 — hoy solo hay una temporada, así que muchos
  árbitros de ligas menores están por debajo de ese umbral.
* **Listo cuando:** Puedes responder “¿cuántas amarillas/partido lleva X
  esta temporada?” con query local. **Cumplido para árbitros ENG/SCO
  frecuentes** (ver `E Duckworth` arriba); no cumplido (ni cumplible sin
  otra fuente) para el resto de ligas.
* **Siguiente:** Paso 5.


### Paso 5 — Justificación + árbitro (texto)

* **Estado (2026-09-21): hecho.** `scripts/justify_predictions.py`
  (`level1+availability+referee-v1`) lee:
  - `output/referees_<date>.json` (Paso 3) → nombre (+ país si viene),
  - `data_sets/referees/referee_matches.csv` (Paso 4) → tasas solo si el
    árbitro tiene **≥ 20** partidos con columnas de tarjetas en el catálogo.
* **Comportamiento (no inventa cifras):**
  - Con tasas: *«Árbitro: Taylor A. (Eng). Histórico local: 3.9 amarillas/partido ·
    6% con ≥1 roja · 9.8 córners/partido (n=31)…»*
  - Sin tasas (casi toda la slate diaria fuera de ENG/SCO hoy): *«Árbitro: …
    Sin tasas / histórico insuficiente… Solo nombre Flashscore»*
  - Sin `referees_<date>.json` o `referee_name=null`: no añade bloque.
* **Qué NO hace:** no toca el pick 1X2/O/U ni `predict_matches.py`. El texto
  deja claro que es contexto tipster.
* **Cómo regenerar:**
  ```bash
  python3 scripts/justify_predictions.py --date YYYY-MM-DD --print
  ```
  (requiere haber corrido predicciones + `extract_referees` ese día; el
  catálogo ya está seedeado desde Paso 4).
* **Listo cuando:** UI/Telegram muestran bloque árbitro sin inventar cifras.
  Cumplido en JSON/TXT; la columna Justification de la UI lo lee del mismo
  fichero que antes.
* **Siguiente:** Paso 6 hecho (2026-09-21). Mientras: validar §4.1 (bajas)
  y opcionalmente sembrar más temporadas ENG/SCO.

### Paso 6 — Mercado tarjetas (Fase E) — solo después de 3–4

* **Estado (2026-09-21): hecho — pipeline aislado; gate OOF PASS; serve OFF.**
  Producto primario fijado: **binario `P(total amarillas HY+AY > 3.5)`**
  (`CARD_LINE = 3.5` en `ml_project/cards/constants.py`). Secundario/diagnóstico:
  cabeza Poisson sobre `total_yellows` (`cards_total` en el registry) — no
  sustituye al binario. **Fuera de alcance v1:** amarillas 1ª parte, player
  props, mercado de rojas, córners (Paso 7), adjuster 1X2 (Paso 8).

* **Aislamiento (ley del proyecto):** tarjetas ≠ 1X2. Nuevo target → nuevo
  pipeline. **No** se meten features de amarillas en el XGBoost 1X2/O/U de
  producción, ni se tocan lanes / bankrolls / `use_league_calibration`.

* **Cobertura MatchHistory** (`python3 scripts/audit_cards_coverage.py`,
  2026-09-21): **44/60** ficheros con `HY`/`AY` → **15.443** filas; **22**
  ligas. Las 16 "extra leagues" (`/new/`: ARG, AUT, BRA, CHN, DEN, FIN, IRL,
  JPN, MEX, NOR, POL, ROU, RUS, SUI, SWE, USA) **no** traen columnas de
  tarjetas — no entrenables. Solo ENG+SCO tienen además `Referee` histórico.
  **No hay cuotas de tarjetas** en el corpus → evaluación solo con métricas
  de probabilidad (sin EV/ROI inventado).

  | Liga | filas | mean(HY+AY) | P(>3.5) | Referee histórico |
  | --- | ---: | ---: | ---: | --- |
  | ENG-Championship | 1104 | 3.83 | 53.5% | sí |
  | ENG-Conference | 1104 | 3.73 | 52.1% | sí |
  | ENG-League 1 | 1104 | 3.82 | 54.2% | sí |
  | ENG-League 2 | 1104 | 3.82 | 53.9% | sí |
  | ESP-Segunda | 924 | 4.95 | 73.7% | no |
  | ENG-Premier League | 760 | 3.96 | 58.2% | sí |
  | ESP-La Liga | 760 | 4.59 | 64.3% | no |
  | ITA-Serie B | 760 | 4.81 | 72.5% | no |
  | ITA-Serie A | 760 | 4.01 | 58.0% | no |
  | TUR-Ligi 1 | 685 | 4.51 | 62.8% | no |
  | FRA-Ligue 2 | 684 | 3.74 | 50.6% | no |
  | BEL-Jupiler League | 623 | 3.83 | 53.9% | no |
  | GER / FRA / NED / POR / GER2 | ~612 c/u | 3.1–5.0 | 37–72% | no |
  | GR-Super League | 475 | 5.28 | 76.4% | no |
  | SCO (Prem + Div 1–3) | 1536 | ~3.8–3.9 | ~53–56% | sí |

* **Layout:**
  - `ml_project/cards/` — `constants`, `data_loader`, `feature_engineering`,
    `train_cards`, `calibration`, `predict_cards`, `config` (lectura aislada
    de `sports.football.cards`, **no** pasa por `sports_config`/LANES).
  - `models/xgb_model_cards.json` + `features_cards.json` + `model_meta_cards.json`
  - `data_sets/cards_calibration.json` (gitignored) + template trackeado
  - `output/predictions_cards_<date>.csv`, `output/experiments/cards_<ts>.*`
  - Registry: mercados nuevos `cards` / `cards_total` en
    [`model_registry.py`](../ml_project/model_registry.py)

* **Features (leakage-free):** forma de tarjetas L5/L10 por equipo (+ venue
  home/away L5), tasas de roja, proxy combinado, prior expanding de liga,
  bloque árbitro con gate `n≥20` partidos **previos** con hy/ay (si no →
  NaN + `missing_ref=1`), `league_cat`, ELO diff. Disponibilidad/bajas:
  **no** en train (sin backfill); flag `use_availability_at_serve=false`.
  Paridad train/serve verificada: 40 samples, **0 mismatches**.

* **Calibración:** un solo Platt **global** sobre P(over) (no per-liga —
  lección del 1X2). Guards `MIN_PLATT_SLOPE` + accuracy/AUC. Fit 2026-09-21
  **aceptado** (`a=0.66`, ΔBrier −0.0018).

* **Gate OOF** (`scripts/experiment_cards.py`, n=12.660, 22 ligas):

  | arm | Brier | logloss |
  | --- | ---: | ---: |
  | league_mean (mejor baseline) | 0.2383 | 0.6694 |
  | referee_mean | 0.2385 | 0.6698 |
  | model_raw | 0.2386 | 0.6702 |
  | **model_cal** | **0.2368** | **0.6659** |
  | placebo (form+ref shuffled) | 0.2575 | 0.7123 |

  - `model_cal − league_mean` = **−0.0016** CI95 **[−0.0028, −0.0003]**
    (excluye cero)
  - placebo **no** bate al modelo real
  - **PASS=True.** Mejora pequeña pero real frente al prior de liga; el
    placebo demuestra que el bloque de features aporta (no es ruido de
    ancho). **Sin claim de edge/ROI** — no hay cuotas de tarjetas.

* **Comandos:**

  ```bash
  source venv/bin/activate
  export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ml_project

  python3 scripts/audit_cards_coverage.py          # cobertura
  python3 -m ml_project.cards.train_cards          # train + OOF
  python3 -m ml_project.cards.calibration          # Platt global
  python3 scripts/experiment_cards.py              # gate (baselines+placebo)
  python3 -m ml_project.cards.predict_cards YYYY-MM-DD --force   # serve smoke
  CARDS_ENABLED=1 ./bin/run_predictions.sh         # hook diario (no fatal)
  ```

* **Serve:** solo emite filas cuando la liga está en el universo de train y
  hay prior de liga + forma L5 de ambos equipos. El resto se **salta** (log),
  nunca se rellena. Hook en `run_predictions.sh` **ON** por defecto
  (`CARDS_ENABLED=0` para desactivar). `justify_predictions` añade una línea factual solo si existe
  `predictions_cards_<date>.csv` y se **fusiona** en `predictions_<date>.csv`
  (cluster Cards junto a 1X2/O/U; `auto_wager` lee el mismo CSV). Cuotas reales
  cuando el scrape las rellene (`Over Cards Odd` / `Under Cards Odd`);
  `allow_synthetic_fallback=false`. **Fuente de cuotas (2026-09-21):** Flashscore /
  BetExplorer / OddsPortal **no publican** Number of Cards O/U; Pamestoixima
  headless → Akamai *Access Denied*. **Winamax.es** sí: `PRELOADED_STATE` por
  HTTPS (`betType` 2603, `specialBetValue=total=3.5`) — sin Playwright.
  `python3 -m ml_project.cards.winamax_odds <date>` (hook en `run_predictions.sh`)
  enlaza por nombre+fecha y escribe `over_cards_3_5` / `under_cards_3_5` +
  `cards_odds_source=winamax`. Cobertura liga-dependiente (LATAM / divisiones
  bajas a menudo; Big-5 lejanos a veces sin mercado). Nota: Winamax puntúa
  amarilla=1 / roja=2; el modelo predice HY+AY.
  Liquidación solo con HY+AY; si no, OPEN.

* **Qué aún no puede funcionar:** ligas extra sin HY/AY; tasas de árbitro
  fuera de ENG/SCO (catálogo forward sin box-score); features de
  suspensión/bajas en train. Sembrar más temporadas ENG/SCO ayuda al bloque
  de árbitro, no al resto.

* **Listo cuando:** gate PASS + serve smoke sin crash + docs. **Cumplido
  2026-09-21.** Cuotas reales: plumbing listo (spider + CSV + auto_wager);
  Flashscore odds-comparison **no** publica Number of Cards O/U (probe FAIL);
  fuente **Winamax** locked (probe PASS 5/40, 2026-09-21); cobertura parcial
  por liga. No claim de edge Cards sin settled + Spearman EV.

* **Siguiente — precisión Cards (prioridad explícita, 2026-09-21):** el gancho
  OOF sobre el prior de liga es pequeño (−0.0016 Brier). Lo que falta no es
  más forma L5→L15 (misma lección que 1X2: el mercado/prior ya ve resultados
  pasados), sino **información que el prior no ve**:

  1. **Bajas / sancionados con impacto en tarjetas** — pivotes duros,
     mediocentros de falta, jugadores a 1 amarilla de sanción, ya
     suspendidos. Fuente: `availability_<date>.json` (`reason_class` ∈
     `{suspension, injury, …}`). **Solo después** del gate §4.1 (calidad del
     scrape). En Cards el canal natural es feature/adjuster **del head de
     tarjetas** (no el adjuster 1X2 Paso 8). Importancia ≠ SoFIFA OVR (ya
     rechazado en D4); hace falta proxy de “perfil de tarjeta” (faltas /
     amarillas/partido del ausente), no rating de calidad.
  2. **Derbis / rivalidades** — flag por pareja o liga (H2H local, misma
     ciudad, clasico-list). Señal de intensidad arbitral / falta que el
     rolling form no captura en un solo partido.
  3. **Presión clasificatoria** — contexto de jornada: pelea por título /
     Europa / descenso / nada en juego (derivable de standings + jornada
     restante). Motiva agresividad o “partido muerto”.

  Gates de aceptación (mismo espíritu que `experiment_cards.py`):
  - A/B OOF o forward con **placebo** de mismo ancho (no confiar en gain-share).
  - Brier/logloss vs `league_mean` + brazo sin la feature nueva.
  - **No** alargar ventanas de forma de tarjetas como primer lever.
  - En paralelo (no sustituye lo anterior): alinear target↔cuota Winamax
    (booking points vs HY+AY) y acumular árbitro forward fuera de ENG/SCO.

  Orden sugerido tras §4.1 abierto: (1) suspensiones/pivotes → (2) derbis →
  (3) presión de tabla. Paso 7 (córners) puede esperar a que (1)–(2) tengan
  al menos un experiment PASS/FAIL documentado.

### Paso 6b — Señales contextuales Cards (plan; no implementado)

* **Qué:** Features / adjuster capped solo en `ml_project/cards/`, detrás de
  flags (`use_availability_at_serve`, futuros `use_derby`, `use_table_pressure`).
* **Por qué:** Única vía razonable a precisión incremental una vez el head
  ya bate al prior por poco; el gap es información de contexto del partido,
  no resolución de form.
* **Precondición:** gate §4.1 para el brazo de bajas; derbis/presión pueden
  prototiparse antes (datos ya en standings / calendar) pero con el mismo
  gate experimental.
* **Listo cuando:** experiment Cards con placebo PASS y, si hay cuotas,
  Spearman EV en settled no peor que el brazo base (idealmente mejor).

### Paso 7 — Mercado córners (Fase F)

* **Qué:** Otro target (córners equipo / over 9.5 / córner temprano…).
  Features: estilo de equipo (presión, centros), cuotas corners si hay,
  árbitro solo como señal débil.
* **Por qué:** Misma lógica: mercado distinto = pipeline distinto.
* **Listo cuando:** Misma barra de evaluación que tarjetas.

### Paso 8 (opcional) — Adjuster 1X2 por bajas

* **Qué:** Post-modelo capped, flag off, log-only primero (diseño D4 N3).
* **Por qué:** Puede ayudar en casos; el OVR SoFIFA ya se descartó como
  peso de importancia — hace falta otra métrica o pesos manuales por
  `reason_class` sin star-rating falso.
* **Precondición:** pasar el gate de §4.1 (1ª semana validada). Sin eso,
  no hay adjuster.
* **Listo cuando:** Forward Brier con adjuster ≤ sin adjuster (o mejora
  clara). Si no, dejar flag off.

---

## 4.1 Gate — Validar la 1ª semana de bajas (antes de meterlas en el modelo)

**Objetivo:** demostrar que `availability_<date>.json` es fiable y estable
como *dato*, no que mejore el 1X2. Solo después tiene sentido un adjuster
(Paso 8) o features de entrenamiento.

**Qué NO es esta validación:** no mide edge, ROI ni Brier del pick. El modelo
sigue igual. Aquí solo se valida la **calidad del scrape**.

### Duración y muestra

* **≥ 7 días consecutivos** con `run_predictions` (o extractor manual) y
  fichero no vacío.
* Preferible cubrir **varias ligas** (no solo una jornada de una sola liga).
* Anotar fechas en una mini-tabla (abajo) o en notas personales.

### A) Chequeos diarios (automáticos / 2 minutos)

Tras cada `./bin/run_predictions.sh` (o al día):

1. **Existe el fichero**  
   `ls -l output/availability_YYYY-MM-DD.json`
2. **JSON válido y no vacío**
   ```bash
   python3 -c "
   import json,sys
   d=json.load(open(sys.argv[1]))
   assert isinstance(d,dict) and len(d)>0
   print(len(d),'matches', sum(len(v.get('home',[]))+len(v.get('away',[])) for v in d.values()),'absentees')
   " output/availability_YYYY-MM-DD.json
   ```
3. **Cobertura vs slate** — casi todos los `match_id` de
   `matches_<date>.json` (con `base_url`) aparecen en availability.
   *Umbral orientativo:* ≥ **85%** de partidos con entrada (aunque
   `home`/`away` estén vacíos = “nadie en Will not play”, eso es OK).
4. **Pipeline no fatal** — si el extractor falla, `predictions_*.csv` igual
   se escribió y el exit del wrapper no debe tumbar el día por bajas.
5. **Justificación regenerada** (opcional pero útil):
   `python3 scripts/justify_predictions.py --date YYYY-MM-DD`  
   Comprobar en UI que las frases de bajas coinciden con el JSON.

Si un día falla el scrape (0 matches, DOM roto, Playwright): **anotar la
fecha y la causa**; no cuenta como día “verde” de la semana.

### B) Spot-check manual vs Flashscore (calidad del parse)

**2–3 partidos por día**, o al menos **10–15 en la semana**, eligiendo:

* 1 con muchas bajas, 1 con pocas/ninguna, 1 de liga “rara” si hay.
* Abrir `lineups_url` del JSON en el navegador y comparar.

Para cada partido, marcar:

| Check | Criterio de OK |
| --- | --- |
| Lado correcto | Nombres en `home` / `away` coinciden con local/visitante de la página (no invertidos). Recordar: el slug de la URL **no** es autoridad; el título sí. |
| Nombres | Cada baja “Will not play” relevante aparece (lesión/sanción). |
| Motivo | `reason` ≈ texto Flashscore; `reason_class` coherente (`injury` / `suspension` / `doubtful`; `inactive` puede existir en el JSON pero la justificación lo omite). |
| Sin inventados | No hay jugadores en el JSON que no estén en la lista de la web. |
| Vacío real | Si Flashscore no muestra “Will not play”, `home`/`away` vacíos (o solo inactive) está bien — no rellenar a mano. |

**Umbral:** ≥ **90%** de los spot-checks sin error de lado ni de jugador fantasma.
Un flip home/away sistemático = **gate fallido** (arreglar extractor antes
de tocar el modelo).

### C) Chequeos de schema (una vez al inicio + si cambia Flashscore)

Sobre un JSON cualquiera de la semana:

* Cada entrada tiene: `home_team`, `away_team`, `league`, `lineups_url`,
  `ts`, `home`, `away`.
* Cada ausente: `name`, `player_id`, `reason`, `reason_class`.
* `reason_class` ∈ `{injury, suspension, doubtful, inactive, other}`.

```bash
python3 -c "
import json,sys
REQ={'home_team','away_team','league','lineups_url','ts','home','away'}
ABS={'name','player_id','reason','reason_class'}
CLS={'injury','suspension','doubtful','inactive','other'}
d=json.load(open(sys.argv[1]))
for mid,r in d.items():
    assert REQ<=set(r), (mid, REQ-set(r))
    for side in ('home','away'):
        for a in r[side]:
            assert ABS<=set(a), a
            assert a['reason_class'] in CLS, a
print('schema OK', len(d), 'matches')
" output/availability_YYYY-MM-DD.json
```

### D) Lo que esta semana NO prueba (aún)

* Que las bajas mejoren Brier / ROI → eso es **Paso 8** (forward test con
  flag off / log-only).
* Importancia del jugador (SoFIFA OVR ya rechazado).
* Que “inactive” deba entrar en el adjuster (de momento se ignora en texto).

### Mini-tabla de seguimiento (rellenar a mano)

| Fecha | availability OK | cobertura ≥85% | spot-checks (n / fallos) | notas |
| --- | --- | --- | --- | --- |
| 2026-09-21 | sí (smoke) | — | smoke extractor OK | 7 matches / 41 absentees |
| … | | | | |

### Criterio de gate ABIERTO (se puede plantear Paso 8)

1. ≥7 días verdes (A1–A4).
2. Spot-checks B con ≥90% OK y **cero** flips home/away sistemáticos.
3. Schema C OK en al menos 2 fechas distintas.
4. Ningún cambio de DOM de Flashscore sin haber re-validado el parser
   (`--from-html` / spot-check).

Hasta entonces: bajas solo en **JSON + justificación**; **prohibido**
meterlas en `predict_matches` / heurísticas / features de train.

---

## 5. Cadencia diaria cuando A+B estén vivos

```text
./bin/update_leagues_data.sh
./bin/run_predictions.sh                              # incluye extract_availability + extract_referees
# python3 scripts/d4_injuries/extract_availability.py YYYY-MM-DD  # solo si hace falta re-correr
# python3 scripts/d4_referees/extract_referees.py YYYY-MM-DD      # solo si hace falta re-correr
python3 scripts/justify_predictions.py                # Pasos 2+5 (bajas + árbitro)
# opcional: Telegram con justificaciones enriquecidas

# al día siguiente
./bin/run_verification.sh
# incluye el append no fatal al histórico de árbitros (Paso 4, hecho):
#   python3 scripts/referees/build_referee_history.py --append-verification YYYY-MM-DD
```

Semanal: `./bin/retrain_pipeline.sh` sigue siendo solo el modelo
**equipo/1X2/O/U**. El mercado de tarjetas (Paso 6) se reentrena aparte:

```bash
python3 -m ml_project.cards.train_cards
python3 -m ml_project.cards.calibration
python3 scripts/experiment_cards.py
```

---

## 6. Qué no hacer todavía

* Meter bajas en el **modelo / adjuster** antes de pasar el gate §4.1.
* Meter “noticias” genéricas / NLP de periódicos sin schema fijo.
* Reentrenar 1X2 cada día esperando milagros.
* Alargar ventanas de forma de tarjetas (L10→L15…) como primer lever de
  precisión Cards — el prior ya ve esos resultados; priorizar Paso 6b.
* Construir modelo de córners **antes** de tener alineaciones + (para
  tarjetas) histórico de árbitro **y** al menos un experiment 6b
  documentado (PASS o FAIL con placebo).
* Confiar en SoFIFA OVR como impacto de baja (ya rechazado en D4) — tampoco
  como proxy de “perfil de tarjeta” del ausente.
* Activar apuestas reales sobre tarjetas/córners sin validación forward.

---

## 7. Criterio de “vamos bien”

1. Cada día hay `availability_*.json` y (luego) `referees_*.json`.
2. La 1ª semana de bajas cumple el gate §4.1 **antes** de cualquier
   adjuster 1X2 **o** brazo de bajas en Cards (Paso 6b.1).
3. El histórico de árbitros **crece** con cada verification.
4. Las justificaciones citan **solo** hechos presentes en esos JSON.
5. Cualquier mercado nuevo tiene baseline y métrica antes de UI/Telegram
   tipster.
6. Precisión Cards: experiment 6b (bajas / derbis / presión) con placebo
   antes de reivindicar mejora; no L15 de forma como atajo.

---

## 8. Próxima acción concreta (una sola)

**Ahora:** validar la 1ª semana de bajas (§4.1) mientras sigue la cadencia
diaria (`run_predictions` → availability → referees → `justify_predictions`).
Pasos 3–6 (árbitro + histórico + texto + **mercado tarjetas**) ya están
hechos; el histórico de árbitros crece solo en cada `run_verification.sh`.
Tarjetas: reentrenar semanal aparte; serve ON por defecto
(`CARDS_ENABLED=0` para apagar). **Tras §4.1:** Paso **6b** (señales que el
prior no ve: sancionados/pivotes → derbis → presión de tabla) antes que
Paso 7 (córners).

Opcional en paralelo: **backfill Big-5 Sofascore** (ya implementado)::

  ```bash
  pip install curl_cffi   # si falta
  python3 scripts/referees/seed_sofascore.py --all-big5 --years 23/24,24/25,25/26
  # log de una corrida larga: logs/seed_sofascore.log
  ```

  Luego alinear target Cards↔booking points Winamax.

**No** implementar Paso 8 (adjuster 1X2) hasta gate §4.1 abierto.
**No** alargar L15 de forma de tarjetas como siguiente experimento.
**No** añadir stakes/lanes de tarjetas sin cuotas + estudio ROI settled.

Frase de arranque para el agente: *“Ayúdame a rellenar / automatizar el
checklist §4.1 del roadmap”* — o *“Implementa el Paso 6b.1 (sancionados /
pivotes) del roadmap de tarjetas…”*.