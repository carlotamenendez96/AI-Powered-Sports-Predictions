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
| Justificación nivel 1 (probs/ELO/EV + bajas) | Hecho (Paso 2) | `justify_predictions.py` lee `availability_*.json`; contexto tipster, no input del pick |
| Bajas / “Will not play” Flashscore | **Cableado en `run_predictions.sh` (paso no fatal) — validando 1ª semana** | `scripts/d4_injuries/extract_availability.py` → `output/availability_<date>.json` |
| Importancia jugador (SoFIFA OVR) | Hecho pero D4 aparcado (OVR ≠ impacto) | `ml_project/availability/sofifa_importance.py` |
| Adjuster 1X2 por bajas | No hecho (shelved N3) | No priorizar hasta medir |
| Árbitro del partido | **Cableado en `run_predictions.sh` (paso no fatal)** | `scripts/d4_referees/extract_referees.py` → `output/referees_<date>.json` |
| Histórico árbitros (tarjetas…) | No existe | Fase B–C |
| Modelo / mercado tarjetas | No existe | Fase E |
| Modelo / mercado córners | No existe | Fase F |

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

* **Qué:** Tabla acumulativa `referee_matches.csv` (o JSONL) con, por partido
  pitado: tarjetas, rojas, resultado, liga, fecha; opcional corners/faltas.
  Crecer: (a) al verificar ayer, guardar stats del partido + árbitro; y/o
  (b) backfill de partidos recientes por liga.
* **Por qué:** Un listado de nombres sin números no sirve; el valor está en
  **tasas** (amarillas/90, % partidos con roja, etc.).
* **Entregable:** Catálogo + ≥ N partidos/árbitro frecuente (definir N, p.ej.
  20) antes de usar en tipster/modelo.
* **Listo cuando:** Puedes responder “¿cuántas amarillas/partido lleva X
  esta temporada?” con query local.
* **Siguiente:** Paso 5.

### Paso 5 — Justificación + árbitro (texto)

* **Qué:** Incluir en la justificación: nombre + resumen corto del histórico
  (p.ej. media de amarillas en últimos K partidos).
* **Por qué:** Cierra el loop tipster con hechos; prepara Telegram.
* **Listo cuando:** UI/Telegram muestran bloque árbitro sin inventar cifras.

### Paso 6 — Mercado tarjetas (Fase E) — solo después de 3–4

* **Qué:** Target propio (p.ej. over amarillas, amarilla 1ª parte). Features:
  árbitro (rates), bajas de jugadores “agresivos” si se puede etiquetar,
  ligas, cuotas si existen.
* **Por qué:** El head 1X2 **no** predice tarjetas; hace falta otro modelo o
  reglas + calibración.
* **Listo cuando:** Backtest o forward test con métrica clara (Brier / ROI
  paper) vs baseline “media de liga” o “media del árbitro”.

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
python3 scripts/justify_predictions.py                # Pasos 2+5
# opcional: Telegram con justificaciones enriquecidas

# al día siguiente
./bin/run_verification.sh
# + append stats del partido al histórico de árbitros (Paso 4)
```

Semanal: `./bin/retrain_pipeline.sh` sigue siendo solo el modelo
**equipo/1X2/O/U**. Los mercados nuevos tendrán su propio train cuando existan.

---

## 6. Qué no hacer todavía

* Meter bajas en el **modelo / adjuster** antes de pasar el gate §4.1.
* Meter “noticias” genéricas / NLP de periódicos sin schema fijo.
* Reentrenar 1X2 cada día esperando milagros.
* Construir modelo de córners **antes** de tener alineaciones + (para
  tarjetas) histórico de árbitro.
* Confiar en SoFIFA OVR como impacto de baja (ya rechazado en D4).
* Activar apuestas reales sobre tarjetas/córners sin validación forward.

---

## 7. Criterio de “vamos bien”

1. Cada día hay `availability_*.json` y (luego) `referees_*.json`.
2. La 1ª semana de bajas cumple el gate §4.1 **antes** de cualquier
   adjuster 1X2.
3. El histórico de árbitros **crece** con cada verification.
4. Las justificaciones citan **solo** hechos presentes en esos JSON.
5. Cualquier mercado nuevo tiene baseline y métrica antes de UI/Telegram
   tipster.

---

## 8. Próxima acción concreta (una sola)

**Ahora:** validar la 1ª semana de bajas (§4.1) mientras sigue la cadencia
diaria (`run_predictions` → availability → referees → `justify_predictions`).

En paralelo o después: **Paso 4** (histórico de árbitros / tarjetas).

**No** implementar Paso 8 (adjuster) hasta gate §4.1 abierto.

Frase de arranque para el agente: *“Implementa el Paso 4 del roadmap
docs/enriched_match_data_roadmap.md”* — o, si el gate está cerrado:
*“Ayúdame a rellenar / automatizar el checklist §4.1 del roadmap”*.