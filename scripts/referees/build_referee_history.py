#!/usr/bin/env python3
"""D4 / Paso 4 — catálogo histórico de árbitros (capa de DATOS, no modelo).

Ver docs/enriched_match_data_roadmap.md §4 Paso 4. Construye y hace crecer
``data_sets/referees/referee_matches.csv`` (una fila por partido pitado) y
``data_sets/referees/referees.json`` (índice id/slug → nombre canónico +
alias), a partir de DOS fuentes independientes que conviven en el mismo CSV
(columna ``source``):

  (A) SEED / backfill — ``data_sets/MatchHistory/*.csv`` (columnas
      football-data.co.uk: ``Referee, HY, AY, HR, AR, HC, AC, Date,
      HomeTeam, AwayTeam, FTHG, FTAG``; liga derivada del nombre de
      fichero). Solo los ficheros de las "main leagues" de football-data
      (descargados vía ``mmz4281/<season>/data.zip``, p.ej.
      ``ENG-Premier_League_25-26.csv``) traen columna ``Referee`` — ver
      ``--coverage``. Las "extra leagues" (ARG, AUT, BRA, ... consolidadas
      multi-temporada desde ``/new/``) NUNCA la traen: no hay backfill
      posible para ellas, solo crecerán hacia delante.

  (B) FORWARD / append — al verificar un día (``bin/run_verification.sh``),
      si ``output/referees_<date>.json`` (D4 Paso 3, asignación del día)
      conoce el árbitro de un ``match_id``, se añade una fila con el
      resultado (``output/results_<date>.json``) y el partido (nombres
      canónicos desde ``output/predictions_<date>.csv`` cuando existe).
      Estas filas NO traen tarjetas/córners — Flashscore en modo
      verificación solo expone el marcador final, no el box score — así
      que ``hy/ay/hr/ar/hc/ac`` quedan vacíos y ``source=verification``.
      Si Paso 3 no corrió ese día (o el árbitro salió ``null``), esta
      función no falla: se salta el partido y lo deja documentado.

Ambas vías son IDÉMPOTENTES: cada fila tiene un ``match_key`` estable
(``mh:<liga>:<fecha>:<home>:<away>`` para seed, ``fs:<match_id>`` para
forward) y el upsert hace ``drop_duplicates(subset=['match_key'])`` —
re-ejecutar el seed o re-verificar un día no duplica filas.

Uso
----
    # Auditoría (qué ficheros de MatchHistory tienen árbitro)
    python3 scripts/referees/build_referee_history.py --coverage

    # Backfill — todas las temporadas en disco con columna Referee
    python3 scripts/referees/build_referee_history.py --seed-from-matchhistory

    # Backfill acotado (ej. solo desde una fecha, o las 3 temporadas más
    # recientes por liga si hubiera varias en disco)
    python3 scripts/referees/build_referee_history.py --seed-from-matchhistory --since 2023-07-01
    python3 scripts/referees/build_referee_history.py --seed-from-matchhistory --seasons 3

    # Forward — llamado por bin/run_verification.sh tras resolver apuestas
    python3 scripts/referees/build_referee_history.py --append-verification 2026-09-20

    # Consulta
    python3 scripts/referees/build_referee_history.py --stats "A Taylor"
    python3 scripts/referees/build_referee_history.py --stats "Letexier F."

No descarga nada por red y no requiere temporadas concretas: usa lo que ya
esté en ``data_sets/MatchHistory/`` (ver ``bin/setup_data.sh <season_code>``
para añadir más temporadas de las "main leagues", p.ej. ``2324``/``2425``).
"""

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
import warnings
from collections import defaultdict

import pandas as pd

# Forward ("verification") rows have no card/corner data at all (hy/ay/hr/
# ar/hc/ac all None), which makes pandas warn about a future dtype-inference
# change when concatenating an all-NA column. Harmless here — the columns
# stay NaN either way — so it's silenced rather than worked around.
warnings.filterwarnings(
    "ignore", message=".*empty or all-NA entries.*", category=FutureWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MATCHHISTORY_DIR = os.path.join(ROOT, "data_sets", "MatchHistory")
REFEREES_DIR = os.path.join(ROOT, "data_sets", "referees")
OUTPUT_DIR = os.path.join(ROOT, "output")

MATCHES_CSV = os.path.join(REFEREES_DIR, "referee_matches.csv")
CATALOG_JSON = os.path.join(REFEREES_DIR, "referees.json")

CSV_COLUMNS = [
    "match_key", "date", "league", "home", "away",
    "score_home", "score_away",
    "referee_name", "referee_id",
    "hy", "ay", "hr", "ar", "hc", "ac",
    "source", "ingested_at",
]

_SEASON_SUFFIX_RE = re.compile(r"_(\d{2})-(\d{2})\.csv$")
_INITIAL_RE = re.compile(r"^[A-Za-z]\.?$")


# ---------------------------------------------------------------------------
# Name / slug normalization
# ---------------------------------------------------------------------------

def normalize_referee(raw_name):
    """Return ``(referee_id, canonical_display)`` for a raw referee string.

    football-data.co.uk uses "Initial Surname" (``"A Taylor"``); Flashscore
    (D4 Paso 3, ``extract_referees.py``) uses "Surname Initial." (``"Letexier
    F."``). Detecting which token is the bare initial lets both collapse to
    the same id so the seed (MatchHistory) and forward (verification) rows
    for the same real person share one catalog entry. Best-effort only —
    two different referees sharing a surname + initial will still collide;
    this is a catalog for rate lookups, not full entity resolution.
    """
    raw = str(raw_name).strip()
    if not raw or raw.lower() == "nan":
        return None, None
    tokens = [t for t in raw.replace(".", " ").split() if t]
    if not tokens:
        return None, None
    if len(tokens) == 1:
        surname, initial = tokens[0], ""
    elif _INITIAL_RE.match(tokens[-1] + "."):
        surname, initial = " ".join(tokens[:-1]), tokens[-1]
    elif _INITIAL_RE.match(tokens[0] + "."):
        initial, surname = tokens[0], " ".join(tokens[1:])
    else:
        surname, initial = " ".join(tokens), ""
    slug_parts = [re.sub(r"[^a-z0-9]", "", surname.lower())]
    if initial:
        slug_parts.append(re.sub(r"[^a-z0-9]", "", initial.lower()))
    slug = "-".join(p for p in slug_parts if p)
    display = surname.title() + (f" {initial.upper()}." if initial else "")
    return (slug or None), display


def _slugify_team(name):
    return re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")


def _num(v):
    """Best-effort numeric coercion; ``None`` for missing/unparseable."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else f


def _parse_mh_date(raw):
    ts = pd.to_datetime(raw, dayfirst=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# MatchHistory audit / seeding (source A)
# ---------------------------------------------------------------------------

def list_matchhistory_files():
    return sorted(glob.glob(os.path.join(MATCHHISTORY_DIR, "*.csv")))


def file_has_referee_column(path):
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return False
    return "Referee" in header


def _count_rows(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f) - 1
    except OSError:
        return -1


def coverage_rows():
    rows = []
    for path in list_matchhistory_files():
        rows.append({
            "file": os.path.basename(path),
            "has_referee": file_has_referee_column(path),
            "rows": _count_rows(path),
        })
    return rows


def print_coverage():
    rows = coverage_rows()
    rows.sort(key=lambda r: (not r["has_referee"], r["file"]))
    print(f"{'REF?':4} {'FILAS':>6}  FICHERO")
    n_yes = 0
    for r in rows:
        flag = "SI" if r["has_referee"] else "no"
        if r["has_referee"]:
            n_yes += 1
        print(f"{flag:4} {r['rows']:>6}  {r['file']}")
    print(f"\n{n_yes}/{len(rows)} ficheros de data_sets/MatchHistory/ tienen columna "
          f"Referee — hoy solo ENG (Premier/Championship/League 1/League "
          f"2/Conference) y SCO (Premier/Division 1-3). football-data.co.uk NO "
          f"publica Referee para el resto de 'main leagues' con sufijo de "
          f"temporada (ESP, FRA, GER, ITA, NED, BEL, POR, TUR, GR) ni para "
          f"ninguna 'extra league' consolidada desde /new/ (ARG, AUT, BRA, CHN, "
          f"DEN, FIN, IRL, JPN, MEX, NOR, POL, ROU, RUS, SUI, SWE, USA): no es un "
          f"bug del downloader, es lo que la fuente expone. No hay backfill "
          f"posible para esas ligas — solo crecen hacia delante vía "
          f"--append-verification (sin tarjetas/córners, ver docstring).")


def _season_key(basename):
    m = _SEASON_SUFFIX_RE.search(basename)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def _league_group(basename):
    m = _SEASON_SUFFIX_RE.search(basename)
    return basename[: m.start()] if m else basename[:-4]


def league_label_from_path(path):
    basename = os.path.basename(path)
    m = _SEASON_SUFFIX_RE.search(basename)
    stem = basename[: m.start()] if m else basename[:-4]
    return stem.replace("_", " ")


def select_seed_files(seasons=None):
    """Files with a Referee column, optionally capped to the N most recent
    seasons per league (by the ``_AA-BB.csv`` filename suffix). With only
    one season on disk per league today this is a no-op; it exists so
    dropping in ``bin/setup_data.sh 2324`` / ``2425`` later (more seasons of
    the same 'main league' files) is picked up without code changes.
    """
    ref_files = [f for f in list_matchhistory_files() if file_has_referee_column(f)]
    if seasons is None:
        return ref_files
    groups = defaultdict(list)
    for f in ref_files:
        groups[_league_group(os.path.basename(f))].append(f)
    selected = []
    for flist in groups.values():
        flist_sorted = sorted(
            flist, key=lambda f: _season_key(os.path.basename(f)) or "", reverse=True)
        selected.extend(flist_sorted[:seasons])
    return sorted(selected)


def rows_from_matchhistory(path, ingested_at):
    df = pd.read_csv(path)
    if "Referee" not in df.columns:
        return []
    league = league_label_from_path(path)
    out = []
    for _, r in df.iterrows():
        ref_raw = r.get("Referee")
        if pd.isna(ref_raw) or not str(ref_raw).strip():
            continue
        date_iso = _parse_mh_date(r.get("Date"))
        if date_iso is None:
            continue
        home = str(r.get("HomeTeam", "")).strip()
        away = str(r.get("AwayTeam", "")).strip()
        ref_id, _ = normalize_referee(ref_raw)
        match_key = f"mh:{league}:{date_iso}:{_slugify_team(home)}:{_slugify_team(away)}"
        out.append({
            "match_key": match_key,
            "date": date_iso,
            "league": league,
            "home": home,
            "away": away,
            "score_home": _num(r.get("FTHG")),
            "score_away": _num(r.get("FTAG")),
            "referee_name": str(ref_raw).strip(),
            "referee_id": ref_id,
            "hy": _num(r.get("HY")),
            "ay": _num(r.get("AY")),
            "hr": _num(r.get("HR")),
            "ar": _num(r.get("AR")),
            "hc": _num(r.get("HC")),
            "ac": _num(r.get("AC")),
            "source": "matchhistory",
            "ingested_at": ingested_at,
        })
    return out


# ---------------------------------------------------------------------------
# Persistence (idempotent upsert + catalog rebuild)
# ---------------------------------------------------------------------------

def load_existing_matches():
    if os.path.exists(MATCHES_CSV):
        return pd.read_csv(MATCHES_CSV, dtype={"referee_id": str, "league": str})
    return pd.DataFrame(columns=CSV_COLUMNS)


def upsert_matches(new_rows):
    """Idempotent append: dedups on ``match_key``, last write wins (so a
    re-seed with corrected data can update a row without creating a dup).
    Returns ``(net_new_rows, total_rows_after)``.
    """
    existing = load_existing_matches()
    before = len(existing)
    if not new_rows:
        return 0, before
    new_df = pd.DataFrame(new_rows, columns=CSV_COLUMNS)
    combined = new_df if existing.empty else pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["match_key"], keep="last")
    combined = combined.sort_values(["date", "league", "home"], na_position="last")
    os.makedirs(REFEREES_DIR, exist_ok=True)
    combined.to_csv(MATCHES_CSV, index=False)
    return len(combined) - before, len(combined)


def rebuild_catalog():
    """Rebuilds referees.json fully from referee_matches.csv (derived
    index, not a second source of truth) — groups by referee_id, collects
    every distinct raw string seen as an alias, and picks the "Surname I."
    style string as canonical when available.
    """
    catalog = {}
    if os.path.exists(MATCHES_CSV):
        df = pd.read_csv(MATCHES_CSV, dtype=str)
        df = df[df["referee_id"].notna() & (df["referee_id"] != "")]
        for rid, group in df.groupby("referee_id"):
            names = sorted(set(group["referee_name"].dropna().astype(str)))
            canonical = next((n for n in names if re.search(r"\b[A-Za-z]\.$", n)), None)
            canonical = canonical or (names[0] if names else rid)
            catalog[rid] = {
                "id": rid,
                "canonical_name": canonical,
                "aliases": names,
                "n_matches": int(len(group)),
                "sources": sorted(set(group["source"].dropna().astype(str))),
            }
    os.makedirs(REFEREES_DIR, exist_ok=True)
    with open(CATALOG_JSON, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, sort_keys=True)
    return catalog


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_seed(seasons=None, since=None):
    files = select_seed_files(seasons=seasons)
    if not files:
        print("[referees] Ningún fichero de MatchHistory con columna Referee encontrado.")
        return
    ingested_at = dt.datetime.now(dt.timezone.utc).isoformat()
    all_rows = []
    per_file = {}
    for f in files:
        rows = rows_from_matchhistory(f, ingested_at)
        if since:
            rows = [r for r in rows if r["date"] >= since]
        per_file[os.path.basename(f)] = len(rows)
        all_rows.extend(rows)
    added, total = upsert_matches(all_rows)
    catalog = rebuild_catalog()
    print(f"[referees] Seed desde {len(files)} fichero(s)"
          f"{f' (>= {since})' if since else ''}:")
    for name in sorted(per_file):
        print(f"  {name}: {per_file[name]} filas con árbitro")
    print(f"[referees] +{added} filas netas nuevas -> {total} filas totales en "
          f"{MATCHES_CSV}; {len(catalog)} árbitros distintos en {CATALOG_JSON}.")


def cmd_append_verification(date_str):
    predictions_path = os.path.join(OUTPUT_DIR, f"predictions_{date_str}.csv")
    results_path = os.path.join(OUTPUT_DIR, f"results_{date_str}.json")
    referees_path = os.path.join(OUTPUT_DIR, f"referees_{date_str}.json")

    if not os.path.exists(referees_path):
        print(f"[referees] {date_str}: no existe {os.path.relpath(referees_path, ROOT)} "
              "(D4 Paso 3 no corrió ese día, o el scraper falló). No es fatal — el "
              "histórico de árbitros simplemente no crece hoy. Requiere Paso 3 "
              "(extract_referees.py) para la asignación del día.")
        return 0
    if not os.path.exists(results_path):
        print(f"[referees] {date_str}: no existe {os.path.relpath(results_path, ROOT)} "
              "(sin resultados verificados aún). No es fatal.")
        return 0

    with open(referees_path) as f:
        referees = json.load(f)
    with open(results_path) as f:
        results = json.load(f)
    # results_<date>.json's home_team/away_team are unreliable in
    # verification (ID-based) mode — the spider re-parses the summary page
    # and both fields can end up equal to the home team's name. Prefer the
    # canonical names from predictions_<date>.csv when available.
    results_by_id = {r.get("match_id"): r for r in results if r.get("match_id")}

    pred_by_id = {}
    if os.path.exists(predictions_path):
        pdf = pd.read_csv(predictions_path, dtype={"match_id": str})
        for _, row in pdf.iterrows():
            mid = row.get("match_id")
            if isinstance(mid, str) and mid:
                pred_by_id[mid] = row

    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    new_rows = []
    skipped_no_ref = 0
    skipped_no_result = 0
    for mid, rinfo in referees.items():
        ref_name = rinfo.get("referee_name")
        if not ref_name:
            skipped_no_ref += 1
            continue
        res = results_by_id.get(mid)
        if res is None:
            skipped_no_result += 1
            continue
        pred = pred_by_id.get(mid)
        if pred is not None:
            home = pred.get("Home Team")
            away = pred.get("Away Team")
            league = pred.get("League") or rinfo.get("league")
            date_val = str(pred.get("Date"))[:10] if pd.notna(pred.get("Date")) else date_str
        else:
            home = rinfo.get("home_team")
            away = rinfo.get("away_team")
            league = rinfo.get("league")
            date_val = date_str
        ref_id, _ = normalize_referee(ref_name)
        new_rows.append({
            "match_key": f"fs:{mid}",
            "date": date_val,
            "league": league,
            "home": home,
            "away": away,
            "score_home": _num(res.get("home_score")),
            "score_away": _num(res.get("away_score")),
            "referee_name": ref_name,
            "referee_id": ref_id,
            "hy": None, "ay": None, "hr": None, "ar": None, "hc": None, "ac": None,
            "source": "verification",
            "ingested_at": now_iso,
        })

    added, total = upsert_matches(new_rows)
    rebuild_catalog()
    print(f"[referees] {date_str}: +{added} filas nuevas -> {total} filas totales "
          f"en {MATCHES_CSV}. Sin árbitro asignado: {skipped_no_ref}; sin "
          f"resultado todavía: {skipped_no_result}. (Filas forward no llevan "
          f"tarjetas/córners — Flashscore en modo verificación no expone el "
          f"box score.)")
    return added


def cmd_stats(name_query, since=None):
    if not os.path.exists(MATCHES_CSV):
        print(f"[referees] No existe {MATCHES_CSV} todavía — ejecuta "
              "--seed-from-matchhistory (y/o deja que --append-verification crezca "
              "el histórico unos días) antes de pedir --stats.")
        return
    df = pd.read_csv(MATCHES_CSV, dtype={"referee_id": str})
    rid, _ = normalize_referee(name_query)
    query_lower = name_query.strip().lower()
    mask = (df["referee_id"] == rid) | (
        df["referee_name"].astype(str).str.lower() == query_lower)
    sub = df[mask]
    if since:
        sub = sub[sub["date"].astype(str) >= since]
    if sub.empty:
        print(f"[referees] Sin partidos para '{name_query}' (id normalizado: {rid!r}). "
              "¿Está seedeada esa liga/temporada? Prueba --coverage.")
        return

    n = len(sub)
    by_source = sub.groupby("source").size().to_dict()
    print(f"=== {name_query}  (id: {rid}) ===")
    print(f"Partidos en catálogo: {n}   por fuente: {by_source}")
    if since:
        print(f"  (filtrado desde {since})")

    with_cards = sub.dropna(subset=["hy", "ay"])
    n_cards = len(with_cards)
    if n_cards:
        yellows = pd.to_numeric(with_cards["hy"], errors="coerce") + \
            pd.to_numeric(with_cards["ay"], errors="coerce")
        reds = pd.to_numeric(with_cards["hr"], errors="coerce") + \
            pd.to_numeric(with_cards["ar"], errors="coerce")
        print(f"Amarillas/partido: {yellows.mean():.2f}  (n={n_cards} partidos con datos de tarjetas)")
        print(f"% partidos con >=1 roja: {(reds.fillna(0) > 0).mean() * 100:.1f}%")
        corners = pd.to_numeric(with_cards["hc"], errors="coerce") + \
            pd.to_numeric(with_cards["ac"], errors="coerce")
        if corners.notna().any():
            print(f"Córners/partido: {corners.mean():.2f}")
    else:
        print("Sin datos de tarjetas/córners para este árbitro (solo filas "
              "'verification', que no traen box score). Necesitas backfill de "
              "MatchHistory para tasas de tarjetas.")
    if n - n_cards:
        print(f"({n - n_cards} partido(s) sin datos de tarjetas, típicamente source=verification)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coverage", action="store_true",
                         help="Tabla de cobertura Referee en MatchHistory y salir.")
    parser.add_argument("--seed-from-matchhistory", action="store_true",
                         help="Backfill desde data_sets/MatchHistory/*.csv (ligas con Referee).")
    parser.add_argument("--seasons", type=int, default=None,
                         help="Con --seed-from-matchhistory: limita a las N temporadas más "
                              "recientes por liga (según sufijo _AA-BB.csv en el nombre de "
                              "fichero). Por defecto usa todo lo que haya en disco.")
    parser.add_argument("--since", type=str, default=None, metavar="YYYY-MM-DD",
                         help="Filtra filas de seed (o de --stats) a partir de esta fecha.")
    parser.add_argument("--append-verification", metavar="YYYY-MM-DD",
                         help="Anexa forward desde output/referees_<date>.json + "
                              "results_<date>.json + predictions_<date>.csv. No falla si "
                              "faltan ficheros.")
    parser.add_argument("--stats", metavar="REFEREE_NAME",
                         help='Imprime tasas para un árbitro, p.ej. --stats "A Taylor".')
    parser.add_argument("--rebuild-catalog", action="store_true",
                         help="Solo regenerar referees.json desde el CSV existente.")
    args = parser.parse_args()

    if args.coverage:
        print_coverage()
    elif args.rebuild_catalog:
        catalog = rebuild_catalog()
        print(f"[referees] Catálogo regenerado: {len(catalog)} árbitros -> {CATALOG_JSON}")
    elif args.seed_from_matchhistory:
        cmd_seed(seasons=args.seasons, since=args.since)
    elif args.append_verification:
        cmd_append_verification(args.append_verification)
    elif args.stats:
        cmd_stats(args.stats, since=args.since)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
