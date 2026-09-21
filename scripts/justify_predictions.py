#!/usr/bin/env python3
"""Level-1 prediction justifications (template text, no LLM / no news).

Reads `output/predictions_YYYY-MM-DD.csv` and writes a companion JSON (+ optional
plain-text report) explaining each pick in Spanish from fields the model
already produces: probs, odds, EV, ELO, heuristic Adj Logs.

If `output/availability_<date>.json` exists (from extract_availability / Paso 1),
adds a short factual block for relevant absences (injury / suspension / doubtful).
Those lines are context only — the 1X2/O/U model does **not** use bajas yet.

If `output/referees_<date>.json` exists (Paso 3) and the local referee catalog
(`data_sets/referees/referee_matches.csv`, Paso 4) has enough matches with
card data for that referee, adds a short factual block with yellows/game etc.
Never invents rates: name-only when the catalog is thin / empty for that ref.

Usage:
    python3 scripts/justify_predictions.py
    python3 scripts/justify_predictions.py --date 2026-09-21
    python3 scripts/justify_predictions.py --date 2026-09-21 --print
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
REFEREES_DIR = os.path.join(PROJECT_ROOT, "data_sets", "referees")
REFEREE_MATCHES_CSV = os.path.join(REFEREES_DIR, "referee_matches.csv")

# Import shared name→id normalizer from Paso 4 (same slug for "A Taylor" /
# "Taylor A.") so justification lookups hit the same catalog rows.
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
try:
    from referees.build_referee_history import normalize_referee  # noqa: E402
except ImportError:  # pragma: no cover — keep justify runnable if script moves
    def normalize_referee(raw_name):  # type: ignore[misc]
        return None, None

JUSTIFY_VERSION = "level1+availability+referee-v1"

# Absences worth mentioning in tipster text. "inactive" is rotation/noise — skip.
_RELEVANT_REASON_CLASSES = frozenset({"injury", "suspension", "doubtful"})
_MAX_ABSENTEES_PER_SIDE = 4

# Gate for citing card rates in tipster text (roadmap Paso 4/5). Below this we
# still name the referee when known, but never invent amarillas/partido.
_MIN_MATCHES_FOR_REF_RATES = 20

PICK_1X2 = {"1": "victoria local", "X": "empate", "2": "victoria visitante",
            "Home": "victoria local", "Draw": "empate", "Away": "victoria visitante"}


def _f(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(str(v).replace("+", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _pct(x) -> str:
    if x is None:
        return "?"
    # Model stores 0.46 or sometimes already percent-like
    if x > 1.5:
        return f"{x:.0f}%"
    return f"{x * 100:.0f}%"


def _label_1x2(code) -> str:
    return PICK_1X2.get(str(code).strip(), str(code))


def _conf_band(c) -> str:
    if c is None:
        return "confianza desconocida"
    if c >= 0.65:
        return "confianza alta"
    if c >= 0.50:
        return "confianza media"
    return "confianza baja"


def _ev_phrase(ev) -> str:
    if ev is None:
        return "sin EV calculado"
    if ev > 0.15:
        return f"EV positivo alto ({ev:+.2f}) según cuota×prob — tómalo con cautela: el EV del modelo suele estar inflado donde más se equivoca"
    if ev > 0:
        return f"EV ligeramente positivo ({ev:+.2f}) frente a la cuota"
    if ev > -0.15:
        return f"EV cerca de cero / negativo leve ({ev:+.2f})"
    return f"EV negativo ({ev:+.2f}): la cuota no compensa la probabilidad del modelo"


def _elo_phrase(home, away, home_elo, away_elo) -> str:
    he, ae = _f(home_elo), _f(away_elo)
    if he is None or ae is None:
        return ""
    diff = he - ae
    if abs(diff) < 25:
        return f"Los ELO están equilibrados ({home} {he:.0f} vs {away} {ae:.0f})."
    if diff > 0:
        return f"{home} parte con ventaja de rating (ELO {he:.0f} vs {ae:.0f}, Δ{diff:+.0f})."
    return f"{away} parte con ventaja de rating (ELO {ae:.0f} vs {he:.0f}, Δ{diff:+.0f})."


def _humanize_adj_logs(raw: str) -> list[str]:
    """Turn a few known Adj Logs tokens into short Spanish bullets."""
    if not raw or not str(raw).strip() or str(raw).strip() in ("", "nan"):
        return []
    text = str(raw)
    out = []
    if re.search(r"No Standings Data", text, re.I):
        out.append("Sin datos de clasificación/form frescos para este partido (standings no disponibles).")
    if re.search(r"Draw Cap", text, re.I):
        out.append("La heurística limitó la probabilidad de empate (tope de draw).")
    if re.search(r"Rank Boost Home", text, re.I):
        out.append("Ajuste por ranking: empujó hacia el local.")
    if re.search(r"Rank Boost Away", text, re.I):
        out.append("Ajuste por ranking: empujó hacia el visitante.")
    if re.search(r"form|streak|momentum|Heating|Cooling|Fade", text, re.I):
        out.append("Hubo ajustes de forma/racha en las heurísticas post-modelo.")
    if re.search(r"Goal.?fest|O/U Boost|OU Boost", text, re.I):
        out.append("Heurística de partido abierto / más goles en el mercado O/U.")
    # Keep short — don't dump the raw log into the user-facing paragraph.
    return out[:4]


def load_availability(date: str) -> dict:
    """Load output/availability_<date>.json or {} if missing / unreadable."""
    path = os.path.join(OUTPUT_DIR, f"availability_{date}.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        print(f"[!] Could not read {path}: {e}", file=sys.stderr)
        return {}


def _format_absentee(a: dict) -> str:
    name = (a.get("name") or "?").strip()
    reason = (a.get("reason") or "").strip()
    return f"{name} ({reason})" if reason else name


def _relevant_absentees(side_list) -> list[dict]:
    if not side_list:
        return []
    out = []
    for a in side_list:
        if not isinstance(a, dict):
            continue
        cls = (a.get("reason_class") or "other").lower()
        if cls in _RELEVANT_REASON_CLASSES:
            out.append(a)
    return out[:_MAX_ABSENTEES_PER_SIDE]


def availability_phrase(avail_rec: dict | None) -> str | None:
    """Spanish sentence listing relevant bajas, or None if nothing worth saying.

    Honest: Flashscore context only — the production model does not adjust for these.
    """
    if not avail_rec:
        return None
    home = _relevant_absentees(avail_rec.get("home") or [])
    away = _relevant_absentees(avail_rec.get("away") or [])
    if not home and not away:
        return None

    bits = []
    if home:
        bits.append("Bajas locales: " + ", ".join(_format_absentee(a) for a in home))
    if away:
        label = "visitantes" if home else "Bajas visitantes"
        bits.append(f"{label}: " + ", ".join(_format_absentee(a) for a in away))
    body = "; ".join(bits) + "."
    return (
        body
        + " (Dato Flashscore «Will not play»; el modelo 1X2/O/U aún no ajusta por bajas.)"
    )


# ---------------------------------------------------------------------------
# Referee (Paso 3 assignment + Paso 4 catalog rates)
# ---------------------------------------------------------------------------

def load_referees_assignment(date: str) -> dict:
    """Load output/referees_<date>.json or {} if missing / unreadable."""
    path = os.path.join(OUTPUT_DIR, f"referees_{date}.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        print(f"[!] Could not read {path}: {e}", file=sys.stderr)
        return {}


def load_referee_rate_index() -> dict:
    """Build referee_id → rate stats from the local catalog (Paso 4).

    Only matches with card columns (hy/ay present) feed yellows/reds/corners.
    Returns {} if the CSV is missing — justification still works name-only.
    """
    if not os.path.isfile(REFEREE_MATCHES_CSV):
        return {}
    try:
        df = pd.read_csv(REFEREE_MATCHES_CSV, dtype={"referee_id": str})
    except (OSError, pd.errors.EmptyDataError, ValueError) as e:
        print(f"[!] Could not read {REFEREE_MATCHES_CSV}: {e}", file=sys.stderr)
        return {}
    if df.empty or "referee_id" not in df.columns:
        return {}

    index: dict = {}
    for rid, group in df.groupby("referee_id"):
        if not isinstance(rid, str) or not rid or rid == "nan":
            continue
        n_all = len(group)
        with_cards = group.dropna(subset=["hy", "ay"]) if "hy" in group.columns else group.iloc[0:0]
        n_cards = len(with_cards)
        stats = {
            "n_matches": n_all,
            "n_with_cards": n_cards,
            "yellows_pg": None,
            "reds_rate": None,
            "corners_pg": None,
            "sample_name": None,
        }
        names = group["referee_name"].dropna().astype(str)
        if len(names):
            stats["sample_name"] = names.iloc[-1]
        if n_cards:
            hy = pd.to_numeric(with_cards["hy"], errors="coerce")
            ay = pd.to_numeric(with_cards["ay"], errors="coerce")
            yellows = hy + ay
            stats["yellows_pg"] = float(yellows.mean())
            if "hr" in with_cards.columns and "ar" in with_cards.columns:
                reds = (
                    pd.to_numeric(with_cards["hr"], errors="coerce").fillna(0)
                    + pd.to_numeric(with_cards["ar"], errors="coerce").fillna(0)
                )
                stats["reds_rate"] = float((reds > 0).mean())
            if "hc" in with_cards.columns and "ac" in with_cards.columns:
                corners = (
                    pd.to_numeric(with_cards["hc"], errors="coerce")
                    + pd.to_numeric(with_cards["ac"], errors="coerce")
                )
                if corners.notna().any():
                    stats["corners_pg"] = float(corners.mean())
        index[rid] = stats
    return index


def referee_phrase(ref_rec: dict | None, rate_index: dict | None) -> str | None:
    """Spanish sentence with referee name (+ rates when catalog is thick enough).

    Never invents figures. Name alone is OK when Flashscore assigned a ref but
    the local catalog has no / too few card rows (typical outside ENG/SCO).
    """
    if not ref_rec:
        return None
    name = (ref_rec.get("referee_name") or "").strip()
    if not name:
        return None

    country = (ref_rec.get("referee_country") or "").strip()
    label = f"{name} ({country})" if country else name

    rid, _ = normalize_referee(name)
    stats = (rate_index or {}).get(rid) if rid else None
    n_cards = int(stats["n_with_cards"]) if stats else 0
    n_all = int(stats["n_matches"]) if stats else 0

    if stats and n_cards >= _MIN_MATCHES_FOR_REF_RATES and stats.get("yellows_pg") is not None:
        bits = [f"Árbitro: {label}."]
        bits.append(f"Histórico local: {stats['yellows_pg']:.1f} amarillas/partido")
        if stats.get("reds_rate") is not None:
            bits.append(f"{stats['reds_rate'] * 100:.0f}% partidos con ≥1 roja")
        if stats.get("corners_pg") is not None:
            bits.append(f"{stats['corners_pg']:.1f} córners/partido")
        rates = " · ".join(bits[1:])
        return (
            f"{bits[0]} {rates} (n={n_cards} partidos con tarjetas en el catálogo). "
            "Contexto tipster; el modelo 1X2/O/U no usa al árbitro."
        )

    # Name known, rates not citable — be explicit why.
    if n_all > 0 and n_cards < _MIN_MATCHES_FOR_REF_RATES:
        return (
            f"Árbitro: {label}. Histórico local insuficiente para tasas de tarjetas "
            f"(n={n_cards} con amarillas; umbral {_MIN_MATCHES_FOR_REF_RATES}). "
            "Solo nombre Flashscore; el modelo no ajusta por árbitro."
        )
    return (
        f"Árbitro: {label}. Sin tasas de tarjetas en el catálogo local "
        "(típicamente fuera de ENG/SCO, o seed aún corto). "
        "Solo nombre Flashscore; el modelo no ajusta por árbitro."
    )


def justify_row(
    row: dict,
    availability_by_id: dict | None = None,
    referees_by_id: dict | None = None,
    referee_rates: dict | None = None,
) -> dict:
    home = str(row.get("Home Team") or row.get("Home") or "?")
    away = str(row.get("Away Team") or row.get("Away") or "?")
    league = str(row.get("League") or "")
    pick = str(row.get("Prediction 1X2") or "").strip()
    conf = _f(row.get("Conf 1X2"))
    odds = _f(row.get("Prediction 1X2 Odd"))
    ev = _f(row.get("EV 1X2"))
    p_h = _f(row.get("Home Win %"))
    p_d = _f(row.get("Draw %"))
    p_a = _f(row.get("Away Win %"))
    ou = str(row.get("Prediction O/U") or "").strip()
    conf_ou = _f(row.get("Conf O/U"))
    odds_ou = _f(row.get("Prediction O/U Odd"))
    ev_ou = _f(row.get("EV O/U"))
    p_o = _f(row.get("Over %"))
    p_u = _f(row.get("Under %"))
    match_id = str(row.get("match_id") or "").strip() or None

    parts = []
    parts.append(
        f"**{home} vs {away}** ({league}). "
        f"El modelo apunta a **{_label_1x2(pick)}** "
        f"con {_conf_band(conf)} ({_pct(conf)})"
        + (f" a cuota {odds:.2f}" if odds else "")
        + "."
    )
    if p_h is not None and p_d is not None and p_a is not None:
        parts.append(
            f"Reparto 1X2: local {_pct(p_h)}, empate {_pct(p_d)}, visitante {_pct(p_a)}."
        )
    elo = _elo_phrase(home, away, row.get("Home ELO"), row.get("Away ELO"))
    if elo:
        parts.append(elo)
    ev_line = _ev_phrase(ev)
    parts.append(ev_line[:1].upper() + ev_line[1:] + ".")

    if ou:
        ou_label = "más de 2.5 goles" if "Over" in ou else "menos de 2.5 goles" if "Under" in ou else ou
        line = (
            f"En goles, favorece **{ou_label}** "
            f"({_conf_band(conf_ou)}, {_pct(conf_ou)}"
            + (f", cuota {odds_ou:.2f}" if odds_ou else "")
            + ")."
        )
        if p_o is not None and p_u is not None:
            line += f" Over {_pct(p_o)} / Under {_pct(p_u)}."
        if ev_ou is not None:
            line += f" EV O/U {ev_ou:+.2f}."
        parts.append(line)

    for bullet in _humanize_adj_logs(row.get("Adj Logs", "")):
        parts.append(bullet)

    used_availability = False
    if availability_by_id and match_id:
        phrase = availability_phrase(availability_by_id.get(match_id))
        if phrase:
            parts.append(phrase)
            used_availability = True

    used_referee = False
    used_referee_rates = False
    if referees_by_id and match_id:
        ref_rec = referees_by_id.get(match_id)
        phrase = referee_phrase(ref_rec, referee_rates)
        if phrase:
            parts.append(phrase)
            used_referee = True
            rid, _ = normalize_referee((ref_rec or {}).get("referee_name") or "")
            st = (referee_rates or {}).get(rid) if rid else None
            if st and int(st.get("n_with_cards") or 0) >= _MIN_MATCHES_FOR_REF_RATES:
                used_referee_rates = True

    # Closing honesty note — one of four states.
    if used_availability and used_referee:
        note = (
            "Nota: probs/cuotas/ELO/heurísticas del modelo; bajas y árbitro son "
            "contexto tipster, no input del pick."
        )
        if used_referee and not used_referee_rates:
            note = (
                "Nota: probs/cuotas/ELO/heurísticas del modelo; bajas y nombre de "
                "árbitro son contexto tipster (sin tasas citables), no input del pick."
            )
        parts.append(note)
    elif used_availability:
        parts.append(
            "Nota: probs/cuotas/ELO/heurísticas del modelo; las bajas anteriores son "
            "contexto tipster, no input del pick."
        )
    elif used_referee:
        if used_referee_rates:
            parts.append(
                "Nota: probs/cuotas/ELO/heurísticas del modelo; el bloque de árbitro "
                "es contexto tipster (tasas del catálogo local), no input del pick."
            )
        else:
            parts.append(
                "Nota: probs/cuotas/ELO/heurísticas del modelo; el nombre del árbitro "
                "es contexto tipster (sin tasas citables aún), no input del pick."
            )
    else:
        parts.append(
            "Nota: esta justificación usa datos del modelo (probs, cuotas, ELO, heurísticas). "
            "Sin bajas relevantes ni árbitro asignado (o sin ficheros) para este partido."
        )

    text = " ".join(parts)
    return {
        "match_id": match_id,
        "date": str(row.get("Date") or "").split(" ")[0],
        "league": league,
        "home": home,
        "away": away,
        "pred_1x2": pick,
        "pred_ou": ou,
        "justification": text,
        "used_availability": used_availability,
        "used_referee": used_referee,
        "used_referee_rates": used_referee_rates,
        "version": JUSTIFY_VERSION,
    }


def latest_pred_date() -> str | None:
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "predictions_*.csv")))
    if not files:
        return None
    name = os.path.basename(files[-1])
    return name.replace("predictions_", "").replace(".csv", "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="ISO date of predictions_*.csv (default: latest)")
    parser.add_argument("--print", action="store_true", help="Print justifications to stdout")
    parser.add_argument(
        "--txt",
        action="store_true",
        default=True,
        help="Also write a .txt report (default: on)",
    )
    parser.add_argument("--no-txt", action="store_true", help="Skip the .txt report")
    args = parser.parse_args()

    date = args.date or latest_pred_date()
    if not date:
        print("[-] No predictions_*.csv found under output/", file=sys.stderr)
        return 1

    pred_path = os.path.join(OUTPUT_DIR, f"predictions_{date}.csv")
    if not os.path.isfile(pred_path):
        print(f"[-] Missing {pred_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(pred_path)
    availability = load_availability(date)
    if availability:
        print(f"[*] Loaded availability for {len(availability)} matches "
              f"(output/availability_{date}.json)")
    else:
        print(f"[*] No availability_{date}.json — justificaciones sin bloque de bajas")

    referees = load_referees_assignment(date)
    if referees:
        n_named = sum(1 for v in referees.values() if (v or {}).get("referee_name"))
        print(f"[*] Loaded referees for {len(referees)} matches "
              f"({n_named} with name) (output/referees_{date}.json)")
    else:
        print(f"[*] No referees_{date}.json — justificaciones sin bloque de árbitro")

    referee_rates = load_referee_rate_index()
    if referee_rates:
        n_thick = sum(
            1 for s in referee_rates.values()
            if int(s.get("n_with_cards") or 0) >= _MIN_MATCHES_FOR_REF_RATES
        )
        print(f"[*] Referee catalog: {len(referee_rates)} ids, "
              f"{n_thick} with ≥{_MIN_MATCHES_FOR_REF_RATES} card matches "
              f"(for rate citations)")
    else:
        print("[*] No data_sets/referees/referee_matches.csv — "
              "árbitro solo por nombre si hay assignment")

    items = [
        justify_row(r.to_dict(), availability, referees, referee_rates)
        for _, r in df.iterrows()
    ]
    n_with_bajas = sum(1 for m in items if m.get("used_availability"))
    n_with_ref = sum(1 for m in items if m.get("used_referee"))
    n_with_rates = sum(1 for m in items if m.get("used_referee_rates"))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_json = os.path.join(OUTPUT_DIR, f"justifications_{date}.json")
    payload = {
        "date": date,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": os.path.basename(pred_path),
        "availability_source": (
            f"availability_{date}.json" if availability else None
        ),
        "referees_source": (
            f"referees_{date}.json" if referees else None
        ),
        "referee_catalog": (
            "data_sets/referees/referee_matches.csv" if referee_rates else None
        ),
        "version": JUSTIFY_VERSION,
        "count": len(items),
        "with_availability": n_with_bajas,
        "with_referee": n_with_ref,
        "with_referee_rates": n_with_rates,
        "matches": items,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[+] Wrote {len(items)} justifications "
          f"({n_with_bajas} con bajas, {n_with_ref} con árbitro, "
          f"{n_with_rates} con tasas) → {out_json}")

    if args.txt and not args.no_txt:
        out_txt = os.path.join(OUTPUT_DIR, f"justifications_{date}.txt")
        lines = [f"Justificaciones · {date} · {len(items)} partidos", ""]
        for i, m in enumerate(items, 1):
            lines.append(f"--- {i}. {m['home']} vs {m['away']} ---")
            lines.append(m["justification"].replace("**", ""))
            lines.append("")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[+] Wrote text report → {out_txt}")

    if args.print:
        for m in items:
            print()
            print(m["justification"].replace("**", ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
