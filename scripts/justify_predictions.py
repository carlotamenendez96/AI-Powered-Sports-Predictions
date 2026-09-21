#!/usr/bin/env python3
"""Level-1 prediction justifications (template text, no LLM / no news).

Reads `output/predictions_YYYY-MM-DD.csv` and writes a companion JSON (+ optional
plain-text report) explaining each pick in Spanish from fields the model
already produces: probs, odds, EV, ELO, heuristic Adj Logs.

This is intentionally boring and honest — it does NOT invent injuries,
referees, or corners. Later layers (availability, referee feeds, LLM prose)
can plug into the same output schema.

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


def justify_row(row: dict) -> dict:
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

    parts.append(
        "Nota: esta justificación solo usa datos del modelo (probs, cuotas, ELO, heurísticas). "
        "No incluye lesiones, alineaciones ni árbitro."
    )

    text = " ".join(parts)
    return {
        "match_id": str(row.get("match_id") or "").strip() or None,
        "date": str(row.get("Date") or "").split(" ")[0],
        "league": league,
        "home": home,
        "away": away,
        "pred_1x2": pick,
        "pred_ou": ou,
        "justification": text,
        "version": "level1-template-v1",
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
    items = [justify_row(r.to_dict()) for _, r in df.iterrows()]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_json = os.path.join(OUTPUT_DIR, f"justifications_{date}.json")
    payload = {
        "date": date,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": os.path.basename(pred_path),
        "version": "level1-template-v1",
        "count": len(items),
        "matches": items,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[+] Wrote {len(items)} justifications → {out_json}")

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
