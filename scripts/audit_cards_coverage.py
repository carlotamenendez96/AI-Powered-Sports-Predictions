#!/usr/bin/env python3
"""Paso 6 — cobertura HY/AY/HR/AR (+ Referee) en MatchHistory.

Solo lectura. Imprime una tabla por fichero y un resumen por liga, pensada
para pegarse en docs/enriched_match_data_roadmap.md §Paso 6.

    python3 scripts/audit_cards_coverage.py
    python3 scripts/audit_cards_coverage.py --markdown
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHHISTORY = os.path.join(ROOT, "data_sets", "MatchHistory")
CARD_LINE = 3.5


def _league_label(basename: str) -> str:
    """Stem without season suffix (ENG-Premier_League_25-26 → ENG-Premier_League)."""
    name = basename[:-4] if basename.endswith(".csv") else basename
    # Drop trailing _AA-BB season codes
    import re
    m = re.search(r"_(\d{2})-(\d{2})$", name)
    return name[: m.start()] if m else name


def scan() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(MATCHHISTORY, "*.csv"))):
        base = os.path.basename(path)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            rows.append({
                "file": base, "league": _league_label(base), "n": -1,
                "has_hy_ay": False, "pct_hy_ay": 0.0, "mean_yellows": None,
                "p_over": None, "has_referee": False, "error": str(e),
            })
            continue
        has_hy = "HY" in df.columns and "AY" in df.columns
        has_ref = "Referee" in df.columns
        n = len(df)
        if has_hy:
            hy = pd.to_numeric(df["HY"], errors="coerce")
            ay = pd.to_numeric(df["AY"], errors="coerce")
            tot = hy + ay
            nn = int(tot.notna().sum())
            pct = 100.0 * nn / n if n else 0.0
            mean = float(tot.mean()) if nn else None
            pover = float((tot > CARD_LINE).mean()) if nn else None
        else:
            nn, pct, mean, pover = 0, 0.0, None, None
        rows.append({
            "file": base,
            "league": _league_label(base),
            "n": n,
            "n_with_cards": nn if has_hy else 0,
            "has_hy_ay": has_hy,
            "pct_hy_ay": round(pct, 1),
            "mean_yellows": round(mean, 2) if mean is not None else None,
            "p_over": round(pover * 100, 1) if pover is not None else None,
            "has_referee": has_ref,
            "error": None,
        })
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame, markdown: bool = False) -> None:
    usable = df[df["has_hy_ay"] & (df["n_with_cards"] > 0)]
    print(f"MatchHistory: {len(df)} ficheros; "
          f"{len(usable)} con HY/AY utilizables; "
          f"{int(usable['n_with_cards'].sum())} filas con tarjetas; "
          f"CARD_LINE={CARD_LINE}")
    print()
    if markdown:
        print("| Fichero | n | %HY+AY | mean(HY+AY) | P(>3.5) | Referee |")
        print("| --- | ---: | ---: | ---: | ---: | --- |")
        for _, r in df.iterrows():
            mean = f"{r['mean_yellows']:.2f}" if r["mean_yellows"] is not None else "—"
            pover = f"{r['p_over']:.1f}%" if r["p_over"] is not None else "—"
            ref = "sí" if r["has_referee"] else "no"
            pct = f"{r['pct_hy_ay']:.1f}%" if r["has_hy_ay"] else "0%"
            print(f"| `{r['file']}` | {r['n']} | {pct} | {mean} | {pover} | {ref} |")
    else:
        hdr = f"{'REF':3} {'n':>5} {'%HY+AY':>7} {'mean':>6} {'P>3.5':>7}  FICHERO"
        print(hdr)
        for _, r in df.iterrows():
            flag = "SI" if r["has_referee"] else "no"
            mean_ok = r["mean_yellows"] is not None and r["mean_yellows"] == r["mean_yellows"]
            pover_ok = r["p_over"] is not None and r["p_over"] == r["p_over"]
            mean = f"{r['mean_yellows']:6.2f}" if mean_ok else "   — "
            pover = f"{r['p_over']:5.1f}%" if pover_ok else "   — "
            pct = f"{r['pct_hy_ay']:5.1f}%" if r["has_hy_ay"] else "  0.0%"
            print(f"{flag:3} {r['n']:5d} {pct:>7} {mean} {pover:>7}  {r['file']}")

    print()
    # League aggregate
    by = (usable.groupby("league", as_index=False)
          .agg(n=("n_with_cards", "sum"),
               mean_yellows=("mean_yellows", "mean"),
               p_over=("p_over", "mean"),
               has_referee=("has_referee", "any")))
    by = by.sort_values("n", ascending=False)
    print(f"Ligas con tarjetas: {len(by)}")
    if markdown:
        print()
        print("| Liga | filas | mean(HY+AY) | P(>3.5) | Referee histórico |")
        print("| --- | ---: | ---: | ---: | --- |")
        for _, r in by.iterrows():
            print(f"| {r['league']} | {int(r['n'])} | "
                  f"{r['mean_yellows']:.2f} | {r['p_over']:.1f}% | "
                  f"{'sí (ENG/SCO)' if r['has_referee'] else 'no'} |")
    else:
        for _, r in by.iterrows():
            ref = "ref" if r["has_referee"] else "—"
            print(f"  {r['league']:30s} n={int(r['n']):5d}  "
                  f"mean={r['mean_yellows']:.2f}  P(>3.5)={r['p_over']:.1f}%  {ref}")

    no_cards = df[~df["has_hy_ay"]]
    print()
    print(f"Sin columnas HY/AY (extra leagues /new/): {len(no_cards)} ficheros — "
          "no entrenables para tarjetas.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true",
                        help="Emitir tablas en Markdown (para el roadmap).")
    args = parser.parse_args()
    if not os.path.isdir(MATCHHISTORY):
        print(f"[-] Missing {MATCHHISTORY}", file=sys.stderr)
        return 1
    df = scan()
    print_table(df, markdown=args.markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
