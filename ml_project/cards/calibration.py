"""Global Platt calibrator for the cards over-line head (Paso 6).

ONE global (a, b) on P(over) — NOT per-league Platt. Football 1X2 already
burned on per-league fits with tiny n; cards has even thinner slices outside
ENG/SCO.

Reuses fit_platt_binary / apply_platt_binary / MIN_PLATT_SLOPE from
ml_project.calibration.fit. Guards reject a calibrator that flattens or
destroys discrimination.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ml_project.calibration.fit import (
    MIN_PLATT_SLOPE,
    apply_platt_binary,
    fit_platt_binary,
)

_EPS = 1e-6


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    y = y.astype(int)
    acc = float(((p >= 0.5).astype(int) == y).mean())
    try:
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "brier": round(float(brier_score_loss(y, p)), 6),
        "log_loss": round(float(log_loss(y, p)), 6),
        "accuracy": round(acc, 4),
        "auc": None if auc != auc else round(auc, 4),
    }


def fit_global_cards_calibrator(
    oof_probs: np.ndarray,
    targets: np.ndarray,
    min_slope: float = MIN_PLATT_SLOPE,
) -> Tuple[Optional[dict], Optional[str]]:
    """Fit global Platt on OOF P(over). Returns (payload, rejection_reason).

    payload is None when a guard rejects the fit (caller keeps prior file /
    serves raw probs).
    """
    mask = ~np.isnan(oof_probs) & ~np.isnan(targets)
    p = np.asarray(oof_probs)[mask].astype(float)
    y = np.asarray(targets)[mask].astype(int)
    if len(p) < 100:
        return None, f"too few OOF rows ({len(p)})"
    if len(np.unique(y)) < 2:
        return None, "degenerate target (single class)"

    a, b = fit_platt_binary(p, y)
    cal = apply_platt_binary(p, a, b)
    before = _metrics(p, y)
    after = _metrics(cal, y)

    reasons = []
    if a < min_slope:
        reasons.append(f"slope {a:.4f} < {min_slope}")
    if after["accuracy"] < before["accuracy"]:
        reasons.append(
            f"accuracy {before['accuracy']:.4f} → {after['accuracy']:.4f}")
    if (after["auc"] is not None and before["auc"] is not None
            and after["auc"] < before["auc"] - 0.01):
        reasons.append(f"auc {before['auc']:.4f} → {after['auc']:.4f}")
    if reasons:
        return None, "; ".join(reasons)

    return {
        "market": "cards",
        "kind": "global_platt",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n": int(len(p)),
        "platt": {"over": {"a": round(float(a), 4), "b": round(float(b), 4)}},
        "before": before,
        "after": after,
        "brier_delta": round(after["brier"] - before["brier"], 6),
        "worst_slope": round(float(a), 4),
        "accepted": True,
        "min_slope_gate": min_slope,
    }, None


def apply_cards_calibrator(p_over: float, cal_data: dict,
                           enabled: bool = True) -> Tuple[float, bool, str]:
    """Apply global Platt. Returns (p_cal, applied, source)."""
    if not enabled or not cal_data or not cal_data.get("accepted"):
        return float(p_over), False, ""
    platt = cal_data.get("platt", {}).get("over")
    if not platt:
        return float(p_over), False, ""
    a, b = float(platt["a"]), float(platt["b"])
    cal = float(apply_platt_binary(np.array([p_over]), a, b)[0])
    cal = max(_EPS, min(1.0 - _EPS, cal))
    return cal, True, "global"


def load_cards_calibration(path: str = "data_sets/cards_calibration.json") -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_cards_calibration(payload: dict,
                           path: str = "data_sets/cards_calibration.json") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def fit_and_save_from_oof(
    oof_path: str = "models/oof_cards.npz",
    out_path: str = "data_sets/cards_calibration.json",
) -> dict:
    """CLI helper: load OOF from train_cards, fit, persist (or write rejection)."""
    data = np.load(oof_path, allow_pickle=True)
    payload, reason = fit_global_cards_calibrator(data["oof"], data["y"])
    if payload is None:
        rejected = {
            "market": "cards",
            "kind": "global_platt",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "accepted": False,
            "rejection_reason": reason,
        }
        save_cards_calibration(rejected, out_path)
        print(f"[cards cal] REJECTED: {reason} → {out_path}")
        return rejected
    save_cards_calibration(payload, out_path)
    print(f"[cards cal] ACCEPTED: a={payload['platt']['over']['a']} "
          f"b={payload['platt']['over']['b']}  "
          f"ΔBrier={payload['brier_delta']:+.5f} → {out_path}")
    return payload


if __name__ == "__main__":
    fit_and_save_from_oof()
