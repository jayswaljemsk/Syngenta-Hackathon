"""
explainer.py -- SHAP-based deterministic top-3 reason-string generator.

For each outlet row, compute SHAP values against the trained priority ranker,
take the features with the highest positive contribution, and render each via
a hand-crafted human-readable template that uses ONLY the row's actual feature
values (no hallucination, no LLM).

CLI:
    python explainer.py --date 2026-02-15 --rep-id REP_0001

Importable:
    from explainer import explain_row
    reasons: list[str] = explain_row(row, top_k=3)

Honest scoping (folded into MODELS.md):
- SHAP values are computed with shap.TreeExplainer against the XGBoost ranker
  from priority_model. They reflect per-row contributions to the RAW score
  (pre-batch-normalization). Positive SHAP means the feature pushed this
  outlet UP in the ranking.
- Templates are deterministic. Each template gates on a per-feature threshold
  so we don't cite e.g. low pest pressure as a "reason to visit".
- Falls back to a small heuristic generator (similar to MOCK_REASONS) when
  the ranker isn't trained or the explainer fails. The API stays up.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


# ---------------------------------------------------------------------------
# Lazy explainer
# ---------------------------------------------------------------------------
_explainer = None
_feats = None
_loaded_failed = False


def _load_explainer():
    """Build and cache shap.TreeExplainer against the trained ranker."""
    global _explainer, _feats, _loaded_failed
    if _explainer is not None or _loaded_failed:
        return _explainer, _feats
    try:
        from priority_model import _load as _load_model  # private helper exposes booster+feats
        import shap
        booster, feats = _load_model()
        _explainer = shap.TreeExplainer(booster)
        _feats = list(feats)
    except Exception as e:
        print(f"[explainer] disabled (ranker/SHAP load failed): {e}")
        _loaded_failed = True
        _explainer = None
        _feats = None
    return _explainer, _feats


def is_available() -> bool:
    expl, _ = _load_explainer()
    return expl is not None


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
STAGE_NAMES = {
    0.0: "sowing", 1.0: "vegetative", 1.5: "tuberization",
    2.0: "flowering", 2.5: "pod filling", 3.0: "harvest",
}


def _stage_name(v) -> str:
    try:
        x = float(v)
    except Exception:
        return "vegetative"
    closest = min(STAGE_NAMES.keys(), key=lambda k: abs(k - x))
    return STAGE_NAMES[closest]


def _render(feat: str, row: pd.Series) -> Optional[str]:
    """Render a reason string for the given feature, using the row's value.
    Returns None when the value isn't strong enough to cite as a positive reason."""
    crop = str(row.get("dominant_crop_in_radius_5km", "wheat")).lower()
    crop_cap = crop.capitalize()
    sku = str(row.get("recommended_sku_pick", "recommended SKU"))
    district = str(row.get("district", "this district"))
    v = row.get(feat)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    val = float(v)

    if feat == "pest_pressure_idx":
        if val < 0.35: return None
        return f"{crop_cap} pest pressure {val:.2f} -- elevated for {district} this week"

    if feat == "stock_of_recommended_sku":
        if val > 50: return None
        return f"Only {int(val)} units of {sku} in stock"

    if feat == "safety_stock_gap":
        if val <= 0: return None
        return f"{int(val)} units below safety stock on {sku}"

    if feat == "days_since_last_visit":
        if val < 14: return None
        if val >= 999:
            return "This tehsil has no recorded visits this season"
        return f"{int(val)} days since last visit to this tehsil"

    if feat == "spray_window_open_flag":
        if val < 0.5: return None
        return f"Spray window open in 48h -- actionable now for {crop}"

    if feat == "rain_mm_next_48h":
        if val < 2: return None
        return f"{val:.0f}mm rain expected in 48h -- act before spray window closes"

    if feat == "rain_probability_72h":
        if val < 0.5: return None
        return f"High rain probability ({val:.0%}) over next 72h"

    if feat == "recent_stockout_flag":
        if val < 0.5: return None
        return f"Recent stockout on {sku} in last 4 weeks"

    if feat == "ndvi_current":
        if 0.35 <= val <= 0.55:
            return None  # neutral range
        if val > 0.55:
            return f"Crop vigor NDVI {val:.2f} -- healthy stand, demand likely"
        return f"Crop vigor NDVI {val:.2f} -- low, scout for stress"

    if feat == "ndvi_delta_14d":
        if abs(val) < 0.03: return None
        if val > 0:
            return f"NDVI up {val:+.2f} over last 14 days -- crop accelerating"
        return f"NDVI down {val:+.2f} over last 14 days -- check for stress"

    if feat == "crop_match_score":
        if val < 0.3: return None
        return f"{crop_cap} catchment fit -- recommended SKU is on-target"

    if feat == "growth_stage_estimate":
        stage = _stage_name(val)
        return f"{crop_cap} in {stage} stage -- treatment window"

    if feat == "days_to_next_critical_stage":
        if val > 21 or val <= 0: return None
        return f"{int(val)} days to next critical growth stage"

    if feat == "total_sales_180d":
        if val < 5000: return None
        return f"Active outlet -- INR {int(val):,} in sales over last 180 days"

    if feat == "avg_order_value_inr":
        if val < 2000: return None
        return f"Avg order value INR {int(val):,} -- premium customer"

    if feat == "visits_last_90d":
        if val == 0:
            return "Tehsil has had no visits in last 90 days"
        if val >= 6:
            return f"Tehsil visited {int(val)} times in last 90 days -- engaged"
        return None

    if feat == "last_purchase_value_inr":
        if val < 2000: return None
        return f"Last purchase INR {int(val):,}"

    if feat == "rep_close_rate_at_outlet":
        if val < 0.10: return None
        return f"Rep close rate {val:.0%} in this tehsil"

    if feat == "rep_visits_to_outlet_lifetime":
        if val < 3: return None
        return f"Rep has visited this tehsil {int(val)} times historically"

    if feat == "rep_familiarity_score":
        if val < 0.6: return None
        return "Strong rep familiarity with this catchment"

    if feat == "weather_volatility_7d":
        if val < 3: return None
        return f"Volatile weather last 7 days (rain std {val:.1f}mm)"

    if feat == "temp_anomaly_c":
        if abs(val) < 2: return None
        sign = "+" if val > 0 else ""
        return f"Temp anomaly {sign}{val:.1f}C vs 7-day mean"

    if feat == "mandi_price_anomaly":
        if abs(val) < 0.05: return None
        sign = "+" if val > 0 else ""
        return f"Mandi {crop} price {sign}{val:.0%} vs national average"

    if feat == "is_market_day_flag":
        if val < 0.5: return None
        return "Today is a market day -- high foot traffic expected"

    if feat == "days_since_competitor_visit":
        if val > 30 or val < 7: return None
        return f"Competitor last seen {int(val)} days ago -- contest window"

    if feat == "days_of_cover":
        if val > 14 or val <= 0: return None
        return f"Stock cover {val:.0f} days at current velocity"

    # Suppressed (not useful as a spoken reason): day_of_week, week_of_month,
    # week_of_season, rep_avg_dwell_minutes.
    return None


# ---------------------------------------------------------------------------
# Fallback (used if explainer is unavailable or all renderings return None)
# ---------------------------------------------------------------------------
def _fallback(row: pd.Series, top_k: int) -> list[str]:
    reasons = []
    crop = str(row.get("dominant_crop_in_radius_5km", "wheat")).capitalize()
    sku = str(row.get("recommended_sku_pick", "recommended SKU"))
    if row.get("pest_pressure_idx", 0) > 0.55:
        reasons.append(
            f"{crop} pest pressure {row['pest_pressure_idx']:.2f} -- elevated for this district this week"
        )
    if row.get("safety_stock_gap", 0) > 0:
        reasons.append(f"{int(row['safety_stock_gap'])} units below safety stock on {sku}")
    elif row.get("recent_stockout_flag", 0):
        reasons.append(f"Recent stockout on {sku} in last 4 weeks")
    if row.get("days_since_last_visit", 0) >= 21:
        reasons.append(f"{int(row['days_since_last_visit'])} days since last visit to this tehsil")
    if row.get("spray_window_open_flag", 0) and row.get("pest_pressure_idx", 0) > 0.4:
        reasons.append(f"Spray window open in 48h -- actionable now for {crop.lower()}")
    if not reasons:
        reasons = [
            f"Steady {crop.lower()} demand in catchment",
            "Active retailer, regular POS flow",
            f"Stock at {int(row.get('stock_of_recommended_sku', 0))} units on recommended SKU",
        ]
    return reasons[:top_k]


# ---------------------------------------------------------------------------
# Public: explain_row
# ---------------------------------------------------------------------------
def explain_row(row: pd.Series, top_k: int = 3) -> list[str]:
    """Return up to top_k human-readable reason strings, ordered by SHAP contribution."""
    expl, feats = _load_explainer()
    if expl is None or feats is None:
        return _fallback(row, top_k)

    try:
        X = pd.DataFrame([[float(row[c]) for c in feats]], columns=feats)
        sv = expl.shap_values(X)
        if hasattr(sv, "shape") and sv.ndim == 2:
            sv = sv[0]
        else:
            sv = np.asarray(sv).flatten()
    except Exception as e:
        print(f"[explainer] shap_values failed on row {row.get('retailer_id','?')}: {e}")
        return _fallback(row, top_k)

    # Sort features by SHAP value descending (most positive first)
    contribs = sorted(zip(feats, sv), key=lambda kv: kv[1], reverse=True)

    out: list[str] = []
    for feat, shap_val in contribs:
        if shap_val <= 0:
            break  # all remaining are <= 0
        text = _render(feat, row)
        if text and text not in out:
            out.append(text)
            if len(out) >= top_k:
                break

    if not out:
        return _fallback(row, top_k)

    # Pad with heuristics if SHAP top-3 didn't render enough templates
    if len(out) < top_k:
        for f in _fallback(row, top_k):
            if f not in out:
                out.append(f)
            if len(out) >= top_k:
                break
    return out[:top_k]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="2026-02-15")
    p.add_argument("--rep-id", default="REP_0001")
    args = p.parse_args()

    from features import get_dataset_root
    df = pd.read_parquet(DATA / f"features_{args.date}.parquet")
    reps = pd.read_csv(get_dataset_root() / "reps_territory.csv")
    rep_row = reps[reps["rep_id"] == args.rep_id].iloc[0]
    outlets = df[df["territory_id"] == rep_row["territory_id"]].reset_index(drop=True)

    print(f"[explainer] available: {is_available()}")
    print(f"[explainer] rep={args.rep_id} territory={rep_row['territory_id']} outlets={len(outlets)}\n")
    for _, row in outlets.iterrows():
        reasons = explain_row(row, top_k=3)
        print(f"{row['retailer_id']}  {row['name']}")
        for r in reasons:
            print(f"  - {r}")
        print()


if __name__ == "__main__":
    main()