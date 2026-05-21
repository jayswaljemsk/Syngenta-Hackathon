"""
main.py -- FastAPI app exposing CONTRACT.md Section 4 endpoints.

Endpoints:
- GET  /health                            -- liveness + model-trained flags
- GET  /plan/today?rep_id=X&date=Y        -- ranked outlets + route + recs + anomalies
- GET  /nba/{outlet_id}?date=Y            -- next-best-product for one outlet
- POST /outcome                           -- log a visit outcome (bandit instrumentation)

Loads features_<date>.parquet on demand. Caches in-memory per date.

Wired-in real modules:
- priority_model.score      -- XGBoost LambdaMART; artifacts in models/
- route_solver.solve_route  -- OR-Tools VRP on Haversine matrix, depot = centroid

Remaining MOCK blocks (each replaced as its dedicated module ships):
- MOCK_NBA         -> nba_engine.py       (rule filter + logistic + templated justification)
- MOCK_REASONS     -> explainer.py        (SHAP top-3 reason strings)
- MOCK_ANOMALIES   -> anomaly_detector.py (IsolationForest + STL)

Outlets are returned in VRP-optimal visit order (route_position = 1..N along
the optimal path). priority_score remains the per-outlet priority axis. So
priority and route are decoupled per CONTRACT.md Section 4.1.

Run:
    cd backend
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Pull the SKU -> target-crops map from features.py so it's in one place.
from features import SKU_CROPS, get_dataset_root

# Real priority ranker if trained; graceful fallback to MOCK_PRIORITY if not.
try:
    from priority_model import score as priority_score_real, is_trained as priority_is_trained
except Exception:
    priority_score_real = None
    def priority_is_trained() -> bool:
        return False

# Real VRP solver if OR-Tools is present; graceful fallback to identity order.
try:
    from route_solver import solve_route as route_solve_real
    _ROUTE_SOLVER_OK = True
except Exception:
    route_solve_real = None
    _ROUTE_SOLVER_OK = False

# Real SHAP-based explainer if ranker + shap are present; falls back internally.
try:
    from explainer import explain_row as explain_row_real, is_available as explainer_is_available
    _EXPLAINER_IMPORTED = True
except Exception:
    explain_row_real = None
    def explainer_is_available() -> bool:
        return False
    _EXPLAINER_IMPORTED = False

# Real NBA engine (rule filter + calibrated logistic) if trained; otherwise
# falls back internally to a rule-only pick using SKU_PRODUCT_META.
try:
    from nba_engine import recommend as nba_recommend_real, is_trained as nba_is_trained
    _NBA_IMPORTED = True
except Exception:
    nba_recommend_real = None
    def nba_is_trained() -> bool:
        return False
    _NBA_IMPORTED = False

# Real anomaly detector if IsolationForest is trained; falls back to MOCK_ANOMALIES.
try:
    from anomaly_detector import detect as anomaly_detect_real, is_trained as anomaly_is_trained
    _ANOMALY_IMPORTED = True
except Exception:
    anomaly_detect_real = None
    def anomaly_is_trained() -> bool:
        return False
    _ANOMALY_IMPORTED = False


# ---------------------------------------------------------------------------
# Paths and app setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

app = FastAPI(
    title="Syngenta Field Copilot API",
    version="0.6.0",
    description="AI-Guided Field Force Intelligence -- backend (Krishna)",
)

# CORS so Jems' Next.js dev server (localhost:3000) can hit this on :8000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_features_cache: dict[str, pd.DataFrame] = {}
_reps_cache: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Product catalog meta (product_id, category, base INR uplift)
# ---------------------------------------------------------------------------
SKU_PRODUCT_META = {
    "Actara 25 WG":      ("ACT_25",  "Insecticide", 6200),
    "Alto 5 SC":         ("ALT_5",   "Fungicide",   4100),
    "Amistar 250 SC":    ("AMS_250", "Fungicide",   5500),
    "Axial 50 EC":       ("AXI_50",  "Herbicide",   3200),
    "Cruiser 350 FS":    ("CRU_350", "Insecticide", 2800),
    "Kavach 75 WP":      ("KAV_75",  "Fungicide",   3400),
    "Movondo":           ("MOV",     "Insecticide", 4900),
    "Score 250 EC":      ("SCR_250", "Fungicide",   5100),
    "Tilt 250 EC":       ("TIL_250", "Fungicide",   3700),
    "Topik 15 WP":       ("TPK_15",  "Herbicide",   3900),
    "Vertimec 1.8 EC":   ("VER_18",  "Insecticide", 4400),
    "Vibrance Integral": ("VIB",     "Fungicide",   3100),
}


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
def load_features(date_str: str) -> pd.DataFrame:
    if date_str in _features_cache:
        return _features_cache[date_str]
    fp = DATA / f"features_{date_str}.parquet"
    if not fp.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No features file for {date_str}. Generate first: "
                f"python features.py --date {date_str}"
            ),
        )
    df = pd.read_parquet(fp)
    _features_cache[date_str] = df
    return df


def load_reps() -> pd.DataFrame:
    global _reps_cache
    if _reps_cache is None:
        root = get_dataset_root()
        _reps_cache = pd.read_csv(root / "reps_territory.csv")
    return _reps_cache


# ---------------------------------------------------------------------------
# Priority scoring: real ranker if trained, MOCK_PRIORITY fallback
# ---------------------------------------------------------------------------
def MOCK_PRIORITY(outlets: pd.DataFrame) -> pd.Series:
    """Hand-crafted weighted sum. Used only if the trained ranker is absent."""
    return (
        0.30 * outlets["pest_pressure_idx"]
        + 0.20 * outlets["recent_stockout_flag"].astype(float)
        + 0.15 * (outlets["safety_stock_gap"].clip(0, 8) / 8)
        + 0.15 * (outlets["days_since_last_visit"].clip(0, 30) / 30)
        + 0.10 * outlets["crop_match_score"]
        + 0.10 * outlets["spray_window_open_flag"].astype(float)
    ).clip(0, 1)


def compute_priority(outlets: pd.DataFrame) -> pd.Series:
    if priority_is_trained() and priority_score_real is not None:
        try:
            return priority_score_real(outlets)
        except Exception as e:
            print(f"[plan_today] priority ranker failed, MOCK_PRIORITY fallback: {e}")
            return MOCK_PRIORITY(outlets)
    return MOCK_PRIORITY(outlets)


# ---------------------------------------------------------------------------
# Route solving: OR-Tools VRP if available, identity-order fallback
# ---------------------------------------------------------------------------
def compute_route_order(outlets: pd.DataFrame) -> tuple[list[int], float]:
    """Return (order_indices, total_distance_m). Identity fallback on failure."""
    if not _ROUTE_SOLVER_OK or route_solve_real is None:
        return list(range(len(outlets))), float("inf")
    try:
        return route_solve_real(outlets)
    except Exception as e:
        print(f"[plan_today] route_solver failed, identity-order fallback: {e}")
        return list(range(len(outlets))), float("inf")


# ---------------------------------------------------------------------------
# Reason strings: SHAP-based explainer if available, MOCK_REASONS fallback
# ---------------------------------------------------------------------------
def compute_reasons(row: pd.Series, top_k: int = 3) -> list[str]:
    if _EXPLAINER_IMPORTED and explain_row_real is not None:
        try:
            return explain_row_real(row, top_k=top_k)
        except Exception as e:
            print(f"[plan_today] explainer failed, MOCK_REASONS fallback: {e}")
    return MOCK_REASONS(row)


# ---------------------------------------------------------------------------
# NBA: real engine if trained, MOCK_NBA fallback
# ---------------------------------------------------------------------------
def compute_nba(row: pd.Series, top_n: int = 4) -> tuple[dict, list[dict]]:
    """Return (primary_dict, alternates_list). Falls back to (MOCK_NBA, [])."""
    if _NBA_IMPORTED and nba_recommend_real is not None:
        try:
            return nba_recommend_real(row, top_n=top_n)
        except Exception as e:
            print(f"[plan_today] nba_engine failed, MOCK_NBA fallback: {e}")
    return MOCK_NBA(row), []


# ---------------------------------------------------------------------------
# Anomalies: IsolationForest+STL detector if trained, MOCK_ANOMALIES fallback
# ---------------------------------------------------------------------------
def compute_anomalies(rep_outlets: pd.DataFrame, date_str: str) -> list[dict]:
    if _ANOMALY_IMPORTED and anomaly_detect_real is not None:
        try:
            return anomaly_detect_real(rep_outlets, date_str)
        except Exception as e:
            print(f"[plan_today] anomaly_detector failed, MOCK_ANOMALIES fallback: {e}")
    return MOCK_ANOMALIES(rep_outlets)


# ---------------------------------------------------------------------------
# Remaining MOCK blocks (replaced as modules ship)
# ---------------------------------------------------------------------------
def MOCK_REASONS(row: pd.Series) -> list[str]:
    """Top-3 reason strings. Replaced by explainer.py (SHAP-based)."""
    reasons = []
    crop = str(row["dominant_crop_in_radius_5km"]).capitalize()
    if row["pest_pressure_idx"] > 0.55:
        reasons.append(
            f"{crop} pest pressure {row['pest_pressure_idx']:.2f} -- elevated for this district this week"
        )
    if row["safety_stock_gap"] > 0:
        reasons.append(
            f"{int(row['safety_stock_gap'])} units below safety stock on {row['recommended_sku_pick']}"
        )
    elif row["recent_stockout_flag"]:
        reasons.append(f"Recent stockout on {row['recommended_sku_pick']} in last 4 weeks")
    if row["days_since_last_visit"] >= 21:
        reasons.append(f"{int(row['days_since_last_visit'])} days since last visit to this tehsil")
    if row["spray_window_open_flag"] and row["pest_pressure_idx"] > 0.4:
        reasons.append(f"Spray window open in 48h -- actionable now for {crop.lower()}")
    if not reasons:
        reasons = [
            f"Steady {crop.lower()} demand in catchment",
            "Active retailer, regular POS flow",
            f"Stock at {int(row['stock_of_recommended_sku'])} units on recommended SKU",
        ]
    return reasons[:3]


def MOCK_NBA(row: pd.Series) -> dict:
    """recommended_product block. Replaced by nba_engine.py."""
    sku = row["recommended_sku_pick"]
    pid, category, base_uplift = SKU_PRODUCT_META.get(sku, ("UNK", "Crop Protection", 3000))
    crop = str(row["dominant_crop_in_radius_5km"]).lower()

    parts = [f"{crop.capitalize()} pest pressure {row['pest_pressure_idx']:.2f} in this district"]
    parts.append(f"current stock {int(row['stock_of_recommended_sku'])} units")
    if row["spray_window_open_flag"]:
        parts.append("spray window open in 48h")
    if row["safety_stock_gap"] > 0:
        parts.append(f"{int(row['safety_stock_gap'])} units below safety threshold")
    justification = "; ".join(parts) + "."

    uplift = int(base_uplift * (1 + 0.5 * row["pest_pressure_idx"]) * (1 + row["safety_stock_gap"] / 10))

    return {
        "product_id": pid,
        "product_name": sku,
        "category": category,
        "expected_uplift_inr": uplift,
        "justification": justification,
    }


def MOCK_ANOMALIES(rep_outlets: pd.DataFrame) -> list[dict]:
    """List of anomaly objects. Replaced by anomaly_detector.py."""
    out = []
    high_pest = rep_outlets[rep_outlets["pest_pressure_idx"] > 0.65]
    if len(high_pest) >= 2:
        crop = str(high_pest.iloc[0]["dominant_crop_in_radius_5km"]).capitalize()
        out.append({
            "type": "elevated_pest_pressure",
            "severity": "medium",
            "description": (
                f"{crop} pest pressure elevated in {len(high_pest)} outlets in this territory "
                f"(mean {high_pest['pest_pressure_idx'].mean():.2f})"
            ),
            "affected_outlets": high_pest["retailer_id"].head(3).tolist(),
        })
    stockouts = rep_outlets[rep_outlets["recent_stockout_flag"] == 1]
    if len(stockouts) >= 2:
        out.append({
            "type": "recurring_stockout_cluster",
            "severity": "low",
            "description": f"Recent stockouts at {len(stockouts)} outlets -- supply review suggested",
            "affected_outlets": stockouts["retailer_id"].head(3).tolist(),
        })
    return out


def weather_summary(rep_outlets: pd.DataFrame) -> str:
    avg_rain = float(rep_outlets["rain_mm_next_48h"].mean())
    spray_frac = float(rep_outlets["spray_window_open_flag"].mean())
    if avg_rain > 5:
        return f"Rain expected (avg {avg_rain:.0f}mm in 48h). Spray windows closing in this territory."
    if spray_frac > 0.7:
        return "Clear spray window for next 48h. Good fungicide push opportunity."
    return "Mixed conditions across territory. Check per-outlet recommendations."


# ---------------------------------------------------------------------------
# Outlet block assembly (CONTRACT.md Section 4.1)
# ---------------------------------------------------------------------------
def build_outlet_block(row: pd.Series, route_position: int) -> dict:
    priority = float(row["_priority_score"])
    confidence = "High" if priority > 0.55 else "Medium" if priority > 0.35 else "Low"
    return {
        "outlet_id":   str(row["retailer_id"]),
        "name":        str(row["name"]),
        "address":     str(row["address"]),
        "lat":         float(row["lat"]),
        "lng":         float(row["lng"]),
        "priority_score": round(priority, 3),
        "confidence":  confidence,
        "route_position": int(route_position),
        "reason_strings": compute_reasons(row),
        "recommended_product": compute_nba(row)[0],
        "estimated_visit_minutes": int(row["rep_avg_dwell_minutes"]),
    }


# ---------------------------------------------------------------------------
# Outcome model (POST /outcome per CONTRACT.md Section 4.3)
# ---------------------------------------------------------------------------
class Outcome(BaseModel):
    rep_id: str
    outlet_id: str
    date: str
    sale_made: bool
    sale_value_inr: float = 0.0
    product_id: str = ""
    dismissal_reason: Optional[str] = None
    rep_notes: Optional[str] = None


_outcome_counter = 8740


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "syngenta-field-copilot",
        "version": app.version,
        "priority_model_trained": priority_is_trained(),
        "route_solver_available": _ROUTE_SOLVER_OK,
        "explainer_available": _EXPLAINER_IMPORTED and explainer_is_available(),
        "nba_model_trained": _NBA_IMPORTED and nba_is_trained(),
        "anomaly_model_trained": _ANOMALY_IMPORTED and anomaly_is_trained(),
    }


@app.get("/plan/today")
def plan_today(rep_id: str, date: str):
    df = load_features(date)
    reps = load_reps()

    rep_row = reps[reps["rep_id"] == rep_id]
    if len(rep_row) == 0:
        raise HTTPException(status_code=404, detail=f"No such rep_id: {rep_id}")
    rep_row = rep_row.iloc[0]
    territory_id = rep_row["territory_id"]

    rep_outlets = df[df["territory_id"] == territory_id].copy()
    if len(rep_outlets) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No outlets for rep {rep_id} (territory {territory_id}) on {date}",
        )

    # 1) Priority: score the whole territory, keep top 10 by priority
    rep_outlets["_priority_score"] = compute_priority(rep_outlets).values
    top = rep_outlets.sort_values("_priority_score", ascending=False).head(10).reset_index(drop=True)

    # 2) Route: VRP-order the top-N. Reorder the DataFrame so route_position
    #    1..N reflects the optimal path. priority_score is unchanged.
    order, total_m = compute_route_order(top)
    routed = top.iloc[order].reset_index(drop=True)
    route_distance_km = round(total_m / 1000.0, 2) if total_m != float("inf") else None

    outlets = [build_outlet_block(row, idx + 1) for idx, row in routed.iterrows()]
    route_polyline = [[float(r["lat"]), float(r["lng"])] for _, r in routed.iterrows()]

    return {
        "rep_id":               rep_id,
        "date":                 date,
        "territory":            str(rep_row["territory_name"]),
        "synced_at":            f"{date}T05:12:00+05:30",
        "weather_summary":      weather_summary(routed),
        "outlets":              outlets,
        "route_polyline":       route_polyline,
        "route_distance_km":    route_distance_km,
        "anomalies":            compute_anomalies(routed, date),
    }


@app.get("/nba/{outlet_id}")
def nba(outlet_id: str, date: str):
    df = load_features(date)
    row_df = df[df["retailer_id"] == outlet_id]
    if len(row_df) == 0:
        raise HTTPException(status_code=404, detail=f"No outlet {outlet_id} on {date}")
    row = row_df.iloc[0].copy()
    row["_priority_score"] = float(compute_priority(row_df).iloc[0])

    primary, alternates = compute_nba(row, top_n=4)

    return {
        "outlet_id": outlet_id,
        "primary": primary,
        "alternates": alternates,
    }


@app.post("/outcome")
def outcome(o: Outcome):
    global _outcome_counter
    _outcome_counter += 1
    outcome_id = f"OUTCOME_{_outcome_counter}"

    log_path = DATA / "outcomes_log.csv"
    new_row = pd.DataFrame([{**o.model_dump(), "id": outcome_id, "logged_at": datetime.utcnow().isoformat()}])
    if log_path.exists():
        new_row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        new_row.to_csv(log_path, index=False)

    return {"status": "logged", "id": outcome_id}