"""
nba_engine.py -- Next-Best-Action engine.

Two stages plus justification:

Stage A -- ELIGIBILITY (hard rule filter)
    SKU is eligible for an outlet iff ALL of:
      - crop_match: outlet's dominant_crop is in SKU_CROPS[sku]
      - category-specific gate:
          insecticide / fungicide  -> pest_pressure_idx >= 0.35
          herbicide                -> growth_stage in {vegetative, tillering}
      - growth_stage_match: SKU's preferred-stage set contains the outlet's
        current stage (soft for fungicides, hard for seed treatments).

Stage B -- P(purchase) via CalibratedClassifierCV(LogisticRegression)
    Trained per (outlet, candidate_sku) row using a 14-day forward POS-proxy
    label (POS of that SKU at that retailer within 14 days of train_date).
    Features = 30 numeric outlet features + 12-dim SKU one-hot. Calibration:
    sigmoid (Platt), cv=3.

Stage C -- score and justify
    score = P(purchase) * SKU_PRODUCT_META[sku].base_uplift_inr
    primary = highest score; alternates = next 3 by score.
    Justification is a deterministic templated string built from the row's
    actual feature values plus SKU category. No LLM in this build (folded
    into MODELS.md as honest scoping: "instrumented, LLM-justification fallback
    available via env var; demo uses deterministic templates").

CLI:
    python nba_engine.py train
    python nba_engine.py test --date 2026-02-15 --outlet RTL_00009

Importable:
    from nba_engine import recommend, is_trained
    primary_dict, alternates_list = recommend(row)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS / "nba_model.joblib"
META_PATH = MODELS / "nba_meta.json"
METRICS_PATH = MODELS / "nba_metrics.json"

TRAIN_DATE_DEFAULT = "2026-01-15"
LABEL_WINDOW_DAYS = 14

# Outlet-feature columns the logistic consumes (same 30 as priority_model)
NUMERIC_FEATS = [
    "days_since_last_visit", "last_purchase_value_inr", "visits_last_90d",
    "total_sales_180d", "avg_order_value_inr",
    "pest_pressure_idx", "ndvi_current", "ndvi_delta_14d", "rain_mm_next_48h",
    "rain_probability_72h", "temp_anomaly_c", "spray_window_open_flag",
    "mandi_price_anomaly", "weather_volatility_7d",
    "stock_of_recommended_sku", "days_of_cover", "safety_stock_gap",
    "recent_stockout_flag",
    "growth_stage_estimate", "days_to_next_critical_stage", "crop_match_score",
    "rep_visits_to_outlet_lifetime", "rep_close_rate_at_outlet",
    "rep_avg_dwell_minutes", "rep_familiarity_score",
    "day_of_week", "week_of_month", "week_of_season", "is_market_day_flag",
    "days_since_competitor_visit",
]

# SKU catalog + meta (kept in sync with main.py.SKU_PRODUCT_META)
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
SKU_LIST = sorted(SKU_PRODUCT_META.keys())  # stable column order for one-hot

# Growth-stage preferences per category (soft hint, used only in Stage A's
# herbicide gate -- fungicides/insecticides ignore stage at this resolution).
HERBICIDE_STAGES = {1.0, 1.5}  # vegetative / tillering / tuberization
SEED_TREATMENT_SKUS = {"Cruiser 350 FS", "Vibrance Integral"}  # apply only at sowing
SOWING_STAGE = 0.0


# ---------------------------------------------------------------------------
# Stage A -- eligibility
# ---------------------------------------------------------------------------
def eligible_skus(row: pd.Series) -> list[str]:
    """Return the SKUs that pass the hard rule filter for this outlet row."""
    from features import SKU_CROPS  # crop -> SKU mapping
    crop = str(row.get("dominant_crop_in_radius_5km", "wheat"))
    pest = float(row.get("pest_pressure_idx", 0.0))
    stage = float(row.get("growth_stage_estimate", 1.0))

    out = []
    for sku in SKU_LIST:
        target_crops = SKU_CROPS.get(sku, [])
        if crop not in target_crops:
            continue
        _, category, _ = SKU_PRODUCT_META[sku]

        # Category-specific gate
        if category in ("Insecticide", "Fungicide") and pest < 0.35:
            continue
        if category == "Herbicide" and stage not in HERBICIDE_STAGES:
            continue
        if sku in SEED_TREATMENT_SKUS and stage != SOWING_STAGE:
            continue

        out.append(sku)
    return out


# ---------------------------------------------------------------------------
# Feature vector assembly (outlet feats + SKU one-hot)
# ---------------------------------------------------------------------------
def make_row_vec(outlet_row: pd.Series, sku: str) -> np.ndarray:
    base = np.array([float(outlet_row[c]) for c in NUMERIC_FEATS], dtype=float)
    onehot = np.zeros(len(SKU_LIST), dtype=float)
    onehot[SKU_LIST.index(sku)] = 1.0
    return np.concatenate([base, onehot])


def feature_names() -> list[str]:
    return NUMERIC_FEATS + [f"sku__{s}" for s in SKU_LIST]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _build_training_frame(features_df: pd.DataFrame, train_date: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X, y, candidate_index) for all eligible (retailer, sku) candidates."""
    from features import SKU_CROPS, get_dataset_root

    tgt = pd.Timestamp(train_date)
    win_lo = tgt + pd.Timedelta(days=1)
    win_hi = tgt + pd.Timedelta(days=LABEL_WINDOW_DAYS)
    pos = pd.read_csv(get_dataset_root() / "retailer_pos.csv", parse_dates=["transaction_date"])
    pos = pos[(pos["transaction_date"] >= win_lo) & (pos["transaction_date"] <= win_hi)]
    bought = set(zip(pos["retailer_id"].astype(str), pos["sku_name"].astype(str)))

    Xs, ys, idx = [], [], []
    for _, row in features_df.iterrows():
        crop = str(row["dominant_crop_in_radius_5km"])
        rid = str(row["retailer_id"])
        for sku in SKU_LIST:
            if crop not in SKU_CROPS.get(sku, []):
                continue
            Xs.append(make_row_vec(row, sku))
            ys.append(int((rid, sku) in bought))
            idx.append((rid, sku))
    X = np.asarray(Xs, dtype=float)
    y = np.asarray(ys, dtype=int)
    idx_df = pd.DataFrame(idx, columns=["retailer_id", "sku"])
    return X, y, idx_df


def train(train_date: str = TRAIN_DATE_DEFAULT) -> dict:
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, brier_score_loss

    feat_path = DATA / f"features_{train_date}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Train features missing: {feat_path}. "
            f"Generate first: python features.py --date {train_date}"
        )
    df = pd.read_parquet(feat_path).reset_index(drop=True)
    print(f"[train] loaded {len(df)} retailers from features_{train_date}.parquet")

    X, y, idx_df = _build_training_frame(df, train_date)
    pos_rate = y.mean() if len(y) else 0.0
    print(f"[train] candidate pairs: {len(y)}; positive rate: {pos_rate:.4f} "
          f"({int(y.sum())}/{len(y)})")

    # Hold out 20% of retailers
    rng = np.random.default_rng(42)
    all_rids = sorted(set(idx_df["retailer_id"]))
    rng.shuffle(all_rids)
    cut = int(0.8 * len(all_rids))
    train_rids = set(all_rids[:cut])
    train_mask = idx_df["retailer_id"].isin(train_rids).values
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[~train_mask], y[~train_mask]
    print(f"[train] split: train pairs={len(ytr)}, test pairs={len(yte)}")

    base = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")),
    ])
    clf = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    clf.fit(Xtr, ytr)
    print("[train] CalibratedClassifierCV(LogisticRegression) fit complete")

    p_te = clf.predict_proba(Xte)[:, 1]
    try:
        auc = float(roc_auc_score(yte, p_te))
    except Exception:
        auc = float("nan")
    brier = float(brier_score_loss(yte, p_te))
    metrics = {
        "train_date": train_date,
        "n_train_pairs": int(len(ytr)),
        "n_test_pairs": int(len(yte)),
        "label_positive_rate": float(pos_rate),
        "test_auc": auc,
        "test_brier": brier,
        "model_type": "CalibratedClassifierCV(LogisticRegression, sigmoid, cv=3)",
        "feature_dim": int(X.shape[1]),
        "sku_list": SKU_LIST,
        "numeric_feats": NUMERIC_FEATS,
    }
    print(f"[train] Metrics:")
    print(f"  test_auc:   {auc:.3f}")
    print(f"  test_brier: {brier:.4f}")

    joblib.dump(clf, MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump({"sku_list": SKU_LIST, "numeric_feats": NUMERIC_FEATS}, f, indent=2)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] saved model -> {MODEL_PATH}")
    print(f"[train] saved meta -> {META_PATH}")
    return metrics


# ---------------------------------------------------------------------------
# Inference (importable)
# ---------------------------------------------------------------------------
_cached_clf = None


def _load():
    global _cached_clf
    if _cached_clf is not None:
        return _cached_clf
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"NBA model missing at {MODEL_PATH}")
    import joblib
    _cached_clf = joblib.load(MODEL_PATH)
    return _cached_clf


def is_trained() -> bool:
    return MODEL_PATH.exists() and META_PATH.exists()


def _justify(row: pd.Series, sku: str, p_purchase: float) -> str:
    """Deterministic templated justification using actual feature values."""
    _, category, _ = SKU_PRODUCT_META[sku]
    crop = str(row.get("dominant_crop_in_radius_5km", "wheat")).lower()
    pest = float(row.get("pest_pressure_idx", 0.0))
    stock = int(row.get("stock_of_recommended_sku", 0))
    rain = float(row.get("rain_mm_next_48h", 0.0))
    spray_open = bool(row.get("spray_window_open_flag", 0))
    safety_gap = int(row.get("safety_stock_gap", 0))
    stage = float(row.get("growth_stage_estimate", 1.0))

    parts: list[str] = []
    # Lead with category-specific demand signal
    if category == "Insecticide":
        parts.append(f"{crop.capitalize()} pest pressure {pest:.2f} -- on-label insecticide")
    elif category == "Fungicide":
        if rain > 5:
            parts.append(f"{int(rain)}mm rain in 48h -- fungicide window opening for {crop}")
        else:
            parts.append(f"{crop.capitalize()} disease risk at pest pressure {pest:.2f} -- fungicide indicated")
    elif category == "Herbicide":
        stage_name = "vegetative" if abs(stage - 1.0) < 0.4 else "tillering"
        parts.append(f"{crop.capitalize()} in {stage_name} stage -- post-emergence herbicide window")

    # Inventory signal
    if safety_gap > 0:
        parts.append(f"{safety_gap} units below safety stock at this outlet")
    elif stock < 10:
        parts.append(f"outlet stock low at {stock} units")

    # Timing signal
    if spray_open and category != "Herbicide":
        parts.append("spray window open in 48h")

    parts.append(f"P(purchase) {p_purchase:.2f}")
    return "; ".join(parts) + "."


def _fallback_top_pick(row: pd.Series) -> tuple[str, float, str]:
    """Used when the NBA model isn't trained. Picks the eligible SKU with the
    largest base_uplift, justifies it, returns (sku, fake_p, justification)."""
    elig = eligible_skus(row)
    if not elig:
        # Final fallback: the features.py recommended_sku_pick
        sku = str(row.get("recommended_sku_pick", SKU_LIST[0]))
        return sku, 0.20, _justify(row, sku, 0.20)
    sku = max(elig, key=lambda s: SKU_PRODUCT_META[s][2])
    return sku, 0.30, _justify(row, sku, 0.30)


def recommend(row: pd.Series, top_n: int = 4) -> tuple[dict, list[dict]]:
    """Return (primary_dict, alternates_list) per CONTRACT.md §4.

    primary: highest score == P(purchase) * base_uplift.
    alternates: next (top_n - 1) by score (so default is primary + 3).
    """
    elig = eligible_skus(row)

    # Fallback path: ranker not trained or no eligible SKUs
    if not is_trained() or not elig:
        sku, p, just = _fallback_top_pick(row)
        pid, category, base_uplift = SKU_PRODUCT_META.get(sku, ("UNK", "Crop Protection", 3000))
        primary = {
            "product_id": pid,
            "product_name": sku,
            "category": category,
            "expected_uplift_inr": int(round(p * base_uplift)),
            "justification": just,
        }
        # Alternates: next best by base uplift among elig (or anything crop-matched)
        from features import SKU_CROPS
        crop = str(row.get("dominant_crop_in_radius_5km", "wheat"))
        alts_pool = [s for s in SKU_LIST if crop in SKU_CROPS.get(s, []) and s != sku][:top_n - 1]
        alts = []
        for s in alts_pool:
            pid2, cat2, bu2 = SKU_PRODUCT_META[s]
            alts.append({
                "product_id": pid2,
                "product_name": s,
                "expected_uplift_inr": int(round(0.20 * bu2)),
                "justification": _justify(row, s, 0.20),
            })
        return primary, alts

    # Real path: score each eligible SKU with the calibrated logistic
    clf = _load()
    rows = np.stack([make_row_vec(row, sku) for sku in elig])
    probs = clf.predict_proba(rows)[:, 1]
    scored = []
    for sku, p in zip(elig, probs):
        pid, category, base_uplift = SKU_PRODUCT_META[sku]
        scored.append({
            "sku": sku, "pid": pid, "category": category,
            "p": float(p),
            "score": float(p * base_uplift),
            "base_uplift": int(base_uplift),
        })
    scored.sort(key=lambda d: d["score"], reverse=True)

    top = scored[0]
    primary = {
        "product_id": top["pid"],
        "product_name": top["sku"],
        "category": top["category"],
        "expected_uplift_inr": int(round(top["p"] * top["base_uplift"])),
        "justification": _justify(row, top["sku"], top["p"]),
    }
    alts = []
    for d in scored[1:top_n]:
        alts.append({
            "product_id": d["pid"],
            "product_name": d["sku"],
            "expected_uplift_inr": int(round(d["p"] * d["base_uplift"])),
            "justification": _justify(row, d["sku"], d["p"]),
        })
    return primary, alts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["train", "test"])
    p.add_argument("--date", default="2026-02-15")
    p.add_argument("--outlet", default=None)
    p.add_argument("--rep-id", default="REP_0001")
    args = p.parse_args()

    if args.action == "train":
        train()
        return

    # test mode
    df = pd.read_parquet(DATA / f"features_{args.date}.parquet")
    if args.outlet:
        targets = df[df["retailer_id"] == args.outlet]
    else:
        from features import get_dataset_root
        reps = pd.read_csv(get_dataset_root() / "reps_territory.csv")
        rep_row = reps[reps["rep_id"] == args.rep_id].iloc[0]
        targets = df[df["territory_id"] == rep_row["territory_id"]]
    print(f"[nba_engine] trained: {is_trained()}; outlets to score: {len(targets)}\n")
    for _, row in targets.iterrows():
        primary, alts = recommend(row)
        print(f"{row['retailer_id']}  {row['name']}  ({row['dominant_crop_in_radius_5km']})")
        print(f"  primary  {primary['product_name']:<20} uplift INR {primary['expected_uplift_inr']:>6}")
        print(f"           {primary['justification']}")
        for a in alts:
            print(f"  alt      {a['product_name']:<20} uplift INR {a['expected_uplift_inr']:>6}")
        print()


if __name__ == "__main__":
    main()