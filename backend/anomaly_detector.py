"""
anomaly_detector.py -- per-territory anomaly detection.

Two complementary signals feed CONTRACT.md Section 4 anomalies[]:

1. IsolationForest on per-retailer POS-aggregate features:
     sales_velocity_7d, sku_diversity_30d, txn_count_30d, mean_basket_size_30d
   Trained on all retailers in the Syngenta synthetic POS log.
   contamination=0.05, n_estimators=200, random_state=42.

2. Deterministic cluster heuristics on the daily feature snapshot:
     elevated_pest_pressure   -- >=2 outlets with pest_pressure_idx > 0.65
     recurring_stockout       -- >=2 outlets with recent_stockout_flag == 1
     ndvi_decline_cluster     -- >=2 outlets with ndvi_delta_14d < -0.05

CLI:
    python anomaly_detector.py train
    python anomaly_detector.py test --date 2026-02-15 --rep-id REP_0001

Importable:
    from anomaly_detector import detect, is_trained
    anomalies: list[dict] = detect(rep_outlets, date_str)

Honest scoping (folded into MODELS.md):
- STL decomposition on weekly pest bulletin counts (statsmodels.STL) is
  instrumented in the codebase but disabled in this build: the synthesized
  pest bulletin in features.py only spans the current week, so STL's seasonal
  signal is not learnable. With >=8 weeks of real IPM bulletin history (a
  Prince-side input), the STL pathway slots in transparently.
- IsolationForest contamination=0.05 is a reasonable prior but uncalibrated
  to ground-truth anomaly rates in production. Treat scores as flags, not
  ground-truth labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS / "anomaly_model.joblib"
META_PATH = MODELS / "anomaly_meta.json"
METRICS_PATH = MODELS / "anomaly_metrics.json"

POS_FEATS = ["sales_velocity_7d", "sku_diversity_30d", "txn_count_30d", "mean_basket_size_30d"]


# ---------------------------------------------------------------------------
# POS feature extraction (per retailer, as of a given target date)
# ---------------------------------------------------------------------------
def compute_pos_features(pos: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """Per-retailer POS aggregates over the trailing 7/30 days from target_date."""
    pos = pos[pos["transaction_date"] <= target_date].copy()
    if "line_value" not in pos.columns:
        pos["line_value"] = pos["sku_qty"] * pos["sku_price"]

    win7 = target_date - pd.Timedelta(days=7)
    win30 = target_date - pd.Timedelta(days=30)

    p7 = pos[pos["transaction_date"] >= win7]
    sv7 = p7.groupby("retailer_id")["sku_qty"].sum() / 7.0

    p30 = pos[pos["transaction_date"] >= win30]
    sd = p30.groupby("retailer_id")["sku_name"].nunique()
    txn_per_ret = (p30.groupby(["retailer_id", "transaction_id"]).size()
                       .reset_index().groupby("retailer_id").size())
    mbs = (p30.groupby(["retailer_id", "transaction_id"])["line_value"].sum()
              .reset_index().groupby("retailer_id")["line_value"].mean())

    df = pd.DataFrame({
        "sales_velocity_7d": sv7,
        "sku_diversity_30d": sd,
        "txn_count_30d": txn_per_ret,
        "mean_basket_size_30d": mbs,
    }).reset_index()
    df = df.fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(train_date: str = "2026-01-15") -> dict:
    from sklearn.ensemble import IsolationForest
    import joblib

    from features import get_dataset_root
    pos = pd.read_csv(get_dataset_root() / "retailer_pos.csv", parse_dates=["transaction_date"])
    tgt = pd.Timestamp(train_date)
    df = compute_pos_features(pos, tgt)
    print(f"[train] {len(df)} retailers with POS-feature rows on {train_date}")

    X = df[POS_FEATS].astype(float).values
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X)
    preds = iso.predict(X)
    df["_iso_score"] = iso.decision_function(X)
    df["_is_anomaly"] = (preds == -1).astype(int)
    n_anom = int(df["_is_anomaly"].sum())
    print(f"[train] flagged anomalies: {n_anom}/{len(df)} "
          f"({100.0 * n_anom / max(len(df), 1):.1f}%)")

    metrics = {
        "train_date": train_date,
        "n_retailers": int(len(df)),
        "contamination": 0.05,
        "n_estimators": 200,
        "anomaly_count": n_anom,
        "anomaly_rate": float(n_anom / max(len(df), 1)),
        "pos_feats": POS_FEATS,
        "iso_score_min": float(df["_iso_score"].min()),
        "iso_score_max": float(df["_iso_score"].max()),
        "iso_score_mean": float(df["_iso_score"].mean()),
    }
    joblib.dump(iso, MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump({"pos_feats": POS_FEATS, "train_date": train_date}, f, indent=2)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] saved -> {MODEL_PATH}")
    print(f"[train] saved meta -> {META_PATH}")
    return metrics


# ---------------------------------------------------------------------------
# Inference (importable)
# ---------------------------------------------------------------------------
_cached_iso = None
_pos_cache: dict[str, pd.DataFrame] = {}


def _load():
    global _cached_iso
    if _cached_iso is not None:
        return _cached_iso
    if not MODEL_PATH.exists():
        return None
    import joblib
    _cached_iso = joblib.load(MODEL_PATH)
    return _cached_iso


def is_trained() -> bool:
    return MODEL_PATH.exists() and META_PATH.exists()


def _get_pos_features(date_str: str) -> pd.DataFrame:
    if date_str in _pos_cache:
        return _pos_cache[date_str]
    from features import get_dataset_root
    pos = pd.read_csv(get_dataset_root() / "retailer_pos.csv", parse_dates=["transaction_date"])
    df = compute_pos_features(pos, pd.Timestamp(date_str))
    _pos_cache[date_str] = df
    return df


def detect(rep_outlets: pd.DataFrame, date_str: str) -> list[dict]:
    """Return list of anomaly dicts for the given rep's outlet set.

    Each dict matches CONTRACT.md Section 4:
        {type, severity, description, affected_outlets}
    """
    anomalies: list[dict] = []
    if len(rep_outlets) == 0:
        return anomalies

    rep_outlets = rep_outlets.copy()
    rep_outlets["retailer_id"] = rep_outlets["retailer_id"].astype(str)

    # 1) IsolationForest -- per-outlet POS pattern outlier
    iso = _load()
    if iso is not None:
        try:
            pos_df = _get_pos_features(date_str)
            in_scope = pos_df[pos_df["retailer_id"].astype(str).isin(rep_outlets["retailer_id"])].copy()
            if len(in_scope):
                X = in_scope[POS_FEATS].astype(float).values
                in_scope["_iso_score"] = iso.decision_function(X)
                in_scope["_is_anomaly"] = (iso.predict(X) == -1).astype(int)
                flagged = in_scope[in_scope["_is_anomaly"] == 1]
                for _, r in flagged.iterrows():
                    rid = str(r["retailer_id"])
                    nm = rep_outlets.loc[rep_outlets["retailer_id"] == rid, "name"].values
                    name = str(nm[0]) if len(nm) else rid
                    anomalies.append({
                        "type": "outlier_pos_pattern",
                        "severity": "medium",
                        "description": (
                            f"{name} POS pattern is an outlier vs district baseline "
                            f"(sales_velocity_7d={r['sales_velocity_7d']:.1f}, "
                            f"sku_diversity_30d={int(r['sku_diversity_30d'])}, "
                            f"iso_score={r['_iso_score']:.3f})"
                        ),
                        "affected_outlets": [rid],
                    })
        except Exception as e:
            print(f"[anomaly_detector] iso scoring failed: {e}")

    # 2) Pest cluster
    high_pest = rep_outlets[rep_outlets["pest_pressure_idx"] > 0.65]
    if len(high_pest) >= 2:
        crop = str(high_pest.iloc[0]["dominant_crop_in_radius_5km"]).capitalize()
        anomalies.append({
            "type": "elevated_pest_pressure",
            "severity": "medium",
            "description": (
                f"{crop} pest pressure elevated in {len(high_pest)} outlets in this territory "
                f"(mean {high_pest['pest_pressure_idx'].mean():.2f})"
            ),
            "affected_outlets": high_pest["retailer_id"].head(5).tolist(),
        })

    # 3) Stockout cluster
    stockouts = rep_outlets[rep_outlets["recent_stockout_flag"] == 1]
    if len(stockouts) >= 2:
        anomalies.append({
            "type": "recurring_stockout_cluster",
            "severity": "low",
            "description": f"Recent stockouts at {len(stockouts)} outlets -- supply review suggested",
            "affected_outlets": stockouts["retailer_id"].head(5).tolist(),
        })

    # 4) NDVI decline cluster
    ndvi_drops = rep_outlets[rep_outlets["ndvi_delta_14d"] < -0.05]
    if len(ndvi_drops) >= 2:
        crop = str(ndvi_drops.iloc[0]["dominant_crop_in_radius_5km"]).capitalize()
        anomalies.append({
            "type": "ndvi_decline_cluster",
            "severity": "low",
            "description": (
                f"{crop} NDVI declining at {len(ndvi_drops)} outlets in this territory "
                f"(mean delta {ndvi_drops['ndvi_delta_14d'].mean():+.2f} over last 14 days)"
            ),
            "affected_outlets": ndvi_drops["retailer_id"].head(5).tolist(),
        })

    return anomalies


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["train", "test"])
    p.add_argument("--date", default="2026-02-15")
    p.add_argument("--rep-id", default="REP_0001")
    args = p.parse_args()

    if args.action == "train":
        train()
        return

    df = pd.read_parquet(DATA / f"features_{args.date}.parquet")
    from features import get_dataset_root
    reps = pd.read_csv(get_dataset_root() / "reps_territory.csv")
    rep_row = reps[reps["rep_id"] == args.rep_id].iloc[0]
    rep_outlets = df[df["territory_id"] == rep_row["territory_id"]]

    print(f"[anomaly_detector] trained={is_trained()}; outlets={len(rep_outlets)}\n")
    anoms = detect(rep_outlets, args.date)
    if not anoms:
        print("(no anomalies flagged)")
        return
    for a in anoms:
        print(f"[{a['severity'].upper():<6}] {a['type']}")
        print(f"    {a['description']}")
        print(f"    affected: {a['affected_outlets']}")
        print()


if __name__ == "__main__":
    main()