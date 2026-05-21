"""
priority_model.py — XGBoost LambdaMART (rank:pairwise) ranker for outlet visit priority.

Pipeline:
1. Synthesize per-(retailer, train_date) labels: sale_made = 1 if the retailer
   had a POS transaction within 14 days AFTER train_date for an SKU whose target
   crops include the retailer's dominant_crop_in_radius_5km; else 0.
2. Load features for train_date from data/features_<train_date>.parquet.
3. Train XGBRanker with group=territory_id; held-out test set is 20% of
   territories via GroupShuffleSplit so a territory never bleeds across train/test.
4. Save model + feature list + SHAP global importance PNG + metrics JSON to models/.

CLI:
    python priority_model.py train
    python priority_model.py test-score --date 2026-02-15

Importable:
    from priority_model import score, is_trained
    s = score(features_df)   # pd.Series in [0, 1], min-max normalized within batch

Honest scoping (folded into MODELS.md):
- Labels are POS-proxy, not rep-causal: source data has no rep_visit->sale link.
- Train date 2026-01-15 leaves 1-month separation from demo dates 2026-02-15/17.
  14-day forward label window stays inside POS range (data ends 2026-03-29).
- Stratification done by GroupShuffleSplit on territory_id; secondary stratification
  by (state, dominant_crop) would require a custom splitter we skip for the 24h cycle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Lazy heavy imports kept out of module load so that main.py can `from priority_model
# import score, is_trained` cheaply at startup even when models/ is absent.

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS / "priority_model.json"
FEATS_PATH = MODELS / "priority_features.json"
METRICS_PATH = MODELS / "priority_metrics.json"
SHAP_PNG = MODELS / "priority_shap_importance.png"

# Numeric feature columns the ranker consumes (excludes the two categoricals
# dominant_crop_in_radius_5km and season_flag_kharif_rabi_zaid; the categorical
# signal is already captured numerically by crop_match_score).
NUMERIC_FEATS = [
    # Bucket A — RFM (5)
    "days_since_last_visit", "last_purchase_value_inr", "visits_last_90d",
    "total_sales_180d", "avg_order_value_inr",
    # Bucket B — Geo (9)
    "pest_pressure_idx", "ndvi_current", "ndvi_delta_14d", "rain_mm_next_48h",
    "rain_probability_72h", "temp_anomaly_c", "spray_window_open_flag",
    "mandi_price_anomaly", "weather_volatility_7d",
    # Bucket C — Inventory (4)
    "stock_of_recommended_sku", "days_of_cover", "safety_stock_gap",
    "recent_stockout_flag",
    # Bucket D — Growth (3 numeric; dominant_crop and season excluded)
    "growth_stage_estimate", "days_to_next_critical_stage", "crop_match_score",
    # Bucket E — Rep (4)
    "rep_visits_to_outlet_lifetime", "rep_close_rate_at_outlet",
    "rep_avg_dwell_minutes", "rep_familiarity_score",
    # Bucket F — Temporal (5)
    "day_of_week", "week_of_month", "week_of_season", "is_market_day_flag",
    "days_since_competitor_visit",
]  # total: 30

TRAIN_DATE_DEFAULT = "2026-01-15"
LABEL_WINDOW_DAYS = 14


# ---------------------------------------------------------------------------
# Label synthesis
# ---------------------------------------------------------------------------
def synthesize_labels(features_df: pd.DataFrame, train_date: str) -> pd.Series:
    """
    Build a binary label per retailer: did this retailer make a crop-matched
    POS sale in the LABEL_WINDOW_DAYS days following train_date?
    """
    from features import SKU_CROPS, get_dataset_root  # local import; avoids cycle on bare module load

    syn = get_dataset_root()
    pos = pd.read_csv(syn / "retailer_pos.csv", parse_dates=["transaction_date"])

    tgt = pd.Timestamp(train_date)
    window_start = tgt + pd.Timedelta(days=1)
    window_end = tgt + pd.Timedelta(days=LABEL_WINDOW_DAYS)
    pos = pos[(pos["transaction_date"] >= window_start) & (pos["transaction_date"] <= window_end)].copy()

    # Map retailer -> dominant crop from the features file
    rid_to_crop = dict(zip(features_df["retailer_id"], features_df["dominant_crop_in_radius_5km"]))
    pos["crop"] = pos["retailer_id"].map(rid_to_crop)

    # SKU -> target crops list. Apply as a vectorized membership check.
    sku_targets = pos["sku_name"].map(lambda s: SKU_CROPS.get(s, []))
    pos["match"] = [c in t if isinstance(t, list) else False for c, t in zip(pos["crop"], sku_targets)]

    matched_retailers = set(pos.loc[pos["match"], "retailer_id"].unique())
    labels = features_df["retailer_id"].isin(matched_retailers).astype(int)
    return labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(train_date: str = TRAIN_DATE_DEFAULT) -> dict:
    import xgboost as xgb
    from sklearn.model_selection import GroupShuffleSplit
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    feat_path = DATA / f"features_{train_date}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Train features missing: {feat_path}. "
            f"Generate first: python features.py --date {train_date}"
        )
    df = pd.read_parquet(feat_path).reset_index(drop=True)
    print(f"[train] loaded {len(df)} rows of features_{train_date}.parquet")

    df["label"] = synthesize_labels(df, train_date)
    pos_rate = df["label"].mean()
    print(f"[train] label positive rate: {pos_rate:.3f} "
          f"({int(df['label'].sum())}/{len(df)})")
    if pos_rate < 0.02 or pos_rate > 0.98:
        print(f"[train] WARNING: degenerate label distribution — ranking signal may be weak")

    # GroupShuffleSplit on territory_id
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["territory_id"]))
    train_df = df.iloc[train_idx].sort_values("territory_id").reset_index(drop=True)
    test_df = df.iloc[test_idx].sort_values("territory_id").reset_index(drop=True)

    train_groups = train_df.groupby("territory_id").size().values
    test_groups = test_df.groupby("territory_id").size().values
    print(f"[train] split: train={len(train_df)} ({len(train_groups)} territories), "
          f"test={len(test_df)} ({len(test_groups)} territories)")

    # Cast / sanitize numeric features (Parquet bool -> int64 etc. handled by xgboost,
    # but force float for safety)
    Xtr = train_df[NUMERIC_FEATS].astype(float)
    Xte = test_df[NUMERIC_FEATS].astype(float)
    ytr = train_df["label"].astype(int)
    yte = test_df["label"].astype(int)

    ranker = xgb.XGBRanker(
        objective="rank:pairwise",
        learning_rate=0.1,
        n_estimators=200,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )
    ranker.fit(Xtr, ytr, group=train_groups, verbose=False)
    print("[train] fit complete")

    # Eval: top-k recall at territory level
    test_df["_raw_score"] = ranker.predict(Xte)
    top1_recall = top_k_recall_at_group(test_df, k=1)
    top3_recall = top_k_recall_at_group(test_df, k=3)
    metrics = {
        "train_date": train_date,
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_territories": int(len(train_groups)),
        "n_test_territories": int(len(test_groups)),
        "label_positive_rate": float(pos_rate),
        "test_top1_recall_at_territory": float(top1_recall),
        "test_top3_recall_at_territory": float(top3_recall),
        "features": NUMERIC_FEATS,
        "model_type": "xgboost.XGBRanker",
        "objective": "rank:pairwise",
    }
    print(f"[train] Metrics:")
    print(f"  test_top1_recall_at_territory: {top1_recall:.3f}")
    print(f"  test_top3_recall_at_territory: {top3_recall:.3f}")

    # SHAP global importance — use TreeExplainer
    try:
        import shap
        explainer = shap.TreeExplainer(ranker)
        sample = Xte.sample(min(500, len(Xte)), random_state=0)
        sv = explainer.shap_values(sample)
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        feats_sorted = [NUMERIC_FEATS[i] for i in order]
        vals_sorted = mean_abs[order]

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.barh(feats_sorted[::-1], vals_sorted[::-1], color="#2E7D32")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Priority Ranker — Global Feature Importance\n"
                     "(XGBoost LambdaMART, rank:pairwise, "
                     f"top-1 recall@territory = {top1_recall:.2f})")
        plt.tight_layout()
        plt.savefig(SHAP_PNG, dpi=120)
        plt.close(fig)
        metrics["shap_top5"] = list(zip(feats_sorted[:5], [float(v) for v in vals_sorted[:5]]))
        print(f"[train] saved SHAP PNG -> {SHAP_PNG}")
    except Exception as e:
        print(f"[train] SHAP plot skipped: {e}")

    # Persist
    ranker.save_model(str(MODEL_PATH))
    with open(FEATS_PATH, "w") as f:
        json.dump(NUMERIC_FEATS, f, indent=2)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[train] saved model -> {MODEL_PATH}")
    print(f"[train] saved features -> {FEATS_PATH}")
    print(f"[train] saved metrics -> {METRICS_PATH}")

    return metrics


def top_k_recall_at_group(df: pd.DataFrame, k: int = 1) -> float:
    """Within each territory, did any of the top-k by _raw_score have label=1?
    Averaged over territories that have at least one positive label."""
    def per_group(g):
        if g["label"].sum() == 0:
            return np.nan
        topk = g.nlargest(k, "_raw_score")
        return float(topk["label"].max())
    vals = df.groupby("territory_id").apply(per_group).dropna()
    return float(vals.mean()) if len(vals) else 0.0


# ---------------------------------------------------------------------------
# Inference (importable)
# ---------------------------------------------------------------------------
_cached_model = None
_cached_feats = None


def _load():
    global _cached_model, _cached_feats
    if _cached_model is not None:
        return _cached_model, _cached_feats
    import xgboost as xgb
    if not MODEL_PATH.exists() or not FEATS_PATH.exists():
        raise FileNotFoundError(f"Trained ranker not found at {MODEL_PATH}")
    booster = xgb.XGBRanker()
    booster.load_model(str(MODEL_PATH))
    with open(FEATS_PATH) as f:
        feats = json.load(f)
    _cached_model = booster
    _cached_feats = feats
    return booster, feats


def is_trained() -> bool:
    return MODEL_PATH.exists() and FEATS_PATH.exists()


def score(features_df: pd.DataFrame) -> pd.Series:
    """
    Score a DataFrame of features and return a pd.Series in [0, 1].
    Raw ranker outputs are min-max normalized WITHIN this call so priority is
    relative to the cohort being scored (e.g. one rep's territory).
    """
    model, feats = _load()
    missing = [c for c in feats if c not in features_df.columns]
    if missing:
        raise ValueError(f"score(): missing columns: {missing}")
    X = features_df[feats].astype(float)
    raw = model.predict(X)
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-9:
        return pd.Series(np.full(len(raw), 0.5), index=features_df.index)
    return pd.Series((raw - lo) / (hi - lo), index=features_df.index)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["train", "test-score"])
    p.add_argument("--date", default=TRAIN_DATE_DEFAULT,
                   help="Train: training-features date. test-score: features date to score.")
    args = p.parse_args()

    if args.action == "train":
        train(train_date=args.date)
    elif args.action == "test-score":
        df = pd.read_parquet(DATA / f"features_{args.date}.parquet")
        df["_score"] = score(df)
        print(df[["retailer_id", "name", "territory_id", "_score"]]
              .sort_values("_score", ascending=False).head(15).to_string(index=False))