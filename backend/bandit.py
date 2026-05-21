"""
bandit.py -- LinUCB contextual bandit scaffold for SKU recommendation.

The scaffold is fully instrumented but is NOT expected to converge in this
build. Two reasons:

1. The /outcome endpoint logs to data/outcomes_log.csv but no rep has
   produced calibration-grade observations yet.
2. LinUCB requires real arm-reward signal across the SKU catalog; with the
   demo dataset we have no rep-causal reward data.

The honest scoping (folded into MODELS.md) is: "Instrumented, awaiting pilot
data. In production the bandit fine-tunes recommendations on top of the
calibrated logistic by exploring with confidence-bounded exploration."

CLI:
    python bandit.py status                       -- counts logged outcomes
    python bandit.py simulate --n 200             -- generate synthetic outcomes to test fit
    python bandit.py fit                          -- fit on whatever's logged

Importable:
    from bandit import LinUCB, log_outcome_path, status
    s = status()
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

LOG_PATH = DATA / "outcomes_log.csv"
BANDIT_STATE = MODELS / "bandit_state.npz"
BANDIT_META = MODELS / "bandit_meta.json"


# SKU catalog mirrors main.py.SKU_PRODUCT_META keys (kept stable for arm-index identity)
SKU_LIST = [
    "Actara 25 WG", "Alto 5 SC", "Amistar 250 SC", "Axial 50 EC",
    "Cruiser 350 FS", "Kavach 75 WP", "Movondo", "Score 250 EC",
    "Tilt 250 EC", "Topik 15 WP", "Vertimec 1.8 EC", "Vibrance Integral",
]

# Context vector schema: must match the order at /outcome ingest time
CONTEXT_FEATS = [
    "pest_pressure_idx", "stock_of_recommended_sku", "safety_stock_gap",
    "days_since_last_visit", "crop_match_score", "spray_window_open_flag",
    "days_to_next_critical_stage", "growth_stage_estimate",
]


def log_outcome_path() -> Path:
    return LOG_PATH


# ---------------------------------------------------------------------------
# LinUCB (Li et al. 2010, disjoint linear UCB, per-arm A_a / b_a)
# ---------------------------------------------------------------------------
class LinUCB:
    """Disjoint LinUCB. One (A_a, b_a) per SKU arm.

    score(x, a) = theta_a . x  +  alpha * sqrt(x.T A_a^{-1} x)
    where theta_a = A_a^{-1} b_a.
    """

    def __init__(self, n_arms: int, d_context: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.d = d_context
        self.alpha = alpha
        # A_a = I_d  (regularized covariance); b_a = 0
        self.A = np.stack([np.eye(d_context) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, d_context))
        self.n_updates = np.zeros(n_arms, dtype=int)

    def _theta(self, a: int) -> np.ndarray:
        return np.linalg.solve(self.A[a], self.b[a])

    def score(self, x: np.ndarray, a: int) -> float:
        theta = self._theta(a)
        mean = float(theta @ x)
        A_inv = np.linalg.inv(self.A[a])
        bonus = self.alpha * float(np.sqrt(x @ A_inv @ x))
        return mean + bonus

    def recommend(self, x: np.ndarray, eligible: Optional[list[int]] = None) -> tuple[int, float]:
        arms = eligible if eligible is not None else list(range(self.n_arms))
        scored = [(a, self.score(x, a)) for a in arms]
        a_star, s_star = max(scored, key=lambda t: t[1])
        return a_star, s_star

    def update(self, x: np.ndarray, a: int, reward: float) -> None:
        self.A[a] = self.A[a] + np.outer(x, x)
        self.b[a] = self.b[a] + reward * x
        self.n_updates[a] += 1

    # Persistence ----------------------------------------------------------
    def save(self, path: Path = BANDIT_STATE) -> None:
        np.savez(
            path,
            A=self.A, b=self.b, n_updates=self.n_updates,
            alpha=np.array([self.alpha]),
        )

    @classmethod
    def load(cls, path: Path = BANDIT_STATE) -> "LinUCB":
        z = np.load(path)
        A = z["A"]
        n_arms, d, _ = A.shape
        obj = cls(n_arms=n_arms, d_context=d, alpha=float(z["alpha"][0]))
        obj.A = A
        obj.b = z["b"]
        obj.n_updates = z["n_updates"]
        return obj


# ---------------------------------------------------------------------------
# Outcomes -> bandit fit
# ---------------------------------------------------------------------------
def _features_for_date(date_str: str) -> pd.DataFrame:
    fp = DATA / f"features_{date_str}.parquet"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return pd.read_parquet(fp)


def _build_context_vector(features_row: pd.Series) -> np.ndarray:
    return np.array([float(features_row[c]) for c in CONTEXT_FEATS], dtype=float)


def fit_from_outcomes(min_rows: int = 25) -> dict:
    """Fit the bandit on the outcomes log if there's enough signal.

    Returns a metadata dict with the convergence indicator.
    """
    if not LOG_PATH.exists():
        return {"status": "no_log", "n_rows": 0, "converged": False}

    log = pd.read_csv(LOG_PATH)
    n_rows = len(log)
    if n_rows < min_rows:
        # Persist an empty state so /health can still claim instrumentation
        b = LinUCB(n_arms=len(SKU_LIST), d_context=len(CONTEXT_FEATS), alpha=1.0)
        b.save()
        meta = {
            "status": "instrumented_awaiting_data",
            "n_rows": int(n_rows),
            "min_rows_for_fit": int(min_rows),
            "n_arms": len(SKU_LIST),
            "d_context": len(CONTEXT_FEATS),
            "context_feats": CONTEXT_FEATS,
            "sku_list": SKU_LIST,
            "converged": False,
        }
        with open(BANDIT_META, "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    bandit = LinUCB(n_arms=len(SKU_LIST), d_context=len(CONTEXT_FEATS), alpha=1.0)
    fit_count = 0
    skipped = 0
    sku_to_idx = {s: i for i, s in enumerate(SKU_LIST)}
    # We need feature snapshots per (outlet, date) -- read lazily
    feat_cache: dict[str, pd.DataFrame] = {}

    for _, row in log.iterrows():
        try:
            date_str = str(row["date"])
            if date_str not in feat_cache:
                feat_cache[date_str] = _features_for_date(date_str)
            feats = feat_cache[date_str]
            outlet = feats[feats["retailer_id"] == row["outlet_id"]]
            if len(outlet) == 0:
                skipped += 1
                continue
            x = _build_context_vector(outlet.iloc[0])

            # Find the arm index by product_name (preferred) else by product_id
            sku_name = row.get("product_id", "")
            arm = None
            # Try direct SKU name match in the outcome row
            if sku_name in sku_to_idx:
                arm = sku_to_idx[sku_name]
            if arm is None:
                skipped += 1
                continue

            reward = float(row.get("sale_value_inr", 0.0)) if bool(row.get("sale_made", False)) else 0.0
            bandit.update(x, arm, reward)
            fit_count += 1
        except Exception:
            skipped += 1
            continue

    bandit.save()
    meta = {
        "status": "fit",
        "n_rows": int(n_rows),
        "n_updates_applied": int(fit_count),
        "n_skipped": int(skipped),
        "min_rows_for_fit": int(min_rows),
        "n_arms": len(SKU_LIST),
        "d_context": len(CONTEXT_FEATS),
        "context_feats": CONTEXT_FEATS,
        "sku_list": SKU_LIST,
        "updates_per_arm": bandit.n_updates.tolist(),
        "converged": bool(fit_count >= len(SKU_LIST) * 10),  # very rough heuristic
    }
    with open(BANDIT_META, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def status() -> dict:
    """Report current bandit instrumentation status."""
    n_rows = int(len(pd.read_csv(LOG_PATH))) if LOG_PATH.exists() else 0
    fitted = BANDIT_STATE.exists()
    meta = {}
    if BANDIT_META.exists():
        with open(BANDIT_META) as f:
            meta = json.load(f)
    return {
        "instrumented": True,
        "outcomes_logged": n_rows,
        "fitted_state_present": fitted,
        "last_meta": meta,
    }


# ---------------------------------------------------------------------------
# Synthetic outcomes (for the dry-run demo only -- DO NOT ship to production)
# ---------------------------------------------------------------------------
def simulate(n: int = 200, date_str: str = "2026-02-15", seed: int = 42) -> None:
    """Append n synthetic outcome rows to the log, then fit. Useful only as a
    demo aid to show that the bandit COULD be fit if real outcomes existed."""
    rng = np.random.default_rng(seed)
    feats = _features_for_date(date_str)
    rows = []
    for _ in range(n):
        outlet = feats.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        crop = str(outlet["dominant_crop_in_radius_5km"])
        sku = SKU_LIST[int(rng.integers(0, len(SKU_LIST)))]
        # Sale probability biased by pest pressure
        p_sale = float(np.clip(0.10 + 0.6 * outlet["pest_pressure_idx"], 0.05, 0.85))
        sale = bool(rng.random() < p_sale)
        sale_value = float(np.clip(rng.gamma(2.5, 1800), 0, 50000)) if sale else 0.0
        rows.append({
            "rep_id": "REP_0001",
            "outlet_id": str(outlet["retailer_id"]),
            "date": date_str,
            "sale_made": sale,
            "sale_value_inr": sale_value,
            "product_id": sku,
            "dismissal_reason": None,
            "rep_notes": None,
            "id": f"SIM_{rng.integers(10000,99999)}",
            "logged_at": pd.Timestamp.utcnow().isoformat(),
        })
    df = pd.DataFrame(rows)
    if LOG_PATH.exists():
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)
    print(f"[simulate] appended {n} synthetic outcomes to {LOG_PATH}")
    meta = fit_from_outcomes()
    print(f"[simulate] fit_from_outcomes result: {json.dumps(meta, indent=2)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["status", "fit", "simulate"])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--date", default="2026-02-15")
    args = p.parse_args()

    if args.action == "status":
        print(json.dumps(status(), indent=2, default=str))
    elif args.action == "fit":
        print(json.dumps(fit_from_outcomes(), indent=2, default=str))
    elif args.action == "simulate":
        simulate(n=args.n, date_str=args.date)


if __name__ == "__main__":
    main()