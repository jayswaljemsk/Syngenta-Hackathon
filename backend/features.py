"""
features.py — turn raw Syngenta data into 32 model-ready features per CONTRACT.md §5.

Output: data/features_<date>.parquet, one row per retailer for the target date.

Data adaptations vs CONTRACT.md (honest scoping):
- retailers.csv has no lat/lng: synthesized per tehsil with deterministic
  hash-jitter around district centroids.
- retailer_visit_log.csv has no retailer_id: visits broadcast to all retailers
  in the visited tehsil. days_since_last_visit / visits_last_90d are
  tehsil-broadcast values, not strictly per-outlet.
- No outcome / sale_made column: labels are synthesized separately in
  make_labels.py (POS within N days post-visit-to-tehsil where promoted
  product was sold).
- weather/NDVI/pest/mandi Parquet files (CONTRACT.md §6) absent: synthesized
  fallback in-memory with matching schema. Drop real Parquet into data/ to
  swap in — loaders detect file presence.
- season_flag_kharif_rabi_zaid is always 'rabi' (dataset Oct 2025 – Mar 2026).
- Pilot district pivoted from Yavatmal (absent in data) to Sehore, MP.

Usage:
    cd backend
    python features.py                    # latest date in data
    python features.py --date 2026-03-15  # explicit date
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REAL = DATA / "real"
SYN = DATA / "syngenta_synthetic"


def get_dataset_root() -> Path:
    """Prefer the real dataset if present, else fall back to synthetic."""
    required = [
        "retailers.csv",
        "reps_territory.csv",
        "retailer_pos.csv",
        "retailer_inventory_weekly.csv",
        "retailer_visit_log.csv",
        "growers.csv",
    ]
    if REAL.exists() and all((REAL / f).exists() for f in required):
        return REAL
    return SYN


# ---------------------------------------------------------------------------
# Geography lookup for synthesizing lat/lng from tehsil ids
# ---------------------------------------------------------------------------
DISTRICT_CENTROIDS = {
    ("Madhya Pradesh", "Sehore"):      (23.2030, 77.0807),
    ("Madhya Pradesh", "Ratlam"):      (23.3315, 75.0367),
    ("Madhya Pradesh", "Indore"):      (22.7196, 75.8577),
    ("Madhya Pradesh", "Ujjain"):      (23.1765, 75.7885),
    ("Maharashtra",    "Jalgaon"):     (20.9999, 75.5667),
    ("Maharashtra",    "Akola"):       (20.7000, 77.0000),
    ("Maharashtra",    "Amravati"):    (20.9333, 77.7500),
    ("Bihar",          "Patna"):       (25.5941, 85.1376),
    ("Bihar",          "Muzaffarpur"): (26.1209, 85.3647),
    ("Haryana",        "Hisar"):       (29.1492, 75.7217),
    ("Haryana",        "Sirsa"):       (29.5333, 75.0167),
    ("Haryana",        "Karnal"):      (29.6857, 76.9905),
    ("Haryana",        "Rohtak"):      (28.8955, 76.6066),
    ("Uttar Pradesh",  "Varanasi"):    (25.3176, 82.9739),
    ("Uttar Pradesh",  "Lucknow"):     (26.8467, 80.9462),
    ("Uttar Pradesh",  "Agra"):        (27.1767, 78.0081),
    ("Uttar Pradesh",  "Kanpur Nagar"): (26.4499, 80.3319),
    ("Uttar Pradesh",  "Meerut"):      (28.9845, 77.7064),
    ("Rajasthan",      "Bharatpur"):   (27.2152, 77.5030),
    ("Rajasthan",      "Bikaner"):     (28.0229, 73.3119),
    ("Rajasthan",      "Sikar"):       (27.6094, 75.1399),
    ("Rajasthan",      "Jaipur"):      (26.9124, 75.7873),
    ("Punjab",         "Ludhiana"):    (30.9010, 75.8573),
    ("Punjab",         "Amritsar"):    (31.6340, 74.8723),
    ("Punjab",         "Bathinda"):    (30.2100, 74.9455),
    ("Punjab",         "Patiala"):     (30.3398, 76.3869),
    ("Karnataka",      "Kalaburagi"):  (17.3297, 76.8343),
    ("Karnataka",      "Vijayapura"):  (16.8302, 75.7100),
    ("Gujarat",        "Ahmedabad"):   (23.0225, 72.5714),
    ("Gujarat",        "Mehsana"):     (23.5880, 72.3693),
    ("Gujarat",        "Rajkot"):      (22.3039, 70.8022),
    ("West Bengal",    "Bardhaman"):   (23.2329, 87.8615),
    ("West Bengal",    "Nadia"):       (23.4057, 88.4907),
}
STATE_CENTROIDS = {
    "Madhya Pradesh":   (23.4733, 77.9479),
    "Maharashtra":      (19.7515, 75.7139),
    "Bihar":            (25.0961, 85.3131),
    "Haryana":          (29.0588, 76.0856),
    "Uttar Pradesh":    (26.8467, 80.9462),
    "Rajasthan":        (27.0238, 74.2179),
    "Punjab":           (31.1471, 75.3412),
    "Karnataka":        (15.3173, 75.7139),
    "Gujarat":          (22.2587, 71.1924),
    "West Bengal":      (22.9868, 87.8550),
}
INDIA = (22.5937, 78.9629)


def _hash01(s: str) -> tuple[float, float]:
    h = hashlib.md5(s.encode()).digest()
    return (
        int.from_bytes(h[0:4], "big") / 2**32,
        int.from_bytes(h[4:8], "big") / 2**32,
    )


def tehsil_latlng(state: str, district: str, tehsil: str,
                  retailer_id: str | None = None,
                  scale: float = 0.15, retailer_scale: float = 0.006):
    """Deterministic synthetic lat/lng. If retailer_id is given, add a small
    extra jitter so retailers sharing a tehsil don't collide on the map.
    Tehsil-level jitter: ~16km. Retailer-level jitter: ~600m."""
    base = (
        DISTRICT_CENTROIDS.get((state, district))
        or STATE_CENTROIDS.get(state)
        or INDIA
    )
    a, b = _hash01(tehsil)
    lat = base[0] + (a - 0.5) * 2 * scale
    lng = base[1] + (b - 0.5) * 2 * scale
    if retailer_id:
        ra, rb = _hash01(retailer_id)
        lat += (ra - 0.5) * 2 * retailer_scale
        lng += (rb - 0.5) * 2 * retailer_scale
    return lat, lng


# ---------------------------------------------------------------------------
# Retailer name synthesis (dataset has no names; demo needs them)
# ---------------------------------------------------------------------------
NAME_TEMPLATES = [
    "{s} Krishi Kendra", "{s} Agri Mart", "{s} Beej Bhandar",
    "{s} Khad Beej", "{s} Agro Centre", "{s} Krishi Seva",
    "{s} Farm Solutions", "Shri {s} Krishi",
]
SURNAMES = [
    "Sharma", "Patel", "Yadav", "Kumar", "Verma", "Singh", "Gupta",
    "Mishra", "Pandey", "Rajput", "Tiwari", "Joshi", "Reddy", "Naidu",
    "Iyer", "Kulkarni", "Choudhary", "Agarwal", "Jain", "Mehta",
    "Saxena", "Trivedi",
]


def retailer_name(retailer_id: str) -> str:
    a, b = _hash01(retailer_id)
    s = SURNAMES[int(a * len(SURNAMES))]
    tmpl = NAME_TEMPLATES[int(b * len(NAME_TEMPLATES))]
    return tmpl.format(s=s)


# ---------------------------------------------------------------------------
# SKU → target crops mapping (used for crop_match and growth-stage features)
# ---------------------------------------------------------------------------
SKU_CROPS = {
    "Actara 25 WG":      ["chickpea", "wheat", "potato", "mustard", "safflower"],
    "Alto 5 SC":         ["wheat", "mustard", "cumin"],
    "Amistar 250 SC":    ["wheat", "potato", "chickpea", "cumin"],
    "Axial 50 EC":       ["wheat"],
    "Cruiser 350 FS":    ["wheat", "chickpea", "mustard", "potato", "maize"],
    "Kavach 75 WP":      ["potato", "wheat", "cumin"],
    "Movondo":           ["wheat", "mustard", "safflower"],
    "Score 250 EC":      ["mustard", "wheat", "cumin"],
    "Tilt 250 EC":       ["wheat", "safflower"],
    "Topik 15 WP":       ["wheat"],
    "Vertimec 1.8 EC":   ["potato", "mustard", "safflower"],
    "Vibrance Integral": ["wheat", "chickpea"],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_syngenta():
    root = get_dataset_root()
    print(f"[load] Syngenta dataset from {root}")
    reps = pd.read_csv(root / "reps_territory.csv")
    retailers = pd.read_csv(root / "retailers.csv")
    visits = pd.read_csv(root / "retailer_visit_log.csv", parse_dates=["visit_date"])
    inv = pd.read_csv(root / "retailer_inventory_weekly.csv", parse_dates=["week_end_date"])
    pos = pd.read_csv(root / "retailer_pos.csv", parse_dates=["transaction_date"])
    growers = pd.read_csv(root / "growers.csv")
    return reps, retailers, visits, inv, pos, growers


def _synth_weather(tehsils, target_date, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in tehsils.iterrows():
        lat, lng = tehsil_latlng(r["state"], r["district"], r["tehsil"])
        for d in range(-7, 8):
            dt = target_date + timedelta(days=d)
            doy = dt.timetuple().tm_yday
            # Rabi temperature curve: cold Dec-Jan, warming through Mar
            seasonal = -6 * np.cos(2 * np.pi * (doy - 15) / 365)
            rows.append({
                "date": dt,
                "pin_code": r["tehsil"],
                "lat": float(lat), "lng": float(lng),
                "temp_max_c": float(24 + seasonal + rng.normal(0, 2.5)),
                "temp_min_c": float(11 + seasonal + rng.normal(0, 2.0)),
                "rain_mm": float(max(0.0, rng.gamma(0.4, 4))),
                "rain_probability": float(np.clip(rng.beta(2, 6), 0, 1)),
                "humidity": float(np.clip(55 + rng.normal(0, 12), 20, 95)),
                "wind_kmh": float(max(0.0, rng.gamma(2.0, 3.0))),
            })
    return pd.DataFrame(rows)


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def load_or_synth_weather(retailers, target_date):
    fp = _first_existing([
        REAL / "weather.parquet",
        REAL / "weather_yavatmal.parquet",
        DATA / "weather.parquet",
        DATA / "weather_yavatmal.parquet",
    ])
    if fp:
        print(f"[load] weather: {fp.name}")
        return pd.read_parquet(fp)
    print("[synth] weather (fallback)")
    return _synth_weather(retailers[["state","district","tehsil"]].drop_duplicates(), target_date)


def load_or_synth_ndvi(retailers, target_date):
    fp = _first_existing([
        REAL / "ndvi.parquet",
        REAL / "ndvi_yavatmal.parquet",
        DATA / "ndvi.parquet",
        DATA / "ndvi_yavatmal.parquet",
    ])
    if fp:
        print(f"[load] ndvi: {fp.name}")
        return pd.read_parquet(fp)
    print("[synth] ndvi (fallback)")
    rng = np.random.default_rng(7)
    tehsils = retailers[["state","district","tehsil"]].drop_duplicates()
    rows = []
    doy_now = target_date.timetuple().tm_yday
    doy_then = (target_date - timedelta(days=14)).timetuple().tm_yday
    for _, r in tehsils.iterrows():
        lat, lng = tehsil_latlng(r["state"], r["district"], r["tehsil"])
        # Rabi NDVI: rises Nov→Jan, plateaus, declines after Feb
        def rabi_ndvi(d):
            # Rabi growing season approx Nov 15 (doy 319) → Apr 1 (doy 91 of next year)
            # Treat by mapping doy to a 0..1 phase in the rabi window
            if d >= 319:
                phase = (d - 319) / (366 - 319 + 91)
            elif d <= 91:
                phase = (47 + d) / (366 - 319 + 91)
            else:
                phase = 0  # off-season
            # Bell curve peak at phase=0.55
            return 0.2 + 0.55 * np.exp(-((phase - 0.55) / 0.30) ** 2)
        base_now = rabi_ndvi(doy_now)
        base_then = rabi_ndvi(doy_then)
        ndvi_now = float(np.clip(base_now + rng.normal(0, 0.04), 0, 1))
        ndvi_then = float(np.clip(base_then + rng.normal(0, 0.04), 0, 1))
        rows.append({
            "date": target_date,
            "tile_id": r["tehsil"],
            "lat": float(lat), "lng": float(lng),
            "ndvi": ndvi_now,
            "ndvi_delta_14d": ndvi_now - ndvi_then,
        })
    return pd.DataFrame(rows)


def load_or_synth_pest(retailers, target_date):
    fp = _first_existing([
        REAL / "pest_bulletin.parquet",
        REAL / "pest.parquet",
        DATA / "pest_bulletin.parquet",
        DATA / "pest.parquet",
    ])
    if fp:
        print(f"[load] pest: {fp.name}")
        return pd.read_parquet(fp)
    print("[synth] pest (fallback)")
    rng = np.random.default_rng(11)
    crops = ["wheat","mustard","chickpea","potato","lentil","barley"]
    pests = {
        "wheat":["aphid","rust","termite"], "mustard":["aphid","alternaria_blight"],
        "chickpea":["pod_borer","wilt"], "potato":["late_blight","jassid"],
        "lentil":["pod_borer","wilt"], "barley":["aphid","rust"],
    }
    week_start = target_date - timedelta(days=target_date.weekday())
    geo = retailers[["state","district"]].drop_duplicates()
    rows = []
    for _, r in geo.iterrows():
        for c in crops:
            for p in pests[c]:
                pressure = float(np.clip(rng.beta(2, 5) + rng.normal(0, 0.05), 0.05, 0.95))
                rows.append({
                    "week_start": week_start,
                    "state": r["state"], "district": r["district"],
                    "crop": c, "pest": p,
                    "pressure_index": pressure,
                })
    return pd.DataFrame(rows)


def load_or_synth_mandi(retailers, target_date):
    fp = _first_existing([
        REAL / "mandi_prices.parquet",
        REAL / "mandi.parquet",
        DATA / "mandi_prices.parquet",
        DATA / "mandi.parquet",
    ])
    if fp:
        print(f"[load] mandi: {fp.name}")
        return pd.read_parquet(fp)
    print("[synth] mandi (fallback)")
    rng = np.random.default_rng(17)
    commods = {"wheat":2200, "mustard":5500, "chickpea":5800, "potato":1400,
               "lentil":7200, "barley":1900}
    geo = retailers[["state","district"]].drop_duplicates()
    rows = []
    for _, r in geo.iterrows():
        mandi_name = f"{r['district']} Krishi Upaj Mandi"
        for c, base in commods.items():
            modal = base * (1 + rng.normal(0, 0.06))
            rows.append({
                "date": target_date,
                "mandi": mandi_name,
                "commodity": c,
                "price_min_inr_qtl":   float(modal * 0.93),
                "price_max_inr_qtl":   float(modal * 1.07),
                "price_modal_inr_qtl": float(modal),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dominant crop per retailer (mode of growers' crop in retailer's tehsil)
# ---------------------------------------------------------------------------
def compute_dominant_crops(retailers, growers):
    def parse(s):
        try:
            return json.loads(s).get("crop")
        except Exception:
            return None
    g = growers.copy()
    g["crop"] = g["grower_crop_calendar"].apply(parse)
    g = g.dropna(subset=["crop"])
    tehsil_crop = g.groupby("tehsil")["crop"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) else "wheat"
    ).to_dict()
    district_crop = g.groupby("district")["crop"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) else "wheat"
    ).to_dict()
    result = {}
    for _, r in retailers.iterrows():
        crop = tehsil_crop.get(r["tehsil"]) or district_crop.get(r["district"], "wheat")
        result[r["retailer_id"]] = crop
    return result, tehsil_crop


# ---------------------------------------------------------------------------
# Growth stage estimate from Rabi calendar
# ---------------------------------------------------------------------------
RABI_STAGES = {
    "wheat":    [("sowing","2025-11-01"), ("tillering","2026-01-15"),
                 ("flowering","2026-02-20"), ("grain_filling","2026-03-10"),
                 ("harvest","2026-03-25")],
    "mustard":  [("sowing","2025-10-15"), ("vegetative","2025-12-01"),
                 ("flowering","2026-01-20"), ("pod_filling","2026-02-15"),
                 ("harvest","2026-03-10")],
    "chickpea": [("sowing","2025-10-25"), ("vegetative","2025-12-15"),
                 ("flowering","2026-01-25"), ("pod_filling","2026-02-20"),
                 ("harvest","2026-03-25")],
    "potato":   [("sowing","2025-10-20"), ("vegetative","2025-12-01"),
                 ("tuberization","2026-01-05"), ("bulking","2026-02-01"),
                 ("harvest","2026-03-01")],
    "lentil":   [("sowing","2025-11-01"), ("vegetative","2025-12-15"),
                 ("flowering","2026-01-30"), ("pod_filling","2026-02-25"),
                 ("harvest","2026-03-30")],
    "barley":   [("sowing","2025-11-05"), ("tillering","2026-01-10"),
                 ("flowering","2026-02-15"), ("grain_filling","2026-03-05"),
                 ("harvest","2026-03-25")],
}
STAGE_INDEX = {  # ordinal index for growth_stage_estimate
    "sowing":0, "vegetative":1, "tillering":1, "tuberization":1.5,
    "flowering":2, "pod_filling":2.5, "grain_filling":2.5, "bulking":2.5,
    "harvest":3,
}


def growth_stage(crop, target_date):
    cal = RABI_STAGES.get(crop, RABI_STAGES["wheat"])
    parsed = [(name, pd.Timestamp(d).date()) for name, d in cal]
    current_stage = parsed[0][0]
    next_stage_date = None
    for name, d in parsed:
        if d <= target_date:
            current_stage = name
        elif next_stage_date is None:
            next_stage_date = d
            break
    days_to_next = (next_stage_date - target_date).days if next_stage_date else 0
    return current_stage, max(0, days_to_next), STAGE_INDEX.get(current_stage, 0)


# ---------------------------------------------------------------------------
# Feature buckets
# ---------------------------------------------------------------------------
def bucket_A_rfm(retailers, visits, pos, target_date):
    """A. Outlet RFM (5)."""
    tgt = pd.Timestamp(target_date)
    pos_p = pos[pos["transaction_date"] <= tgt].copy()
    pos_p["line_value"] = pos_p["sku_qty"] * pos_p["sku_price"]
    vis_p = visits[visits["visit_date"] <= tgt]

    last_visit_by_tehsil = vis_p.groupby("visit_tehsil")["visit_date"].max().to_dict()
    cutoff90 = tgt - pd.Timedelta(days=90)
    visits90 = vis_p[vis_p["visit_date"] >= cutoff90]
    visits90_count = visits90.groupby("visit_tehsil").size().to_dict()

    last_purchase = pos_p.sort_values("transaction_date").groupby("retailer_id")["line_value"].last().to_dict()
    cutoff180 = tgt - pd.Timedelta(days=180)
    pos180 = pos_p[pos_p["transaction_date"] >= cutoff180]
    total180 = pos180.groupby("retailer_id")["line_value"].sum().to_dict()
    txn = pos_p.groupby(["retailer_id","transaction_id"])["line_value"].sum().reset_index()
    aov = txn.groupby("retailer_id")["line_value"].mean().to_dict()

    rows = []
    for _, r in retailers.iterrows():
        rid = r["retailer_id"]; th = r["tehsil"]
        lv = last_visit_by_tehsil.get(th)
        dslv = (tgt - lv).days if lv is not None and pd.notna(lv) else 999
        rows.append({
            "retailer_id": rid,
            "days_since_last_visit":   int(dslv),
            "last_purchase_value_inr": float(last_purchase.get(rid, 0.0)),
            "visits_last_90d":         int(visits90_count.get(th, 0)),
            "total_sales_180d":        float(total180.get(rid, 0.0)),
            "avg_order_value_inr":     float(aov.get(rid, 0.0)),
        })
    return pd.DataFrame(rows)


def bucket_B_geo(retailers, weather, ndvi, pest, mandi, target_date, dom_crops):
    """B. Geo signals (8) — plus weather_volatility_7d for bucket F."""
    tgt = pd.Timestamp(target_date)
    w = weather.copy()
    w["date"] = pd.to_datetime(w["date"])
    wnext48 = w[(w["date"] > tgt) & (w["date"] <= tgt + pd.Timedelta(days=2))]
    rain48 = wnext48.groupby("pin_code")["rain_mm"].sum().to_dict()
    wnext72 = w[(w["date"] > tgt) & (w["date"] <= tgt + pd.Timedelta(days=3))]
    rainprob72 = wnext72.groupby("pin_code")["rain_probability"].max().to_dict()
    wlast = w[(w["date"] >= tgt - pd.Timedelta(days=7)) & (w["date"] < tgt)]
    base_temp = wlast.groupby("pin_code")["temp_max_c"].mean().to_dict()
    today_temp = w[w["date"].dt.normalize() == tgt.normalize()].groupby("pin_code")["temp_max_c"].mean().to_dict()
    vol7 = wlast.groupby("pin_code")["rain_mm"].std().to_dict()

    ndvi_now = ndvi.set_index("tile_id")["ndvi"].to_dict()
    ndvi_d14 = ndvi.set_index("tile_id")["ndvi_delta_14d"].to_dict()

    pest_idx = pest.set_index(["state","district","crop"])["pressure_index"].to_dict()
    overall_modal = mandi.groupby("commodity")["price_modal_inr_qtl"].mean().to_dict()
    mandi_lookup = mandi.assign(
        mandi_district=mandi["mandi"].str.replace(" Krishi Upaj Mandi", "", regex=False)
    ).set_index(["mandi_district","commodity"])["price_modal_inr_qtl"].to_dict()

    rows = []
    for _, r in retailers.iterrows():
        rid = r["retailer_id"]; th = r["tehsil"]; st = r["state"]; dt = r["district"]
        crop = dom_crops.get(rid, "wheat")
        r48 = float(rain48.get(th, 0.0))
        p72 = float(rainprob72.get(th, 0.0))
        bt  = float(base_temp.get(th, 22.0))
        tt  = float(today_temp.get(th, bt))
        spray = 1 if (r48 < 5 and p72 < 0.4) else 0
        pp = float(pest_idx.get((st, dt, crop), 0.3))
        modal_here = float(mandi_lookup.get((dt, crop), overall_modal.get(crop, 2000)))
        modal_mean = float(overall_modal.get(crop, modal_here))
        anomaly = (modal_here - modal_mean) / modal_mean if modal_mean else 0.0
        rows.append({
            "retailer_id": rid,
            "pest_pressure_idx":      pp,
            "ndvi_current":           float(ndvi_now.get(th, 0.4)),
            "ndvi_delta_14d":         float(ndvi_d14.get(th, 0.0)),
            "rain_mm_next_48h":       r48,
            "rain_probability_72h":   p72,
            "temp_anomaly_c":         tt - bt,
            "spray_window_open_flag": int(spray),
            "mandi_price_anomaly":    float(anomaly),
            "weather_volatility_7d":  float(vol7.get(th, 0.0)),
        })
    return pd.DataFrame(rows)


def bucket_C_inventory(retailers, inv, pos, target_date, dom_crops):
    """C. Inventory gap (4) — uses retailer's dominant SKU among those that match their dominant crop."""
    tgt = pd.Timestamp(target_date)
    # Inventory: most recent snapshot ≤ target_date per (retailer, sku)
    inv_p = inv[inv["week_end_date"] <= tgt].copy()
    latest_inv = inv_p.sort_values("week_end_date").groupby(["retailer_id","sku_name"]).tail(1)
    last4w = tgt - pd.Timedelta(weeks=4)
    inv_4w = inv_p[inv_p["week_end_date"] >= last4w]

    # Sales velocity last 30d per (retailer, sku)
    cutoff30 = tgt - pd.Timedelta(days=30)
    pos_p = pos[(pos["transaction_date"] <= tgt) & (pos["transaction_date"] >= cutoff30)]
    velocity = pos_p.groupby(["retailer_id","sku_name"])["sku_qty"].sum() / 30.0

    SAFETY_STOCK = 8

    # Per retailer: choose a recommended_sku = SKU with highest stock among those that match dom crop
    rows = []
    for _, r in retailers.iterrows():
        rid = r["retailer_id"]
        crop = dom_crops.get(rid, "wheat")
        candidate_skus = [s for s, cs in SKU_CROPS.items() if crop in cs]
        ret_inv = latest_inv[latest_inv["retailer_id"] == rid]
        # If retailer's latest inventory has none of the candidate SKUs, fall back to its top-stocked SKU
        if len(ret_inv) == 0:
            stock = 0; sku_pick = candidate_skus[0] if candidate_skus else "Topik 15 WP"
        else:
            ret_inv_cand = ret_inv[ret_inv["sku_name"].isin(candidate_skus)]
            if len(ret_inv_cand) == 0:
                # Pick the candidate with highest national avg stock proxy via latest_inv
                sku_pick = candidate_skus[0] if candidate_skus else ret_inv.iloc[0]["sku_name"]
                stock = 0
            else:
                pick_row = ret_inv_cand.sort_values("sku_qty", ascending=False).iloc[0]
                sku_pick = pick_row["sku_name"]
                stock = int(pick_row["sku_qty"])
        v = float(velocity.get((rid, sku_pick), 0.0))
        days_cover = float(stock / v) if v > 0 else float(999.0 if stock > 0 else 0.0)
        gap = max(0, SAFETY_STOCK - stock)
        stockout = int((inv_4w[(inv_4w["retailer_id"] == rid) & (inv_4w["sku_name"] == sku_pick)]["sku_qty"] == 0).any())
        rows.append({
            "retailer_id": rid,
            "recommended_sku_pick": sku_pick,  # carried for downstream NBA, not a CONTRACT feature
            "stock_of_recommended_sku": int(stock),
            "days_of_cover":           float(min(days_cover, 999.0)),
            "safety_stock_gap":        int(gap),
            "recent_stockout_flag":    int(stockout),
        })
    return pd.DataFrame(rows)


def bucket_D_growth(retailers, target_date, dom_crops, tehsil_crop):
    """D. Growth stage relevance (5)."""
    rows = []
    for _, r in retailers.iterrows():
        rid = r["retailer_id"]
        crop = dom_crops.get(rid, "wheat")
        stage_name, days_to_next, stage_idx = growth_stage(crop, target_date)
        # crop_match_score against the recommended SKU is computed downstream where SKU is known;
        # here, we compute the per-retailer score against its OWN dominant crop matching all candidate SKUs
        # crop_match_score := fraction of SKUs in catalog that target this retailer's crop (proxy of market fit)
        match = sum(1 for cs in SKU_CROPS.values() if crop in cs) / len(SKU_CROPS)
        rows.append({
            "retailer_id": rid,
            "dominant_crop_in_radius_5km": crop,  # tehsil-mode of growers' crops
            "growth_stage_estimate":       float(stage_idx),
            "days_to_next_critical_stage": int(days_to_next),
            "crop_match_score":            float(match),
            "season_flag_kharif_rabi_zaid": "rabi",
        })
    return pd.DataFrame(rows)


def bucket_E_rep(retailers, reps, visits, pos, target_date):
    """E. Rep affinity (4)."""
    tgt = pd.Timestamp(target_date)
    vis_p = visits[visits["visit_date"] <= tgt]
    # Map territory_id → rep_id
    terr_to_rep = dict(zip(reps["territory_id"], reps["rep_id"]))
    # For each retailer, the rep is the rep of its territory
    rows = []
    # Pre-compute: visits by (rep_id, visit_tehsil)
    rep_tehsil_visits = vis_p.groupby(["rep_id","visit_tehsil"]).size().to_dict()
    # Close rate proxy: count rep's tehsil visits where a POS sale of the promoted product happened
    # within 14 days at any retailer in that tehsil.
    pos_p = pos[pos["transaction_date"] <= tgt][["retailer_id","sku_name","transaction_date"]].merge(
        retailers[["retailer_id","tehsil"]], on="retailer_id"
    )
    # Build a quick lookup: set of (tehsil, sku, date) for fast membership
    pos_p_set = pos_p.set_index(["tehsil","sku_name"])  # date column carried

    # Approximate close rate per rep: faster via vectorized join
    vis_join = vis_p.merge(
        pos_p, left_on=["visit_tehsil","product_recommended"],
        right_on=["tehsil","sku_name"], how="left"
    )
    vis_join["delta_days"] = (vis_join["transaction_date"] - vis_join["visit_date"]).dt.days
    vis_join["converted"] = ((vis_join["delta_days"] >= 0) & (vis_join["delta_days"] <= 14)).astype(int)
    rep_close = vis_join.groupby("rep_id")["converted"].mean().to_dict()

    for _, r in retailers.iterrows():
        rid = r["retailer_id"]; th = r["tehsil"]
        rep_id = terr_to_rep.get(r["territory_id"], None)
        lifetime = int(rep_tehsil_visits.get((rep_id, th), 0)) if rep_id else 0
        close_rate = float(rep_close.get(rep_id, 0.0)) if rep_id else 0.0
        # dwell minutes: not in data — synthesize deterministically by retailer_id
        a, _ = _hash01(rid)
        dwell = 20 + 25 * a  # 20–45 min range
        familiarity = float(np.log1p(lifetime) * (0.3 + close_rate))
        rows.append({
            "retailer_id": rid,
            "rep_visits_to_outlet_lifetime": lifetime,
            "rep_close_rate_at_outlet":      close_rate,
            "rep_avg_dwell_minutes":         float(dwell),
            "rep_familiarity_score":         familiarity,
        })
    return pd.DataFrame(rows)


def bucket_F_temporal(retailers, target_date, weather_volatility):
    """F. Temporal (6) — note: weather_volatility_7d already computed in bucket B."""
    season_start = date(2025, 10, 1)
    rows = []
    for _, r in retailers.iterrows():
        rid = r["retailer_id"]; th = r["tehsil"]
        dow = target_date.weekday()  # Mon=0
        wom = (target_date.day - 1) // 7 + 1
        wos = ((target_date - season_start).days // 7) + 1
        # Market day: Tue (1) and Fri (4) by MP convention
        is_market = 1 if dow in (1, 4) else 0
        # Days since competitor visit: NOT in data — synthesize deterministically
        a, _ = _hash01(rid + "competitor")
        days_competitor = int(7 + a * 30)  # 7–37 days
        rows.append({
            "retailer_id": rid,
            "day_of_week":               int(dow),
            "week_of_month":             int(wom),
            "week_of_season":            int(wos),
            "is_market_day_flag":        int(is_market),
            "days_since_competitor_visit": days_competitor,
            # weather_volatility_7d sourced from bucket B
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD")
    args = p.parse_args()

    reps, retailers, visits, inv, pos, growers = load_syngenta()

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        # latest Sunday week_end_date in inventory
        target_date = pd.to_datetime(inv["week_end_date"].max()).date()
    print(f"[date] target: {target_date}")

    weather = load_or_synth_weather(retailers, target_date)
    ndvi    = load_or_synth_ndvi(retailers, target_date)
    pest    = load_or_synth_pest(retailers, target_date)
    mandi   = load_or_synth_mandi(retailers, target_date)

    dom_crops, tehsil_crop = compute_dominant_crops(retailers, growers)
    print(f"[crops] retailers with dom crop assigned: {len(dom_crops)}")

    print("[bucket A] RFM …");        A = bucket_A_rfm(retailers, visits, pos, target_date)
    print("[bucket B] Geo …");        B = bucket_B_geo(retailers, weather, ndvi, pest, mandi, target_date, dom_crops)
    print("[bucket C] Inventory …");  C = bucket_C_inventory(retailers, inv, pos, target_date, dom_crops)
    print("[bucket D] Growth …");     D = bucket_D_growth(retailers, target_date, dom_crops, tehsil_crop)
    print("[bucket E] Rep …");        E = bucket_E_rep(retailers, reps, visits, pos, target_date)
    print("[bucket F] Temporal …");   F = bucket_F_temporal(retailers, target_date, B["weather_volatility_7d"])

    # Identity / geography columns the API needs
    ident = retailers[["retailer_id","territory_id","state","district","tehsil"]].copy()
    ident["name"] = ident["retailer_id"].apply(retailer_name)
    ident["address"] = ident["tehsil"] + ", " + ident["district"] + ", " + ident["state"]
    latlng = pd.DataFrame(
        [(rid, *tehsil_latlng(s, d, t, retailer_id=rid)) for rid, s, d, t in zip(
            ident["retailer_id"], ident["state"], ident["district"], ident["tehsil"])],
        columns=["retailer_id","lat","lng"]
    )
    ident = ident.merge(latlng, on="retailer_id")

    # Merge all
    out = ident
    for X in [A, B, C, D, E, F]:
        out = out.merge(X, on="retailer_id", how="left")
    out["date"] = pd.Timestamp(target_date)

    # Verify column inventory matches CONTRACT.md §5
    expected_features = {
        # A
        "days_since_last_visit","last_purchase_value_inr","visits_last_90d",
        "total_sales_180d","avg_order_value_inr",
        # B
        "pest_pressure_idx","ndvi_current","ndvi_delta_14d","rain_mm_next_48h",
        "rain_probability_72h","temp_anomaly_c","spray_window_open_flag","mandi_price_anomaly",
        # C
        "stock_of_recommended_sku","days_of_cover","safety_stock_gap","recent_stockout_flag",
        # D
        "dominant_crop_in_radius_5km","growth_stage_estimate","days_to_next_critical_stage",
        "crop_match_score","season_flag_kharif_rabi_zaid",
        # E
        "rep_visits_to_outlet_lifetime","rep_close_rate_at_outlet",
        "rep_avg_dwell_minutes","rep_familiarity_score",
        # F (6)
        "day_of_week","week_of_month","week_of_season","is_market_day_flag",
        "days_since_competitor_visit","weather_volatility_7d",
    }
    missing = expected_features - set(out.columns)
    extra_features = set(out.columns) - expected_features - {
        "retailer_id","territory_id","state","district","tehsil","name","address","lat","lng",
        "date","recommended_sku_pick",
    }
    print(f"[verify] expected features present: {len(expected_features) - len(missing)}/32")
    if missing:
        print(f"[verify] MISSING: {sorted(missing)}")
    if extra_features:
        print(f"[verify] extra columns: {sorted(extra_features)}")

    DATA.mkdir(parents=True, exist_ok=True)
    out_path = DATA / f"features_{target_date}.parquet"
    out.to_parquet(out_path, index=False)

    # Summary
    print()
    print("=" * 60)
    print("FEATURES SUMMARY")
    print("=" * 60)
    print(f"Output:    {out_path}")
    print(f"Rows:      {len(out)}")
    print(f"Columns:   {len(out.columns)}")
    print(f"Date:      {target_date}")
    print(f"Nulls:")
    null_counts = out.isna().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts):
        print(null_counts.to_string())
    else:
        print("  (none)")
    print()
    print("Crop distribution among retailers:")
    print(out["dominant_crop_in_radius_5km"].value_counts().to_string())
    print()
    print("Sehore-only sanity check:")
    sehore = out[out["district"] == "Sehore"]
    print(f"  retailers: {len(sehore)}")
    print(sehore[["retailer_id","name","tehsil","lat","lng",
                  "pest_pressure_idx","stock_of_recommended_sku",
                  "days_since_last_visit"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()