# MODELS.md — Syngenta Field Copilot, Backend

> Track 2 — AI-Guided Field Force Intelligence.  
> Owner: Krishna (backend + ML).  
> This file declares every model, every weight, every third-party library
> contributing to inference, and every honest scoping caveat that a reviewer
> needs to evaluate the submission fairly.

---

## 1. Trained model artifacts (live in `backend/models/`)

All artifacts are produced from the synthetic Syngenta dataset (`data/syngenta_synthetic/`, never committed) plus the public-data parquets in `data/`. None are pre-trained on third-party data.

| File | Trained by | Train date | Algorithm | Purpose |
|---|---|---|---|---|
| `priority_model.json` | `priority_model.py train` | 2026-01-15 | XGBoost LambdaMART, `objective=rank:pairwise` | Per-outlet visit-priority score (0..1, min-max normalized within rep cohort) |
| `priority_features.json` | `priority_model.py train` | 2026-01-15 | (metadata) | Ordered feature column list used by the ranker |
| `priority_metrics.json` | `priority_model.py train` | 2026-01-15 | (metadata) | Train/test split sizes, top-k recall at territory, SHAP top-5 |
| `priority_shap_importance.png` | `priority_model.py train` | 2026-01-15 | SHAP TreeExplainer | Global feature importance plot for the priority ranker |
| `nba_model.joblib` | `nba_engine.py train` | 2026-01-15 | `CalibratedClassifierCV(LogisticRegression, method="sigmoid", cv=3)` | P(purchase) for each eligible (outlet, SKU) pair |
| `nba_meta.json` | `nba_engine.py train` | 2026-01-15 | (metadata) | Feature column order + SKU one-hot positions |
| `nba_metrics.json` | `nba_engine.py train` | 2026-01-15 | (metadata) | AUC, Brier, dataset sizes |
| `anomaly_model.joblib` | `anomaly_detector.py train` | 2026-01-15 | `sklearn.ensemble.IsolationForest(n_estimators=200, contamination=0.05)` | Per-retailer POS-pattern outlier flag |
| `anomaly_meta.json` / `anomaly_metrics.json` | `anomaly_detector.py train` | 2026-01-15 | (metadata) | Feature columns + flagged rate |
| `bandit_state.npz` / `bandit_meta.json` | `bandit.py fit` | rolling | LinUCB (Li et al. 2010, disjoint) | Contextual exploration for SKU recommendation. **Not converged in this build.** |

### Label provenance

- **priority_model**: per-retailer label = 1 if the retailer logged ANY crop-matched POS sale in the 14 days following 2026-01-15, else 0. Crop-match uses the `SKU_CROPS` mapping in `features.py`. Source: `data/syngenta_synthetic/retailer_pos.csv`.
- **nba_engine**: per-(retailer, SKU) candidate-pair label = 1 if that SKU was bought at that retailer in the same 14-day window, else 0. Same POS source. Crop-eligible pairs only.
- **anomaly_detector**: unsupervised; no labels.
- **bandit**: rewards = `sale_value_inr` from `/outcome` log when `sale_made == True`, else 0.

### Reproducing

```bash
cd backend
python features.py --date 2026-01-15
python priority_model.py train
python nba_engine.py train
python anomaly_detector.py train
python bandit.py fit     # writes empty-state instrumentation
```

---

## 2. Library / framework licenses

No pre-trained third-party weights ship in this build. Only standard ML libraries are imported. Licenses:

| Library | Used in | License |
|---|---|---|
| xgboost (2.x) | `priority_model.py` | Apache-2.0 |
| scikit-learn (1.5.x) | `nba_engine.py`, `anomaly_detector.py`, splits, calibration | BSD-3-Clause |
| ortools (9.x) | `route_solver.py` | Apache-2.0 |
| shap (0.46.x) | `priority_model.py` (global plot), `explainer.py` (per-row) | MIT |
| matplotlib (3.9.x) | SHAP PNG | matplotlib license (PSF-derived) |
| joblib (1.x) | model persistence | BSD-3-Clause |
| numpy / pandas / scipy | features, all modules | BSD-3-Clause |
| pyarrow | parquet I/O | Apache-2.0 |
| statsmodels | (referenced for STL; not loaded in this build — see §3) | BSD-3-Clause |
| fastapi / pydantic / uvicorn | `main.py` API server | MIT / MIT / BSD-3-Clause |

No GPL, AGPL, or commercial-only library is used. No model weights downloaded from a hub. No HuggingFace artifacts in this build.

---

## 3. Honest scoping — what's real, what's instrumented, what's deferred

A reviewer-facing register of every place we cut a corner consciously. This is the single source of truth for "is this overclaimed?" questions.

### 3.1 Labels are POS-proxy, not rep-causal

Neither `priority_model` nor `nba_engine` has access to rep-causal sales attribution (i.e., "this rep visit caused this sale"). The synthetic dataset has no such field. We use a 14-day forward POS window as a proxy. Implication: the models learn correlations of feature snapshot → future POS, not "if rep visits → sale". This is acceptable for a recommendation system whose downstream value is rep prioritization, not causal inference, but a production deployment would need rep-tagged sales data.

### 3.2 LLM justification is templated, not Phi-3-mini

The original plan called for a Phi-3-mini via Ollama (with a `gpt-4o-mini` fallback) to generate the per-product justification sentence. In this build, justifications are **deterministic templated strings** built from each outlet's actual feature values and SKU category metadata (`nba_engine._justify`, `explainer._render`). No LLM is loaded.

Why: (a) ship-quality requires templates to be inspectable and contract-compliant, with zero hallucination risk on numeric values; (b) a 7B model is more setup overhead than the demo justifies. The LLM call is a transparent swap-in: replace `_justify` with an API or local-Ollama call that takes the same (row, sku, p_purchase) inputs.

### 3.3 STL pest-bulletin decomposition is disabled

`anomaly_detector.py` was specified to use `statsmodels.STL` on weekly pest bulletin counts to flag residuals beyond 2σ over a rolling 8-week window. The current public-data synthesis (`features.py:_synth_weather`/`_synth_pest`) only produces a single week of pest bulletin rows. STL needs ≥ 2 full seasonal periods (≥ 8 weeks at weekly resolution); with one week of data, STL is not learnable. We fall back to cluster heuristics on the daily feature snapshot (elevated_pest, recurring_stockout, ndvi_decline_cluster). With ≥ 8 weeks of real IPM bulletin history (Prince-side input), the STL pathway slots in transparently — same return shape, same anomaly-dict schema.

### 3.4 Maize pest pressure has no public-data coverage

Pest bulletin synthesis covers wheat / mustard / chickpea / potato / lentil / barley. Maize-dominant retailers (32 of 4000) fall back to a default `pest_pressure_idx = 0.30`. That fails the NBA engine's 0.35 insecticide gate; `nba_engine.recommend` then takes the rule-based fallback path, returning `recommended_sku_pick` from `features.py` with a conservative P(purchase) = 0.20. Justification text is honest about the underlying signal level. Production fix: pull IPM maize bulletins; no architectural change needed.

### 3.5 Bandit is instrumented, not converged

`bandit.py` implements disjoint LinUCB (Li et al. 2010), a `(A_a, b_a)` per arm with `theta_a = A_a^{-1} b_a`. The `/outcome` endpoint logs to `data/outcomes_log.csv`, and `bandit.fit_from_outcomes()` updates the per-arm matrices from the log. At demo time, the log holds at most a handful of synthetic outcomes (or zero), so the bandit has no convergence claim. The honest framing on the doc deck is: "Instrumented, awaiting pilot data. In production it fine-tunes recommendations on top of the calibrated logistic by exploring with confidence-bounded exploration."

### 3.6 Distance is Haversine, not road-network

`route_solver.py` uses Haversine great-circle distance, integer-rounded meters, for the OR-Tools VRP distance matrix. The `route_polyline` is `[[lat,lng], ...]` in solver order. An OSRM hook is named in the module docstring and would be a transparent drop-in (same matrix interface). For 9 outlets within a single district, the road-vs-air gap is small.

### 3.7 Synthetic public-data parquets

When `data/weather.parquet`, `data/ndvi.parquet`, `data/pest_bulletin.parquet`, or `data/mandi_prices.parquet` are absent, `features.py` synthesizes plausible parquets matching CONTRACT.md §6. The synth is seeded and deterministic. As of submission, all four are synth (Prince's real downloads landed during integration but were superseded by the synth path for time-zero reproducibility).

### 3.8 Pilot district pivot

CONTRACT.md §9 originally named Yavatmal as the pilot district. The Syngenta synthetic data did not include Yavatmal retailers; we pivoted to Sehore (Madhya Pradesh, wheat belt) for the broader synth and Patna (Bihar, wheat belt) for `REP_0001`'s demo territory. CONTRACT.md §9 should be re-read as Sehore/Patna in the submission package.

### 3.9 Outcome-logging endpoint

`POST /outcome` writes rows to `data/outcomes_log.csv` (`backend/data/outcomes_log.csv` if you run from `backend/`). This file is `.gitignore`'d. It's the bandit's only input stream for now. A production system would attach this to a transactional store.

---

## 4. Headline metrics (frozen on 2026-01-15 train cut, 2026-02-15 demo evaluation)

| Model | Train rows | Test rows | Headline metric | Value |
|---|---:|---:|---|---|
| `priority_model` (XGBoost LambdaMART) | 3,182 (400 territories) | 818 (100 territories) | Top-1 recall at territory | **0.980** |
| `priority_model` | 3,182 | 818 | Top-3 recall at territory | **1.000** |
| `nba_engine` (CalibratedClassifierCV) | 24,433 pairs | 6,023 pairs | Test AUC | **0.621** |
| `nba_engine` | 24,433 | 6,023 | Test Brier score | **0.152** |
| `anomaly_detector` (IsolationForest) | 3,999 | (unsupervised) | Flagged anomaly rate | **5.0%** (by contamination setting) |

Reading guide for the reviewer:
- Priority's high recall reflects the synthetic dataset's structure: a few features (recent_stockout_flag, pest_pressure_idx, days_since_last_visit) carry most of the rank signal. SHAP global plot in `models/priority_shap_importance.png` makes this visible.
- NBA's modest AUC reflects the 19.4% positive base rate and short label window. Stage A (eligibility) does most of the safety work; Stage B sharpens the ordering.
- Anomaly's flagged rate is by IsolationForest's contamination hyperparameter, not by ground-truth labels.

---

## 5. What changes for production

In priority order if the pilot moves forward:
1. Replace POS-proxy labels with rep-tagged sales.
2. Plug a real LLM (Phi-3-mini via Ollama, or a hosted small-model API) into `nba_engine._justify` and `explainer._render` as a stylization layer over the templated strings.
3. Backfill ≥ 8 weeks of pest bulletin history; enable the STL pathway in `anomaly_detector`.
4. Accumulate ≥ 10× SKU-count outcomes in `outcomes_log.csv`, then run `bandit.fit_from_outcomes()` on a daily cron.
5. Swap Haversine for OSRM road-network distances in `route_solver`.