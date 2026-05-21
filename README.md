# Syngenta Field Copilot — Backend

Track 2, AI-Guided Field Force Intelligence. FastAPI + scikit-learn + XGBoost + OR-Tools backend that returns a ranked, routed, justified daily plan for a Syngenta sales rep.

> See `MODELS.md` for model artifacts, licenses, and honest scoping.  
> See `../CONTRACT.md` for the locked API + data schemas.

---

## File map

```
backend/
  main.py                 FastAPI app (CONTRACT.md §4 endpoints)
  features.py             Feature engineering pipeline (32 features, CONTRACT.md §5)
  priority_model.py       XGBoost LambdaMART ranker (rank:pairwise)
  route_solver.py         OR-Tools VRP, Haversine matrix
  nba_engine.py           Rule filter + CalibratedClassifierCV(LogisticRegression)
  explainer.py            SHAP per-row top-3 reason strings (deterministic templates)
  anomaly_detector.py     IsolationForest + cluster heuristics
  bandit.py               LinUCB scaffold (instrumented, awaiting pilot data)
  MODELS.md               Model + license declarations
  README.md               (this file)
  models/                 Persisted trained artifacts (NOT committed; rebuild locally)
  data/                   features_<date>.parquet outputs, synthetic Syngenta data
    syngenta_synthetic/   (NOT committed; provided by Syngenta)
    outcomes_log.csv      (NOT committed; written by POST /outcome)
```

---

## Setup

```powershell
# From the repo root
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt    # or: pip install fastapi uvicorn pydantic xgboost scikit-learn ortools shap matplotlib joblib pyarrow pandas numpy
```

Place the Syngenta dataset CSVs in `backend/data/syngenta_synthetic/` (or in the path `get_dataset_root()` resolves — currently `backend/data/real/` based on the project setup).

---

## One-shot rebuild (cold start to green API)

```powershell
# 1. Generate features for the train date AND the demo dates
python features.py --date 2026-01-15
python features.py --date 2026-02-15
python features.py --date 2026-02-17
python features.py --date 2026-03-29

# 2. Train every model
python priority_model.py train       # writes models/priority_model.json + SHAP PNG
python nba_engine.py train           # writes models/nba_model.joblib
python anomaly_detector.py train     # writes models/anomaly_model.joblib
python bandit.py fit                 # writes empty-state bandit instrumentation

# 3. Start the API
uvicorn main:app --reload --port 8000
```

Verify in another terminal:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
```
Expected: all of `priority_model_trained`, `route_solver_available`, `explainer_available`, `nba_model_trained`, `anomaly_model_trained` are `True`.

---

## Endpoints (per CONTRACT.md §4)

### `GET /health`

Liveness + module-trained flags. No params.

### `GET /plan/today?rep_id=REP_0001&date=2026-02-15`

Returns the rep's daily plan: ranked + routed outlets, per-outlet recommendation, anomalies. Top-10 outlets by priority, then ordered by OR-Tools VRP; `route_position` reflects the optimal path, `priority_score` reflects the model's priority axis. They are decoupled.

### `GET /nba/{outlet_id}?date=2026-02-15`

Returns the primary SKU recommendation + 3 alternates for a single outlet, with deterministic templated justifications using actual feature values.

### `POST /outcome`

Logs a visit outcome to `data/outcomes_log.csv`. Body shape per CONTRACT.md §4.3. Drives the `bandit.py` learning loop in production.

---

## Architecture in one paragraph

`features.py` produces a per-(outlet, date) snapshot of 32 features by joining the synthetic Syngenta data with public-data parquets (real or synth). `priority_model.py` ranks the rep's territory with XGBoost LambdaMART. The top-10 by priority feed `route_solver.py` (OR-Tools VRP, Haversine matrix, depot = centroid). Each outlet is then passed through `nba_engine.py`: Stage A rule filter (crop_match + category-specific pest/stage gates) → Stage B `CalibratedClassifierCV(LogisticRegression)` for P(purchase) → score = P × base_uplift → Stage C templated justification. `explainer.py` runs SHAP TreeExplainer against the priority ranker per row to surface the top-3 reasons in human-readable templates (no LLM, no hallucination). `anomaly_detector.py` runs IsolationForest on POS aggregates + cluster heuristics on the feature snapshot. `bandit.py` is instrumented through `POST /outcome` but not converged. `main.py` glues all of this behind three endpoints, each module having a graceful fallback so the API stays up if any artifact is missing.

---

## Known limitations

Read `MODELS.md` §3 for the full reviewer-facing register. Summary:

- Labels are POS-proxy, not rep-causal (synthetic data has no rep→sale link).
- Justifications are deterministic templates, not LLM-generated (Phi-3-mini swap-in documented).
- STL on pest series is instrumented but disabled (single-week pest synth).
- Maize outlets fall back to rule-based SKU (no pest data for maize).
- Distance is Haversine, not road-network.
- Bandit is instrumented, not converged.

All seven gaps are transparent swap-ins, not architecture rewrites.

---

## Repo hygiene

`.gitignore` covers: `data/`, `models/`, `*.parquet`, `*.csv`, `*.json` (with `!MODELS.md`), `*.zip`, `.env`, `.venv/`, `__pycache__/`, `node_modules/`. The Syngenta synthetic dataset MUST NEVER be committed.

---

## Owner

Krishna — backend + ML. Branch: `backend`. Merge to `main` per the team's hackathon cadence.