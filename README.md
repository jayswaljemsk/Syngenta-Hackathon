# Syngenta Field Copilot — Full Project (Backend + Frontend)

Track 2, AI-Guided Field Force Intelligence. This repository contains the backend (FastAPI + ML + OR-Tools) and a Next.js frontend that together provide a ranked, routed, and justified daily plan for a Syngenta sales rep.

This root README documents how to set up, run, and contribute to the project locally.

---

## Table of contents

- Project overview
- Repo layout
- Quick start (fastest path to run locally)
- Backend: setup, training, and API
- Frontend: setup and running the UI
- Running the full stack
- API examples (curl)
- Data & models (privacy and storage)
- Tests, linting, and CI
- Troubleshooting
- Contributing & license

---

## Project overview

The backend produces per-outlet priority scores, routes the top N outlets using OR-Tools, and returns per-outlet recommendations with deterministic justifications. The frontend displays a map and plan view for a rep to review daily tasks.

Key components:

- `backend/`: FastAPI app, feature pipeline, training scripts, models, and utilities.
- `frontend/`: Next.js app (React + TypeScript) that consumes the backend API.

---

## Repo layout

```
.
├─ backend/                # FastAPI app, ML training scripts, models, data helpers
├─ frontend/               # Next.js app, UI components
├─ models/                 # Generated model artifacts (ignored in VCS)
├─ data/                   # Local data outputs and demos (ignored)
└─ README.md               # (this file)
```

---

## Quick start

Prerequisites:

- Python 3.10+ (backend)
- Node 18+ / npm (frontend)
- `git` and network access for dependencies

Fastest way to run both locally (from repo root):

1. Start backend in a terminal:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

2. Start frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the frontend at `http://localhost:3000` and the backend health at `http://localhost:8000/health`.

---

## Backend — setup & rebuild

Location: [backend](backend)

Prereqs: Python 3.10+, system build tools for some packages (wheels available for most platforms).

Install and run (Windows example):

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

One-shot rebuild (cold start -> green API):

```powershell
# 1. Generate features for required dates
python features.py --date 2026-01-15
python features.py --date 2026-02-15
python features.py --date 2026-02-17
python features.py --date 2026-03-29

# 2. Train models
python priority_model.py train
python nba_engine.py train
python anomaly_detector.py train
python bandit.py fit

# 3. Start API
uvicorn main:app --reload --port 8000
```

Health check:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

Expected response includes flags such as `priority_model_trained`, `route_solver_available`, `explainer_available`, `nba_model_trained`, and `anomaly_model_trained`.

---

## Frontend — setup & run

Location: [frontend](frontend)

The frontend is a Next.js TypeScript app. It expects the backend API to be available (default `http://localhost:8000`). If you need to change the backend URL, see `frontend/lib/backendConfig.ts`.

Install and run:

```powershell
cd frontend
npm install
npm run dev
```

Build for production:

```powershell
npm run build
npm run start
```

If port 3000 is occupied, either stop the blocking process or set `PORT` before starting:

```powershell
$env:PORT=3001; npm run dev
```

---

## Running the full stack (recommended order)

1. Start backend (port 8000).
2. Start frontend (port 3000).
3. Visit `http://localhost:3000` and interact with the UI. The frontend calls the backend endpoints documented below.

For remote testing with a publicly accessible URL, expose the frontend via `ngrok` or a similar tunneling service and set the relevant backend host where needed.

---

## API examples

Health check:

```bash
curl http://localhost:8000/health
```

Get today's plan for a rep (example):

```bash
curl "http://localhost:8000/plan/today?rep_id=REP_0001&date=2026-02-15"
```

Get NBA recommendation for an outlet:

```bash
curl "http://localhost:8000/nba/OUTLET_123?date=2026-02-15"
```

Post an outcome (example JSON):

```bash
curl -X POST http://localhost:8000/outcome \
  -H "Content-Type: application/json" \
  -d '{"rep_id":"REP_0001","outlet_id":"OUTLET_123","date":"2026-02-15","result":"visit","skus":[{"sku":"SKU_A","qty":2}]}'
```

Refer to `CONTRACT.md` for full API schemas and request/response shapes.

---

## Data & models

- `models/` contains trained artifacts produced by training scripts. These files are ignored by VCS and must be rebuilt locally if missing.
- `data/` contains generated feature snapshots and logs (ignored by VCS). `data/outcomes_log.csv` is appended by `POST /outcome`.
- The synthetic Syngenta dataset MUST NEVER be committed. See `.gitignore` for rules.

If you need to share model artifacts for demo, produce a small sanitized package and share out-of-band.

---

## Tests & linting

Backend tests (if present) can be run via `pytest` from the `backend` folder. Frontend tests can be run via `npm test` in `frontend` if configured.

Add CI workflows in `.github/workflows/` to run linters and tests on PRs.

---

## Troubleshooting

- Git: if you see `fatal: refusing to merge unrelated histories` when pulling, avoid forcing merges blindly. Safer options:

  1. Back up your current work: `git branch backup/local-main`
  2. Fetch remote: `git fetch origin`
  3. If you want to make your local match remote (discard local divergent commits):

```powershell
git fetch origin
git reset --hard origin/main
```

  4. If you intended to merge two unrelated histories, use:

```powershell
git pull origin main --allow-unrelated-histories
```

  Use these carefully; prefer creating a backup branch first.

- Frontend: if `npm run dev` fails because port 3000 is in use, find and kill the process or change `PORT`.
- Backend: if packages fail to install, ensure your Python version matches requirements and install system build tools.

---

## Contributing

1. Fork the repo and create a feature branch.
2. Keep secrets and synthetic datasets out of commits.
3. Open a PR against `main` with a clear description and tests where applicable.

Add a `CONTRIBUTING.md` with your preferred workflow and reviewer checklist if you expect outside contributions.

---

## License & owner

See `MODELS.md` for model-specific licensing. Add a top-level `LICENSE` file to declare the project license.

Owner: Krishna (backend + ML). For questions about frontend, see `frontend/README.md` (add if you want a dedicated frontend README).

---

Further improvements (examples): add a dedicated `frontend/README.md`, include screenshots or a short demo, or add CI workflows in `.github/workflows/`.

To propose changes or report issues, please open an issue or a pull request against `main`.