# backend-ev-energy-flow

FastAPI backend for the Grid Sense / EV energy flow demo. It provides:

- Clerk-backed user and admin identity resolution
- single-EV prediction and simulation APIs
- fleet optimization and forecasting endpoints
- MongoDB persistence for user identities and session history
- a WebSocket channel for live simulation telemetry

## Audience, Goal, Scope

This documentation targets developers and demo operators who need to run, understand, or extend the backend.

- Goal: get the API running locally and understand how the major services fit together
- Included: setup, API surface, simulation flow, auth model, forecasting dependency, persistence model
- Excluded: production deployment hardening and backend test strategy, because those are not defined in this repository

## Documentation Map

This backend doc set follows the Diataxis structure:

- Tutorial: [Run the backend locally](docs/tutorial-run-the-backend-locally.md)
- How-to: [Common backend tasks](docs/how-to-common-backend-tasks.md)
- Reference: [API and configuration reference](docs/reference-api-and-configuration.md)
- Explanation: [Backend architecture and data flow](docs/explanation-backend-architecture.md)

## Quick Start

1. Create a Python environment and install dependencies.
2. Copy `.env.example` to `.env` and fill in MongoDB and Clerk values.
3. Confirm the sibling directory `../openvpp_forecasting_stack` exists.
4. Start the API with Uvicorn.

```bash
cd backend-ev-energy-flow
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Once running, open `http://localhost:8000/health`.

## Core Capabilities

- `GET /health` reports MongoDB and in-memory simulation session status.
- `POST /api/predict/site-decision` scores one EV session against weather, tariff, grid stress, and inference demand.
- `WS /ws/simulation` streams state updates and final summaries for a location-scoped simulation session.
- `/api/fleet/*` endpoints run the fleet optimizer, seasonal comparison, annual projection, week forecast, and upgrade ROI analysis.
- `/history/runs` stores and retrieves completed session summaries for authenticated users.

## Important Local Assumptions

- MongoDB is required at startup. The backend fails fast if `MONGODB_URI` is missing.
- Clerk is required for authenticated REST endpoints such as `/auth/me`, `/api/predict/site-decision`, and `/history/runs`.
- The forecasting-backed features depend on `openvpp_forecasting_stack` being present beside this project. Those model assets are loaded dynamically from `../openvpp_forecasting_stack/scripts/train_three_year_models.py`.
- The live simulation engine is in-memory. Restarting the backend clears active simulation sessions.

## Default Seeded Admins

These emails are seeded into the admin collection on startup:

- `orian.neo007@gmail.com`
- `ppranayreddy5454@gmail.com`

## Repository Layout

```text
backend-ev-energy-flow/
├── app/
│   ├── main.py
│   ├── clerk_auth.py
│   ├── db.py
│   ├── prediction.py
│   ├── simulation.py
│   ├── fleet.py
│   └── forecasting_model_adapter.py
├── artifacts/
├── mock/
├── .env.example
└── requirements.txt
```
