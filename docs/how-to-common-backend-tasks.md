# How-to: Common Backend Tasks

This guide is for developers and demo operators who already know what the backend is and need task-focused instructions.

## Run the Backend for Frontend Development

1. Start MongoDB or point `MONGODB_URI` at an existing instance.
2. Set Clerk and CORS values in `.env`.
3. Start the backend with `uvicorn app.main:app --reload --port 8000`.
4. Point the frontend to `http://localhost:8000` and `ws://localhost:8000/ws/simulation`.

## Seed Admin Users Again

The backend seeds admin emails during startup, but you can trigger it manually:

```bash
curl -X POST http://localhost:8000/auth/bootstrap/admins
```

This uses the admin email tuple in `app/settings.py`.

## Sync a Clerk User into the Backend

The normal frontend flow calls `POST /auth/sync` after sign-in. To do it manually:

```bash
curl http://localhost:8000/auth/sync \
  -X POST \
  -H "Authorization: Bearer <clerk-session-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "full_name": "Example User"
  }'
```

Use this when:

- `/auth/me` fails because the user has no synced email
- you are debugging role assignment and user upserts

## Launch a Simulation from a Known Session Draft

Initialize the session:

```bash
curl http://localhost:8000/simulation/init \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "location_id": "101",
    "arrival_soc_kwh": 68,
    "hours_until_departure": 9,
    "charger_location": "Phoenix Demo Charger",
    "charger_type": "BIDIRECTIONAL",
    "start_date_local": "2026-01-15",
    "start_time_local": "18:00"
  }'
```

Apply environment overrides if needed:

```bash
curl http://localhost:8000/simulation/environment \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "location_id": "101",
    "grid_stress": "HIGH",
    "inference_demand": "HIGH",
    "tariff_mode": "PEAK"
  }'
```

Start playback:

```bash
curl http://localhost:8000/simulation/play \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "location_id": "101",
    "speed_multiplier": 4
  }'
```

## Troubleshoot Missing Forecasting Data

If prediction or fleet analytics fail:

1. Confirm backend-local forecast assets exist at `forecasting_assets/data/training_3yr`.
2. If you are still using the sibling project for code loading, confirm `../openvpp_forecasting_stack` exists and has `scripts/train_three_year_models.py`.
3. Confirm Python can import `pandas` and `scikit-learn` dependencies used by the forecasting adapter.

The simulation engine degrades more gracefully than the prediction and fleet endpoints:

- prediction and fleet APIs depend directly on the forecasting stack
- simulation initialization falls back to a non-ML context if forecasting is unavailable

## Inspect Saved History

Fetch the latest runs for an authenticated user:

```bash
curl http://localhost:8000/history/runs \
  -H "Authorization: Bearer <clerk-session-token>"
```

Save a new completed run:

```bash
curl http://localhost:8000/history/runs \
  -X POST \
  -H "Authorization: Bearer <clerk-session-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "location_id": "101",
    "charger_location": "Phoenix Demo Charger",
    "charger_type": "BIDIRECTIONAL",
    "start_time_local": "18:00",
    "summary": {"net_profit_usd": 12.4}
  }'
```

## Reset a Stuck Simulation

For a given `location_id`:

```bash
curl -X POST "http://localhost:8000/simulation/reset?location_id=101"
```

This clears the current simulation state for that location in memory, but it does not delete any persisted history records.
