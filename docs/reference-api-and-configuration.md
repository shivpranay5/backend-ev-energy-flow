# Reference: API and Configuration

This reference describes the backend's configuration surface, runtime dependencies, and public endpoints.

## Runtime Dependencies

## Python Packages

Declared in `requirements.txt`:

- `fastapi`
- `uvicorn[standard]`
- `motor`
- `httpx`
- `PyJWT[crypto]`
- `python-dotenv`
- `scipy`
- `numpy`

## External Services

- MongoDB for users, admins, and simulation history
- Clerk for session token verification
- `openvpp_forecasting_stack` for forecast-driven prediction and fleet analytics

## Environment Variables

Loaded from `.env` by `app/settings.py`.

| Variable | Required | Purpose |
| --- | --- | --- |
| `MONGODB_URI` | Yes | MongoDB connection string |
| `MONGODB_DB` | No | MongoDB database name, defaults to `openvpp` |
| `CLERK_JWKS_URL` | No | Clerk JWKS URL, defaults to `https://api.clerk.com/v1/jwks` |
| `CLERK_SECRET_KEY` | No in code, but expected operationally | Clerk secret key |
| `CLERK_ISSUER` | No | Optional issuer validation |
| `CORS_ORIGINS` | No | Comma-separated allowlist of exact origins |
| `CORS_ORIGIN_REGEX` | No | Regex allowlist for origins |
| `ALLOW_GUEST_WS` | No | Boolean flag for permissive local WebSocket usage |

## Seeded Admin Emails

Defined in `Settings.admin_emails`:

- `orian.neo007@gmail.com`
- `ppranayreddy5454@gmail.com`

## Collections and Indexes

MongoDB collections:

- `admins`
- `users`
- `simulation_runs`

Indexes created on startup:

- `admins.email` unique
- `users.email` unique sparse
- `users.clerk_user_id` unique sparse
- `simulation_runs.created_at`
- `simulation_runs.user_id + created_at`
- `simulation_runs.role + created_at`
- `simulation_runs.location_id + created_at`

## REST Endpoints

## Health and Auth

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | No | Runtime health, MongoDB name, active simulation sessions |
| `GET` | `/auth/me` | Yes | Resolve current backend principal |
| `POST` | `/auth/sync` | Yes | Upsert Clerk user email and full name |
| `GET` | `/auth/admin/me` | Yes | Resolve current admin principal |
| `POST` | `/auth/bootstrap/admins` | No | Re-run admin seeding |

## Prediction

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `POST` | `/api/predict/site-decision` | Yes | Return the model-driven site recommendation and simulation seed |

`SiteDecisionRequest` fields:

- `location_id`
- `location_name`
- `charger_type`
- `arrival_soc_kwh`
- `hours_until_departure`
- `start_date_local`
- `start_time_local`
- `ownership_mode`
- `latitude`
- `longitude`
- `overrides`

`SiteDecisionResponse` includes:

- `weather`
- `forecast`
- `economics`
- `recommendation`
- `environment_patch`
- `simulation_seed`
- `model_context`

## Fleet APIs

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `POST` | `/api/fleet/optimize` | No | Run the LP or greedy fleet optimizer |
| `GET` | `/api/fleet/scenario/grid-event` | No | Demo scenario using a date-driven fleet and ML prediction |
| `POST` | `/api/fleet/session-correlated` | No | Compare a single session against fleet allocation |
| `GET` | `/api/fleet/demo-vehicles` | No | Return the generated demo fleet |
| `GET` | `/api/fleet/demo-jobs` | No | Return the generated inference queue |
| `GET` | `/api/fleet/compare-seasons` | No | Compare the same fleet across seasonal dates |
| `GET` | `/api/fleet/annual-projection` | No | Estimate annual revenue and net value |
| `GET` | `/api/fleet/week-forecast` | No | Return a seven-day ML-backed fleet outlook |
| `GET` | `/api/fleet/upgrade-roi` | No | Estimate ROI of converting chargers to bidirectional |

## History

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `POST` | `/history/runs` | Yes | Save a simulation summary for the current principal |
| `GET` | `/history/runs` | Yes | List the current principal's saved history |

## Simulation Control

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `POST` | `/simulation/init` | No | Initialize an in-memory simulation session |
| `POST` | `/simulation/environment` | No | Patch runtime environment fields |
| `POST` | `/simulation/play` | No | Start automatic playback |
| `POST` | `/simulation/pause` | No | Pause playback |
| `POST` | `/simulation/step` | No | Advance one simulation tick |
| `POST` | `/simulation/reset` | No | Reset the in-memory simulation |

## WebSocket Endpoint

Endpoint:

```text
WS /ws/simulation?location_id=<id>[&role=user|admin][&token=<token>]
```

Supported incoming message types:

- `init_session`
- `update_environment`
- `play`
- `pause`
- `step`
- `reset`

Primary outgoing message types:

- `state_update`
- `simulation_complete`
- `error`

Notes:

- sessions are keyed by `location_id`
- the current frontend relies on location scoping and UI restrictions more than backend role enforcement
- if `role=user` is supplied in the query string, mutating WebSocket actions are rejected

## Mock and Artifact Directories

- `mock/` contains JSON fixtures used by the simulation engine and demo flows
- `artifacts/` contains generated backtest and prediction outputs used by the forecasting workstream
