# Tutorial: Run the Backend Locally

This tutorial is for a developer who wants a working local backend for the EV energy flow demo.

By the end, you will have:

- the FastAPI app running on port `8000`
- MongoDB connected
- Clerk-backed auth endpoints working
- forecasting-backed prediction endpoints available

## Before You Start

You need:

- Python
- MongoDB access
- a Clerk app with a valid session token flow
- forecast assets under `backend-ev-energy-flow/forecasting_assets`
- the sibling project directory `openvpp_forecasting_stack` for training code, unless you vendor that code too

The backend expects this layout:

```text
aee_hackathon/
├── backend-ev-energy-flow/
└── openvpp_forecasting_stack/
```

## 1. Install Dependencies

```bash
cd backend-ev-energy-flow
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure Environment Variables

Copy the example file:

```bash
cp .env.example .env
```

Set the following values in `.env`:

- `MONGODB_URI`: Mongo connection string
- `MONGODB_DB`: database name, default is `openvpp`
- `CLERK_JWKS_URL`: Clerk JWKS endpoint
- `CLERK_SECRET_KEY`: Clerk secret key
- `CLERK_ISSUER`: optional issuer pin
- `CORS_ORIGINS`: comma-separated frontend origins
- `CORS_ORIGIN_REGEX`: regex for allowed origins
- `ALLOW_GUEST_WS`: whether unauthenticated WebSocket connections are allowed

## 3. Start the API

```bash
uvicorn app.main:app --reload --port 8000
```

On startup, the app:

- loads environment settings
- loads the mock simulation bundle from `mock/`
- connects to MongoDB and creates indexes
- seeds the configured admin emails
- initializes the in-memory simulation session registry

## 4. Check Health

Open:

```text
http://localhost:8000/health
```

You should get a JSON payload with:

- `ok: true`
- `mongo_db`
- `sessions`

## 5. Verify Authentication

Use a valid Clerk session token and call:

```bash
curl http://localhost:8000/auth/me \
  -H "Authorization: Bearer <clerk-session-token>"
```

Expected result:

- the backend verifies the Clerk token
- the user is upserted into MongoDB if an email is present
- the response includes `user_id`, `email`, `role`, and `source`

If the token is valid but the email has not been synced yet, the frontend normally solves that by calling `POST /auth/sync`.

## 6. Verify Prediction

Call the site decision endpoint with an authenticated user:

```bash
curl http://localhost:8000/api/predict/site-decision \
  -H "Authorization: Bearer <clerk-session-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "location_id": "101",
    "location_name": "Phoenix Demo Charger",
    "charger_type": "BIDIRECTIONAL",
    "arrival_soc_kwh": 72,
    "hours_until_departure": 10,
    "start_date_local": "2025-07-15",
    "start_time_local": "18:00",
    "ownership_mode": "fleet"
  }'
```

This returns:

- weather and forecast context
- economics inputs
- an action recommendation
- an environment patch
- a simulation seed that the frontend can launch directly

## 7. Verify Simulation

You can initialize a simulation over HTTP:

```bash
curl http://localhost:8000/simulation/init \
  -H "Content-Type: application/json" \
  -d '{
    "location_id": "101",
    "arrival_soc_kwh": 72,
    "hours_until_departure": 10,
    "charger_type": "BIDIRECTIONAL",
    "start_date_local": "2025-07-15",
    "start_time_local": "18:00"
  }'
```

Then connect the frontend or a WebSocket client to:

```text
ws://localhost:8000/ws/simulation?location_id=101
```

You will receive:

- `state_update` messages
- a `simulation_complete` message when the session ends

## What To Read Next

- For common operations, see [Common backend tasks](how-to-common-backend-tasks.md).
- For endpoint details, see [API and configuration reference](reference-api-and-configuration.md).
- For architectural context, see [Backend architecture and data flow](explanation-backend-architecture.md).
