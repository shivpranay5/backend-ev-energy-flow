# OpenVPP Backend

FastAPI backend for the WebSocket-based EV energy orchestration MVP.

## What this backend provides

- `GET /health`
- `GET /auth/me`
- `GET /auth/admin/me`
- `POST /api/predict/site-decision`
- `WS /ws/simulation`

It uses:

- MongoDB for user/admin records and simulation run storage
- Clerk session token verification for authenticated requests
- a single in-memory simulation engine for the live WebSocket flow

## Environment

Copy `.env.example` to `.env` and set:

- `MONGODB_URI`
- `MONGODB_DB`
- `CLERK_JWKS_URL`
- `CLERK_SECRET_KEY`
- `CLERK_ISSUER` if you want to pin the token issuer

## Run

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Notes

- Admin emails are seeded automatically:
  - `orian.neo007@gmail.com`
  - `ppranayreddy5454@gmail.com`
- Users are not seeded. They are created on first authenticated request.
- The WebSocket simulation is permissive by default for local development, but the auth layer is ready for Clerk tokens when you connect the frontend.
- Site prediction is now historical-date based only.
- The backend uses `openvpp_forecasting_stack` and its cached Arizona daily datasets to rebuild the trained forecasting models in the current environment and score past dates.
- The selected charger location affects charger capability and user flow, but the forecast itself is driven by the Arizona-wide historical forecasting stack rather than live current APIs.
