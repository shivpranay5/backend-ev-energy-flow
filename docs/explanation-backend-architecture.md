# Explanation: Backend Architecture and Data Flow

This document explains how the backend is structured and why the application is split the way it is.

## System Role

The backend has three distinct jobs:

1. Resolve identity and role information from Clerk into backend-native principals.
2. Produce forecast-driven decisions for one EV session or a whole fleet.
3. Maintain an interactive in-memory simulation that streams telemetry to the frontend.

These jobs are related, but they are not implemented in the same way:

- auth is request-driven and persisted in MongoDB
- prediction and fleet analytics are compute-driven and rely on the forecasting stack
- simulation is stateful, location-scoped, and kept in process memory

## Main Components

## `app/main.py`

This is the composition root. It:

- loads settings
- loads the mock bundle
- connects to MongoDB
- seeds admins
- configures CORS
- registers REST and WebSocket routes
- stores live simulation sessions in `app.state.sim_sessions`

## `app/clerk_auth.py`

This module translates a Clerk session token into a backend `Principal`.

Important behavior:

- it accepts bearer tokens from headers, cookies, or query params
- it validates signatures through Clerk JWKS
- it optionally validates issuer
- it upserts the user into MongoDB if an email is available
- it derives the role from the `admins` collection

That design keeps role assignment simple and backend-controlled.

## `app/db.py`

MongoDB is used for durable identity and session history, not for live simulation state.

That division matters:

- user/admin records need to survive restarts
- live simulation sessions are cheap to reconstruct and easier to keep in memory

## `app/prediction.py`

The single-session prediction endpoint computes the recommended action before a simulation starts.

It combines:

- requested session details
- charger capability
- date and time
- weather and grid conditions from the forecasting adapter
- tariff and value heuristics

The output is not only a recommendation. It also produces an `environment_patch` and `simulation_seed`, which lets the frontend launch the exact scenario the model reasoned about.

## `app/simulation.py`

The simulation engine is a separate concern from the predictor.

Why:

- the predictor decides what should happen at launch time
- the simulator shows how the session evolves over time

The engine tracks:

- state of charge
- active mode
- revenues and costs
- V2G export and inference energy
- OCPP-style charger context
- ML-derived environment hints

It emits two event streams:

- `state_update` on every tick
- `simulation_complete` at the end

## `app/fleet.py`

Fleet analysis is deliberately not built on top of the single-session engine.

Instead, it solves a separate allocation problem:

- classify likely grid stress through the forecasting stack
- estimate flexible energy per vehicle
- assign V2G or inference work across the fleet
- maximize value with a MILP solver when available
- fall back to greedy logic when SciPy MILP is unavailable

This keeps the fleet dashboard focused on allocation economics rather than timeline playback.

## Forecasting Dependency

The backend is tightly coupled to `openvpp_forecasting_stack`.

The adapter in `forecasting_model_adapter.py` dynamically imports the training script and rebuilds model bundles in the current environment. That means the backend is not just reading a static file of predictions. It is reconstructing the runtime forecasting context from the neighboring project.

This explains two important operational behaviors:

- prediction and fleet endpoints are only as available as the forecasting stack
- simulation initialization can still continue with fallback behavior when forecasting is unavailable

## End-to-End Flow

## User enters through the frontend

The frontend signs the user in with Clerk and calls `/auth/sync`.

## Backend resolves a principal

Subsequent authenticated calls use `/auth/me` and receive a backend role:

- `admin`
- `user`

## Prediction runs before simulation

The frontend sends charger type, arrival SoC, departure window, and local start time to `/api/predict/site-decision`.

The backend returns:

- the recommended action
- supporting model context
- a launch-ready simulation seed

## Simulation begins

The frontend initializes the session through REST and listens over WebSocket using the selected `location_id`.

The engine then:

- initializes state
- accepts environment updates
- plays or steps the session
- broadcasts snapshots to all connected clients for that location

## Completion is persisted

When the frontend receives a summary, it saves that summary through `/history/runs`.

## Design Tradeoffs

## Why MongoDB plus in-memory sessions

This is a pragmatic demo architecture:

- MongoDB handles durable user and run metadata well
- in-memory sessions keep live orchestration simple and fast

The tradeoff is that simulation state is not durable across backend restarts.

## Why REST plus WebSocket

REST is used for:

- setup
- prediction
- fleet analytics
- persistence

WebSocket is used for:

- live state streaming
- low-latency control messages

This split keeps the API surface easy to debug while still supporting a real-time dashboard.
