from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .clerk_auth import Principal, upsert_user_identity, verify_clerk_identity, verify_clerk_session
from .db import DatabaseBundle, connect_mongo, seed_admins
from .mock_loader import load_bundle
from .settings import Settings, get_settings
from .simulation import SimulationEngine


class AuthResponse(BaseModel):
    user_id: str
    email: str
    role: str
    full_name: str | None = None
    source: str


class AuthSyncRequest(BaseModel):
    email: str
    full_name: str | None = None


class HistorySaveRequest(BaseModel):
    location_id: str
    charger_location: str | None = None
    charger_type: str | None = None
    start_time_local: str | None = None
    summary: dict[str, Any]
    latest_state: dict[str, Any] | None = None


class HistoryItem(BaseModel):
    id: str
    user_id: str
    email: str
    role: str
    location_id: str
    charger_location: str | None = None
    charger_type: str | None = None
    start_time_local: str | None = None
    created_at: str
    summary: dict[str, Any]
    latest_state: dict[str, Any] | None = None


class SessionRequest(BaseModel):
    location_id: str | None = None
    arrival_soc_kwh: float = Field(ge=0, le=100)
    hours_until_departure: int = Field(ge=0, le=168)
    charger_location: str | None = None
    charger_type: str | None = None
    start_time_local: str | None = None


class EnvironmentPatch(BaseModel):
    location_id: str | None = None
    grid_stress: str | None = None
    inference_demand: str | None = None
    tariff_mode: str | None = None
    charge_price_per_kwh: float | None = None
    v2g_price_per_kwh: float | None = None
    inference_value_per_kwh: float | None = None
    inference_power_kwh_per_hour: float | None = None


class WSMessage(BaseModel):
    type: str
    location_id: str | None = None
    arrival_soc_kwh: float | None = None
    hours_until_departure: int | None = None
    charger_location: str | None = None
    charger_type: str | None = None
    start_time_local: str | None = None
    grid_stress: str | None = None
    inference_demand: str | None = None
    tariff_mode: str | None = None
    speed_multiplier: float | None = None
    charge_price_per_kwh: float | None = None
    v2g_price_per_kwh: float | None = None
    inference_value_per_kwh: float | None = None
    inference_power_kwh_per_hour: float | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    bundle = load_bundle()
    mongo = await connect_mongo(settings)
    await seed_admins(mongo, settings.admin_emails)
    app.state.settings = settings
    app.state.mongo = mongo
    app.state.bundle = bundle
    app.state.sim_sessions = {}
    app.state.sim_sessions_lock = asyncio.Lock()

    try:
        yield
    finally:
        mongo.client.close()


app = FastAPI(title="OpenVPP Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins or ["*"],
    allow_origin_regex=get_settings().cors_origin_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db(request: Request) -> DatabaseBundle:
    return request.app.state.mongo


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_engine(request: Request) -> SimulationEngine:
    session = request.app.state.sim_sessions.get("default")
    if session is None:
        session = _ensure_session_sync(request.app, "default")
    return session["engine"]


async def get_principal(request: Request, db: DatabaseBundle = Depends(get_db), settings: Settings = Depends(get_settings_dep)) -> Principal:
    return await verify_clerk_identity(request, settings, db)


def _make_engine(bundle: dict[str, Any]) -> SimulationEngine:
    return SimulationEngine(bundle, bundle.get("ocpp_transitions.json", []))


def _ensure_session_sync(app: FastAPI, location_id: str) -> dict[str, Any]:
    existing = app.state.sim_sessions.get(location_id)
    if existing is not None:
        return existing

    engine = _make_engine(app.state.bundle)

    async def emit_state(payload: dict[str, Any], *, _location_id: str = location_id) -> None:
        await _broadcast(app, payload, _location_id)

    async def emit_summary(payload: dict[str, Any], *, _location_id: str = location_id) -> None:
        await _broadcast(app, payload, _location_id)

    engine.on_tick = emit_state
    engine.on_complete = emit_summary
    session = {"engine": engine, "clients": set()}
    app.state.sim_sessions[location_id] = session
    return session


async def _ensure_session(app: FastAPI, location_id: str) -> dict[str, Any]:
    async with app.state.sim_sessions_lock:
        return _ensure_session_sync(app, location_id)


async def _broadcast(app: FastAPI, payload: dict[str, Any], location_id: str) -> None:
    async with app.state.sim_sessions_lock:
        session = app.state.sim_sessions.get(location_id)
        clients = list(session["clients"]) if session else []
    stale = []
    for ws in clients:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            stale.append(ws)
    if stale:
        async with app.state.sim_sessions_lock:
            session = app.state.sim_sessions.get(location_id)
            if session is None:
                return
            for ws in stale:
                session["clients"].discard(ws)


async def _broadcast_state(app: FastAPI, location_id: str) -> None:
    session = await _ensure_session(app, location_id)
    engine: SimulationEngine = session["engine"]
    await _broadcast(app, await engine.current_snapshot(), location_id)


@app.get("/health")
async def health(request: Request):
    db = request.app.state.mongo
    sessions = request.app.state.sim_sessions
    return {
        "ok": True,
        "time": datetime.now(timezone.utc).isoformat(),
        "mongo_db": db.db.name,
        "sessions": {
            location_id: {
                "initialized": session["engine"].initialized,
                "completed": session["engine"].completed,
                "clients": len(session["clients"]),
            }
            for location_id, session in sessions.items()
        },
    }


@app.get("/auth/me", response_model=AuthResponse)
async def auth_me(principal: Principal = Depends(get_principal)):
    return AuthResponse(
        user_id=principal.user_id,
        email=principal.email,
        role=principal.role,
        full_name=principal.full_name,
        source=principal.source,
    )


@app.post("/auth/sync", response_model=AuthResponse)
async def auth_sync(payload: AuthSyncRequest, request: Request, settings: Settings = Depends(get_settings_dep)):
    session = await verify_clerk_session(request, settings)
    db = request.app.state.mongo if request else None
    if db is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database unavailable")
    synced = await upsert_user_identity(
        bundle=db,
        user_id=session.user_id,
        email=payload.email,
        full_name=payload.full_name,
        source="clerk-sync",
    )
    return AuthResponse(
        user_id=synced.user_id,
        email=synced.email,
        role=synced.role,
        full_name=synced.full_name,
        source=synced.source,
    )


@app.get("/auth/admin/me", response_model=AuthResponse)
async def auth_admin_me(principal: Principal = Depends(get_principal)):
    if principal.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return AuthResponse(
        user_id=principal.user_id,
        email=principal.email,
        role=principal.role,
        full_name=principal.full_name,
        source=principal.source,
    )


def _serialize_history_run(doc: dict[str, Any]) -> HistoryItem:
    return HistoryItem(
        id=str(doc.get("_id")),
        user_id=str(doc.get("user_id") or ""),
        email=str(doc.get("email") or ""),
        role=str(doc.get("role") or "user"),
        location_id=str(doc.get("location_id") or "default"),
        charger_location=doc.get("charger_location"),
        charger_type=doc.get("charger_type"),
        start_time_local=doc.get("start_time_local"),
        created_at=doc.get("created_at").isoformat() if doc.get("created_at") else datetime.now(timezone.utc).isoformat(),
        summary=doc.get("summary") or {},
        latest_state=doc.get("latest_state"),
    )


@app.post("/history/runs", response_model=HistoryItem)
async def save_history_run(
    payload: HistorySaveRequest,
    request: Request,
    principal: Principal = Depends(get_principal),
):
    bundle = request.app.state.mongo
    now = datetime.now(timezone.utc)
    doc = {
        "user_id": principal.user_id,
        "email": principal.email,
        "role": principal.role,
        "location_id": payload.location_id,
        "charger_location": payload.charger_location,
        "charger_type": payload.charger_type,
        "start_time_local": payload.start_time_local,
        "summary": payload.summary,
        "latest_state": payload.latest_state,
        "created_at": now,
        "updated_at": now,
    }
    result = await bundle.simulation_runs.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _serialize_history_run(doc)


@app.get("/history/runs", response_model=list[HistoryItem])
async def list_history_runs(
    request: Request,
    principal: Principal = Depends(get_principal),
    limit: int = 20,
):
    bundle = request.app.state.mongo
    cursor = (
        bundle.simulation_runs.find({"user_id": principal.user_id, "role": principal.role})
        .sort("created_at", -1)
        .limit(max(1, min(limit, 100)))
    )
    docs = await cursor.to_list(length=max(1, min(limit, 100)))
    return [_serialize_history_run(doc) for doc in docs]


@app.post("/auth/bootstrap/admins")
async def bootstrap_admins(request: Request):
    settings = request.app.state.settings
    bundle = request.app.state.mongo
    seeded = await seed_admins(bundle, settings.admin_emails)
    return {"ok": True, "admins": seeded}


@app.post("/simulation/init")
async def init_simulation(payload: SessionRequest, request: Request):
    location_id = payload.location_id or "default"
    session = await _ensure_session(request.app, location_id)
    engine: SimulationEngine = session["engine"]
    state = await engine.init_session(
        payload.arrival_soc_kwh,
        payload.hours_until_departure,
        location_id=location_id,
        charger_location=payload.charger_location,
        charger_type=payload.charger_type,
        start_time_local=payload.start_time_local,
    )
    await _broadcast_state(request.app, location_id)
    return state


@app.post("/simulation/environment")
async def update_environment(payload: EnvironmentPatch, request: Request):
    location_id = payload.location_id or "default"
    session = await _ensure_session(request.app, location_id)
    engine: SimulationEngine = session["engine"]
    state = await engine.update_environment(payload.model_dump(exclude_none=True))
    await _broadcast_state(request.app, location_id)
    return state


@app.post("/simulation/play")
async def play_simulation(payload: dict[str, Any], request: Request):
    location_id = str(payload.get("location_id") or "default")
    session = await _ensure_session(request.app, location_id)
    engine: SimulationEngine = session["engine"]
    speed = float(payload.get("speed_multiplier", 1))
    state = await engine.play(speed)
    await _broadcast_state(request.app, location_id)
    return state


@app.post("/simulation/pause")
async def pause_simulation(request: Request):
    location_id = request.query_params.get("location_id") or "default"
    session = await _ensure_session(request.app, location_id)
    engine: SimulationEngine = session["engine"]
    state = await engine.pause()
    await _broadcast_state(request.app, location_id)
    return state


@app.post("/simulation/step")
async def step_simulation(request: Request):
    location_id = request.query_params.get("location_id") or "default"
    session = await _ensure_session(request.app, location_id)
    engine: SimulationEngine = session["engine"]
    state = await engine.step()
    await _broadcast_state(request.app, location_id)
    if engine.completed and engine.summary:
        await _broadcast(request.app, engine.summary, location_id)
    return state


@app.post("/simulation/reset")
async def reset_simulation(request: Request):
    location_id = request.query_params.get("location_id") or "default"
    session = await _ensure_session(request.app, location_id)
    engine: SimulationEngine = session["engine"]
    state = await engine.reset()
    await _broadcast_state(request.app, location_id)
    return state


@app.websocket("/ws/simulation")
async def ws_simulation(websocket: WebSocket):
    app = websocket.app
    location_id = websocket.query_params.get("location_id") or "default"
    session = await _ensure_session(app, location_id)
    engine: SimulationEngine = session["engine"]
    role = websocket.query_params.get("role")

    # Optional role / token support for future frontend integration.
    await websocket.accept()
    async with app.state.sim_sessions_lock:
        session = _ensure_session_sync(app, location_id)
        session["clients"].add(websocket)

    try:
        if engine.initialized:
            await websocket.send_text(json.dumps(await engine.current_snapshot()))

        while True:
            raw = await websocket.receive_text()
            try:
                msg = WSMessage.model_validate_json(raw)
            except Exception as exc:  # noqa: BLE001
                await websocket.send_text(json.dumps({"type": "error", "message": f"Invalid message: {exc}"}))
                continue

            if role == "user" and msg.type in {"update_environment", "play", "pause", "step", "reset"}:
                await websocket.send_text(json.dumps({"type": "error", "message": "Admin access required"}))
                continue

            if msg.type == "init_session":
                if msg.arrival_soc_kwh is None or msg.hours_until_departure is None:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Missing init_session fields"}))
                    continue
                state = await engine.init_session(
                    msg.arrival_soc_kwh,
                    msg.hours_until_departure,
                    location_id=msg.location_id or location_id,
                    charger_location=msg.charger_location,
                    charger_type=msg.charger_type,
                    start_time_local=msg.start_time_local,
                )
                await _broadcast(app, state, location_id)
                continue

            if msg.type == "update_environment":
                patch = {
                    "grid_stress": msg.grid_stress,
                    "inference_demand": msg.inference_demand,
                    "tariff_mode": msg.tariff_mode,
                    "charge_price_per_kwh": msg.charge_price_per_kwh,
                    "v2g_price_per_kwh": msg.v2g_price_per_kwh,
                    "inference_value_per_kwh": msg.inference_value_per_kwh,
                    "inference_power_kwh_per_hour": msg.inference_power_kwh_per_hour,
                }
                state = await engine.update_environment({k: v for k, v in patch.items() if v is not None})
                await _broadcast(app, state, location_id)
                continue

            if msg.type == "play":
                speed = msg.speed_multiplier or 1.0
                state = await engine.play(speed)
                await _broadcast(app, state, location_id)
                continue

            if msg.type == "pause":
                state = await engine.pause()
                await _broadcast(app, state, location_id)
                continue

            if msg.type == "step":
                state = await engine.step()
                await _broadcast(app, state, location_id)
                if engine.completed and engine.summary:
                    await _broadcast(app, engine.summary, location_id)
                continue

            if msg.type == "reset":
                state = await engine.reset()
                await _broadcast(app, state, location_id)
                continue

            await websocket.send_text(json.dumps({"type": "error", "message": f"Unsupported message type: {msg.type}"}))
    except WebSocketDisconnect:
        pass
    finally:
        async with app.state.sim_sessions_lock:
            session = app.state.sim_sessions.get(location_id)
            if session is not None:
                session["clients"].discard(websocket)
