from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .auth import Principal, verify_clerk_identity
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


class SessionRequest(BaseModel):
    arrival_soc_kwh: float = Field(ge=0, le=100)
    hours_until_departure: int = Field(ge=0, le=168)


class EnvironmentPatch(BaseModel):
    grid_stress: str | None = None
    inference_demand: str | None = None
    tariff_mode: str | None = None
    charge_price_per_kwh: float | None = None
    v2g_price_per_kwh: float | None = None
    inference_value_per_kwh: float | None = None
    inference_power_kwh_per_hour: float | None = None


class WSMessage(BaseModel):
    type: str
    arrival_soc_kwh: float | None = None
    hours_until_departure: int | None = None
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
    engine = SimulationEngine(bundle, bundle.get("ocpp_transitions.json", []))

    app.state.settings = settings
    app.state.mongo = mongo
    app.state.engine = engine
    app.state.bundle = bundle
    app.state.ws_clients = set()
    app.state.ws_lock = asyncio.Lock()

    async def emit_state(payload: dict[str, Any]) -> None:
        await _broadcast(app, payload)

    async def emit_summary(payload: dict[str, Any]) -> None:
        await _broadcast(app, payload)

    engine.on_tick = emit_state
    engine.on_complete = emit_summary

    try:
        yield
    finally:
        mongo.client.close()


app = FastAPI(title="OpenVPP Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db(request: Request) -> DatabaseBundle:
    return request.app.state.mongo


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_engine(request: Request) -> SimulationEngine:
    return request.app.state.engine


async def get_principal(request: Request, db: DatabaseBundle = Depends(get_db), settings: Settings = Depends(get_settings_dep)) -> Principal:
    return await verify_clerk_identity(request, settings, db)


async def _broadcast(app: FastAPI, payload: dict[str, Any]) -> None:
    async with app.state.ws_lock:
        clients = list(app.state.ws_clients)
    stale = []
    for ws in clients:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            stale.append(ws)
    if stale:
        async with app.state.ws_lock:
            for ws in stale:
                app.state.ws_clients.discard(ws)


async def _broadcast_state(app: FastAPI) -> None:
    engine: SimulationEngine = app.state.engine
    await _broadcast(app, await engine.current_snapshot())


@app.get("/health")
async def health(request: Request):
    db = request.app.state.mongo
    engine: SimulationEngine = request.app.state.engine
    return {
        "ok": True,
        "time": datetime.now(timezone.utc).isoformat(),
        "mongo_db": db.db.name,
        "initialized": engine.initialized,
        "completed": engine.completed,
        "clients": len(request.app.state.ws_clients),
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


@app.post("/auth/bootstrap/admins")
async def bootstrap_admins(request: Request):
    settings = request.app.state.settings
    bundle = request.app.state.mongo
    seeded = await seed_admins(bundle, settings.admin_emails)
    return {"ok": True, "admins": seeded}


@app.post("/simulation/init")
async def init_simulation(payload: SessionRequest, request: Request):
    engine: SimulationEngine = request.app.state.engine
    state = await engine.init_session(payload.arrival_soc_kwh, payload.hours_until_departure)
    await _broadcast_state(request.app)
    return state


@app.post("/simulation/environment")
async def update_environment(payload: EnvironmentPatch, request: Request):
    engine: SimulationEngine = request.app.state.engine
    state = await engine.update_environment(payload.model_dump(exclude_none=True))
    await _broadcast_state(request.app)
    return state


@app.post("/simulation/play")
async def play_simulation(payload: dict[str, Any], request: Request):
    engine: SimulationEngine = request.app.state.engine
    speed = float(payload.get("speed_multiplier", 1))
    state = await engine.play(speed)
    await _broadcast_state(request.app)
    return state


@app.post("/simulation/pause")
async def pause_simulation(request: Request):
    engine: SimulationEngine = request.app.state.engine
    state = await engine.pause()
    await _broadcast_state(request.app)
    return state


@app.post("/simulation/step")
async def step_simulation(request: Request):
    engine: SimulationEngine = request.app.state.engine
    state = await engine.step()
    await _broadcast_state(request.app)
    if engine.completed and engine.summary:
        await _broadcast(request.app, engine.summary)
    return state


@app.post("/simulation/reset")
async def reset_simulation(request: Request):
    engine: SimulationEngine = request.app.state.engine
    state = await engine.reset()
    await _broadcast_state(request.app)
    return state


@app.websocket("/ws/simulation")
async def ws_simulation(websocket: WebSocket):
    app = websocket.app
    engine: SimulationEngine = app.state.engine
    role = websocket.query_params.get("role")

    # Optional role / token support for future frontend integration.
    await websocket.accept()
    async with app.state.ws_lock:
        app.state.ws_clients.add(websocket)

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
                state = await engine.init_session(msg.arrival_soc_kwh, msg.hours_until_departure)
                await _broadcast(app, state)
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
                await _broadcast(app, state)
                continue

            if msg.type == "play":
                speed = msg.speed_multiplier or 1.0
                state = await engine.play(speed)
                await _broadcast(app, state)
                continue

            if msg.type == "pause":
                state = await engine.pause()
                await _broadcast(app, state)
                continue

            if msg.type == "step":
                state = await engine.step()
                await _broadcast(app, state)
                if engine.completed and engine.summary:
                    await _broadcast(app, engine.summary)
                continue

            if msg.type == "reset":
                state = await engine.reset()
                await _broadcast(app, state)
                continue

            await websocket.send_text(json.dumps({"type": "error", "message": f"Unsupported message type: {msg.type}"}))
    except WebSocketDisconnect:
        pass
    finally:
        async with app.state.ws_lock:
            app.state.ws_clients.discard(websocket)
