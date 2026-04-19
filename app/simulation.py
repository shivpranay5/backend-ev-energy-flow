from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class EnvironmentState:
    grid_stress: str = "LOW"
    inference_demand: str = "MEDIUM"
    tariff_mode: str = "OFF_PEAK"
    charge_price_per_kwh: float | None = None
    v2g_price_per_kwh: float | None = None
    inference_value_per_kwh: float | None = None
    inference_power_kwh_per_hour: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_stress": self.grid_stress,
            "inference_demand": self.inference_demand,
            "tariff_mode": self.tariff_mode,
        }


@dataclass(slots=True)
class OcppContext:
    charger_status: str = "CONNECTED_IDLE"
    connector_state: str = "CONNECTED"
    heartbeat_ts: str = field(default_factory=_utc_now_iso)
    transaction_state: str = "STARTED"
    session_state: str = "ACTIVE"
    meter_kwh_imported: float = 0.0
    meter_kwh_exported: float = 0.0
    charging_power_kw: float = 0.0
    discharging_power_kw: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "charger_status": self.charger_status,
            "connector_state": self.connector_state,
            "heartbeat_ts": self.heartbeat_ts,
            "transaction_state": self.transaction_state,
            "session_state": self.session_state,
            "meter_kwh_imported": round(self.meter_kwh_imported, 3),
            "meter_kwh_exported": round(self.meter_kwh_exported, 3),
            "charging_power_kw": round(self.charging_power_kw, 3),
            "discharging_power_kw": round(self.discharging_power_kw, 3),
        }


@dataclass(slots=True)
class ActionResult:
    action: str
    feasible: bool
    delta_soc_kwh: float
    immediate_value_usd: float
    recovery_cost_usd: float
    marginal_net_profit_usd: float
    reason: str | None = None


@dataclass(slots=True)
class SimulationSnapshot:
    simulation_hour: int
    hours_remaining: int
    current_mode: str
    battery_level_kwh: float
    mobility_buffer_kwh: float
    flexible_kwh: float
    current_revenue_usd: float
    current_cost_usd: float
    accumulated_revenue_usd: float
    accumulated_charge_cost_usd: float
    net_profit_usd: float
    grid_export_kwh: float
    inference_energy_kwh: float
    inference_hours: float
    current_prices: dict[str, float]
    environment_state: dict[str, Any]
    ocpp_context: dict[str, Any]
    accumulated_inference_revenue_usd: float = 0.0
    accumulated_v2g_revenue_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "type": "state_update",
            "simulation_hour": self.simulation_hour,
            "hours_remaining": self.hours_remaining,
            "current_mode": self.current_mode,
            "battery_level_kwh": round(self.battery_level_kwh, 3),
            "mobility_buffer_kwh": round(self.mobility_buffer_kwh, 3),
            "flexible_kwh": round(self.flexible_kwh, 3),
            "current_revenue_usd": round(self.current_revenue_usd, 3),
            "current_cost_usd": round(self.current_cost_usd, 3),
            "accumulated_revenue_usd": round(self.accumulated_revenue_usd, 3),
            "accumulated_charge_cost_usd": round(self.accumulated_charge_cost_usd, 3),
            "net_profit_usd": round(self.net_profit_usd, 3),
            "grid_export_kwh": round(self.grid_export_kwh, 3),
            "inference_energy_kwh": round(self.inference_energy_kwh, 3),
            "inference_hours": round(self.inference_hours, 3),
            "current_prices": self.current_prices,
            "environment_state": self.environment_state,
            "ocpp_context": self.ocpp_context,
            "accumulated_inference_revenue_usd": round(self.accumulated_inference_revenue_usd, 3),
            "accumulated_v2g_revenue_usd": round(self.accumulated_v2g_revenue_usd, 3),
        }
        payload.update(self.extra)
        return payload


class SimulationEngine:
    BATTERY_CAPACITY_KWH = 100.0
    CHARGE_POWER_KWH_PER_HOUR = 7.0
    V2G_DISCHARGE_POWER_KWH_PER_HOUR = 10.0
    V2G_EVENT_CAP_KWH = 30.0

    def __init__(self, defaults: dict[str, Any], ocpp_transitions: list[dict[str, Any]]) -> None:
        self.defaults = defaults
        self.ocpp_transitions = ocpp_transitions or []
        self._lock = asyncio.Lock()
        self._play_task: asyncio.Task[None] | None = None
        self.on_tick: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        self.on_complete: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        self.playing = False
        self.speed_multiplier = 1.0
        self.initialized = False
        self.completed = False
        self.seed = {
            "arrival_soc_kwh": float(defaults.get("session_seed.json", {}).get("arrival_soc_kwh", 68)),
            "hours_until_departure": int(defaults.get("session_seed.json", {}).get("hours_until_departure", 9)),
            "location_id": str(defaults.get("session_seed.json", {}).get("location_id", "default")),
            "charger_location": str(defaults.get("session_seed.json", {}).get("charger_location", "Arizona charger location")),
            "charger_type": str(defaults.get("session_seed.json", {}).get("charger_type", "UNIDIRECTIONAL")).upper(),
            "start_time_local": str(defaults.get("session_seed.json", {}).get("start_time_local", "18:00")),
        }
        self.environment = EnvironmentState()
        self.state = self._blank_state()
        self.mode_durations: dict[str, int] = {"CHARGING": 0, "V2G_DISCHARGE": 0, "INFERENCE_ACTIVE": 0, "IDLE": 0}
        self.summary: dict[str, Any] | None = None

    def _blank_state(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            simulation_hour=0,
            hours_remaining=0,
            current_mode="IDLE",
            battery_level_kwh=0.0,
            mobility_buffer_kwh=0.0,
            flexible_kwh=0.0,
            current_revenue_usd=0.0,
            current_cost_usd=0.0,
            accumulated_revenue_usd=0.0,
            accumulated_charge_cost_usd=0.0,
            net_profit_usd=0.0,
            grid_export_kwh=0.0,
            inference_energy_kwh=0.0,
            inference_hours=0.0,
            current_prices={"charge_price_per_kwh": 0.08, "v2g_price_per_kwh": 0.35, "inference_value_per_kwh": 0.4},
            environment_state=self.environment.to_dict(),
            ocpp_context=OcppContext().to_dict(),
            accumulated_inference_revenue_usd=0.0,
            accumulated_v2g_revenue_usd=0.0,
        )

    async def init_session(
        self,
        arrival_soc_kwh: float,
        hours_until_departure: int,
        location_id: str | None = None,
        charger_location: str | None = None,
        charger_type: str | None = None,
        start_time_local: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            self.seed = {
                "arrival_soc_kwh": float(arrival_soc_kwh),
                "hours_until_departure": int(hours_until_departure),
                "location_id": str(location_id or self.seed.get("location_id") or "default"),
                "charger_location": str(charger_location or self.seed.get("charger_location") or "Arizona charger location"),
                "charger_type": str(charger_type or self.seed.get("charger_type") or "UNIDIRECTIONAL").upper(),
                "start_time_local": str(start_time_local or self.seed.get("start_time_local") or "18:00"),
            }
            self.environment.tariff_mode = self._tariff_mode_for_start_time(self.seed["start_time_local"])
            self.initialized = True
            self.completed = False
            self.mode_durations = {"CHARGING": 0, "V2G_DISCHARGE": 0, "INFERENCE_ACTIVE": 0, "IDLE": 0}
            self.summary = None
            self.state = self._new_state_from_seed()
            return self.state.to_dict()

    async def update_environment(self, patch: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            for field in ("grid_stress", "inference_demand", "tariff_mode"):
                value = patch.get(field)
                if value:
                    setattr(self.environment, field, str(value))

            if "charge_price_per_kwh" in patch:
                self.environment.charge_price_per_kwh = float(patch["charge_price_per_kwh"])
            if "v2g_price_per_kwh" in patch:
                self.environment.v2g_price_per_kwh = float(patch["v2g_price_per_kwh"])
            if "inference_value_per_kwh" in patch:
                self.environment.inference_value_per_kwh = float(patch["inference_value_per_kwh"])
            if "inference_power_kwh_per_hour" in patch:
                self.environment.inference_power_kwh_per_hour = float(patch["inference_power_kwh_per_hour"])

            if self.initialized:
                self.state.environment_state = self.environment.to_dict()
                self.state.current_prices = self._current_prices()
                self.state.ocpp_context = self._ocpp_for_mode(self.state.current_mode)
            return self.state.to_dict()

    async def reset(self) -> dict[str, Any]:
        async with self._lock:
            self.completed = False
            self.playing = False
            if self._play_task and not self._play_task.done():
                self._play_task.cancel()
            self._play_task = None
            self.summary = None
            self.mode_durations = {"CHARGING": 0, "V2G_DISCHARGE": 0, "INFERENCE_ACTIVE": 0, "IDLE": 0}
            self.state = self._new_state_from_seed()
            return self.state.to_dict()

    async def pause(self) -> dict[str, Any]:
        async with self._lock:
            self.playing = False
            if self._play_task and not self._play_task.done():
                self._play_task.cancel()
            self._play_task = None
            return self.state.to_dict()

    async def play(self, speed_multiplier: float) -> dict[str, Any]:
        async with self._lock:
            if not self.initialized:
                self.state = self._new_state_from_seed()
                self.initialized = True
            self.speed_multiplier = max(0.1, float(speed_multiplier or 1.0))
            self.playing = True
            if not self._play_task or self._play_task.done():
                self._play_task = asyncio.create_task(self._run_loop())
            return self.state.to_dict()

    async def step(self) -> dict[str, Any]:
        async with self._lock:
            if not self.initialized:
                self.state = self._new_state_from_seed()
                self.initialized = True
            return self._advance_locked()

    async def current_snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return self.state.to_dict()

    async def current_summary(self) -> dict[str, Any] | None:
        async with self._lock:
            return self.summary

    def _new_state_from_seed(self) -> SimulationSnapshot:
        initial_soc = float(self.seed["arrival_soc_kwh"])
        hours = max(0, int(self.seed["hours_until_departure"]))
        mobility_buffer = self._mobility_buffer(hours)
        flexible = max(0.0, initial_soc - mobility_buffer)
        return SimulationSnapshot(
            simulation_hour=0,
            hours_remaining=hours,
            current_mode="IDLE",
            battery_level_kwh=initial_soc,
            mobility_buffer_kwh=mobility_buffer,
            flexible_kwh=flexible,
            current_revenue_usd=0.0,
            current_cost_usd=0.0,
            accumulated_revenue_usd=0.0,
            accumulated_charge_cost_usd=0.0,
            net_profit_usd=0.0,
            grid_export_kwh=0.0,
            inference_energy_kwh=0.0,
            inference_hours=0.0,
            current_prices=self._current_prices(),
            environment_state=self.environment.to_dict(),
            ocpp_context=self._ocpp_for_mode("IDLE"),
            accumulated_inference_revenue_usd=0.0,
            accumulated_v2g_revenue_usd=0.0,
            extra=self._session_context_extra(),
        )

    def _session_context_extra(self) -> dict[str, Any]:
        charger_type = str(self.seed.get("charger_type") or "UNIDIRECTIONAL").upper()
        return {
            "location_id": self.seed.get("location_id"),
            "charger_location": self.seed.get("charger_location"),
            "charger_type": charger_type,
            "start_time_local": self.seed.get("start_time_local"),
            "supports_v2g": charger_type == "BIDIRECTIONAL",
            "available_modes": ["CHARGING", "INFERENCE_ACTIVE", "IDLE"]
            if charger_type != "BIDIRECTIONAL"
            else ["CHARGING", "V2G_DISCHARGE", "INFERENCE_ACTIVE", "IDLE"],
        }

    def _current_prices(self) -> dict[str, float]:
        charge_defaults = {"OFF_PEAK": 0.08, "NORMAL": 0.11, "PEAK": 0.11}
        v2g_defaults = {"PEAK": 0.35, "OFF_PEAK": 0.35, "NORMAL": 0.35}
        inference_defaults = {"OFF_PEAK": 0.4, "NORMAL": 0.4, "PEAK": 0.4}
        return {
            "charge_price_per_kwh": float(
                self.environment.charge_price_per_kwh
                if self.environment.charge_price_per_kwh is not None
                else charge_defaults.get(self.environment.tariff_mode, 0.11)
            ),
            "v2g_price_per_kwh": float(
                self.environment.v2g_price_per_kwh
                if self.environment.v2g_price_per_kwh is not None
                else v2g_defaults.get(self.environment.tariff_mode, 0.35)
            ),
            "inference_value_per_kwh": float(
                self.environment.inference_value_per_kwh
                if self.environment.inference_value_per_kwh is not None
                else inference_defaults.get(self.environment.tariff_mode, 0.4)
            ),
        }

    def _mobility_buffer(self, hours_until_departure: int) -> float:
        if hours_until_departure <= 4:
            return 30.0
        if hours_until_departure <= 8:
            return 25.0
        return 20.0

    def _inference_power(self) -> float:
        if self.environment.inference_power_kwh_per_hour is not None:
            return float(self.environment.inference_power_kwh_per_hour)
        if self.environment.inference_demand == "HIGH":
            return 8.0
        if self.environment.inference_demand == "MEDIUM":
            return 4.0
        return 2.0

    def _tariff_mode_for_start_time(self, start_time_local: str | None) -> str:
        if not start_time_local:
            return self.environment.tariff_mode
        try:
            hour = int(str(start_time_local).split(":", 1)[0])
        except Exception:  # noqa: BLE001
            return self.environment.tariff_mode
        hour = hour % 24
        if hour >= 22 or hour < 6:
            return "OFF_PEAK"
        if 16 <= hour < 21:
            return "PEAK"
        return "NORMAL"

    def _remaining_charge_time_hours(self, projected_soc: float, mobility_buffer: float, charge_price: float) -> float:
        deficit = max(0.0, mobility_buffer - projected_soc)
        if deficit <= 0:
            return 0.0
        return deficit / self.CHARGE_POWER_KWH_PER_HOUR

    def _can_recover(self, projected_soc: float, mobility_buffer: float, hours_until_departure: int) -> bool:
        if projected_soc >= mobility_buffer:
            return True
        remaining = max(0, hours_until_departure - 1)
        required = self._remaining_charge_time_hours(projected_soc, mobility_buffer, self.environment.charge_price_per_kwh)
        return required <= remaining

    def _evaluate_action(self, action: str, soc: float, mobility_buffer: float, remaining_export_cap: float) -> ActionResult:
        charge_price = self._current_prices()["charge_price_per_kwh"]
        v2g_price = self._current_prices()["v2g_price_per_kwh"]
        inference_value = self._current_prices()["inference_value_per_kwh"]
        hours_until_departure = self.state.hours_remaining
        flexible_kwh = max(0.0, soc - mobility_buffer)

        if action == "CHARGING":
            delta = min(self.CHARGE_POWER_KWH_PER_HOUR, self.BATTERY_CAPACITY_KWH - soc)
            projected = soc + delta
            immediate = -(delta * charge_price)
            recovery = max(0.0, mobility_buffer - projected) * charge_price
            marginal = immediate - recovery
            return ActionResult(action, delta > 0, delta, immediate, recovery, marginal, None if delta > 0 else "battery_full")

        if action == "V2G_DISCHARGE":
            if str(self.seed.get("charger_type") or "UNIDIRECTIONAL").upper() != "BIDIRECTIONAL":
                return ActionResult(action, False, 0.0, 0.0, 0.0, float("-inf"), "charger_type_not_bidirectional")
            if self.environment.grid_stress != "HIGH":
                return ActionResult(action, False, 0.0, 0.0, 0.0, float("-inf"), "grid_stress_not_high")
            delta = -min(self.V2G_DISCHARGE_POWER_KWH_PER_HOUR, flexible_kwh, remaining_export_cap)
            projected = soc + delta
            feasible = projected >= mobility_buffer and self._can_recover(projected, mobility_buffer, hours_until_departure)
            immediate = abs(delta) * v2g_price
            recovery = max(0.0, mobility_buffer - projected) * charge_price
            marginal = immediate - recovery if feasible else float("-inf")
            reason = None if feasible else "insufficient_recovery_time_or_soc"
            return ActionResult(action, feasible, delta, immediate, recovery, marginal, reason)

        if action == "INFERENCE_ACTIVE":
            if self.environment.inference_demand != "HIGH":
                return ActionResult(action, False, 0.0, 0.0, 0.0, float("-inf"), "inference_demand_not_high")
            if hours_until_departure < 4:
                return ActionResult(action, False, 0.0, 0.0, 0.0, float("-inf"), "departure_too_close")
            inference_power = self._inference_power()
            delta = -min(inference_power, flexible_kwh)
            projected = soc + delta
            feasible = projected >= mobility_buffer and self._can_recover(projected, mobility_buffer, hours_until_departure)
            immediate = abs(delta) * inference_value
            recovery = max(0.0, mobility_buffer - projected) * charge_price
            marginal = immediate - recovery if feasible else float("-inf")
            reason = None if feasible else "insufficient_recovery_time_or_soc"
            return ActionResult(action, feasible, delta, immediate, recovery, marginal, reason)

        projected = soc
        return ActionResult("IDLE", True, 0.0, 0.0, 0.0, 0.0, None)

    def _ocpp_for_mode(self, mode: str) -> dict[str, Any]:
        mapping = {
            "CHARGING": "CHARGING",
            "V2G_DISCHARGE": "DISCHARGING",
            "INFERENCE_ACTIVE": "CONNECTED_IDLE",
            "IDLE": "AVAILABLE",
        }
        base = {
            "charger_status": mapping.get(mode, "AVAILABLE"),
            "connector_state": "ACTIVE" if mode in {"CHARGING", "V2G_DISCHARGE"} else "CONNECTED",
            "heartbeat_status": "OK",
            "transaction_state": "RUNNING" if mode in {"CHARGING", "V2G_DISCHARGE"} else "STARTED",
            "session_state": "ACTIVE" if mode != "IDLE" else "IDLE",
            "meter_kwh_imported": 0.0,
            "meter_kwh_exported": 0.0,
            "charging_power_kw": 0.0,
            "discharging_power_kw": 0.0,
        }
        if mode == "CHARGING":
            base["meter_kwh_imported"] = self.CHARGE_POWER_KWH_PER_HOUR
            base["charging_power_kw"] = self.CHARGE_POWER_KWH_PER_HOUR
        elif mode == "V2G_DISCHARGE":
            base["meter_kwh_exported"] = self.V2G_DISCHARGE_POWER_KWH_PER_HOUR
            base["discharging_power_kw"] = self.V2G_DISCHARGE_POWER_KWH_PER_HOUR
        elif mode == "INFERENCE_ACTIVE":
            base["transaction_state"] = "STARTED"
        return base

    def _advance_locked(self) -> dict[str, Any]:
        if self.completed:
            return self.state.to_dict()

        self.state.mobility_buffer_kwh = self._mobility_buffer(self.state.hours_remaining)
        self.state.flexible_kwh = max(0.0, self.state.battery_level_kwh - self.state.mobility_buffer_kwh)
        self.state.current_prices = self._current_prices()
        self.state.environment_state = self.environment.to_dict()
        self.state.extra = self._session_context_extra()

        if self.state.battery_level_kwh <= self.state.mobility_buffer_kwh:
            selected = ActionResult(
                action="CHARGING",
                feasible=True,
                delta_soc_kwh=min(self.CHARGE_POWER_KWH_PER_HOUR, self.BATTERY_CAPACITY_KWH - self.state.battery_level_kwh),
                immediate_value_usd=0.0,
                recovery_cost_usd=0.0,
                marginal_net_profit_usd=0.0,
                reason="hard_override_soc_at_or_below_buffer",
            )
        else:
            remaining_export_cap = self.V2G_EVENT_CAP_KWH
            candidates = [
                self._evaluate_action("CHARGING", self.state.battery_level_kwh, self.state.mobility_buffer_kwh, remaining_export_cap),
                self._evaluate_action("V2G_DISCHARGE", self.state.battery_level_kwh, self.state.mobility_buffer_kwh, remaining_export_cap),
                self._evaluate_action("INFERENCE_ACTIVE", self.state.battery_level_kwh, self.state.mobility_buffer_kwh, remaining_export_cap),
                self._evaluate_action("IDLE", self.state.battery_level_kwh, self.state.mobility_buffer_kwh, remaining_export_cap),
            ]
            feasible = [item for item in candidates if item.feasible]
            selected = max(feasible, key=lambda item: item.marginal_net_profit_usd) if feasible else candidates[-1]

        self._apply_action(selected)
        self.state.simulation_hour += 1
        self.state.hours_remaining = max(0, self.state.hours_remaining - 1)
        self.state.flexible_kwh = max(0.0, self.state.battery_level_kwh - self.state.mobility_buffer_kwh)
        self.state.current_prices = self._current_prices()
        self.state.environment_state = self.environment.to_dict()
        self.state.ocpp_context = self._ocpp_for_mode(selected.action)
        self.state.extra = self._session_context_extra()
        self.mode_durations[selected.action] = self.mode_durations.get(selected.action, 0) + 1
        self.state.inference_hours = self.mode_durations.get("INFERENCE_ACTIVE", 0)
        self.state.net_profit_usd = self.state.accumulated_revenue_usd - self.state.accumulated_charge_cost_usd

        snapshot = self.state.to_dict()
        if self.state.hours_remaining <= 0:
            self.completed = True
            self.summary = self._build_summary()
            snapshot["type"] = "state_update"
        return snapshot

    def _apply_action(self, action: ActionResult) -> None:
        self.state.current_mode = action.action
        self.state.current_revenue_usd = max(0.0, action.immediate_value_usd)
        self.state.current_cost_usd = 0.0

        if action.action == "CHARGING":
            self.state.battery_level_kwh = min(self.BATTERY_CAPACITY_KWH, self.state.battery_level_kwh + action.delta_soc_kwh)
            cost = abs(action.delta_soc_kwh) * self.state.current_prices["charge_price_per_kwh"]
            self.state.current_cost_usd = cost
            self.state.accumulated_charge_cost_usd += cost
            self.state.ocpp_context = self._ocpp_for_mode("CHARGING")
            self.state.ocpp_context["meter_kwh_imported"] = action.delta_soc_kwh
        elif action.action == "V2G_DISCHARGE":
            export_kwh = abs(action.delta_soc_kwh)
            self.state.battery_level_kwh = max(0.0, self.state.battery_level_kwh - export_kwh)
            v2g_revenue = export_kwh * self.state.current_prices["v2g_price_per_kwh"]
            self.state.accumulated_revenue_usd += v2g_revenue
            self.state.accumulated_v2g_revenue_usd += v2g_revenue
            self.state.grid_export_kwh += export_kwh
            self.state.ocpp_context = self._ocpp_for_mode("V2G_DISCHARGE")
            self.state.ocpp_context["meter_kwh_exported"] = export_kwh
        elif action.action == "INFERENCE_ACTIVE":
            used_kwh = abs(action.delta_soc_kwh)
            self.state.battery_level_kwh = max(0.0, self.state.battery_level_kwh - used_kwh)
            inference_revenue = used_kwh * self.state.current_prices["inference_value_per_kwh"]
            self.state.accumulated_revenue_usd += inference_revenue
            self.state.accumulated_inference_revenue_usd += inference_revenue
            self.state.inference_energy_kwh += used_kwh
            self.state.inference_hours += 1
            self.state.ocpp_context = self._ocpp_for_mode("INFERENCE_ACTIVE")
        else:
            self.state.ocpp_context = self._ocpp_for_mode("IDLE")

        self.state.accumulated_revenue_usd = round(self.state.accumulated_revenue_usd, 6)
        self.state.accumulated_charge_cost_usd = round(self.state.accumulated_charge_cost_usd, 6)
        self.state.net_profit_usd = self.state.accumulated_revenue_usd - self.state.accumulated_charge_cost_usd

    async def _run_loop(self) -> None:
        try:
            while True:
                async with self._lock:
                    if not self.playing or self.completed:
                        break
                await asyncio.sleep(max(0.05, 1.0 / max(self.speed_multiplier, 0.1)))
                async with self._lock:
                    if not self.playing or self.completed:
                        break
                    snapshot = self._advance_locked()
                if self.on_tick:
                    await self.on_tick(snapshot)
                if self.completed and self.summary and self.on_complete:
                    await self.on_complete(self.summary)
        except asyncio.CancelledError:
            return

    def _build_summary(self) -> dict[str, Any]:
        summary = {
            "type": "simulation_complete",
            "total_revenue_usd": round(self.state.accumulated_revenue_usd, 3),
            "total_charge_cost_usd": round(self.state.accumulated_charge_cost_usd, 3),
            "inference_revenue_usd": round(self.state.accumulated_inference_revenue_usd, 3),
            "digital_contribution_hours": self.mode_durations.get("INFERENCE_ACTIVE", 0),
            "inference_energy_kwh": round(self.state.inference_energy_kwh, 3),
            "mode_durations": dict(self.mode_durations),
            "final_soc_kwh": round(self.state.battery_level_kwh, 3),
            "net_profit_usd": round(self.state.net_profit_usd, 3),
            "grid_export_kwh": round(self.state.grid_export_kwh, 3),
        }
        return summary
