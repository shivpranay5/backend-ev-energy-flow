from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .energy_model import (
    BATTERY_CAPACITY_KWH,
    CHARGE_POWER_KWH_PER_HOUR,
    V2G_DISCHARGE_POWER_KWH_PER_HOUR,
    available_flexible_energy_kwh,
    count_off_peak_hours,
    inference_power_kwh_per_hour,
    mobility_buffer_kwh,
    preferred_departure_target_kwh,
    tariff_mode_for_hour,
)
from .forecasting_model_adapter import get_historical_forecast_for_date


class ScenarioOverrides(BaseModel):
    grid_stress: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    inference_demand: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    tariff_mode: Literal["OFF_PEAK", "NORMAL", "PEAK"] | None = None
    outage_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    charge_price_per_kwh: float | None = Field(default=None, ge=0.0)
    v2g_price_per_kwh: float | None = Field(default=None, ge=0.0)
    inference_value_per_kwh: float | None = Field(default=None, ge=0.0)


class SiteDecisionRequest(BaseModel):
    location_id: str
    location_name: str | None = None
    charger_type: Literal["UNIDIRECTIONAL", "BIDIRECTIONAL"]
    arrival_soc_kwh: float = Field(ge=0, le=100)
    hours_until_departure: int = Field(ge=1, le=168)
    start_date_local: str
    start_time_local: str
    ownership_mode: Literal["private", "fleet"] = "private"
    latitude: float | None = None
    longitude: float | None = None
    overrides: ScenarioOverrides | None = None


class WeatherContext(BaseModel):
    source: str
    temperature_c: float
    condition_bucket: Literal["normal", "hot", "extreme_hot", "cold"]
    heat_risk_index: float = Field(ge=0.0, le=1.0)
    storm_risk_index: float = Field(ge=0.0, le=1.0)


class ForecastContext(BaseModel):
    tariff_mode: Literal["OFF_PEAK", "NORMAL", "PEAK"]
    grid_stress: Literal["LOW", "MEDIUM", "HIGH"]
    inference_demand: Literal["LOW", "MEDIUM", "HIGH"]
    grid_event_probability: float = Field(ge=0.0, le=1.0)
    outage_probability: float = Field(ge=0.0, le=1.0)
    inference_value_score: float = Field(ge=0.0, le=1.0)
    participation_window_score: float = Field(ge=0.0, le=1.0)


class EconomicsContext(BaseModel):
    charge_price_per_kwh: float
    v2g_price_per_kwh: float
    inference_value_per_kwh: float
    mobility_buffer_kwh: float
    weather_margin_kwh: float
    flexible_kwh: float
    preferred_departure_target_kwh: float
    off_peak_hours_remaining: int


class ActionScore(BaseModel):
    action: Literal["CHARGING", "V2G_DISCHARGE", "INFERENCE_ACTIVE", "IDLE"]
    feasible: bool
    expected_net_value_usd: float
    expected_energy_delta_kwh: float
    reason: str


class RecommendationContext(BaseModel):
    action: Literal["CHARGING", "V2G_DISCHARGE", "INFERENCE_ACTIVE", "IDLE"]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    alternatives: list[ActionScore]


class SimulationSeed(BaseModel):
    location_id: str
    location_name: str | None = None
    charger_type: Literal["UNIDIRECTIONAL", "BIDIRECTIONAL"]
    arrival_soc_kwh: float = Field(ge=0, le=100)
    hours_until_departure: int = Field(ge=1, le=168)
    start_date_local: str
    start_time_local: str
    environment_patch: dict[str, str | float]
    initial_action: Literal["CHARGING", "V2G_DISCHARGE", "INFERENCE_ACTIVE", "IDLE"]
    auto_step_on_launch: bool = True
    auto_play: bool = True
    speed_multiplier: float = Field(default=1.0, gt=0)


class SiteDecisionResponse(BaseModel):
    location_id: str
    location_name: str | None = None
    charger_type: Literal["UNIDIRECTIONAL", "BIDIRECTIONAL"]
    supports_v2g: bool
    datetime_local: str
    ownership_mode: Literal["private", "fleet"]
    weather: WeatherContext
    forecast: ForecastContext
    economics: EconomicsContext
    recommendation: RecommendationContext
    environment_patch: dict[str, str | float]
    simulation_seed: SimulationSeed
    model_context: dict[str, object]


def _parse_datetime(date_value: str, time_value: str) -> datetime:
    return datetime.fromisoformat(f"{date_value}T{time_value}")


def _weather_margin(condition_bucket: str) -> float:
    if condition_bucket == "extreme_hot":
        return 6.0
    if condition_bucket == "hot":
        return 3.0
    if condition_bucket == "cold":
        return 2.0
    return 0.0


def _inference_demand(hour: int, ownership_mode: str) -> tuple[str, float]:
    if 18 <= hour < 24:
        return "HIGH", 0.82 if ownership_mode == "fleet" else 0.72
    if 9 <= hour < 18:
        return "MEDIUM", 0.56 if ownership_mode == "fleet" else 0.48
    return "LOW", 0.26


def _stress_bucket(probability: float, threshold: float) -> str:
    if probability >= max(threshold + 0.18, 0.62):
        return "HIGH"
    if probability >= max(threshold, 0.36):
        return "MEDIUM"
    return "LOW"


def _weather_context_from_historical(
    *,
    tmin_c: float,
    tmax_c: float,
    weather_prob: float,
    hour: int,
    source: str,
) -> WeatherContext:
    if hour <= 6:
        blend = 0.1
    elif hour <= 12:
        blend = 0.45
    elif hour <= 17:
        blend = 0.95
    elif hour <= 21:
        blend = 0.65
    else:
        blend = 0.3
    temperature_c = tmin_c + (tmax_c - tmin_c) * blend
    if weather_prob >= 0.72 or temperature_c >= 40:
        bucket = "extreme_hot"
    elif weather_prob >= 0.45 or temperature_c >= 33:
        bucket = "hot"
    elif temperature_c <= 8:
        bucket = "cold"
    else:
        bucket = "normal"
    storm_risk = min(0.35, max(0.02, weather_prob * 0.18))
    return WeatherContext(
        source=source,
        temperature_c=round(temperature_c, 2),
        condition_bucket=bucket,  # type: ignore[arg-type]
        heat_risk_index=round(max(weather_prob, 0.08 if bucket == "cold" else 0.18), 3),
        storm_risk_index=round(storm_risk, 3),
    )


def _clamp(value: float, floor: float, ceiling: float) -> float:
    return max(floor, min(ceiling, value))


def _price_shape_for_hour(
    base_price_cents: float,
    tariff_mode: str,
    ownership_mode: str,
    inference_score: float,
    grid_probability: float,
) -> tuple[float, float, float]:
    base_kwh = base_price_cents / 100.0
    charge_mult = {"OFF_PEAK": 0.72, "NORMAL": 1.0, "PEAK": 1.28}
    charge_price = round(_clamp(base_kwh * charge_mult[tariff_mode], 0.08, 0.15), 4)

    # Research-backed launch bands:
    # - V2G: $0.20-$0.50 / kWh
    # - Inference: $0.25-$0.45 / kWh-equivalent
    v2g_floor = 0.20 if tariff_mode != "PEAK" else 0.28
    v2g_target = 0.32 if tariff_mode != "PEAK" else 0.42
    if ownership_mode == "fleet":
        v2g_target += 0.03
    v2g_price = round(_clamp(v2g_target + grid_probability * 0.08, v2g_floor, 0.50), 4)

    inference_floor = 0.25 if ownership_mode == "private" else 0.30
    inference_target = 0.31 if ownership_mode == "private" else 0.36
    if tariff_mode == "OFF_PEAK":
        inference_target += 0.03
    elif tariff_mode == "PEAK":
        inference_target -= 0.01
    inference_price = round(_clamp(inference_target + inference_score * 0.06, inference_floor, 0.45), 4)
    return charge_price, v2g_price, inference_price


def predict_site_decision(payload: SiteDecisionRequest) -> SiteDecisionResponse:
    dt_local = _parse_datetime(payload.start_date_local, payload.start_time_local)
    historical = get_historical_forecast_for_date(payload.start_date_local)
    weather = _weather_context_from_historical(
        tmin_c=historical.weather_raw["tmin_c_est"],
        tmax_c=historical.weather_raw["tmax_c_est"],
        weather_prob=historical.weather_prob,
        hour=dt_local.hour,
        source=historical.source,
    )
    weather_margin = _weather_margin(weather.condition_bucket)
    tariff_mode = tariff_mode_for_hour(dt_local.hour)

    inference_demand, inference_score = _inference_demand(dt_local.hour, payload.ownership_mode)
    if payload.ownership_mode == "fleet":
        inference_score = min(1.0, inference_score + 0.08)
    if historical.price_prob < 0.18 and tariff_mode == "OFF_PEAK":
        inference_score = min(1.0, inference_score + 0.06)
    grid_probability = historical.fusion_prob
    charge_price, v2g_price, inference_price = _price_shape_for_hour(
        historical.price_raw["price_all_sectors_cents_per_kwh"],
        tariff_mode,
        payload.ownership_mode,
        inference_score,
        grid_probability,
    )
    eia_prob = historical.eia_prob
    price_prob = historical.price_prob
    off_peak_hours_remaining = count_off_peak_hours(dt_local.hour, payload.hours_until_departure)

    overrides = payload.overrides or ScenarioOverrides()
    if overrides.tariff_mode is not None:
        tariff_mode = overrides.tariff_mode
    if overrides.inference_demand is not None:
        inference_demand = overrides.inference_demand
        inference_score = {"LOW": 0.26, "MEDIUM": 0.56, "HIGH": 0.82}[inference_demand]

    if overrides.charge_price_per_kwh is not None:
        charge_price = overrides.charge_price_per_kwh
    if overrides.v2g_price_per_kwh is not None:
        v2g_price = overrides.v2g_price_per_kwh
    if overrides.inference_value_per_kwh is not None:
        inference_price = overrides.inference_value_per_kwh
    grid_stress = _stress_bucket(grid_probability, historical.fusion_threshold)
    if overrides.grid_stress is not None:
        grid_stress = overrides.grid_stress
        grid_probability = {"LOW": 0.18, "MEDIUM": 0.58, "HIGH": 0.86}[grid_stress]

    outage_probability = overrides.outage_probability
    if outage_probability is None:
        outage_probability = min(0.95, max(weather.storm_risk_index, grid_probability * 0.28))

    mobility_buffer = mobility_buffer_kwh(payload.hours_until_departure) + weather_margin
    flexible_kwh = available_flexible_energy_kwh(payload.arrival_soc_kwh, mobility_buffer)
    preferred_target = preferred_departure_target_kwh(payload.ownership_mode, payload.charger_type)
    participation_window_score = min(1.0, payload.hours_until_departure / 10.0)
    supports_v2g = payload.charger_type == "BIDIRECTIONAL"

    required_to_buffer = max(0.0, mobility_buffer - payload.arrival_soc_kwh)
    off_peak_recovery_capacity = off_peak_hours_remaining * CHARGE_POWER_KWH_PER_HOUR
    can_wait_for_off_peak = required_to_buffer <= off_peak_recovery_capacity

    alternatives: list[ActionScore] = []
    available_modes = ["CHARGING", "INFERENCE_ACTIVE", "IDLE"]
    if supports_v2g:
        available_modes.insert(1, "V2G_DISCHARGE")

    for action in available_modes:
        feasible = True
        delta = 0.0
        reason = "forecast_value"
        expected_value = 0.0

        if action == "CHARGING":
            delta = min(CHARGE_POWER_KWH_PER_HOUR, BATTERY_CAPACITY_KWH - payload.arrival_soc_kwh)
            expected_value = -(delta * charge_price)
            if tariff_mode != "OFF_PEAK" and can_wait_for_off_peak and payload.arrival_soc_kwh >= mobility_buffer:
                expected_value -= 0.35
                reason = "defer_charge_to_off_peak"
            elif payload.arrival_soc_kwh < preferred_target and tariff_mode == "OFF_PEAK":
                expected_value += 0.25
                reason = "cheap_recovery_window"
        elif action == "V2G_DISCHARGE":
            if not supports_v2g:
                feasible = False
                reason = "charger_not_bidirectional"
            elif flexible_kwh <= 0:
                feasible = False
                reason = "no_flexible_energy"
            else:
                delta = -min(V2G_DISCHARGE_POWER_KWH_PER_HOUR, flexible_kwh)
                expected_value = abs(delta) * v2g_price * grid_probability - abs(delta) * charge_price * 0.35
                if grid_stress != "HIGH":
                    expected_value -= 1.2
                    reason = "grid_signal_not_strong_enough"
        elif action == "INFERENCE_ACTIVE":
            if flexible_kwh <= 0:
                feasible = False
                reason = "no_flexible_energy"
            elif payload.hours_until_departure < 4:
                feasible = False
                reason = "parking_window_too_short"
            else:
                power = inference_power_kwh_per_hour(inference_demand)
                delta = -min(power, flexible_kwh)
                expected_value = abs(delta) * inference_price * inference_score - abs(delta) * charge_price * 0.2
                if inference_demand == "LOW":
                    expected_value -= 0.6
                    reason = "compute_demand_soft"
        else:
            expected_value = 0.0
            if tariff_mode != "OFF_PEAK" and can_wait_for_off_peak:
                expected_value += 0.18
                reason = "preserve_headroom_for_off_peak"

        projected_soc = payload.arrival_soc_kwh + delta
        if projected_soc < mobility_buffer:
            feasible = False
            reason = "violates_mobility_buffer"
            expected_value = float("-inf")

        alternatives.append(
            ActionScore(
                action=action,  # type: ignore[arg-type]
                feasible=feasible,
                expected_net_value_usd=round(expected_value if feasible else -999.0, 3),
                expected_energy_delta_kwh=round(delta, 3),
                reason=reason,
            )
        )

    if payload.arrival_soc_kwh <= mobility_buffer or (required_to_buffer > 0 and not can_wait_for_off_peak):
        chosen = next(item for item in alternatives if item.action == "CHARGING")
        explanation = "Charge now because current state cannot safely wait for later recovery without risking departure buffer."
        confidence = 0.92
    else:
        feasible_items = [item for item in alternatives if item.feasible]
        chosen = max(feasible_items, key=lambda item: item.expected_net_value_usd)
        if chosen.action == "V2G_DISCHARGE":
            explanation = "Bidirectional site plus peak-stress conditions make V2G the highest expected value while staying above the weather-adjusted mobility buffer."
            confidence = min(0.9, 0.52 + grid_probability * 0.4)
        elif chosen.action == "INFERENCE_ACTIVE":
            explanation = "Inference wins because compute demand and parked time are both strong enough to monetize flexible energy without crossing reserve."
            confidence = min(0.88, 0.46 + inference_score * 0.45)
        elif chosen.action == "CHARGING":
            explanation = "Off-peak energy is cheap right now and the battery is below the preferred departure target, so charging is favored over leaving value on the table."
            confidence = 0.76
        else:
            explanation = "Stay idle now and preserve the battery for a cheaper charging window or a stronger market signal later."
            confidence = 0.64

    environment_patch = {
        "grid_stress": grid_stress,
        "inference_demand": inference_demand,
        "tariff_mode": tariff_mode,
        "charge_price_per_kwh": round(charge_price, 3),
        "v2g_price_per_kwh": round(v2g_price, 3),
        "inference_value_per_kwh": round(inference_price, 3),
        "inference_power_kwh_per_hour": inference_power_kwh_per_hour(inference_demand),
    }

    return SiteDecisionResponse(
        location_id=payload.location_id,
        location_name=payload.location_name,
        charger_type=payload.charger_type,
        supports_v2g=supports_v2g,
        datetime_local=f"{payload.start_date_local}T{payload.start_time_local}",
        ownership_mode=payload.ownership_mode,
        weather=weather,
        forecast=ForecastContext(
            tariff_mode=tariff_mode,
            grid_stress=grid_stress,
            inference_demand=inference_demand,
            grid_event_probability=round(grid_probability, 3),
            outage_probability=round(outage_probability, 3),
            inference_value_score=round(inference_score, 3),
            participation_window_score=round(participation_window_score, 3),
        ),
        economics=EconomicsContext(
            charge_price_per_kwh=round(charge_price, 3),
            v2g_price_per_kwh=round(v2g_price, 3),
            inference_value_per_kwh=round(inference_price, 3),
            mobility_buffer_kwh=round(mobility_buffer, 3),
            weather_margin_kwh=round(weather_margin, 3),
            flexible_kwh=round(flexible_kwh, 3),
            preferred_departure_target_kwh=round(preferred_target, 3),
            off_peak_hours_remaining=off_peak_hours_remaining,
        ),
        recommendation=RecommendationContext(
            action=chosen.action,
            confidence=round(confidence, 3),
            explanation=explanation,
            alternatives=alternatives,
        ),
        environment_patch=environment_patch,
        simulation_seed=SimulationSeed(
            location_id=payload.location_id,
            location_name=payload.location_name,
            charger_type=payload.charger_type,
            arrival_soc_kwh=payload.arrival_soc_kwh,
            hours_until_departure=payload.hours_until_departure,
            start_date_local=payload.start_date_local,
            start_time_local=payload.start_time_local,
            environment_patch=environment_patch,
            initial_action=chosen.action,
            auto_step_on_launch=True,
            auto_play=True,
            speed_multiplier=1.0,
        ),
        model_context={
            "predictor_type": "hybrid_forecast_optimizer_v1",
            "source_mode": "historical_date_only",
            "historical_source": historical.source,
            "requested_date": historical.requested_date,
            "resolved_date": historical.resolved_date,
            "date_match_type": historical.match_type,
            "fusion_threshold": historical.fusion_threshold,
            "fusion_probability": round(historical.fusion_prob, 4),
            "historical_probabilities": {
                "eia_prob": round(historical.eia_prob, 4),
                "weather_prob": round(historical.weather_prob, 4),
                "price_prob": round(historical.price_prob, 4),
            },
            "historical_weather_raw": historical.weather_raw,
            "historical_price_raw": historical.price_raw,
            "historical_eia_raw": historical.eia_raw,
            "features_used": [
                "historical_eia_daily_features",
                "historical_weather_daily_features",
                "historical_price_daily_features",
                "trained_eia_weather_price_submodels",
                "trained_fusion_model",
                "selected_hour_tariff_shape",
                "research_price_bands",
                "charger_type",
                "arrival_soc_kwh",
                "hours_until_departure",
            ],
            "notes": [
                "Prediction is historical-date based only.",
                "The forecasting stack is rebuilt from cached Arizona daily data in the current environment and used end-to-end.",
                "The selected charger location affects charger capability constraints, while the forecast itself is Arizona-wide historical forecasting from the current training set.",
                "Action pricing uses research-backed Arizona TOU and inference/V2G monetization bands, not only retail-price multipliers.",
                "Recommendation is current-hour optimal action, not whole-session optimal control.",
            ],
        },
    )
