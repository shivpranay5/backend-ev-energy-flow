from __future__ import annotations

from typing import Final, Literal

InferenceDemand = Literal["LOW", "MEDIUM", "HIGH"]
TariffMode = Literal["OFF_PEAK", "NORMAL", "PEAK"]

# Keep the core energy assumptions in one place so the simulator, predictor,
# and fleet optimizer all describe the same physical system.
BATTERY_CAPACITY_KWH: Final[float] = 100.0
CHARGE_POWER_KWH_PER_HOUR: Final[float] = 7.0
V2G_DISCHARGE_POWER_KWH_PER_HOUR: Final[float] = 10.0
V2G_EVENT_CAP_KWH: Final[float] = 30.0


def tariff_mode_for_hour(hour: int) -> TariffMode:
    normalized_hour = hour % 24
    if normalized_hour >= 22 or normalized_hour < 6:
        return "OFF_PEAK"
    if 16 <= normalized_hour < 21:
        return "PEAK"
    return "NORMAL"


def tariff_mode_for_time(start_time_local: str | None, default: TariffMode = "OFF_PEAK") -> TariffMode:
    if not start_time_local:
        return default
    try:
        hour = int(str(start_time_local).split(":", 1)[0])
    except Exception:  # noqa: BLE001
        return default
    return tariff_mode_for_hour(hour)


def mobility_buffer_kwh(hours_until_departure: int) -> float:
    if hours_until_departure <= 4:
        return 30.0
    if hours_until_departure <= 8:
        return 25.0
    return 20.0


def count_off_peak_hours(start_hour: int, hours_until_departure: int) -> int:
    off_peak_hours = 0
    for offset in range(hours_until_departure):
        if tariff_mode_for_hour(start_hour + offset) == "OFF_PEAK":
            off_peak_hours += 1
    return off_peak_hours


def preferred_departure_target_kwh(ownership_mode: str, charger_type: str) -> float:
    base_target = 75.0 if ownership_mode == "private" else 70.0
    if charger_type == "BIDIRECTIONAL":
        return base_target + 5.0
    return base_target


def inference_power_kwh_per_hour(
    demand: InferenceDemand,
    override: float | None = None,
) -> float:
    if override is not None:
        return float(override)
    if demand == "HIGH":
        return 8.0
    if demand == "MEDIUM":
        return 4.0
    return 2.0


def available_flexible_energy_kwh(soc_kwh: float, mobility_buffer: float) -> float:
    return max(0.0, soc_kwh - mobility_buffer)
