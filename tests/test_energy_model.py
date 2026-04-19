from app.energy_model import (
    count_off_peak_hours,
    inference_power_kwh_per_hour,
    mobility_buffer_kwh,
    tariff_mode_for_hour,
)


def test_tariff_mode_for_hour_matches_time_of_use_schedule() -> None:
    assert tariff_mode_for_hour(2) == "OFF_PEAK"
    assert tariff_mode_for_hour(10) == "NORMAL"
    assert tariff_mode_for_hour(18) == "PEAK"


def test_mobility_buffer_grows_as_departure_window_shrinks() -> None:
    assert mobility_buffer_kwh(12) == 20.0
    assert mobility_buffer_kwh(6) == 25.0
    assert mobility_buffer_kwh(3) == 30.0


def test_count_off_peak_hours_wraps_across_midnight() -> None:
    assert count_off_peak_hours(20, 8) == 6


def test_inference_power_tracks_demand_bucket() -> None:
    assert inference_power_kwh_per_hour("LOW") == 2.0
    assert inference_power_kwh_per_hour("MEDIUM") == 4.0
    assert inference_power_kwh_per_hour("HIGH") == 8.0
