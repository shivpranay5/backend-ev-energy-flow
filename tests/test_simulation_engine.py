from app.simulation import SimulationEngine


def make_engine(*, soc_kwh: float, hours_until_departure: int, charger_type: str = "BIDIRECTIONAL") -> SimulationEngine:
    engine = SimulationEngine(defaults={}, ocpp_transitions=[])
    engine.seed.update(
        {
            "arrival_soc_kwh": soc_kwh,
            "hours_until_departure": hours_until_departure,
            "charger_type": charger_type,
            "location_id": "test-site",
            "charger_location": "Test Charger",
            "start_date_local": "2026-04-19",
            "start_time_local": "18:00",
        }
    )
    engine.environment.grid_stress = "HIGH"
    engine.environment.inference_demand = "LOW"
    engine.environment.tariff_mode = "PEAK"
    engine.state = engine._new_state_from_seed()
    engine.initialized = True
    return engine


def test_simulation_forces_charging_at_mobility_floor() -> None:
    engine = make_engine(soc_kwh=25.0, hours_until_departure=6, charger_type="BIDIRECTIONAL")

    snapshot = engine._advance_locked()

    assert snapshot["current_mode"] == "CHARGING"
    assert snapshot["battery_level_kwh"] == 32.0
    assert snapshot["iteration_breakdown"]["selected_action"] == "CHARGING"
    assert snapshot["iteration_breakdown"]["direct_cost_usd"] == 0.77
    assert snapshot["iteration_breakdown"]["reason"] == "hard_override_soc_at_or_below_buffer"
    assert snapshot["action_candidates"][0]["action"] == "CHARGING"


def test_simulation_respects_session_v2g_export_cap() -> None:
    engine = make_engine(soc_kwh=95.0, hours_until_departure=8, charger_type="BIDIRECTIONAL")

    first = engine._advance_locked()
    second = engine._advance_locked()
    third = engine._advance_locked()
    fourth = engine._advance_locked()

    assert first["current_mode"] == "V2G_DISCHARGE"
    assert second["current_mode"] == "V2G_DISCHARGE"
    assert third["current_mode"] == "V2G_DISCHARGE"
    assert fourth["current_mode"] != "V2G_DISCHARGE"
    assert fourth["grid_export_kwh"] == 30.0


def test_simulation_iteration_breakdown_reports_v2g_economics() -> None:
    engine = make_engine(soc_kwh=95.0, hours_until_departure=8, charger_type="BIDIRECTIONAL")

    snapshot = engine._advance_locked()

    assert snapshot["iteration_breakdown"] == {
        "action": "V2G_DISCHARGE",
        "selected": True,
        "feasible": True,
        "energy_delta_kwh": -10.0,
        "energy_transacted_kwh": 10.0,
        "gross_revenue_usd": 3.5,
        "direct_cost_usd": 0.0,
        "recovery_cost_usd": 0.0,
        "projected_net_value_usd": 3.5,
        "reason": None,
        "iteration": 1,
        "selected_action": "V2G_DISCHARGE",
        "candidate_count": 4,
    }
    assert [candidate["action"] for candidate in snapshot["action_candidates"]] == [
        "V2G_DISCHARGE",
        "IDLE",
        "CHARGING",
        "INFERENCE_ACTIVE",
    ]
    assert snapshot["action_candidates"][0]["gross_revenue_usd"] == 3.5
