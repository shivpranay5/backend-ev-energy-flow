from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MOCK_DIR = Path(__file__).resolve().parents[2] / "mock"


def load_mock_json(name: str) -> Any:
    path = MOCK_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_bundle() -> dict[str, Any]:
    files = [
        "simulation_constants.json",
        "session_seed.json",
        "environment_presets.json",
        "vehicle_state.json",
        "ocpp_telemetry.json",
        "ocpp_transitions.json",
        "decision_matrix.json",
        "action_scenarios.json",
        "scenario_runs.json",
        "chart_series.json",
        "ws_messages.json",
        "final_summary.json",
        "summary_reports.json",
        "test_fixtures.json",
    ]
    return {name: load_mock_json(name) for name in files}

