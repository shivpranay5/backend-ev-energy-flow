# Mock Data Bundle

This folder contains the minimal JSON fixtures for the Grid Sense WebSocket MVP.

Files:
- `simulation_constants.json`: hardcoded system defaults and bounds
- `session_seed.json`: initial simulation inputs
- `environment_presets.json`: admin scenario presets
- `vehicle_profiles.json`: canonical vehicle states across the state space
- `vehicle_state.json`: normalized single-vehicle state
- `ocpp_telemetry.json`: mocked charger/session telemetry
- `ocpp_transitions.json`: normalized charger state machine examples
- `decision_matrix.json`: branch coverage for all action selections
- `action_scenarios.json`: action scoring fixtures and expected outcomes
- `scenario_runs.json`: end-to-end traces for each major scenario
- `chart_series.json`: per-tick time-series data for charts
- `ws_messages.json`: WebSocket request/response examples
- `final_summary.json`: simulation completion payload
- `summary_reports.json`: per-scenario final reports
- `test_fixtures.json`: edge cases for backend tests
