from app.fleet import compare_three_seasons, compute_annual_projection, compute_upgrade_roi


def test_annual_projection_day_buckets_sum_to_calendar_year() -> None:
    projection = compute_annual_projection()

    total_days = (
        projection.high_stress_days_per_year
        + projection.medium_stress_days_per_year
        + projection.low_stress_days_per_year
    )

    assert total_days == 365.0


def test_annual_projection_scales_with_fleet_size() -> None:
    projection_small = compute_annual_projection(15)
    projection_large = compute_annual_projection(30)

    assert projection_large.projected_annual_net_usd != projection_small.projected_annual_net_usd
    assert projection_large.bidirectional_count > projection_small.bidirectional_count
    assert projection_large.inference_capable_count > projection_small.inference_capable_count
    assert projection_large.fleet_size == 30


def test_upgrade_roi_uses_positive_v2g_value_signal() -> None:
    roi = compute_upgrade_roi()

    assert roi.current_annual_net_usd > 0
    assert all(scenario.additional_annual_v2g_revenue_usd > 0 for scenario in roi.scenarios)


def test_compare_three_seasons_returns_three_scenarios() -> None:
    comparison = compare_three_seasons()

    assert len(comparison.scenarios) == 3
    assert comparison.insight
