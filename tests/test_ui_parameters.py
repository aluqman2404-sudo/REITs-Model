"""Tests for Stage 7 simulation parameter construction."""

from src.core.config import load_config
from src.ui.loaders import get_region_record, get_region_simulation_summary
from src.ui.views import _scenario_parameters, _scenario_summary_row


def test_dashboard_loader_exposes_enriched_history_columns():
    region_row = get_region_record("London")

    assert "hist_price_growth_vol_annual" in region_row
    assert "unemployment_lag1" in region_row
    assert "approvals_growth" in region_row


def test_scenario_parameters_use_annualized_and_blended_sigma():
    region = "London"
    region_row = get_region_record(region)
    region_summary = get_region_simulation_summary(region)
    stage5_row = _scenario_summary_row(region_summary, "Baseline")

    params, meta = _scenario_parameters(region_row, "Baseline", stage5_row=stage5_row, horizon_years=5)
    price_floor = float(region_row["nominal_house_price"]) * load_config().controls.minimum_price_floor_ratio

    assert meta["sigma_annual"] >= float(region_row["sigma"])
    assert params["sigma"] >= meta["sigma_annual"]
    assert meta["target_fair_value"] >= price_floor
