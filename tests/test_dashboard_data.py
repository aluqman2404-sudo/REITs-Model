"""Tests for validated dashboard data loading."""

from pathlib import Path

from src.ui.loaders import load_dashboard_data


def test_dashboard_loader_returns_expected_regions_and_cache():
    data = load_dashboard_data()

    assert data.metadata["region_count"] == 12
    assert data.metadata["scenario_count"] == 5
    assert data.metadata["master_dataset_file"] == "master_dataset_canonical.csv"
    assert data.metadata["stage4_parameters_file"] == "sde_parameters_bankgrade.csv"
    assert data.metadata["stage5_summary_file"] == "simulation_summary_bankgrade.csv"
    assert data.metadata["stage6_handoff_file"] == "stage7_handoff_bankgrade.csv"
    assert bool(data.metadata["historical_only"])
    assert bool(data.metadata["split_horizon"])
    assert data.metadata["effective_data_as_of"] == "2025-01-01"
    assert data.metadata["max_publication_lag_months"] == 12
    assert data.metadata["descriptive_panel_file"] == "descriptive_market_panel.csv"
    assert data.metadata["descriptive_panel_coverage_end"] == "2026-01-01"
    assert data.metadata["latest_house_price_end"] == "2026-01-01"
    validation = data.metadata["validation_summary"]
    assert validation["rate_stability"]["flagged_unstable_regions"] == 0
    assert not validation["subsample_holdout"]["beats_zero_growth"]
    assert (
        validation["forecast_benchmarks"]["model"]["rmse"]
        < validation["forecast_benchmarks"]["ar1"]["rmse"]
    )
    assert data.region_table["region"].nunique() == 12
    assert "latest_house_price_date" in data.region_table.columns
    assert "latest_house_price" in data.region_table.columns
    assert (data.region_table["latest_house_price_date"] >= data.region_table["date"]).all()

    stage7_dir = Path(__file__).resolve().parents[1] / "data/outputs/stage7"
    assert (stage7_dir / "dashboard_metadata.json").exists()
    assert (stage7_dir / "region_snapshot.csv").exists()
