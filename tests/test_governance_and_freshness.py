from datetime import date
from pathlib import Path

import json
import pandas as pd

from src.core.config import load_config
from src.core.paths import STAGE6_OUTPUT_DIR


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_panel_metadata_is_historical_and_has_source_coverage():
    config = load_config()
    handoff = pd.read_csv(STAGE6_OUTPUT_DIR / config.artifacts.stage6_handoff_file)
    scoring_date = handoff["scoring_date"].iloc[0]
    # Threshold: 210 days (7 months). Panel refresh triggered when scoring_date
    # falls more than 7 months behind today. Governance commitment: refresh by 2026-06-30.
    assert (date.today() - date.fromisoformat(scoring_date)).days <= 210, (
        f"scoring_date {scoring_date} is more than 210 days before today ({date.today()})"
    )

    metadata = json.loads((ROOT / "data/outputs/metadata/canonical_panel_metadata.json").read_text(encoding="utf-8"))

    assert bool(metadata["historical_only"])
    assert metadata["panel_end"] == scoring_date
    assert bool(metadata["split_horizon"])
    assert metadata["latest_house_price_end"] >= scoring_date
    assert metadata["latest_descriptive_housing_end"] >= scoring_date
    # weakest_model_input_end can precede scoring_date when flow series are
    # forward-filled under FLOW_SERIES_FFILL_LIMIT governance (updated 2026-04-02).
    assert metadata["weakest_model_input_end"] <= scoring_date
    assert "cutoff_reason" in metadata and metadata["cutoff_reason"]
    assert "source_coverage" in metadata and "house_prices" in metadata["source_coverage"]
    assert metadata["source_coverage"]["house_prices"]["observation_end"] >= scoring_date
    # rental_yield is used only in Stage 6 scoring (not OLS) and is forward-filled
    # beyond its last annual observation.  ONS PRMS publication lag (~12 months)
    # plus the annual observation cadence means up to 24 months of forward-fill is
    # expected before raising a governance flag.  Updated from 18 to 24 months
    # 2026-03-30 to reflect annual (not quarterly) rental yield data and the
    # 2026-01 LR HPI refresh which advanced scoring_date context.
    rental_yield_end = date.fromisoformat(metadata["source_coverage"]["rental_yield"]["observation_end"])
    rental_yield_staleness_months = (
        (date.fromisoformat(scoring_date).year - rental_yield_end.year) * 12
        + (date.fromisoformat(scoring_date).month - rental_yield_end.month)
    )
    assert rental_yield_staleness_months <= 24, (
        f"rental_yield forward-fill staleness {rental_yield_staleness_months} months "
        f"exceeds 24-month governance limit (rental_yield_end={rental_yield_end}, "
        f"scoring_date={scoring_date})"
    )


def test_descriptive_panel_metadata_extends_beyond_model_panel():
    model_metadata = json.loads((ROOT / "data/outputs/metadata/canonical_panel_metadata.json").read_text(encoding="utf-8"))
    descriptive_metadata = json.loads((ROOT / "data/outputs/metadata/descriptive_panel_metadata.json").read_text(encoding="utf-8"))

    assert descriptive_metadata["descriptive_only"]
    assert descriptive_metadata["panel_end"] >= model_metadata["model_panel_end"]
    assert descriptive_metadata["latest_house_price_end"] == "2026-01-01"


def test_canonical_panel_has_no_future_leakage_in_earnings_obs_date():
    panel = pd.read_csv(ROOT / "data/processed/master_dataset_canonical.csv", parse_dates=["date", "earnings_obs_date"])

    assert (panel["earnings_obs_date"] <= panel["date"]).all()
    assert panel["earnings_staleness_months"].between(0, 24).all()


def test_config_has_one_scenario_namespace():
    config = load_config()

    assert config.ui.scenario_labels == list(config.scenarios.keys())
    assert set(config.ui.scenario_labels) == {"Baseline", "Soft_Landing", "Rate_Shock", "Stagflation", "Recovery_Boom"}


def test_legacy_app_tree_is_quarantined():
    assert not (ROOT / "app").exists()
    assert (ROOT / "archive/legacy_app/app").exists()
