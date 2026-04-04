"""Tests for the canonical Stage 6 scoring helpers."""

from pathlib import Path

from src.core.config import load_config
from src.scoring.engine import (
    consumer_signal_dashboard,
    label_score,
    reit_signal_dashboard,
    score_consumer,
    score_reit,
)


def test_label_score_buy():
    assert label_score(70) == "Supportive"


def test_label_score_hold():
    assert label_score(50) == "Mixed"


def test_label_score_dont_buy():
    assert label_score(20) == "Cautious"


def test_score_consumer_returns_bounded_score():
    result = score_consumer(
        "London",
        income=85000,
        deposit=100000,
        simulation_percentiles={"p10": -15, "p50": 2, "p90": 15, "base_model_score": 58, "start_price": 425000},
        property_price=425000,
        mortgage_rate=5.25,
        mortgage_term_years=30,
        downside_prob=0.32,
    )

    assert 0 <= result["total_score"] <= 100
    assert result["label"] in {"Supportive", "Mixed", "Cautious"}
    assert result["label_mode"] == "descriptive_signal"
    assert result["mortgage_stress"] in {"Contained", "Elevated", "High"}


def test_score_reit_returns_bounded_score():
    result = score_reit(
        "North West",
        gross_yield=6.1,
        simulation_percentiles={"p10": -10, "p50": 6, "p90": 18, "base_model_score": 62},
        target_yield=5.5,
        downside_prob=0.18,
    )

    assert 0 <= result["total_score"] <= 100
    assert result["label"] in {"Supportive", "Mixed", "Cautious"}
    assert result["label_mode"] == "descriptive_signal"
    assert result["yield_gap_pct"] == 0.6


def test_signal_dashboards_expose_named_components_and_no_action_labels():
    consumer = consumer_signal_dashboard(
        "London",
        income=85000,
        deposit=100000,
        simulation_percentiles={"p10": -15, "p50": 2, "p90": 15, "start_price": 425000},
        valuation_signal=62,
        cycle_signal=54,
        property_price=425000,
        mortgage_rate=5.25,
        mortgage_term_years=30,
        downside_prob=0.32,
    )
    reit = reit_signal_dashboard(
        "North West",
        gross_yield=6.1,
        simulation_percentiles={"p10": -10, "p50": 6, "p90": 18, "start_price": 210000},
        valuation_signal=58,
        cycle_signal=52,
        target_yield=5.5,
        downside_prob=0.18,
    )

    assert {signal["display_name"] for signal in consumer["signals"]} == {"Valuation", "Cycle / Momentum", "Downside Resilience", "Affordability"}
    assert {signal["display_name"] for signal in reit["signals"]} == {"Valuation", "Cycle / Momentum", "Downside Resilience", "Yield Support"}
    assert consumer["secondary_composite"]["label_mode"] == "secondary_synthesis_only"
    assert reit["secondary_composite"]["label_mode"] == "secondary_synthesis_only"
    assert consumer["secondary_composite"]["method"] == "evidence_adjusted_secondary_synthesis"
    assert reit["secondary_composite"]["method"] == "evidence_adjusted_secondary_synthesis"
    assert all("simple_status" in signal for signal in consumer["signals"])
    assert all("simple_status" in signal for signal in reit["signals"])


def test_weight_consistency():
    """Both weight sets must sum to 1.0 and no hardcoded weight dicts must remain
    in the source files that previously housed them."""
    from dataclasses import asdict

    config = load_config()

    # --- component weights (app_consumer_weights, app_reit_weights) ---
    consumer_w = asdict(config.scoring.app_consumer_weights)
    reit_w = asdict(config.scoring.app_reit_weights)

    assert consumer_w, "app_consumer_weights must not be empty"
    assert reit_w, "app_reit_weights must not be empty"
    assert abs(sum(consumer_w.values()) - 1.0) < 1e-9, (
        f"app_consumer_weights sum to {sum(consumer_w.values())}, expected 1.0"
    )
    assert abs(sum(reit_w.values()) - 1.0) < 1e-9, (
        f"app_reit_weights sum to {sum(reit_w.values())}, expected 1.0"
    )

    # --- scenario probability weights ---
    scenario_consumer_w = config.scoring.scenario_consumer_weights
    scenario_reit_w = config.scoring.scenario_reit_weights

    assert scenario_consumer_w, "scenario_consumer_weights must not be empty"
    assert scenario_reit_w, "scenario_reit_weights must not be empty"
    assert abs(sum(scenario_consumer_w.values()) - 1.0) < 1e-9, (
        f"scenario_consumer_weights sum to {sum(scenario_consumer_w.values())}, expected 1.0"
    )
    assert abs(sum(scenario_reit_w.values()) - 1.0) < 1e-9, (
        f"scenario_reit_weights sum to {sum(scenario_reit_w.values())}, expected 1.0"
    )

    # --- confirm no hardcoded weight dicts remain in canonical_rebuild.py ---
    ROOT = Path(__file__).resolve().parents[1]
    rebuild_src = (ROOT / "src/model/canonical_rebuild.py").read_text(encoding="utf-8")

    assert '"Baseline": 0.40' not in rebuild_src, (
        "Hardcoded consumer scenario weight found in canonical_rebuild.py — "
        "should be config-driven via scoring.scenario_consumer_weights"
    )
    assert "1.0 / len(load_config().ui.scenario_labels)" not in rebuild_src, (
        "Dynamic 1/N reit_weights still present in canonical_rebuild.py — "
        "should be config-driven via scoring.scenario_reit_weights"
    )


def test_descriptive_signals_are_shrunk_in_secondary_synthesis():
    consumer = consumer_signal_dashboard(
        "London",
        income=120000,
        deposit=200000,
        simulation_percentiles={"p10": 5, "p50": 10, "p90": 20, "start_price": 425000},
        valuation_signal=50,
        cycle_signal=50,
        property_price=425000,
        mortgage_rate=2.0,
        mortgage_term_years=30,
        downside_prob=0.0,
    )

    assert consumer["signal_scores"]["downside"] == 100.0
    assert consumer["signal_scores"]["affordability"] == 100.0
    assert consumer["secondary_composite"]["score"] < 100.0
