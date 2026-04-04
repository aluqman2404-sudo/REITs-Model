import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_independent_validation_app_consistency_checks_pass():
    checks = pd.read_csv(ROOT / "data/outputs/validation/app_artifact_consistency.csv")

    assert checks["passed"].eq(1).all()


def test_independent_validation_lineage_contains_canonical_files():
    lineage = pd.read_csv(ROOT / "data/outputs/validation/artifact_lineage_checks.csv")
    lookup = lineage.set_index("artifact")

    assert bool(lookup.loc["canonical_panel", "exists"])
    assert bool(lookup.loc["stage4_fair_value_panel", "exists"])
    assert bool(lookup.loc["stage7_snapshot", "exists"])


def test_score_calibration_supports_only_longer_horizon_descriptive_bucketing():
    calibration = pd.read_csv(ROOT / "data/outputs/validation/score_calibration_diagnostics.csv")
    valuation = calibration[calibration["signal_name"] == "valuation_signal"].set_index("horizon_months")

    assert valuation.loc[36, "monotonic_mean_return"]
    assert valuation.loc[60, "monotonic_mean_return"]
    assert valuation.loc[60, "supportive_minus_cautious_return_pp"] > 0


def test_component_signal_strength_identifies_split_between_supported_and_descriptive_layers():
    strength = pd.read_csv(ROOT / "data/outputs/validation/component_signal_strength.csv")

    valuation_60 = strength[(strength["signal_name"] == "valuation_signal") & (strength["horizon_months"] == 60)].iloc[0]
    macro_blend_12 = strength[(strength["signal_name"] == "historical_macro_blend_signal") & (strength["horizon_months"] == 12)].iloc[0]

    assert valuation_60["evidence_level"] == "supported"
    assert macro_blend_12["evidence_level"] in {"supported", "partial"}


def test_simulation_transparency_outputs_exist_and_are_populated():
    distribution = pd.read_csv(ROOT / "data/outputs/validation/historical_vs_simulated_distribution.csv")
    moments = pd.read_csv(ROOT / "data/outputs/validation/simulation_moment_comparison.csv")

    assert not distribution.empty
    assert not moments.empty
    assert "UK_12_region_average" in set(moments["region"])


def test_ols_replay_check_passes():
    replay_path = ROOT / "data/outputs/validation/ols_replay_check.json"
    if not replay_path.exists():
        pytest.skip("ols_replay_check.json not yet generated — run independent_validation_runner first")
    result = json.loads(replay_path.read_text(encoding="utf-8"))
    assert result["passed"] is True, (
        f"OLS replay check failed: {result.get('finding', 'no detail')}. "
        f"max_deviation={result.get('replay_max_deviation')}"
    )
    assert result["replay_max_deviation"] < 0.02


def test_subsample_ols_validation_runs():
    subsample_path = ROOT / "data/outputs/validation/subsample_ols_validation.json"
    if not subsample_path.exists():
        pytest.skip("subsample_ols_validation.json not yet generated — run independent_validation_runner first")
    result = json.loads(subsample_path.read_text(encoding="utf-8"))
    required_keys = {
        "train_end", "holdout_start", "holdout_rmse", "zero_growth_rmse",
        "beats_zero_growth", "directional_1m", "directional_6m", "directional_12m",
    }
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )
    dir_12m = result["directional_12m"]
    assert dir_12m is not None, "directional_12m should not be None"
    assert 0.40 <= dir_12m <= 0.80, (
        f"directional_12m={dir_12m:.4f} is outside plausibility range [0.40, 0.80]"
    )
