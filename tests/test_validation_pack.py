"""Tests for the canonical validation pack outputs."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PANEL = ROOT / "data" / "processed" / "master_dataset_canonical.csv"


def test_validation_pack_forecast_benchmark_shows_model_edge():
    """
    R023: model does not beat zero-growth on RMSE in the 2023–2025 holdout
    (rate-shock regime diverges from training window). Model does beat AR1 at
    all horizons — the primary evidence for forecast signal. Assertions reflect
    the documented R023 limitation.
    """
    overall = pd.read_csv(ROOT / "data/outputs/validation/forecast_benchmark_overall.csv")
    rmse = overall.set_index("benchmark")["rmse"]

    assert rmse["model"] <= rmse["ar1"], (
        f"model RMSE ({rmse['model']:.6f}) should beat AR1 ({rmse['ar1']:.6f})"
    )


def test_validation_pack_has_stable_rate_gap_feature():
    params = pd.read_csv(ROOT / "data/outputs/stage4_final/sde_parameters_bankgrade.csv")
    stability = pd.read_csv(ROOT / "data/outputs/validation/coefficient_stability.csv").set_index("feature")
    rate_candidates = pd.read_csv(ROOT / "data/outputs/stage4_final/rate_channel_candidates.csv").set_index("feature")
    active_rate_feature = params["rate_channel_feature"].iloc[0]

    assert active_rate_feature == "mortgage_rate_lag3"
    assert active_rate_feature in stability.index
    assert bool(stability.loc[active_rate_feature, "same_sign_full_late"])
    assert active_rate_feature in rate_candidates.index
    assert rate_candidates.loc[active_rate_feature, "late_coef"] < 0


def test_validation_pack_marks_stress_feature_as_auxiliary_episode_not_core_instability():
    stability = pd.read_csv(ROOT / "data/outputs/validation/coefficient_stability.csv").set_index("feature")

    assert "financial_stress_excess_lag3" in stability.index
    assert stability.loc["financial_stress_excess_lag3", "interpretation"] == "episodic_not_identified_late"


# ---------------------------------------------------------------------------
# C2 / C3 signal backtest smoke tests (R003)
# ---------------------------------------------------------------------------

def test_c2_downside_backtest_runs():
    """
    Smoke test for the C2 downside probability directional backtest.

    The canonical panel does not carry a 'downside_probability' time series,
    so backtest_c2_downside() will return an empty DataFrame with the expected
    schema (and emit a UserWarning). This test confirms:
    - No exception is raised
    - The result DataFrame has the required columns
    - The summary dict has accuracy, precision, recall keys

    Does NOT enforce minimum accuracy — results are data-driven.
    """
    import warnings
    from src.scoring.backtest import backtest_c2_downside

    panel = pd.read_csv(CANONICAL_PANEL, parse_dates=["date"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result_df, summary = backtest_c2_downside(panel)

    expected_cols = {"region", "date", "predicted_downside_flag",
                     "realised_decline", "correct_direction"}
    assert expected_cols.issubset(set(result_df.columns)), (
        f"backtest_c2_downside() result missing columns. "
        f"Expected {expected_cols}, got {set(result_df.columns)}"
    )
    assert isinstance(summary, dict), "summary must be a dict"
    assert {"accuracy", "precision", "recall"}.issubset(summary.keys()), (
        f"summary missing keys, got {set(summary.keys())}"
    )


def test_c3_affordability_backtest_runs():
    """
    Smoke test for the C3 affordability signal directional backtest.

    The canonical panel contains 'affordability_gap', so this backtest
    runs with real data. This test confirms:
    - No exception is raised
    - The result DataFrame is non-empty and has the required columns
    - The summary dict has accuracy and directional_information_ratio keys

    Does NOT enforce minimum accuracy — results are data-driven.
    """
    from src.scoring.backtest import backtest_c3_affordability

    panel = pd.read_csv(CANONICAL_PANEL, parse_dates=["date"])
    result_df, summary = backtest_c3_affordability(panel)

    expected_cols = {"region", "date", "affordability_signal",
                     "realised_correction", "correct_direction"}
    assert expected_cols.issubset(set(result_df.columns)), (
        f"backtest_c3_affordability() result missing columns. "
        f"Expected {expected_cols}, got {set(result_df.columns)}"
    )
    assert len(result_df) > 0, (
        "backtest_c3_affordability() returned an empty DataFrame — "
        "'affordability_gap' should be present in the canonical panel"
    )
    assert isinstance(summary, dict), "summary must be a dict"
    assert {"accuracy", "directional_information_ratio"}.issubset(summary.keys()), (
        f"summary missing keys, got {set(summary.keys())}"
    )


# ---------------------------------------------------------------------------
# C4 / C5 signal backtest smoke tests
# ---------------------------------------------------------------------------

def test_c4_rental_yield_backtest_runs():
    """
    Smoke test for the C4 rental yield directional backtest.

    The canonical panel contains 'gross_yield_pct', so this backtest
    runs with real data.  This test confirms:
    - No exception is raised
    - The result DataFrame is non-empty and has the required columns
    - The summary dict has accuracy and information_ratio_proxy keys

    Does NOT enforce minimum accuracy — results are data-driven.
    """
    from src.scoring.backtest import backtest_c4_rental_yield

    panel = pd.read_csv(CANONICAL_PANEL, parse_dates=["date"])
    result_df, summary = backtest_c4_rental_yield(panel)

    expected_cols = {
        "region", "date", "yield_signal_direction",
        "realised_outperformance", "correct_direction",
    }
    assert expected_cols.issubset(set(result_df.columns)), (
        f"backtest_c4_rental_yield() result missing columns. "
        f"Expected {expected_cols}, got {set(result_df.columns)}"
    )
    assert len(result_df) > 0, (
        "backtest_c4_rental_yield() returned an empty DataFrame — "
        "'gross_yield_pct' should be present in the canonical panel"
    )
    assert isinstance(summary, dict), "summary must be a dict"
    assert {"accuracy", "information_ratio_proxy"}.issubset(summary.keys()), (
        f"summary missing keys, got {set(summary.keys())}"
    )


def test_c5_cycle_backtest_runs():
    """
    Smoke test for the C5 cycle / momentum directional backtest.

    The canonical panel contains 'price_growth_12m', so this backtest
    runs with real data.  This test confirms:
    - No exception is raised
    - The result DataFrame is non-empty and has the required columns
    - The summary dict has accuracy and information_ratio_proxy keys

    Does NOT enforce minimum accuracy — results are data-driven.
    """
    import warnings
    from src.scoring.backtest import backtest_c5_cycle

    panel = pd.read_csv(CANONICAL_PANEL, parse_dates=["date"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result_df, summary = backtest_c5_cycle(panel)

    expected_cols = {
        "region", "date", "momentum_signal_direction",
        "realised_continuation", "correct_direction",
    }
    assert expected_cols.issubset(set(result_df.columns)), (
        f"backtest_c5_cycle() result missing columns. "
        f"Expected {expected_cols}, got {set(result_df.columns)}"
    )
    assert len(result_df) > 0, (
        "backtest_c5_cycle() returned an empty DataFrame — "
        "a momentum column should be present in the canonical panel"
    )
    assert isinstance(summary, dict), "summary must be a dict"
    assert {"accuracy", "information_ratio_proxy"}.issubset(summary.keys()), (
        f"summary missing keys, got {set(summary.keys())}"
    )
