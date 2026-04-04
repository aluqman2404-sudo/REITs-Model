"""Tests for the canonical historical panel to Stage 4-ready export."""

from pathlib import Path

import pandas as pd

from src.data.build_canonical_panel import build_canonical_panel
from src.data.export_stage4_ready_panel import build_stage4_ready_panel


def test_stage4_ready_panel_keeps_enriched_columns():
    build_canonical_panel()
    source = Path(__file__).resolve().parents[1] / "data/processed/master_dataset_canonical.csv"
    panel = build_stage4_ready_panel(pd.read_csv(source, parse_dates=["date"]))

    expected_columns = {
        "financial_stress_lag3",
        "approvals_growth",
        "starts_lag6",
        "starts_lag12",
        "unemployment_rate",
        "unemployment_diff",
        "unemployment_lag1",
        "transaction_growth",
        "payment_burden",
        "payment_burden_gap",
        "affordability_gap",
        "price_to_rent",
        "lender_spread_lag3",
    }

    assert expected_columns.issubset(panel.columns)
    assert panel["region"].nunique() == 12
    assert panel.groupby("region")["date"].apply(lambda series: series.is_monotonic_increasing).all()
