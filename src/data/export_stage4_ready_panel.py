"""Build a canonical Stage 4-ready panel from the canonical historical dataset."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.core.paths import CANONICAL_PANEL_PATH, OUTPUT_DATA_DIR, PROCESSED_DATA_DIR, ensure_directory
from src.data.build_canonical_panel import build_canonical_panel


SOURCE_PATH = CANONICAL_PANEL_PATH
OUTPUT_PATH = PROCESSED_DATA_DIR / "ols_ready_dataset_v2.csv"
METADATA_PATH = ensure_directory(OUTPUT_DATA_DIR / "stage3") / "stage4_ready_panel_metadata.json"

REQUIRED_COLUMNS = [
    "region",
    "date",
    "nominal_house_price",
    "real_house_price",
    "mortgage_rate",
    "base_rate",
    "real_annual_earnings",
    "gross_yield_pct",
    "price_growth",
    "price_growth_lag1",
    "mortgage_rate_lag3",
    "earnings_growth_12m_lag1",
    "log_price_to_income_diff",
]

ENHANCED_COLUMNS = [
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
    "regime_gfc",
    "regime_stamp_duty",
]

MONTH_COLUMNS = [f"month_{month}" for month in range(2, 13)]


@dataclass(frozen=True)
class Stage4ReadyPanelMetadata:
    source_file: str
    output_file: str
    generated_at_utc: str
    row_count: int
    region_count: int
    date_start: str
    date_end: str
    required_columns: list[str]
    enhanced_columns_present: list[str]


def build_stage4_ready_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Select and validate the enriched Stage 4 feature panel."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["region", "date"]).reset_index(drop=True)
    if "earnings_growth_12m_lag1" not in df.columns and "real_annual_earnings" in df.columns:
        df["earnings_growth_12m_lag1"] = (
            df.groupby("region", sort=False)["real_annual_earnings"].pct_change(12).shift(1)
        )
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"master_dataset_v2 is missing required Stage 4 columns: {missing}")

    ordered_columns: list[str] = []
    for column in REQUIRED_COLUMNS + ENHANCED_COLUMNS + MONTH_COLUMNS:
        if column in df.columns and column not in ordered_columns:
            ordered_columns.append(column)

    panel = df[ordered_columns].copy()
    return panel


def export_stage4_ready_panel(
    source_path: Path = SOURCE_PATH,
    output_path: Path = OUTPUT_PATH,
    metadata_path: Path = METADATA_PATH,
) -> Stage4ReadyPanelMetadata:
    """Write the canonical Stage 4-ready panel and a small lineage record."""
    if source_path == CANONICAL_PANEL_PATH and not source_path.exists():
        build_canonical_panel()
    df = pd.read_csv(source_path, parse_dates=["date"])
    panel = build_stage4_ready_panel(df)
    panel.to_csv(output_path, index=False)

    metadata = Stage4ReadyPanelMetadata(
        source_file=source_path.name,
        output_file=output_path.name,
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        row_count=int(len(panel)),
        region_count=int(panel["region"].nunique()),
        date_start=str(panel["date"].min().date()),
        date_end=str(panel["date"].max().date()),
        required_columns=REQUIRED_COLUMNS,
        enhanced_columns_present=[column for column in ENHANCED_COLUMNS if column in panel.columns],
    )
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    metadata = export_stage4_ready_panel()
    print(json.dumps(asdict(metadata), indent=2))


if __name__ == "__main__":
    main()
