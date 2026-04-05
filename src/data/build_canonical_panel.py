"""Build the canonical historical research panel used by Stage 4-7."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.logging_utils import get_logger
from src.core.paths import (
    CANONICAL_PANEL_METADATA_PATH,
    CANONICAL_PANEL_PATH,
    DESCRIPTIVE_PANEL_METADATA_PATH,
    DESCRIPTIVE_PANEL_PATH,
    METADATA_OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ensure_directory,
)
from src.ingestion.release_metadata import read_release_metadata


logger = get_logger("data.canonical_panel")
BASE_PANEL_PATH = PROCESSED_DATA_DIR / "master_dataset_v2.csv"
PROJECT_START = pd.Timestamp("2005-01-01")


@dataclass(frozen=True)
class SourceCoverage:
    source_file: str
    file_modified_utc: str
    observation_start: str
    observation_end: str


@dataclass(frozen=True)
class CanonicalPanelMetadata:
    generated_at_utc: str
    source_panel_file: str
    canonical_panel_file: str
    row_count: int
    region_count: int
    panel_start: str
    panel_end: str
    model_panel_end: str
    historical_only: bool
    split_horizon: bool
    latest_house_price_end: str
    latest_rental_level_end: str
    latest_descriptive_housing_end: str
    weakest_model_input_end: str
    cutoff_reason: str
    source_coverage: dict[str, dict[str, str]]
    max_publication_lag_months: int
    effective_data_as_of: str


@dataclass(frozen=True)
class DescriptivePanelMetadata:
    generated_at_utc: str
    descriptive_panel_file: str
    row_count: int
    region_count: int
    panel_start: str
    panel_end: str
    latest_market_data_end: str
    latest_complete_housing_context_end: str
    latest_house_price_end: str
    latest_rental_level_end: str
    latest_rental_yield_end: str
    descriptive_only: bool
    note: str
    source_coverage: dict[str, dict[str, str]]


def _latest_raw_file(pattern: str) -> Path:
    matches = sorted(RAW_DATA_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No raw file found for pattern {pattern}")
    return matches[-1]


def _coverage_record(path: Path, date_column: str = "date") -> SourceCoverage:
    df = pd.read_csv(path)
    dates = pd.to_datetime(df[date_column], errors="coerce")
    modified = datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    return SourceCoverage(
        source_file=path.name,
        file_modified_utc=modified,
        observation_start=str(dates.min().date()),
        observation_end=str(dates.max().date()),
    )


def _min_coverage_end(coverage: dict[str, dict[str, str]], keys: list[str]) -> str:
    ends = [coverage[key]["observation_end"]
            for key in keys if key in coverage]
    if not ends:
        raise ValueError(
            "No coverage keys provided for weakest-end calculation")
    return min(ends)


def _month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _annual_asof_series(
    monthly_index: pd.DataFrame,
    annual_df: pd.DataFrame,
    value_column: str,
    obs_date_column: str,
    output_value_column: str,
    output_obs_date_column: str,
) -> pd.DataFrame:
    annual = annual_df.copy()
    annual[obs_date_column] = _month_start(annual[obs_date_column])
    annual = annual.sort_values(
        ["region", obs_date_column]).reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for region, month_group in monthly_index.groupby("region", sort=False):
        month_group = month_group.sort_values("date").copy()
        annual_region = annual[annual["region"] ==
                               region][[obs_date_column, value_column]].copy()
        annual_region = annual_region.rename(
            columns={obs_date_column: "obs_date"})
        merged = pd.merge_asof(
            month_group.sort_values("date"),
            annual_region.sort_values("obs_date"),
            left_on="date",
            right_on="obs_date",
            direction="backward",
        )
        merged["region"] = region
        merged[output_value_column] = merged[value_column]
        merged[output_obs_date_column] = merged["obs_date"]
        frames.append(
            merged[["region", "date", output_value_column, output_obs_date_column]])
    return pd.concat(frames, ignore_index=True)


def _months_since_observation(date_series: pd.Series, obs_series: pd.Series) -> pd.Series:
    return (
        (date_series.dt.year - obs_series.dt.year) * 12 +
        (date_series.dt.month - obs_series.dt.month)
    )


def _build_descriptive_panel(
    regions: list[str],
    coverage: dict[str, dict[str, str]],
    output_path: Path = DESCRIPTIVE_PANEL_PATH,
    metadata_path: Path = DESCRIPTIVE_PANEL_METADATA_PATH,
) -> DescriptivePanelMetadata:
    prices = pd.read_csv(_latest_raw_file(
        "land_registry_hpi*.csv"), parse_dates=["date"])
    prices = prices[prices["region"].isin(regions)][["date", "region", "average_price"]].rename(
        columns={"average_price": "descriptive_house_price"}
    )
    rents = pd.read_csv(_latest_raw_file(
        "ons_rental_levels*.csv"), parse_dates=["date"])
    rents = rents[rents["region"].isin(regions)][["date", "region", "median_monthly_rent"]].rename(
        columns={"median_monthly_rent": "descriptive_monthly_rent"}
    )
    yields = pd.read_csv(_latest_raw_file(
        "rental_yield*.csv"), parse_dates=["date"])
    yields = yields[yields["region"].isin(regions)][["date", "region", "gross_yield_pct"]].rename(
        columns={"gross_yield_pct": "descriptive_gross_yield_pct"}
    )

    panel = (
        prices.merge(rents, on=["date", "region"], how="outer")
        .merge(yields, on=["date", "region"], how="outer")
        .loc[lambda df: df["date"] >= PROJECT_START]
        .sort_values(["region", "date"])
        .reset_index(drop=True)
    )
    panel["model_aligned_observation"] = panel["date"].le(
        pd.Timestamp(coverage["rental_yield"]["observation_end"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)

    metadata = DescriptivePanelMetadata(
        generated_at_utc=datetime.now(
            timezone.utc).isoformat(timespec="seconds"),
        descriptive_panel_file=output_path.name,
        row_count=int(len(panel)),
        region_count=int(panel["region"].nunique()),
        panel_start=str(panel["date"].min().date()),
        panel_end=str(panel["date"].max().date()),
        latest_market_data_end=max(
            coverage["house_prices"]["observation_end"],
            coverage["rental_levels"]["observation_end"],
        ),
        latest_complete_housing_context_end=min(
            coverage["house_prices"]["observation_end"],
            coverage["rental_levels"]["observation_end"],
        ),
        latest_house_price_end=coverage["house_prices"]["observation_end"],
        latest_rental_level_end=coverage["rental_levels"]["observation_end"],
        latest_rental_yield_end=coverage["rental_yield"]["observation_end"],
        descriptive_only=True,
        note=(
            "Descriptive market panel only. It extends beyond the harmonised model panel and is not used "
            "for Stage 4 estimation or validated Stage 5/6 calibration."
        ),
        source_coverage=coverage,
    )
    metadata_path.write_text(json.dumps(
        asdict(metadata), indent=2), encoding="utf-8")
    logger.info("Wrote descriptive panel -> %s", output_path)
    logger.info("Wrote descriptive panel metadata -> %s", metadata_path)
    return metadata


def build_canonical_panel(
    source_path: Path = BASE_PANEL_PATH,
    output_path: Path = CANONICAL_PANEL_PATH,
    metadata_path: Path = CANONICAL_PANEL_METADATA_PATH,
) -> CanonicalPanelMetadata:
    """Build the canonical panel with as-of annual features and source coverage metadata."""
    panel = pd.read_csv(source_path, parse_dates=["date"]).sort_values(
        ["region", "date"]).reset_index(drop=True)
    monthly_index = panel[["region", "date"]].copy()

    earnings_path = _latest_raw_file("ons_earnings_regional*.csv")
    earnings_raw = pd.read_csv(earnings_path)
    if "median_annual_earnings" not in earnings_raw.columns:
        raise ValueError(
            "Regional earnings raw file is missing median_annual_earnings")

    earnings_asof = _annual_asof_series(
        monthly_index,
        earnings_raw[["region", "date", "median_annual_earnings"]],
        value_column="median_annual_earnings",
        obs_date_column="date",
        output_value_column="nominal_annual_earnings_asof",
        output_obs_date_column="earnings_obs_date",
    )
    panel = panel.drop(columns=["real_annual_earnings"], errors="ignore").merge(
        earnings_asof,
        on=["region", "date"],
        how="left",
    )

    cpi_base = float(
        panel.loc[panel["date"] == pd.Timestamp("2015-01-01"), "cpi"].iloc[0])
    panel["real_annual_earnings"] = panel["nominal_annual_earnings_asof"] / \
        (panel["cpi"] / cpi_base)
    panel["earnings_obs_date"] = pd.to_datetime(panel["earnings_obs_date"])
    panel["earnings_staleness_months"] = _months_since_observation(
        panel["date"], panel["earnings_obs_date"])
    panel["earnings_anchor_flag"] = panel["date"].dt.month.eq(12)
    panel["earnings_interpolated_flag"] = panel["earnings_staleness_months"].gt(
        0)

    if "real_house_price" in panel.columns:
        panel["price_to_income_ratio_derived"] = panel["real_house_price"] / \
            panel["real_annual_earnings"]
    panel["log_income_asof"] = np.where(panel["real_annual_earnings"] > 0, np.log(
        panel["real_annual_earnings"]), np.nan)
    panel["log_income"] = panel["log_income_asof"]
    panel["log_price_to_income"] = panel["price_to_income_ratio_derived"].where(
        panel["price_to_income_ratio_derived"] > 0
    ).map(lambda value: value if pd.isna(value) else float(np.log(value)))

    grp = panel.groupby("region", sort=False)
    panel["earnings_growth_12m"] = grp["real_annual_earnings"].pct_change(12)
    panel["earnings_growth_12m_lag1"] = grp["earnings_growth_12m"].shift(1)
    panel["log_price_to_income_diff"] = grp["log_price_to_income"].diff()
    panel["official_pti_anchor_flag"] = panel["date"].dt.month.eq(12)
    panel["population_anchor_flag"] = panel["date"].dt.month.eq(6)
    panel["supply_anchor_flag"] = panel["date"].dt.month.eq(4)
    panel["earnings_growth_12m_lag1_is_interpolated"] = panel["earnings_interpolated_flag"]
    panel["net_additions_monthly_is_interpolated"] = ~panel["supply_anchor_flag"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)

    coverage = {
        "house_prices": asdict(_coverage_record(_latest_raw_file("land_registry_hpi*.csv"))),
        "mortgage_rate": asdict(_coverage_record(_latest_raw_file("boe_mortgage_rate*.csv"))),
        "base_rate": asdict(_coverage_record(_latest_raw_file("boe_base_rate*.csv"))),
        "mortgage_approvals": asdict(_coverage_record(_latest_raw_file("boe_mortgage_approvals.csv"))),
        "cpi": asdict(_coverage_record(_latest_raw_file("ons_cpi*.csv"))),
        "earnings": asdict(_coverage_record(earnings_path)),
        "rental_levels": asdict(_coverage_record(_latest_raw_file("ons_rental_levels*.csv"))),
        "rental_yield": asdict(_coverage_record(_latest_raw_file("rental_yield*.csv"))),
        "rental_index": asdict(_coverage_record(_latest_raw_file("ons_rental_index*.csv"))),
        "population": asdict(_coverage_record(_latest_raw_file("ons_population*.csv"))),
        "affordability": asdict(_coverage_record(_latest_raw_file("ons_affordability_ratio*.csv"))),
        "transactions": asdict(_coverage_record(_latest_raw_file("hmrc_transactions*.csv"))),
        "unemployment": asdict(_coverage_record(_latest_raw_file("ons_regional_unemployment*.csv"))),
    }
    latest_house_price_end = coverage["house_prices"]["observation_end"]
    latest_rental_level_end = coverage["rental_levels"]["observation_end"]
    # latest_descriptive_housing_end: latest date at which ANY housing-market
    # context is available for the UI descriptive panel.  Uses max() rather
    # than min() because house prices (published ~8 weeks after reference month)
    # extend well beyond rental levels (ONS PRMS, ~12-month lag) and are the
    # primary market indicator in the descriptive view.
    latest_descriptive_housing_end = max(
        latest_house_price_end, latest_rental_level_end)
    # rental_yield is excluded from weakest-input calculation because it is
    # used only in Stage 6 scoring (yield_score), not in the Stage 4 OLS
    # regression.  It is forward-filled in clean_and_merge for months beyond
    # its last quarterly observation; its staleness is tracked separately in
    # source_coverage but must not cap the OLS estimation panel.
    weakest_model_input_end = _min_coverage_end(
        coverage,
        ["house_prices", "mortgage_approvals", "transactions", "unemployment"],
    )
    _build_descriptive_panel(
        sorted(panel["region"].unique().tolist()), coverage)

    # ── Publication-lag audit ─────────────────────────────────────────────────
    # Map from coverage key to the sidecar series name written by each ingestion
    # script.  Keys without a sidecar (mortgage_approvals, transactions,
    # unemployment) are silently skipped — their lags are not yet tracked.
    _COVERAGE_TO_SIDECAR: dict[str, str] = {
        "base_rate":     "boe_base_rate",
        "mortgage_rate": "boe_mortgage_rate",
        "cpi":           "ons_cpi",
        "earnings":      "ons_earnings_regional",
        "population":    "ons_population",
        "house_prices":  "land_registry_hpi",
        "rental_index":  "ons_rental_index",
        "rental_levels": "ons_rental_levels",
        "rental_yield":  "rental_yield",
        "affordability": "ons_affordability_ratio",
    }
    lags: list[int] = []
    for _cov_key, _series in _COVERAGE_TO_SIDECAR.items():
        _meta = read_release_metadata(_series, RAW_DATA_DIR)
        if _meta is not None:
            lags.append(int(_meta["publication_lag_months_estimated"]))
    max_publication_lag_months: int = max(lags) if lags else 0

    # effective_data_as_of: the latest date at which a real-time analyst could
    # have had all panel series published simultaneously.
    # = panel_end_date − max_publication_lag_months
    _panel_end_ts = pd.Timestamp(str(panel["date"].max().date()))
    _effective_ts = _panel_end_ts - \
        pd.DateOffset(months=max_publication_lag_months)
    effective_data_as_of: str = str(_effective_ts.date())

    metadata = CanonicalPanelMetadata(
        generated_at_utc=datetime.now(
            timezone.utc).isoformat(timespec="seconds"),
        source_panel_file=source_path.name,
        canonical_panel_file=output_path.name,
        row_count=int(len(panel)),
        region_count=int(panel["region"].nunique()),
        panel_start=str(panel["date"].min().date()),
        panel_end=str(panel["date"].max().date()),
        model_panel_end=str(panel["date"].max().date()),
        historical_only=True,
        split_horizon=True,
        latest_house_price_end=latest_house_price_end,
        latest_rental_level_end=latest_rental_level_end,
        latest_descriptive_housing_end=latest_descriptive_housing_end,
        weakest_model_input_end=weakest_model_input_end,
        cutoff_reason=(
            "The harmonised model panel stops at the weakest critical input rather than the freshest single source. "
            f"Current model support ends {weakest_model_input_end}; descriptive house prices run to {latest_house_price_end} "
            f"and descriptive rents to {latest_rental_level_end}."
        ),
        source_coverage=coverage,
        max_publication_lag_months=max_publication_lag_months,
        effective_data_as_of=effective_data_as_of,
    )
    ensure_directory(METADATA_OUTPUT_DIR)
    metadata_path.write_text(json.dumps(
        asdict(metadata), indent=2), encoding="utf-8")
    logger.info("Wrote canonical panel -> %s", output_path)
    logger.info("Wrote canonical panel metadata -> %s", metadata_path)
    return metadata


def main() -> None:
    metadata = build_canonical_panel()
    print(json.dumps(asdict(metadata), indent=2))


if __name__ == "__main__":
    main()
