"""Validation helpers for canonical model inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class DatasetValidationError(ValueError):
    """Raised when a required dataset fails validation."""


@dataclass(frozen=True)
class ValidationResult:
    dataset_name: str
    n_rows: int
    n_columns: int


def ensure_columns(df: pd.DataFrame, dataset_name: str, required: list[str]) -> ValidationResult:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DatasetValidationError(
            f"{dataset_name} is missing required columns: {missing}"
        )
    return ValidationResult(dataset_name=dataset_name, n_rows=len(df), n_columns=len(df.columns))


def ensure_unique_key(df: pd.DataFrame, dataset_name: str, key_columns: list[str]) -> None:
    duplicates = df.duplicated(subset=key_columns).sum()
    if duplicates:
        raise DatasetValidationError(
            f"{dataset_name} contains {duplicates} duplicate rows on key {key_columns}"
        )


def ensure_expected_regions(df: pd.DataFrame, dataset_name: str, expected_regions: list[str]) -> None:
    present = sorted(df["region"].dropna().unique().tolist())
    missing = sorted(set(expected_regions) - set(present))
    unexpected = sorted(set(present) - set(expected_regions))
    if missing or unexpected:
        raise DatasetValidationError(
            f"{dataset_name} region mismatch. Missing={missing} Unexpected={unexpected}"
        )


def ensure_bounds(
    df: pd.DataFrame,
    dataset_name: str,
    column: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    series = pd.to_numeric(df[column], errors="coerce")
    if lower is not None and (series < lower).any():
        raise DatasetValidationError(f"{dataset_name}.{column} has values below {lower}")
    if upper is not None and (series > upper).any():
        raise DatasetValidationError(f"{dataset_name}.{column} has values above {upper}")


def validate_master_dataset(df: pd.DataFrame, expected_regions: list[str]) -> ValidationResult:
    result = ensure_columns(
        df,
        "master_dataset",
        [
            "date",
            "region",
            "nominal_house_price",
            "real_house_price",
            "mortgage_rate",
            "base_rate",
            "real_annual_earnings",
            "gross_yield_pct",
        ],
    )
    ensure_unique_key(df, "master_dataset", ["date", "region"])
    ensure_expected_regions(df, "master_dataset", expected_regions)
    ensure_bounds(df, "master_dataset", "nominal_house_price", lower=1.0)
    ensure_bounds(df, "master_dataset", "mortgage_rate", lower=0.0, upper=20.0)
    if "earnings_obs_date" in df.columns:
        obs_date = pd.to_datetime(df["earnings_obs_date"], errors="coerce")
        date = pd.to_datetime(df["date"], errors="coerce")
        if (obs_date > date).any():
            raise DatasetValidationError("master_dataset.earnings_obs_date contains future leakage relative to panel date")
    if "earnings_staleness_months" in df.columns:
        ensure_bounds(df, "master_dataset", "earnings_staleness_months", lower=0.0, upper=24.0)
    return result


def validate_stage4_parameters(df: pd.DataFrame, expected_regions: list[str]) -> ValidationResult:
    result = ensure_columns(
        df,
        "stage4_parameters",
        ["region", "kappa", "sigma", "mu_equilibrium", "gamma_annual_pp"],
    )
    ensure_unique_key(df, "stage4_parameters", ["region"])
    ensure_expected_regions(df, "stage4_parameters", expected_regions)
    ensure_bounds(df, "stage4_parameters", "kappa", lower=0.0, upper=2.0)
    ensure_bounds(df, "stage4_parameters", "sigma", lower=0.0, upper=1.0)
    return result


def validate_stage5_summary(df: pd.DataFrame, expected_regions: list[str]) -> ValidationResult:
    result = ensure_columns(
        df,
        "stage5_summary",
        [
            "scenario",
            "region",
            "median_5yr_growth",
            "p10_5yr_growth",
            "p90_5yr_growth",
            "prob_terminal_loss_10pct",
        ],
    )
    ensure_unique_key(df, "stage5_summary", ["scenario", "region"])
    ensure_expected_regions(df, "stage5_summary", expected_regions)
    ensure_bounds(df, "stage5_summary", "prob_terminal_loss_10pct", lower=0.0, upper=1.0)
    return result


def validate_stage6_handoff(df: pd.DataFrame, expected_regions: list[str]) -> ValidationResult:
    result = ensure_columns(
        df,
        "stage6_handoff",
        [
            "region",
            "consumer_score",
            "consumer_band",
            "reit_score",
            "reit_band",
            "pct_above_pstar",
            "wtd_return_consumer",
            "p_terminal_loss_10_avg",
        ],
    )
    ensure_unique_key(df, "stage6_handoff", ["region"])
    ensure_expected_regions(df, "stage6_handoff", expected_regions)
    ensure_bounds(df, "stage6_handoff", "consumer_score", lower=0.0, upper=100.0)
    ensure_bounds(df, "stage6_handoff", "reit_score", lower=0.0, upper=100.0)
    ensure_bounds(df, "stage6_handoff", "p_terminal_loss_10_avg", lower=0.0, upper=1.0)
    return result
