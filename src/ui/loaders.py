"""Canonical, validated data access for the Stage 7 dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.core.config import load_config
from src.core.paths import (
    CANONICAL_PANEL_METADATA_PATH,
    CANONICAL_PANEL_PATH,
    DESCRIPTIVE_PANEL_METADATA_PATH,
    DESCRIPTIVE_PANEL_PATH,
    STAGE4_OUTPUT_DIR,
    STAGE5_OUTPUT_DIR,
    STAGE6_OUTPUT_DIR,
    STAGE7_OUTPUT_DIR,
    ensure_directory,
)
from src.core.validation import validate_master_dataset, validate_stage4_parameters, validate_stage5_summary, validate_stage6_handoff
from src.data.build_canonical_panel import build_canonical_panel

# ── Dashboard data bundle (git-tracked, used on Streamlit Cloud) ──────────
_DASHBOARD_DATA_DIR: Path = Path(__file__).resolve().parents[2] / "dashboard_data"


def _resolve(filename: str, original: Path) -> Path:
    """Return dashboard_data/<filename> if it exists, else original path.

    This allows Streamlit Cloud (where data/outputs/ is git-ignored) to read
    from the committed dashboard_data/ snapshot, while local development
    continues to use the live data/outputs/ paths.
    """
    candidate = _DASHBOARD_DATA_DIR / filename
    return candidate if candidate.exists() else original


@dataclass(frozen=True)
class DashboardData:
    master: pd.DataFrame
    fair_panel: pd.DataFrame
    latest: pd.DataFrame
    descriptive_panel: pd.DataFrame
    latest_descriptive: pd.DataFrame
    params: pd.DataFrame
    simulation: pd.DataFrame
    distribution_compare: pd.DataFrame
    moment_compare: pd.DataFrame
    handoff: pd.DataFrame
    region_table: pd.DataFrame
    metadata: dict


def _read_csv(path: Path, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)


def _safe_float(value) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if pd.notna(numeric) else None


def _load_validation_summary(validation_dir: Path) -> dict:
    summary: dict[str, object] = {}

    subsample_path = validation_dir / "subsample_ols_validation.json"
    if subsample_path.exists():
        subsample = json.loads(subsample_path.read_text(encoding="utf-8"))
        summary["subsample_holdout"] = {
            "train_end": subsample.get("train_end"),
            "holdout_start": subsample.get("holdout_start"),
            "holdout_n_obs": int(subsample.get("holdout_n_obs", 0) or 0),
            "holdout_rmse": _safe_float(subsample.get("holdout_rmse")),
            "zero_growth_rmse": _safe_float(subsample.get("zero_growth_rmse")),
            "beats_zero_growth": bool(subsample.get("beats_zero_growth", False)),
            "directional_1m": _safe_float(subsample.get("directional_1m")),
            "directional_6m": _safe_float(subsample.get("directional_6m")),
            "directional_12m": _safe_float(subsample.get("directional_12m")),
            "generated_at_utc": subsample.get("generated_at_utc"),
        }

    overall_path = validation_dir / "forecast_benchmark_overall.csv"
    if overall_path.exists():
        overall = pd.read_csv(overall_path)
        benchmark_summary: dict[str, dict[str, float | int | None]] = {}
        for row in overall.to_dict(orient="records"):
            benchmark = str(row.get("benchmark", "unknown"))
            benchmark_summary[benchmark] = {
                "rmse": _safe_float(row.get("rmse")),
                "mae": _safe_float(row.get("mae")),
                "n_forecasts": int(row.get("n_forecasts", 0) or 0),
            }
        summary["forecast_benchmarks"] = benchmark_summary

    rate_stability_path = validation_dir / "regional_rate_stability.csv"
    if rate_stability_path.exists():
        rate_stability = pd.read_csv(rate_stability_path)
        flagged = int(rate_stability.get("flagged_unstable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        summary["rate_stability"] = {
            "regions_tested": int(len(rate_stability)),
            "flagged_unstable_regions": flagged,
            "all_regions_stable": flagged == 0 and len(rate_stability) > 0,
        }

    return summary


def _build_metadata(
    master_path: Path,
    params_path: Path,
    simulation_path: Path,
    handoff_path: Path,
    config_regions: list[str],
    simulation: pd.DataFrame,
    master: pd.DataFrame,
    panel_metadata: dict,
    descriptive_metadata: dict,
    validation_summary: dict,
) -> dict:
    run_timestamp = max(
        master_path.stat().st_mtime,
        params_path.stat().st_mtime,
        simulation_path.stat().st_mtime,
        handoff_path.stat().st_mtime,
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latest_model_run_utc": datetime.fromtimestamp(run_timestamp, tz=timezone.utc).isoformat(timespec="seconds"),
        "master_dataset_file": master_path.name,
        "descriptive_panel_file": DESCRIPTIVE_PANEL_PATH.name,
        "stage4_parameters_file": params_path.name,
        "stage5_summary_file": simulation_path.name,
        "stage6_handoff_file": handoff_path.name,
        "historical_vs_simulated_distribution_file": "historical_vs_simulated_distribution.csv",
        "simulation_moment_comparison_file": "simulation_moment_comparison.csv",
        "data_coverage_start": str(master["date"].min().date()),
        "data_coverage_end": str(master["date"].max().date()),
        "model_panel_coverage_end": str(master["date"].max().date()),
        "descriptive_panel_coverage_end": descriptive_metadata.get("panel_end"),
        "latest_market_data_end": descriptive_metadata.get("latest_market_data_end"),
        "latest_complete_housing_context_end": descriptive_metadata.get("latest_complete_housing_context_end"),
        "historical_only": bool(panel_metadata.get("historical_only", True)),
        "split_horizon": bool(panel_metadata.get("split_horizon", False)),
        "effective_data_as_of": panel_metadata.get("effective_data_as_of", str(master["date"].max().date())),
        "max_publication_lag_months": panel_metadata.get("max_publication_lag_months"),
        "latest_house_price_end": panel_metadata.get("latest_house_price_end"),
        "latest_rental_level_end": panel_metadata.get("latest_rental_level_end"),
        "latest_descriptive_housing_end": panel_metadata.get("latest_descriptive_housing_end"),
        "weakest_model_input_end": panel_metadata.get("weakest_model_input_end", str(master["date"].max().date())),
        "cutoff_reason": panel_metadata.get("cutoff_reason", ""),
        "source_coverage": panel_metadata.get("source_coverage", {}),
        "descriptive_panel_note": descriptive_metadata.get("note", ""),
        "validation_summary": validation_summary,
        "regions_covered": config_regions,
        "region_count": len(config_regions),
        "scenario_count": int(simulation["scenario"].nunique()),
        "scenario_names": load_config().ui.scenario_labels,
    }

def _latest_descriptive_context(descriptive_panel: pd.DataFrame, config_regions: list[str]) -> pd.DataFrame:
    panel = descriptive_panel[descriptive_panel["region"].isin(config_regions)].sort_values(["region", "date"])
    price_latest = (
        panel.dropna(subset=["descriptive_house_price"])
        .groupby("region", as_index=False)
        .tail(1)[["region", "date", "descriptive_house_price"]]
        .rename(columns={"date": "latest_house_price_date", "descriptive_house_price": "latest_house_price"})
    )
    rent_latest = (
        panel.dropna(subset=["descriptive_monthly_rent"])
        .groupby("region", as_index=False)
        .tail(1)[["region", "date", "descriptive_monthly_rent"]]
        .rename(columns={"date": "latest_rent_date", "descriptive_monthly_rent": "latest_monthly_rent_raw"})
    )
    yield_latest = (
        panel.dropna(subset=["descriptive_gross_yield_pct"])
        .groupby("region", as_index=False)
        .tail(1)[["region", "date", "descriptive_gross_yield_pct"]]
        .rename(columns={"date": "latest_yield_date", "descriptive_gross_yield_pct": "latest_gross_yield_pct"})
    )
    return (
        price_latest.merge(rent_latest, on="region", how="outer")
        .merge(yield_latest, on="region", how="outer")
        .sort_values("region")
        .reset_index(drop=True)
    )


def _percentile_rank(series: pd.Series, value: float) -> float:
    clean = series.dropna().astype(float)
    if clean.empty:
        return float("nan")
    return float((clean <= float(value)).mean() * 100.0)


def _historical_signal_context(fair_panel: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for region, group in fair_panel.groupby("region"):
        grp = group.sort_values("date").reset_index(drop=True)
        latest_row = grp.iloc[-1]
        valuation_pct = _percentile_rank(grp["fair_value_gap_log"], float(latest_row["fair_value_gap_log"]))
        if "transaction_growth" in grp.columns:
            cycle_pct = _percentile_rank(grp["transaction_growth"], float(latest_row.get("transaction_growth", 0.0)))
        else:
            cycle_pct = float("nan")
        rows.append(
            {
                "region": region,
                "valuation_stretch_percentile": valuation_pct,
                "cycle_strength_percentile": cycle_pct,
            }
        )
    context = pd.DataFrame(rows)

    current_latest = latest[["region", "gross_yield_pct"]].copy()
    current_latest["rent_support_percentile"] = current_latest["gross_yield_pct"].rank(method="average", pct=True) * 100.0
    context = context.merge(current_latest[["region", "rent_support_percentile"]], on="region", how="left")
    return context


def write_dashboard_cache(data: DashboardData) -> dict[str, str]:
    """Write app-ready cache files for reproducible Stage 7 delivery."""
    ensure_directory(STAGE7_OUTPUT_DIR)
    region_snapshot_path = STAGE7_OUTPUT_DIR / "region_snapshot.csv"
    metadata_path = STAGE7_OUTPUT_DIR / "dashboard_metadata.json"

    data.region_table.to_csv(region_snapshot_path, index=False)
    metadata_path.write_text(json.dumps(data.metadata, indent=2), encoding="utf-8")
    return {
        "region_snapshot": str(region_snapshot_path),
        "dashboard_metadata": str(metadata_path),
    }


@lru_cache(maxsize=1)
def load_dashboard_data() -> DashboardData:
    """Load the validated datasets required by the dashboard."""
    config = load_config()
    master_path      = _resolve("master_dataset_canonical.csv",    CANONICAL_PANEL_PATH)
    _meta_path       = _resolve("canonical_panel_metadata.json",   CANONICAL_PANEL_METADATA_PATH)
    _desc_path       = _resolve("descriptive_market_panel.csv",    DESCRIPTIVE_PANEL_PATH)
    _desc_meta_path  = _resolve("descriptive_panel_metadata.json", DESCRIPTIVE_PANEL_METADATA_PATH)
    if not master_path.exists() or not _meta_path.exists():
        build_canonical_panel()
        master_path = CANONICAL_PANEL_PATH
        _meta_path  = CANONICAL_PANEL_METADATA_PATH
    if not _desc_path.exists() or not _desc_meta_path.exists():
        build_canonical_panel()
        _desc_path      = DESCRIPTIVE_PANEL_PATH
        _desc_meta_path = DESCRIPTIVE_PANEL_METADATA_PATH
    params_path     = _resolve(config.artifacts.stage4_parameters_file, STAGE4_OUTPUT_DIR / config.artifacts.stage4_parameters_file)
    simulation_path = _resolve(config.artifacts.stage5_summary_file,    STAGE5_OUTPUT_DIR / config.artifacts.stage5_summary_file)
    handoff_path    = _resolve(config.artifacts.stage6_handoff_file,     STAGE6_OUTPUT_DIR / config.artifacts.stage6_handoff_file)
    fair_panel_path = _resolve("fair_value_panel_bankgrade.csv",         STAGE4_OUTPUT_DIR / "fair_value_panel_bankgrade.csv")
    _std_validation_dir = STAGE7_OUTPUT_DIR.parent / "validation"
    validation_dir  = _DASHBOARD_DATA_DIR if _DASHBOARD_DATA_DIR.exists() else _std_validation_dir
    distribution_compare_path = _resolve("historical_vs_simulated_distribution.csv", _std_validation_dir / "historical_vs_simulated_distribution.csv")
    moment_compare_path       = _resolve("simulation_moment_comparison.csv",          _std_validation_dir / "simulation_moment_comparison.csv")

    master = _read_csv(master_path, parse_dates=["date"]).sort_values(["region", "date"]).reset_index(drop=True)
    fair_panel = _read_csv(fair_panel_path, parse_dates=["date"]).sort_values(["region", "date"]).reset_index(drop=True)
    descriptive_panel = _read_csv(_desc_path, parse_dates=["date"]).sort_values(["region", "date"]).reset_index(drop=True)
    params = _read_csv(params_path)
    simulation = _read_csv(simulation_path)
    distribution_compare = _read_csv(distribution_compare_path)
    moment_compare = _read_csv(moment_compare_path)
    handoff = _read_csv(handoff_path)
    panel_metadata = json.loads(_meta_path.read_text(encoding="utf-8"))
    descriptive_metadata = json.loads(_desc_meta_path.read_text(encoding="utf-8"))
    validation_summary = _load_validation_summary(validation_dir)
    latest_descriptive = _latest_descriptive_context(descriptive_panel, config.project.regions)

    validate_master_dataset(master, config.project.regions)
    validate_stage4_parameters(params, config.project.regions)
    validate_stage5_summary(simulation, config.project.regions)
    validate_stage6_handoff(handoff, config.project.regions)

    latest = (
        master.sort_values(["region", "date"])
        .groupby("region", as_index=False)
        .tail(1)
        .sort_values("region")
        .reset_index(drop=True)
    )
    signal_context = _historical_signal_context(fair_panel, latest)

    hist_stats = None
    if "price_growth" in master.columns:
        hist_stats = (
            master.groupby("region", as_index=False)
            .agg(
                hist_price_growth_mean_monthly=("price_growth", "mean"),
                hist_price_growth_vol_monthly=("price_growth", "std"),
            )
        )
        hist_stats["hist_price_growth_mean_annual"] = (
            hist_stats["hist_price_growth_mean_monthly"] * 12.0
        )
        hist_stats["hist_price_growth_vol_annual"] = (
            hist_stats["hist_price_growth_vol_monthly"] * (12.0 ** 0.5)
        )

    params_keep = ["region", "sigma", "mu_equilibrium", "last_updated", "final_spec_note", "earnings_staleness_months"]
    if "sigma_frequency" in params.columns:
        params_keep.append("sigma_frequency")
    params_subset = params[params_keep].copy()
    latest_keep = [
        "region",
        "date",
        "nominal_house_price",
        "real_house_price",
        "mortgage_rate",
        "base_rate",
        "real_annual_earnings",
        "gross_yield_pct",
        "real_monthly_rent",
    ]
    for optional_col in [
        "unemployment_lag1",
        "unemployment_rate",
        "approvals_growth",
        "transaction_growth",
        "financial_stress_lag3",
        "payment_burden",
        "payment_burden_gap",
        "affordability_gap",
        "price_to_rent",
        "lender_spread_lag3",
    ]:
        if optional_col in latest.columns:
            latest_keep.append(optional_col)

    region_table = (
        handoff.merge(
            latest[latest_keep],
            on="region",
            how="left",
        )
        .merge(latest_descriptive, on="region", how="left")
        .merge(params_subset, on="region", how="left")
        .merge(signal_context, on="region", how="left")
        .sort_values("region")
        .reset_index(drop=True)
    )
    if "latest_house_price" in region_table.columns:
        region_table["house_price_gap_vs_model_pct"] = (
            region_table["latest_house_price"] / region_table["nominal_house_price"] - 1.0
        ) * 100.0
    if hist_stats is not None:
        region_table = region_table.merge(hist_stats, on="region", how="left")
    if "p_terminal_loss_10_avg" in region_table.columns:
        region_table["downside_vulnerability_percentile"] = (
            region_table["p_terminal_loss_10_avg"].rank(method="average", pct=True) * 100.0
        )

    metadata = _build_metadata(
        master_path,
        params_path,
        simulation_path,
        handoff_path,
        config.project.regions,
        simulation,
        master,
        panel_metadata,
        descriptive_metadata,
        validation_summary,
    )

    data = DashboardData(
        master=master,
        fair_panel=fair_panel,
        latest=latest,
        descriptive_panel=descriptive_panel,
        latest_descriptive=latest_descriptive,
        params=params,
        simulation=simulation,
        distribution_compare=distribution_compare,
        moment_compare=moment_compare,
        handoff=handoff,
        region_table=region_table,
        metadata=metadata,
    )
    write_dashboard_cache(data)
    return data


def get_region_record(region: str) -> dict:
    data = load_dashboard_data()
    row = data.region_table[data.region_table["region"] == region]
    if row.empty:
        raise KeyError(f"Region not found: {region}")
    return row.iloc[0].to_dict()


def get_region_history(region: str, months: int = 120) -> pd.DataFrame:
    data = load_dashboard_data()
    history = data.master[data.master["region"] == region].sort_values("date")
    return history.tail(months).reset_index(drop=True)


def get_region_simulation_summary(region: str) -> pd.DataFrame:
    data = load_dashboard_data()
    return (
        data.simulation[data.simulation["region"] == region]
        .sort_values("median_5yr_growth")
        .reset_index(drop=True)
    )


def get_region_distribution_comparison(region: str) -> pd.DataFrame:
    data = load_dashboard_data()
    return data.distribution_compare[data.distribution_compare["region"] == region].copy().reset_index(drop=True)


def get_region_moment_comparison(region: str) -> dict:
    data = load_dashboard_data()
    matched = data.moment_compare[data.moment_compare["region"] == region]
    if matched.empty:
        raise KeyError(f"Region not found in simulation moment comparison: {region}")
    return matched.iloc[0].to_dict()
