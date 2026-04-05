"""Independent artifact-driven validation for the canonical housing model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import load_config
from src.core.paths import (
    CANONICAL_PANEL_METADATA_PATH,
    CANONICAL_PANEL_PATH,
    STAGE4_OUTPUT_DIR,
    STAGE5_OUTPUT_DIR,
    STAGE6_OUTPUT_DIR,
    STAGE7_OUTPUT_DIR,
)
from src.core.config import load_config as _load_config_for_replay
from src.model.canonical_rebuild import RATE_FEATURE, run_subsample_ols_validation
from src.model.ols import fit_structural_fair_value


VALIDATION_DIR = STAGE7_OUTPUT_DIR.parent / "validation"
ARTIFACT_LINEAGE_PATH = VALIDATION_DIR / "artifact_lineage_checks.csv"
APP_CONSISTENCY_PATH = VALIDATION_DIR / "app_artifact_consistency.csv"
SIMULATION_REPLAY_PATH = VALIDATION_DIR / "simulation_plausibility_replay.csv"
SCORE_BUCKET_PATH = VALIDATION_DIR / "score_bucket_performance.csv"
SCORE_CALIBRATION_PATH = VALIDATION_DIR / "score_calibration_diagnostics.csv"
COMPONENT_SIGNAL_STRENGTH_PATH = VALIDATION_DIR / "component_signal_strength.csv"
HISTORICAL_VS_SIMULATED_DISTRIBUTION_PATH = VALIDATION_DIR / \
    "historical_vs_simulated_distribution.csv"
SIMULATION_MOMENT_COMPARISON_PATH = VALIDATION_DIR / \
    "simulation_moment_comparison.csv"
INDEPENDENT_SUMMARY_PATH = VALIDATION_DIR / "independent_validation_summary.md"


@dataclass(frozen=True)
class IndependentValidationResult:
    generated_at_utc: str
    artifact_lineage_path: str
    app_consistency_path: str
    simulation_replay_path: str
    score_bucket_path: str
    score_calibration_path: str
    component_signal_strength_path: str
    historical_vs_simulated_distribution_path: str
    simulation_moment_comparison_path: str
    summary_path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_lineage_checks() -> pd.DataFrame:
    config = load_config()
    panel_metadata = json.loads(
        CANONICAL_PANEL_METADATA_PATH.read_text(encoding="utf-8"))
    artifacts = [
        ("canonical_panel", CANONICAL_PANEL_PATH),
        ("canonical_panel_metadata", CANONICAL_PANEL_METADATA_PATH),
        ("stage4_parameters", STAGE4_OUTPUT_DIR /
         config.artifacts.stage4_parameters_file),
        ("stage4_fair_value_panel", STAGE4_OUTPUT_DIR /
         "fair_value_panel_bankgrade.csv"),
        ("stage5_summary", STAGE5_OUTPUT_DIR /
         config.artifacts.stage5_summary_file),
        ("stage6_handoff", STAGE6_OUTPUT_DIR /
         config.artifacts.stage6_handoff_file),
        ("stage7_snapshot", STAGE7_OUTPUT_DIR /
         config.artifacts.stage7_snapshot_file),
        ("stage7_metadata", STAGE7_OUTPUT_DIR /
         config.artifacts.stage7_metadata_file),
    ]
    rows = []
    for label, path in artifacts:
        exists = path.exists()
        row_count = None
        if exists and path.suffix == ".csv":
            row_count = int(len(pd.read_csv(path)))
        rows.append(
            {
                "artifact": label,
                "path": str(path),
                "exists": exists,
                "sha256": _sha256(path) if exists else "",
                "row_count": row_count,
            }
        )

    rows.append(
        {
            "artifact": "historical_only_metadata",
            "path": str(CANONICAL_PANEL_METADATA_PATH),
            "exists": True,
            "sha256": "",
            "row_count": int(bool(panel_metadata.get("historical_only", False))),
        }
    )
    lineage = pd.DataFrame(rows)
    lineage.to_csv(ARTIFACT_LINEAGE_PATH, index=False)
    return lineage


def _replay_simulation_plausibility(master: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, group in master.groupby("region"):
        grp = group.sort_values("date").reset_index(drop=True)
        forward_returns = np.asarray(
            [
                (float(grp.loc[index + 60, "nominal_house_price"]) /
                 float(grp.loc[index, "nominal_house_price"]) - 1.0) * 100.0
                for index in range(len(grp) - 60)
            ],
            dtype=float,
        )
        baseline = summary[(summary["region"] == region) & (
            summary["scenario"] == "Baseline")].iloc[0]
        hist_loss = float(np.mean(forward_returns <= -10.0))
        rows.append(
            {
                "region": region,
                "historical_5y_p10": float(np.percentile(forward_returns, 10)),
                "historical_5y_p50": float(np.percentile(forward_returns, 50)),
                "historical_5y_p90": float(np.percentile(forward_returns, 90)),
                "historical_prob_loss_10pct": hist_loss,
                "baseline_sim_median_5y": float(baseline["median_5yr_growth"]),
                "baseline_sim_prob_loss_10pct": float(baseline["prob_terminal_loss_10pct"]),
                "median_minus_hist_p50": float(baseline["median_5yr_growth"] - np.percentile(forward_returns, 50)),
                "tail_risk_gap": float(baseline["prob_terminal_loss_10pct"] - hist_loss),
                "within_hist_10_90_band": bool(
                    np.percentile(forward_returns, 10) <=
                    float(baseline["median_5yr_growth"]) <=
                    np.percentile(forward_returns, 90)
                ),
            }
        )
    replay = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    replay.to_csv(SIMULATION_REPLAY_PATH, index=False)
    return replay


def _historical_vs_simulated_distribution(master: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    distribution_rows = []
    moment_rows = []
    region_moments = []

    for region, group in master.groupby("region"):
        grp = group.sort_values("date").reset_index(drop=True)
        forward_returns = np.asarray(
            [
                (float(grp.loc[index + 60, "nominal_house_price"]) /
                 float(grp.loc[index, "nominal_house_price"]) - 1.0) * 100.0
                for index in range(len(grp) - 60)
            ],
            dtype=float,
        )
        baseline = summary[(summary["region"] == region) & (
            summary["scenario"] == "Baseline")].iloc[0]
        simulated = {
            "p10": float(baseline["p10_5yr_growth"]),
            "p25": float(baseline["p25_5yr_growth"]),
            "p50": float(baseline["median_5yr_growth"]),
            "p75": float(baseline["p75_5yr_growth"]),
            "p90": float(baseline["p90_5yr_growth"]),
        }
        historical = {
            "p10": float(np.percentile(forward_returns, 10)),
            "p25": float(np.percentile(forward_returns, 25)),
            "p50": float(np.percentile(forward_returns, 50)),
            "p75": float(np.percentile(forward_returns, 75)),
            "p90": float(np.percentile(forward_returns, 90)),
        }
        for percentile in ["p10", "p25", "p50", "p75", "p90"]:
            distribution_rows.append(
                {
                    "region": region,
                    "horizon_months": 60,
                    "percentile": percentile,
                    "historical_return_pct": historical[percentile],
                    "simulated_return_pct": simulated[percentile],
                    "simulated_minus_historical_pct": simulated[percentile] - historical[percentile],
                }
            )
        row = {
            "region": region,
            "horizon_months": 60,
            "historical_mean_pct": float(np.mean(forward_returns)),
            "historical_median_pct": historical["p50"],
            "historical_p10_pct": historical["p10"],
            "historical_p90_pct": historical["p90"],
            "historical_prob_loss_10pct": float(np.mean(forward_returns <= -10.0)),
            "simulated_median_pct": simulated["p50"],
            "simulated_p10_pct": simulated["p10"],
            "simulated_p90_pct": simulated["p90"],
            "simulated_prob_loss_10pct": float(baseline["prob_terminal_loss_10pct"]),
            "within_hist_10_90_band": bool(historical["p10"] <= simulated["p50"] <= historical["p90"]),
            "tail_risk_gap": float(baseline["prob_terminal_loss_10pct"] - np.mean(forward_returns <= -10.0)),
        }
        moment_rows.append(row)
        region_moments.append(row)

    pooled = pd.DataFrame(region_moments)
    pooled_row = {
        "region": "UK_12_region_average",
        "horizon_months": 60,
        "historical_mean_pct": float(pooled["historical_mean_pct"].mean()),
        "historical_median_pct": float(pooled["historical_median_pct"].mean()),
        "historical_p10_pct": float(pooled["historical_p10_pct"].mean()),
        "historical_p90_pct": float(pooled["historical_p90_pct"].mean()),
        "historical_prob_loss_10pct": float(pooled["historical_prob_loss_10pct"].mean()),
        "simulated_median_pct": float(pooled["simulated_median_pct"].mean()),
        "simulated_p10_pct": float(pooled["simulated_p10_pct"].mean()),
        "simulated_p90_pct": float(pooled["simulated_p90_pct"].mean()),
        "simulated_prob_loss_10pct": float(pooled["simulated_prob_loss_10pct"].mean()),
        "within_hist_10_90_band": float(pooled["within_hist_10_90_band"].mean()) >= 0.5,
        "tail_risk_gap": float(pooled["tail_risk_gap"].mean()),
    }
    moment_rows.append(pooled_row)
    for percentile in ["p10", "p25", "p50", "p75", "p90"]:
        if percentile in {"p25", "p75"}:
            historical_value = np.nan
            simulated_value = np.nan
        else:
            historical_value = pooled_row[f"historical_{percentile}_pct"] if percentile != "p50" else pooled_row["historical_median_pct"]
            simulated_value = pooled_row[f"simulated_{percentile}_pct"] if percentile != "p50" else pooled_row["simulated_median_pct"]
        distribution_rows.append(
            {
                "region": "UK_12_region_average",
                "horizon_months": 60,
                "percentile": percentile,
                "historical_return_pct": historical_value,
                "simulated_return_pct": simulated_value,
                "simulated_minus_historical_pct": simulated_value - historical_value if np.isfinite(historical_value) and np.isfinite(simulated_value) else np.nan,
            }
        )

    distribution = pd.DataFrame(distribution_rows).sort_values(
        ["region", "percentile"]).reset_index(drop=True)
    moments = pd.DataFrame(moment_rows).sort_values(
        "region").reset_index(drop=True)
    distribution.to_csv(HISTORICAL_VS_SIMULATED_DISTRIBUTION_PATH, index=False)
    moments.to_csv(SIMULATION_MOMENT_COMPARISON_PATH, index=False)
    return distribution, moments


def _valuation_signal(fair_value_gap_log: pd.Series) -> pd.Series:
    return (50.0 - 90.0 * fair_value_gap_log).clip(lower=0.0, upper=100.0)


def _rate_signal(rate_level: pd.Series) -> pd.Series:
    """
    Map lagged mortgage-rate levels to a simple 0-100 support score.

    The canonical rate feature is now a lagged mortgage-rate level rather than a
    small gap/change term, so the score must be anchored around a neutral level
    rather than scaled as a basis-point shock.
    """
    neutral_rate = 4.5
    return (50.0 - 12.0 * (rate_level - neutral_rate)).clip(lower=0.0, upper=100.0)


def _transaction_signal(transaction_growth: pd.Series) -> pd.Series:
    return (50.0 + 220.0 * transaction_growth).clip(lower=0.0, upper=100.0)


def _stress_overlay_signal(stress_excess: pd.Series) -> pd.Series:
    return (100.0 - 7.5 * stress_excess).clip(lower=0.0, upper=100.0)


def _score_bucket_performance(fair_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_config()
    panel = fair_panel.sort_values(["region", "date"]).copy()
    panel["valuation_signal"] = _valuation_signal(panel["fair_value_gap_log"])
    panel["rate_repricing_signal"] = _rate_signal(panel[RATE_FEATURE])
    panel["transaction_signal"] = _transaction_signal(
        panel["transaction_growth"])
    if "financial_stress_excess_lag3" in panel.columns:
        panel["stress_overlay_signal"] = _stress_overlay_signal(
            panel["financial_stress_excess_lag3"])
    else:
        panel["stress_overlay_signal"] = _stress_overlay_signal(
            panel["financial_stress_lag3"].sub(10.5).clip(lower=0.0))
    panel["valuation_plus_rate_signal"] = (
        0.80 * panel["valuation_signal"] +
        0.20 * panel["rate_repricing_signal"]
    ).clip(lower=0.0, upper=100.0)
    panel["historical_macro_blend_signal"] = (
        0.70 * panel["valuation_signal"] +
        0.15 * panel["rate_repricing_signal"] +
        0.15 * panel["transaction_signal"]
    ).clip(lower=0.0, upper=100.0)
    signal_map = {
        "valuation_signal": "core_empirical",
        "rate_repricing_signal": "auxiliary_macro",
        "transaction_signal": "auxiliary_macro",
        "stress_overlay_signal": "episodic_overlay",
        "valuation_plus_rate_signal": "subset_composite",
        "historical_macro_blend_signal": "proxy_full_composite",
    }

    rows = []
    diagnostics = []
    strength_rows = []
    for horizon in config.validation.score_backtest_horizons_months:
        returns = []
        for region, group in panel.groupby("region"):
            grp = group.sort_values("date").copy()
            grp[f"forward_return_{horizon}m"] = grp["nominal_house_price"].shift(
                -horizon) / grp["nominal_house_price"] - 1.0
            returns.append(grp)
        horizon_panel = pd.concat(returns, ignore_index=True)
        for signal_name, signal_role in signal_map.items():
            subset = horizon_panel.dropna(
                subset=[f"forward_return_{horizon}m", signal_name]).copy()
            subset["signal_bucket"] = subset[signal_name].map(
                lambda score: "Supportive" if score >= config.scoring.thresholds.supportive else "Mixed" if score >= config.scoring.thresholds.mixed else "Cautious"
            )
            for bucket, bucket_df in subset.groupby("signal_bucket"):
                rows.append(
                    {
                        "signal_name": signal_name,
                        "signal_role": signal_role,
                        "horizon_months": horizon,
                        "bucket": bucket,
                        "n_obs": int(len(bucket_df)),
                        "mean_forward_return_pct": float(bucket_df[f"forward_return_{horizon}m"].mean() * 100.0),
                        "median_forward_return_pct": float(bucket_df[f"forward_return_{horizon}m"].median() * 100.0),
                        "prob_loss_pct": float((bucket_df[f"forward_return_{horizon}m"] < 0).mean()),
                    }
                )
            ordered = (
                subset.groupby("signal_bucket")[f"forward_return_{horizon}m"]
                .mean()
                .reindex(["Cautious", "Mixed", "Supportive"])
            )
            supportive = ordered.loc["Supportive"]
            cautious = ordered.loc["Cautious"]
            spread = float((supportive - cautious) * 100.0)
            spearman = float(subset[signal_name].corr(
                subset[f"forward_return_{horizon}m"], method="spearman"))
            monotonic = bool(ordered.is_monotonic_increasing)
            diagnostics.append(
                {
                    "signal_name": signal_name,
                    "signal_role": signal_role,
                    "horizon_months": horizon,
                    "spearman_corr": spearman,
                    "monotonic_mean_return": monotonic,
                    "supportive_minus_cautious_return_pp": spread,
                    "n_obs": int(len(subset)),
                }
            )
            strength_rows.append(
                {
                    "signal_name": signal_name,
                    "signal_role": signal_role,
                    "horizon_months": horizon,
                    "spearman_corr": spearman,
                    "supportive_minus_cautious_return_pp": spread,
                    "monotonic_mean_return": monotonic,
                    "evidence_level": (
                        "supported"
                        if monotonic and spearman >= 0.10 and spread > 5.0
                        else "partial"
                        if spread > 0.0 and spearman > 0.0
                        else "weak"
                    ),
                }
            )

    performance = pd.DataFrame(rows).sort_values(
        ["signal_name", "horizon_months", "bucket"]).reset_index(drop=True)
    calibration = pd.DataFrame(diagnostics).sort_values(
        ["signal_name", "horizon_months"]).reset_index(drop=True)
    strength = pd.DataFrame(strength_rows).sort_values(
        ["horizon_months", "evidence_level", "supportive_minus_cautious_return_pp"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    performance.to_csv(SCORE_BUCKET_PATH, index=False)
    calibration.to_csv(SCORE_CALIBRATION_PATH, index=False)
    strength.to_csv(COMPONENT_SIGNAL_STRENGTH_PATH, index=False)
    return performance, calibration, strength


def _app_consistency_check() -> pd.DataFrame:
    config = load_config()
    snapshot_path = STAGE7_OUTPUT_DIR / config.artifacts.stage7_snapshot_file
    metadata_path = STAGE7_OUTPUT_DIR / config.artifacts.stage7_metadata_file
    snapshot = pd.read_csv(snapshot_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stage6 = pd.read_csv(STAGE6_OUTPUT_DIR /
                         config.artifacts.stage6_handoff_file)
    rows = [
        {
            "check": "region_snapshot_matches_stage6_rows",
            "passed": int(len(snapshot) == len(stage6)),
            "detail": f"snapshot={len(snapshot)} stage6={len(stage6)}",
        },
        {
            "check": "region_snapshot_regions_match_stage6",
            "passed": int(set(snapshot["region"]) == set(stage6["region"])),
            "detail": f"snapshot_regions={snapshot['region'].nunique()} stage6_regions={stage6['region'].nunique()}",
        },
        {
            "check": "dashboard_metadata_historical_only",
            "passed": int(bool(metadata.get("historical_only", False))),
            "detail": str(metadata.get("historical_only", False)),
        },
        {
            "check": "dashboard_metadata_stage4_file",
            "passed": int(metadata.get("stage4_parameters_file") == config.artifacts.stage4_parameters_file),
            "detail": str(metadata.get("stage4_parameters_file")),
        },
        {
            "check": "dashboard_metadata_stage5_file",
            "passed": int(metadata.get("stage5_summary_file") == config.artifacts.stage5_summary_file),
            "detail": str(metadata.get("stage5_summary_file")),
        },
        {
            "check": "dashboard_metadata_stage6_file",
            "passed": int(metadata.get("stage6_handoff_file") == config.artifacts.stage6_handoff_file),
            "detail": str(metadata.get("stage6_handoff_file")),
        },
        {
            "check": "dashboard_metadata_split_horizon_matches_canonical",
            "passed": int(
                bool(metadata.get("split_horizon", False)) ==
                bool(json.loads(CANONICAL_PANEL_METADATA_PATH.read_text(
                    encoding="utf-8")).get("split_horizon", False))
            ),
            "detail": str(metadata.get("split_horizon", False)),
        },
    ]
    consistency = pd.DataFrame(rows)
    consistency.to_csv(APP_CONSISTENCY_PATH, index=False)
    return consistency


def _write_summary(
    lineage: pd.DataFrame,
    replay: pd.DataFrame,
    calibration: pd.DataFrame,
    strength: pd.DataFrame,
    moments: pd.DataFrame,
    app_checks: pd.DataFrame,
) -> None:
    within_band = float(replay["within_hist_10_90_band"].mean())
    avg_tail_gap = float(replay["tail_risk_gap"].mean())
    strongest = calibration.loc[calibration["supportive_minus_cautious_return_pp"].idxmax(
    )]
    supported = strength[strength["evidence_level"] == "supported"]
    supported_signals = ", ".join(
        sorted(supported["signal_name"].unique())) if not supported.empty else "none"
    pooled_moments = moments[moments["region"] ==
                             "UK_12_region_average"].iloc[0]
    summary = f"""# Independent Validation Summary

Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}

## Scope

- This runner consumes canonical artifacts and cache outputs only.
- It does not refit substitute forecasting models.
- Replay diagnostics cover artifact lineage, app cache consistency, simulation plausibility, and valuation-signal calibration.

## Replay checks

- Artifact files present: `{int(lineage['exists'].sum())}` / `{len(lineage[lineage['artifact'] != 'historical_only_metadata'])}`
- App consistency checks passed: `{int(app_checks['passed'].sum())}` / `{len(app_checks)}`
- Baseline median within historical 10-90 band: `{within_band:.0%}`
- Average tail-risk gap: `{avg_tail_gap:.3f}`
- Region-average historical vs simulated median 5y return: `{pooled_moments['historical_median_pct']:.2f}%` vs `{pooled_moments['simulated_median_pct']:.2f}%`

## Score calibration

- Strongest supportive-versus-cautious return spread: `{strongest['supportive_minus_cautious_return_pp']:.2f}` percentage points at `{int(strongest['horizon_months'])}m`
- Signals with supported evidence under the current historical backtest screen: `{supported_signals}`
- This evidence supports valuation-led descriptive bucketing, not action-style advice.
"""
    INDEPENDENT_SUMMARY_PATH.write_text(summary, encoding="utf-8")


OLS_REPLAY_CHECK_PATH = VALIDATION_DIR / "ols_replay_check.json"
SUBSAMPLE_OLS_VALIDATION_PATH = VALIDATION_DIR / "subsample_ols_validation.json"


def run_ols_replay_check() -> dict:
    """Re-estimate the deployed Stage 4 OLS and compare replayed P* to the artifact.

    Loads the canonical panel, re-runs fit_structural_fair_value with the same
    config as canonical_rebuild.py, computes fair_value_nominal for each
    region-month, then compares to the deployed fair_value_panel_bankgrade.csv.

    max_p_star_deviation = max(|replayed - deployed| / deployed) across all pairs.
    Passes if max_p_star_deviation < 0.02 (2%).

    Does NOT raise on failure — sets passed=False and logs the finding.
    Saves result to data/outputs/validation/ols_replay_check.json.
    """
    import warnings as _warn

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    config = _load_config_for_replay()

    # Load canonical panel (same path as canonical_rebuild uses)
    master = pd.read_csv(CANONICAL_PANEL_PATH, parse_dates=["date"])
    master = master.sort_values(["region", "date"]).reset_index(drop=True)

    # Add engineered features (needed for OLS regressors)
    from src.model.canonical_rebuild import _load_master_dataset as _get_master
    master_enriched = _get_master()

    # Re-run fair value OLS with same config
    try:
        replayed_fv = fit_structural_fair_value(
            master_enriched,
            regressors=config.fair_value.regressors,
            anchor_months=config.fair_value.anchor_months,
        )
        replayed_panel = replayed_fv.fitted_panel.copy()

        # Compute replayed fair_value_nominal using 2015-01-01 CPI base
        cpi_base_rows = replayed_panel[replayed_panel["date"] == pd.Timestamp(
            "2015-01-01")]
        if cpi_base_rows.empty:
            cpi_base = float(replayed_panel["cpi"].iloc[0])
        else:
            cpi_base = float(cpi_base_rows["cpi"].iloc[0])
        replayed_panel["fair_value_nominal_replayed"] = replayed_panel["fair_value_real"] * \
            (replayed_panel["cpi"] / cpi_base)

        # Load deployed artifact
        deployed_path = STAGE4_OUTPUT_DIR / "fair_value_panel_bankgrade.csv"
        deployed = pd.read_csv(deployed_path, parse_dates=["date"])

        # Merge on region + date
        merged = deployed[["region", "date", "fair_value_nominal"]].merge(
            replayed_panel[["region", "date", "fair_value_nominal_replayed"]],
            on=["region", "date"],
            how="inner",
        ).dropna(subset=["fair_value_nominal", "fair_value_nominal_replayed"])

        if merged.empty:
            result: dict = {
                "replay_max_deviation": None,
                "n_regions": 0,
                "n_months": 0,
                "passed": False,
                "finding": "No overlapping region-date pairs between replayed and deployed panel",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        else:
            deviations = (
                (merged["fair_value_nominal_replayed"] - merged["fair_value_nominal"]).abs() /
                merged["fair_value_nominal"].abs()
            )
            max_dev = float(deviations.max())
            passed = bool(max_dev < 0.02)
            finding = "PASSED: replayed P* agrees with deployed within 2%." if passed else (
                f"FINDING: max P* deviation {max_dev:.4f} ({max_dev*100:.2f}%) exceeds 2% threshold. "
                f"Worst region-date: {merged.loc[deviations.idxmax(), 'region']} "
                f"{merged.loc[deviations.idxmax(), 'date'].strftime('%Y-%m-%d')}."
            )
            result = {
                "replay_max_deviation": round(max_dev, 6),
                "n_regions": int(merged["region"].nunique()),
                "n_months": int(merged["date"].nunique()),
                "passed": passed,
                "finding": finding,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
    except Exception as exc:
        _warn.warn(
            f"run_ols_replay_check failed: {exc}", UserWarning, stacklevel=2)
        result = {
            "replay_max_deviation": None,
            "n_regions": 0,
            "n_months": 0,
            "passed": False,
            "finding": f"OLS replay raised an exception: {exc}",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    OLS_REPLAY_CHECK_PATH.write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_independent_validation() -> IndependentValidationResult:
    master = pd.read_csv(CANONICAL_PANEL_PATH, parse_dates=["date"])
    fair_panel = pd.read_csv(
        STAGE4_OUTPUT_DIR / "fair_value_panel_bankgrade.csv", parse_dates=["date"])
    summary = pd.read_csv(STAGE5_OUTPUT_DIR /
                          load_config().artifacts.stage5_summary_file)

    lineage = _artifact_lineage_checks()
    replay = _replay_simulation_plausibility(master, summary)
    _, moments = _historical_vs_simulated_distribution(master, summary)
    _, calibration, strength = _score_bucket_performance(fair_panel)
    app_checks = _app_consistency_check()
    _write_summary(lineage, replay, calibration, strength, moments, app_checks)

    return IndependentValidationResult(
        generated_at_utc=datetime.now(
            timezone.utc).isoformat(timespec="seconds"),
        artifact_lineage_path=str(ARTIFACT_LINEAGE_PATH),
        app_consistency_path=str(APP_CONSISTENCY_PATH),
        simulation_replay_path=str(SIMULATION_REPLAY_PATH),
        score_bucket_path=str(SCORE_BUCKET_PATH),
        score_calibration_path=str(SCORE_CALIBRATION_PATH),
        component_signal_strength_path=str(COMPONENT_SIGNAL_STRENGTH_PATH),
        historical_vs_simulated_distribution_path=str(
            HISTORICAL_VS_SIMULATED_DISTRIBUTION_PATH),
        simulation_moment_comparison_path=str(
            SIMULATION_MOMENT_COMPARISON_PATH),
        summary_path=str(INDEPENDENT_SUMMARY_PATH),
    )


def run_all_validations() -> dict:
    """Run all independent validation checks and return a combined summary dict."""
    independent_result = run_independent_validation()
    replay_result = run_ols_replay_check()
    subsample_result = run_subsample_ols_validation()

    combined = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "independent_validation": asdict(independent_result),
        "ols_replay_check": replay_result,
        "subsample_ols_validation": subsample_result,
    }
    return combined


def main() -> None:
    combined = run_all_validations()
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
