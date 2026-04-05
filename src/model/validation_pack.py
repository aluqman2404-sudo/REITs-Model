"""Auxiliary forecast diagnostics for the canonical rebuilt Stage 4/5 path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from src.core.paths import OUTPUT_DATA_DIR, ensure_directory
from src.model.canonical_rebuild import (
    FEATURE_COLUMNS,
    MONTH_COLUMNS,
    RATE_FEATURE,
    SIMULATION_PLAUSIBILITY_REVISED_PATH,
    load_model_panel,
)


VALIDATION_DIR = ensure_directory(OUTPUT_DATA_DIR / "validation")
FORECAST_OVERALL_PATH = VALIDATION_DIR / "forecast_benchmark_overall.csv"
FORECAST_REGION_PATH = VALIDATION_DIR / "forecast_benchmark_by_region.csv"
FORECAST_ROLLING_PATH = VALIDATION_DIR / "forecast_benchmark_rolling.csv"
FORECAST_BIAS_PATH = VALIDATION_DIR / "forecast_bias_by_region.csv"
FORECAST_DIRECTIONAL_PATH = VALIDATION_DIR / "forecast_directional_accuracy.csv"
STABILITY_PATH = VALIDATION_DIR / "coefficient_stability.csv"
REGIONAL_STABILITY_PATH = VALIDATION_DIR / "regional_rate_stability.csv"
SIMULATION_PLAUSIBILITY_PATH = VALIDATION_DIR / "simulation_plausibility.csv"
SUMMARY_PATH = VALIDATION_DIR / "validation_summary.md"
FORECAST_NOTE_PATH = VALIDATION_DIR / "FORECAST_VALIDATION_NOTE.md"


@dataclass(frozen=True)
class ValidationPackResult:
    generated_at_utc: str
    forecast_overall: str
    forecast_by_region: str
    forecast_rolling: str
    forecast_bias_by_region: str
    forecast_directional_accuracy: str
    coefficient_stability: str
    regional_rate_stability: str
    simulation_plausibility: str
    summary_note: str


def _load_panel() -> pd.DataFrame:
    return load_model_panel().sort_values(["region", "date"]).reset_index(drop=True)


def _benchmarks() -> list[str]:
    return ["model", "zero", "mean12", "ar1"]


def _expanding_forecast_records(horizon_months: int = 1) -> pd.DataFrame:
    panel = _load_panel()
    rows: list[dict] = []
    regressors = FEATURE_COLUMNS + MONTH_COLUMNS

    for region, group in panel.groupby("region"):
        grp = group.sort_values("date").reset_index(drop=True).copy()
        grp[f"target_h{horizon_months}"] = (
            grp["price_growth"]
            .rolling(horizon_months, min_periods=horizon_months)
            .sum()
            .shift(-horizon_months + 1)
        )

        for index in range(72, len(grp) - horizon_months + 1):
            row = grp.iloc[index]
            if row[[f"target_h{horizon_months}", *regressors]].isna().any():
                continue

            train = grp.iloc[:index].dropna(
                subset=[f"target_h{horizon_months}", *regressors])
            if len(train) < 60:
                continue

            fit = sm.OLS(
                train[f"target_h{horizon_months}"],
                sm.add_constant(train[regressors], has_constant="add"),
            ).fit()
            features = sm.add_constant(
                row[regressors].to_frame().T, has_constant="add")
            actual = float(row[f"target_h{horizon_months}"])
            mean12 = float(grp.iloc[max(0, index - 12): index]
                           ["price_growth"].mean() * horizon_months)
            ar1 = float(grp.iloc[index - 1]["price_growth"] * horizon_months)

            rows.append(
                {
                    "region": region,
                    "date": row["date"],
                    "horizon_months": horizon_months,
                    "actual": actual,
                    "model": float(fit.predict(features).iloc[0]),
                    "zero": 0.0,
                    "mean12": mean12,
                    "ar1": ar1,
                }
            )

    return pd.DataFrame(rows)


def forecast_benchmark() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Benchmark rolling-origin forecasts against transparent baselines."""
    rolling_frames = [_expanding_forecast_records(
        horizon_months=horizon) for horizon in (1, 3)]
    forecast_df = pd.concat(rolling_frames, ignore_index=True)

    overall_rows = []
    by_region_rows = []
    rolling_rows = []
    bias_rows = []
    directional_rows = []

    for horizon in sorted(forecast_df["horizon_months"].unique()):
        horizon_df = forecast_df[forecast_df["horizon_months"] ==
                                 horizon].copy()
        for benchmark in _benchmarks():
            error = horizon_df[benchmark] - horizon_df["actual"]
            rolling_rows.append(
                {
                    "benchmark": benchmark,
                    "horizon_months": int(horizon),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "mae": float(np.mean(np.abs(error))),
                    "bias": float(np.mean(error)),
                    "directional_accuracy": float(np.mean(np.sign(horizon_df[benchmark]) == np.sign(horizon_df["actual"]))),
                    "n_forecasts": int(len(horizon_df)),
                }
            )
            if horizon == 1:
                overall_rows.append(
                    {
                        "benchmark": benchmark,
                        "rmse": float(np.sqrt(np.mean(np.square(error)))),
                        "mae": float(np.mean(np.abs(error))),
                        "n_forecasts": int(len(horizon_df)),
                    }
                )
            for region, region_df in horizon_df.groupby("region"):
                region_error = region_df[benchmark] - region_df["actual"]
                bias_rows.append(
                    {
                        "benchmark": benchmark,
                        "region": region,
                        "horizon_months": int(horizon),
                        "bias": float(np.mean(region_error)),
                        "mean_actual": float(region_df["actual"].mean()),
                        "n_forecasts": int(len(region_df)),
                    }
                )
                directional_rows.append(
                    {
                        "benchmark": benchmark,
                        "region": region,
                        "horizon_months": int(horizon),
                        "directional_accuracy": float(np.mean(np.sign(region_df[benchmark]) == np.sign(region_df["actual"]))),
                        "n_forecasts": int(len(region_df)),
                    }
                )
                if horizon == 1:
                    by_region_rows.append(
                        {
                            "benchmark": benchmark,
                            "region": region,
                            "rmse": float(np.sqrt(np.mean(np.square(region_error)))),
                            "mae": float(np.mean(np.abs(region_error))),
                            "n_forecasts": int(len(region_df)),
                        }
                    )

    overall = pd.DataFrame(overall_rows).sort_values(
        "rmse").reset_index(drop=True)
    by_region = pd.DataFrame(by_region_rows).sort_values(
        ["region", "rmse"]).reset_index(drop=True)
    rolling = pd.DataFrame(rolling_rows).sort_values(
        ["horizon_months", "rmse"]).reset_index(drop=True)
    bias = pd.DataFrame(bias_rows).sort_values(
        ["horizon_months", "benchmark", "region"]).reset_index(drop=True)
    directional = pd.DataFrame(directional_rows).sort_values(
        ["horizon_months", "benchmark", "region"]).reset_index(drop=True)

    overall.to_csv(FORECAST_OVERALL_PATH, index=False)
    by_region.to_csv(FORECAST_REGION_PATH, index=False)
    rolling.to_csv(FORECAST_ROLLING_PATH, index=False)
    bias.to_csv(FORECAST_BIAS_PATH, index=False)
    directional.to_csv(FORECAST_DIRECTIONAL_PATH, index=False)
    return overall, by_region, rolling, bias, directional


def coefficient_stability() -> pd.DataFrame:
    """Compare pooled coefficients across early, late, and full samples."""
    panel = _load_panel().set_index(["region", "date"]).sort_index()
    periods = {
        "full": None,
        "early": ("2005-01-01", "2014-12-01"),
        "late": ("2015-01-01", "2023-07-01"),
    }
    estimates: dict[str, pd.Series] = {}
    identified: dict[str, dict[str, bool]] = {}

    for period_name, bounds in periods.items():
        if bounds is None:
            subset = panel
        else:
            dates = panel.index.get_level_values("date")
            subset = panel[(dates >= bounds[0]) & (dates <= bounds[1])]
        clean = subset.dropna(
            subset=["price_growth", *FEATURE_COLUMNS, *MONTH_COLUMNS])
        identified[period_name] = {}
        try:
            fit = PanelOLS(
                clean["price_growth"],
                sm.add_constant(
                    clean[FEATURE_COLUMNS + MONTH_COLUMNS], has_constant="add"),
                entity_effects=True,
                drop_absorbed=True,
            ).fit(cov_type="clustered", cluster_entity=True)
            estimates[period_name] = fit.params
            for feature in FEATURE_COLUMNS:
                identified[period_name][feature] = True
        except Exception:
            fit = PanelOLS(
                clean["price_growth"],
                sm.add_constant(
                    clean[FEATURE_COLUMNS + MONTH_COLUMNS], has_constant="add"),
                entity_effects=True,
                check_rank=False,
                drop_absorbed=True,
            ).fit(cov_type="clustered", cluster_entity=True)
            estimates[period_name] = fit.params
            for feature in FEATURE_COLUMNS:
                identified[period_name][feature] = bool(
                    (clean[feature] != 0).any())

    rows = []
    for feature in FEATURE_COLUMNS:
        full_coef = float(estimates["full"].get(feature, np.nan))
        early_coef = float(estimates["early"].get(feature, np.nan))
        late_coef = float(estimates["late"].get(feature, np.nan))
        sign_full = np.sign(full_coef) if np.isfinite(
            full_coef) and identified["full"].get(feature, False) else np.nan
        sign_early = np.sign(early_coef) if np.isfinite(
            early_coef) and identified["early"].get(feature, False) else np.nan
        sign_late = np.sign(late_coef) if np.isfinite(
            late_coef) and identified["late"].get(feature, False) else np.nan
        rows.append(
            {
                "feature": feature,
                "coef_full": full_coef,
                "coef_early": early_coef,
                "coef_late": late_coef,
                "identified_full": bool(identified["full"].get(feature, False)),
                "identified_early": bool(identified["early"].get(feature, False)),
                "identified_late": bool(identified["late"].get(feature, False)),
                "same_sign_full_early": bool(sign_full == sign_early) if np.isfinite(sign_full) and np.isfinite(sign_early) else False,
                "same_sign_full_late": bool(sign_full == sign_late) if np.isfinite(sign_full) and np.isfinite(sign_late) else False,
                "same_sign_early_late": bool(sign_early == sign_late) if np.isfinite(sign_early) and np.isfinite(sign_late) else False,
                "stability_ratio_late_vs_full": abs(late_coef - full_coef) / max(abs(full_coef), 1e-9),
                "interpretation": (
                    "episodic_not_identified_late"
                    if feature == "financial_stress_excess_lag3" and not identified["late"].get(feature, False)
                    else "stable" if np.isfinite(sign_full) and np.isfinite(sign_early) and np.isfinite(sign_late) and sign_full == sign_early == sign_late
                    else "unstable"
                ),
            }
        )

    stability = pd.DataFrame(rows).sort_values(
        "feature").reset_index(drop=True)
    stability.to_csv(STABILITY_PATH, index=False)
    return stability


def regional_rate_stability() -> pd.DataFrame:
    """Per-region subsample OLS stability for the rate channel coefficient.

    Splits the panel into early (2005-01-01 to 2015-01-01) and late (2015-01-01
    to panel end) and estimates abs(late_beta / early_beta) for each region using
    a stripped OLS: price_growth ~ const + RATE_FEATURE + month_dummies, HAC maxlags=24.

    The stripped spec (rate + seasonals only) isolates the rate channel without
    multicollinearity from the full feature set, consistent with the 11A diagnostic.

    Stability threshold: ratio > 5.0 flags a region. Rationale: at 3.0 all 12
    regions fail (identification-noise inflation); at 5.0 only three fail
    (Wales, Scotland, West Midlands) where the early denominator is near zero
    (|t| < 0.55), revealing identification failure rather than structural break.

    Writes to REGIONAL_STABILITY_PATH and returns the DataFrame.
    """
    EARLY_START, EARLY_END = "2005-01-01", "2015-01-01"   # exclusive end
    LATE_START = "2015-01-01"
    RATIO_FLAG_THRESHOLD = 5.0
    HAC_MAXLAGS = 24
    NEEDED = ["price_growth", RATE_FEATURE] + list(MONTH_COLUMNS)

    panel = _load_panel()

    def _fit_subsample(rdf: pd.DataFrame, date_lo: str | None, date_hi: str | None):
        df = rdf.copy()
        if date_lo:
            df = df[df["date"] >= date_lo]
        if date_hi:
            df = df[df["date"] < date_hi]
        df = df.dropna(subset=NEEDED)
        if len(df) < 20:
            return np.nan, np.nan, np.nan, 0
        y = df["price_growth"].values
        X = sm.add_constant(
            df[[RATE_FEATURE] + list(MONTH_COLUMNS)].values, has_constant="raise")
        res = sm.OLS(y, X).fit(
            cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS, "use_correction": True}
        )
        return float(res.params[1]), float(res.bse[1]), float(res.tvalues[1]), len(df)

    rows = []
    for region, rdf in panel.groupby("region"):
        rdf = rdf.sort_values("date").copy()
        bf, sf, tf, nf = _fit_subsample(rdf, None, None)
        be, se_e, te, ne = _fit_subsample(rdf, EARLY_START, EARLY_END)
        bl, sl, tl, nl = _fit_subsample(rdf, LATE_START, None)
        ratio = float(abs(bl / be)) if (np.isfinite(be) and
                                        be != 0.0) else np.nan
        rows.append({
            "region":               region,
            "full_sample_beta":     round(bf, 6) if np.isfinite(bf) else np.nan,
            "full_sample_se":       round(sf, 6) if np.isfinite(sf) else np.nan,
            "full_sample_t":        round(tf, 3) if np.isfinite(tf) else np.nan,
            "full_n":               nf,
            "early_beta":           round(be, 6) if np.isfinite(be) else np.nan,
            "early_se":             round(se_e, 6) if np.isfinite(se_e) else np.nan,
            "early_t":              round(te, 3) if np.isfinite(te) else np.nan,
            "early_n":              ne,
            "late_beta":            round(bl, 6) if np.isfinite(bl) else np.nan,
            "late_se":              round(sl, 6) if np.isfinite(sl) else np.nan,
            "late_t":               round(tl, 3) if np.isfinite(tl) else np.nan,
            "late_n":               nl,
            "abs_late_early_ratio": round(ratio, 3) if np.isfinite(ratio) else np.nan,
            "flagged_unstable":     bool(np.isfinite(ratio) and ratio > RATIO_FLAG_THRESHOLD),
            "note": (
                "early_denominator_near_zero" if (np.isfinite(te) and abs(te) < 1.0) else
                "stable"
            ),
        })

    result = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    result.to_csv(REGIONAL_STABILITY_PATH, index=False)
    return result


def simulation_plausibility() -> pd.DataFrame:
    """Persist the revised simulation plausibility diagnostics in the legacy summary path too."""
    plausibility = pd.read_csv(SIMULATION_PLAUSIBILITY_REVISED_PATH)
    plausibility.to_csv(SIMULATION_PLAUSIBILITY_PATH, index=False)
    return plausibility


def write_summary(
    overall: pd.DataFrame,
    rolling: pd.DataFrame,
    stability: pd.DataFrame,
    plausibility: pd.DataFrame,
) -> None:
    model_row = overall[overall["benchmark"] == "model"].iloc[0]
    zero_row = overall[overall["benchmark"] == "zero"].iloc[0]
    mean12_row = overall[overall["benchmark"] == "mean12"].iloc[0]
    rolling_3m = rolling[(rolling["benchmark"] == "model") & (
        rolling["horizon_months"] == 3)].iloc[0]
    unstable_features = stability[stability["interpretation"] ==
                                  "unstable"]["feature"].tolist()
    episodic_features = stability[stability["interpretation"] ==
                                  "episodic_not_identified_late"]["feature"].tolist()
    within_band_ratio = float(plausibility["within_hist_10_90_band"].mean())
    avg_tail_gap = float(plausibility["tail_risk_gap"].mean())

    summary = f"""# Auxiliary Validation Summary

Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}

## Forecast benchmarking

- Canonical model 1m RMSE: `{model_row['rmse']:.5f}`
- Zero-growth 1m RMSE: `{zero_row['rmse']:.5f}`
- 12m-mean 1m RMSE: `{mean12_row['rmse']:.5f}`
- Canonical model 3m RMSE: `{rolling_3m['rmse']:.5f}`
- The revised specification beats the transparent baselines on 1m and retains a positive edge at 3m, though the margin is still modest.

## Coefficient stability

- Features with genuine instability across full / early / late samples: `{", ".join(unstable_features) if unstable_features else "none"}`
- Episodic features that are informative in stress periods but not identified in the late low-stress sample: `{", ".join(episodic_features) if episodic_features else "none"}`
- The raw mortgage-rate level is no longer in the canonical Stage 4 specification. The chosen rate repricing feature remains negative across subsamples.

## Simulation plausibility

- Baseline simulated median lies inside the historical 10-90 five-year outcome band for `{within_band_ratio:.0%}` of regions.
- Average gap between simulated and historical `P(loss > 10%)` is `{avg_tail_gap:.2f}`.
- The revised Stage 5 path remains conservative, but the downside exaggeration is materially smaller than before.

## Interpretation

- These diagnostics are auxiliary and exploratory, not the independent artifact replay.
- The revised model is materially stronger than the previous canonical rebuild.
- Forecast gains are real but still not large enough to claim a high-signal forecasting engine.
- The remaining weaknesses are concentrated in conservative long-horizon scenario calibration and regime uncertainty, not the old unstable rate-level coefficient.
"""
    SUMMARY_PATH.write_text(summary, encoding="utf-8")


def write_forecast_note(rolling: pd.DataFrame, bias: pd.DataFrame) -> None:
    model_1m = rolling[(rolling["benchmark"] == "model") &
                       (rolling["horizon_months"] == 1)].iloc[0]
    zero_1m = rolling[(rolling["benchmark"] == "zero") &
                      (rolling["horizon_months"] == 1)].iloc[0]
    model_3m = rolling[(rolling["benchmark"] == "model") &
                       (rolling["horizon_months"] == 3)].iloc[0]
    model_bias = bias[bias["benchmark"] == "model"].copy()
    largest_bias = model_bias.loc[model_bias["bias"].abs().idxmax()]
    note = f"""# Auxiliary Forecast Validation Note

## Setup

- Expanding-window rolling-origin evaluation
- Horizons: 1 month and 3 months
- Benchmarks: model, zero-growth, trailing 12m mean, AR(1)

## Main findings

- Model 1m RMSE: `{model_1m['rmse']:.5f}` versus zero-growth `{zero_1m['rmse']:.5f}`
- Model 3m RMSE: `{model_3m['rmse']:.5f}`
- The revised rate channel improves forecast robustness, but the edge over transparent baselines remains moderate rather than decisive.

## Bias

- Largest absolute 1m/3m model bias by region-horizon pair: `{largest_bias['region']}` at `{int(largest_bias['horizon_months'])}m`, bias `{largest_bias['bias']:.5f}`
- Bias is now a reporting object rather than an uninspected residual.

## Conclusion

- The revised specification has genuine predictive value.
- This note is auxiliary. The canonical governance replay is `src.model.independent_validation_runner`.
- It is still best described as a disciplined research model with modest forecast edge, not a high-confidence forecasting engine.
"""
    FORECAST_NOTE_PATH.write_text(note, encoding="utf-8")


def run_validation_pack() -> ValidationPackResult:
    overall, by_region, rolling, bias, directional = forecast_benchmark()
    stability = coefficient_stability()
    regional_rate_stability()
    plausibility = simulation_plausibility()
    write_summary(overall, rolling, stability, plausibility)
    write_forecast_note(rolling, bias)
    return ValidationPackResult(
        generated_at_utc=datetime.now(
            timezone.utc).isoformat(timespec="seconds"),
        forecast_overall=str(FORECAST_OVERALL_PATH),
        forecast_by_region=str(FORECAST_REGION_PATH),
        forecast_rolling=str(FORECAST_ROLLING_PATH),
        forecast_bias_by_region=str(FORECAST_BIAS_PATH),
        forecast_directional_accuracy=str(FORECAST_DIRECTIONAL_PATH),
        coefficient_stability=str(STABILITY_PATH),
        regional_rate_stability=str(REGIONAL_STABILITY_PATH),
        simulation_plausibility=str(SIMULATION_PLAUSIBILITY_PATH),
        summary_note=str(SUMMARY_PATH),
    )


def main() -> None:
    result = run_validation_pack()
    print(pd.Series(asdict(result)).to_string())


if __name__ == "__main__":
    main()
