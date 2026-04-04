"""Empirical confidence interval coverage test for the 90% CI.

Uses a rolling 1-step-ahead approach with time-varying fair value P*(t):
  - For each month t in the holdout (2018+), compute the actual log-gap
    x_t = log(P_t) - log(P*_t) using the in-sample OLS fair value
  - Predict the distribution of x_{t+1} using the OU dynamics:
        x_{t+1} | x_t  ~  N( mu(x_t), sigma^2 * dt )
        where mu(x_t) = x_t * exp(-kappa * dt) + gamma * dt
  - The 90% prediction interval for log(P_{t+1}) is:
        [log(P*_{t+1}) + mu_x - 1.645*sigma*sqrt(dt),
         log(P*_{t+1}) + mu_x + 1.645*sigma*sqrt(dt)]
  - Count what fraction of actual log prices fall inside

A pooled coverage >= 0.80 passes. < 0.70 is a material finding.

Usage:
    python -m src.model.ci_coverage_test

Output:
    data/outputs/validation/ci_coverage_test.json
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "processed" / "master_dataset_canonical.csv"
FAIR_VALUE_PATH = ROOT / "data" / "outputs" / "stage4_final" / "fair_value_panel_bankgrade.csv"
SDE_PARAMS_PATH = ROOT / "data" / "outputs" / "stage4_final" / "sde_parameters_bankgrade.csv"
OUTPUT_DIR = ROOT / "data" / "outputs" / "validation"
OUTPUT_PATH = OUTPUT_DIR / "ci_coverage_test.json"

HOLDOUT_START = pd.Timestamp("2018-01-01")
N_PATHS = 1000
DT = 1.0 / 12.0
PASS_THRESHOLD = 0.80
MATERIAL_THRESHOLD = 0.70
TARGET_COVERAGE = 0.90
Z_90 = 1.645  # two-sided 90% normal critical value


def run_ci_coverage_test() -> dict:
    warnings.filterwarnings("ignore")

    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    fv_panel = pd.read_csv(FAIR_VALUE_PATH, parse_dates=["date"])
    sde = pd.read_csv(SDE_PARAMS_PATH).set_index("region")

    regions = sde.index.tolist()
    regional_coverage: dict[str, float] = {}

    for region in regions:
        row = sde.loc[region]
        kappa = float(row["kappa"])
        sigma = float(row["sigma"])
        gamma_annual = float(row["gamma_annual_pp"]) / 100.0

        # Fair value panel for this region (in-sample OLS P*)
        fv_reg = fv_panel[fv_panel["region"] == region].sort_values("date").reset_index(drop=True)
        pan_reg = panel[panel["region"] == region].sort_values("date").reset_index(drop=True)

        # Merge on date to align actual prices with fair values
        merged = pd.merge(
            pan_reg[["date", "nominal_house_price"]],
            fv_reg[["date", "fair_value_nominal"]],
            on="date", how="inner"
        ).sort_values("date").reset_index(drop=True)

        holdout = merged[merged["date"] >= HOLDOUT_START].reset_index(drop=True)
        if len(holdout) < 6:
            print(f"  {region}: insufficient holdout data ({len(holdout)} obs) — skipping")
            continue

        # Compute log-gap: x_t = log(P_t) - log(P*_t)
        log_p = np.log(holdout["nominal_house_price"].values)
        log_p_star = np.log(holdout["fair_value_nominal"].values)
        x = log_p - log_p_star  # log-gap at each holdout date

        # Rolling 1-step-ahead prediction intervals
        # For t=0..n-2: predict distribution of x_{t+1} given x_t
        # E[x_{t+1}] = x_t * exp(-kappa*DT) + gamma_annual * DT
        # Var[x_{t+1}] = sigma^2 * DT
        sigma_step = sigma * np.sqrt(DT)  # 1-month OU noise SD

        inside_count = 0
        total_count = 0

        for t in range(len(holdout) - 1):
            # Predicted mean of x_{t+1}
            mu_x_next = x[t] * np.exp(-kappa * DT) + gamma_annual * DT
            # 90% CI for log(P_{t+1}):
            #   log(P_{t+1}) = log(P*_{t+1}) + x_{t+1}
            #   x_{t+1} in CI <=> |x_{t+1} - mu_x_next| < Z_90 * sigma_step
            x_next_actual = x[t + 1]
            if abs(x_next_actual - mu_x_next) <= Z_90 * sigma_step:
                inside_count += 1
            total_count += 1

        coverage = float(inside_count / total_count) if total_count > 0 else float("nan")
        regional_coverage[region] = round(coverage, 4)
        print(f"  {region:35s}: coverage={coverage:.3f}  (n={total_count})")

    if not regional_coverage:
        raise RuntimeError("No regions produced coverage results")

    pooled_coverage = float(np.nanmean(list(regional_coverage.values())))
    pooled_passes = pooled_coverage >= PASS_THRESHOLD

    if pooled_coverage < MATERIAL_THRESHOLD:
        finding = (
            f"MATERIAL FINDING: pooled 1-step-ahead CI coverage {pooled_coverage:.3f} is below "
            "0.70. The OU sigma parameters may underestimate realized monthly volatility "
            "in the holdout period. HAC standard errors mitigate inference issues; "
            "sigma recalibration is recommended."
        )
    elif pooled_passes:
        finding = (
            f"PASS: pooled 1-step-ahead CI coverage {pooled_coverage:.3f} >= 0.80 threshold. "
            "The 90% prediction interval is reasonably calibrated on holdout data."
        )
    else:
        finding = (
            f"MARGINAL: pooled coverage {pooled_coverage:.3f} is between 0.70 and 0.80. "
            "Within acceptable range for an in-sample holdout; monitoring recommended."
        )

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "holdout_start": str(HOLDOUT_START.date()),
        "method": (
            "Rolling 1-step-ahead: for each t in holdout, predict distribution of "
            "x_{t+1} = log(P_{t+1}) - log(P*_{t+1}) using OU dynamics with time-varying P*. "
            "Check if actual x_{t+1} falls within 90% normal CI: "
            "|x_{t+1} - E[x_{t+1}]| <= 1.645 * sigma * sqrt(dt)."
        ),
        "target_coverage": TARGET_COVERAGE,
        "n_paths_used": N_PATHS,
        "regional_coverage": regional_coverage,
        "pooled_coverage": round(pooled_coverage, 4),
        "pooled_passes": pooled_passes,
        "finding": finding,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nPooled 1-step-ahead CI coverage: {pooled_coverage:.4f}  (passes={pooled_passes})")
    print(f"Saved to {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    run_ci_coverage_test()
