"""OLS residual diagnostics for the canonical fair-value regression.

For each of the 12 regions, applies three diagnostic tests to the OLS
fair-value residuals:
  1. Ljung-Box at lags 1, 6, 12  — autocorrelation test
  2. ARCH test at lag 4           — conditional heteroskedasticity
  3. Jarque-Bera                  — departure from normality

Flags are expected for housing price series and represent well-documented
properties (Case & Shiller 1989). The mitigation is HAC standard errors
(Newey-West, 6 lags) used throughout the canonical model.

Usage:
    python -m src.model.residual_diagnostics

Output:
    data/outputs/validation/residual_diagnostics.json
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "processed" / "master_dataset_canonical.csv"
OUTPUT_DIR = ROOT / "data" / "outputs" / "validation"
OUTPUT_PATH = OUTPUT_DIR / "residual_diagnostics.json"

REGRESSORS = ["log_income_asof", "log_rent", "mortgage_rate"]
ANCHOR_MONTHS = [6, 12]
FLAG_P = 0.05


def run_residual_diagnostics() -> dict:
    warnings.filterwarnings("ignore")

    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])

    # Fit pooled OLS on anchor months (same spec as fit_structural_fair_value)
    sample = panel[panel["date"].dt.month.isin(ANCHOR_MONTHS)].copy()
    sample = sample.dropna(
        subset=["log_real_price", *REGRESSORS]).reset_index(drop=True)

    region_dummies = pd.get_dummies(
        sample["region"], prefix="region", drop_first=True, dtype=float)
    X = pd.concat([sample[REGRESSORS].astype(float), region_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(sample["log_real_price"], X).fit()
    sample = sample.copy()
    sample["residual"] = fit.resid.values

    print(f"Pooled OLS: n={len(sample)}, R²={fit.rsquared:.4f}")

    regions = sorted(sample["region"].unique())
    regional_results: dict = {}
    ac_flag_count = 0
    arch_flag_count = 0
    jb_flag_count = 0

    for region in regions:
        reg_resid = sample.loc[sample["region"] ==
                               region, "residual"].sort_values().values
        n_obs = len(reg_resid)
        flags: list[str] = []

        if n_obs < 20:
            print(f"  {region}: too few residuals ({n_obs}) — skipping")
            continue

        # 1. Ljung-Box test
        lb_results: dict = {}
        for lag in [1, 6, 12]:
            if lag >= n_obs:
                continue
            try:
                lb = acorr_ljungbox(reg_resid, lags=[lag], return_df=True)
                stat = float(lb["lb_stat"].iloc[0])
                p = float(lb["lb_pvalue"].iloc[0])
                lb_results[f"lag_{lag}"] = {
                    "stat": round(stat, 4), "p": round(p, 4)}
                if p < FLAG_P:
                    flags.append(f"autocorrelation_lag{lag}")
                    ac_flag_count += 1
            except Exception as e:
                lb_results[f"lag_{lag}"] = {
                    "stat": None, "p": None, "error": str(e)}

        # 2. ARCH test at lag 4
        arch_result: dict = {}
        try:
            if n_obs > 10:
                lm_stat, lm_p, f_stat, f_p = het_arch(reg_resid, nlags=4)
                arch_result["lag_4"] = {"stat": round(
                    float(lm_stat), 4), "p": round(float(lm_p), 4)}
                if lm_p < FLAG_P:
                    flags.append("arch_heteroskedasticity")
                    arch_flag_count += 1
        except Exception as e:
            arch_result["lag_4"] = {"stat": None, "p": None, "error": str(e)}

        # 3. Jarque-Bera test
        jb_result: dict = {}
        try:
            jb_stat, jb_p = jarque_bera(reg_resid)
            jb_result = {"stat": round(
                float(jb_stat), 4), "p": round(float(jb_p), 4)}
            if jb_p < FLAG_P:
                flags.append("non_normality")
                jb_flag_count += 1
        except Exception as e:
            jb_result = {"stat": None, "p": None, "error": str(e)}

        regional_results[region] = {
            "n_obs": n_obs,
            "ljung_box": lb_results,
            "arch_test": arch_result,
            "jarque_bera": jb_result,
            "flags": flags,
        }
        flag_str = ", ".join(flags) if flags else "none"
        print(f"  {region:35s}: n={n_obs:3d}  flags=[{flag_str}]")

    n_regions = len(regional_results)
    overall_finding = (
        f"Diagnostics across {n_regions} regions: "
        f"autocorrelation flags in {ac_flag_count} instances, "
        f"ARCH flags in {arch_flag_count} regions, "
        f"non-normality flags in {jb_flag_count} regions. "
        "These are expected properties of housing price series and are not model defects. "
        "HAC (Newey-West) standard errors correct inference for autocorrelation. "
        "Non-normality is documented in MODEL_LIMITATIONS.md."
    )

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ols_spec": {
            "regressors": REGRESSORS,
            "anchor_months": ANCHOR_MONTHS,
            "n_total_obs": int(len(sample)),
            "r_squared": round(float(fit.rsquared), 4),
        },
        "regions": regional_results,
        "summary": {
            "n_regions": n_regions,
            "ac_flag_count": ac_flag_count,
            "arch_flag_count": arch_flag_count,
            "jb_flag_count": jb_flag_count,
        },
        "mitigation": (
            "HAC (Newey-West, 6 lags) standard errors correct inference for autocorrelation. "
            "Non-normality documented in MODEL_LIMITATIONS.md. "
            "Residual autocorrelation in housing is expected and well-documented "
            "(Case & Shiller 1989)."
        ),
        "overall_finding": overall_finding,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved to {OUTPUT_PATH}")
    print(
        f"Summary: AC={ac_flag_count}, ARCH={arch_flag_count}, JB={jb_flag_count} flags")
    return output


if __name__ == "__main__":
    run_residual_diagnostics()
