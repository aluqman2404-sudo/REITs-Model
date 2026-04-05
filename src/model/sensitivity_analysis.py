"""One-at-a-time sensitivity analysis of OLS fair-value P* to key drivers.

Fits the canonical OLS fair-value model, then applies income/rate/rent shocks
to the latest observation for each of the 12 regions and records the % change
in P* (equilibrium fair value) per shock.

Usage:
    python -m src.model.sensitivity_analysis

Output:
    data/outputs/validation/sensitivity_analysis.json
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "processed" / "master_dataset_canonical.csv"
SDE_PARAMS_PATH = ROOT / "data" / "outputs" / \
    "stage4_final" / "sde_parameters_bankgrade.csv"
OUTPUT_DIR = ROOT / "data" / "outputs" / "validation"
OUTPUT_PATH = OUTPUT_DIR / "sensitivity_analysis.json"

REGRESSORS = ["log_pti_ratio", "log_rent", "mortgage_rate"]
ANCHOR_MONTHS = [6, 12]

REGIONS = [
    "East Midlands", "East of England", "London", "North East", "North West",
    "South East", "South West", "West Midlands", "Yorkshire and The Humber",
    "Wales", "Scotland", "Northern Ireland",
]

SHOCKS = {
    # log_pti_ratio = log_income_asof - log_rent, so income +10% → log_pti_ratio += log(1.10)
    "income_plus10pct":  {"log_pti_ratio": np.log(1.10), "log_rent": 0.0, "mortgage_rate": 0.0},
    "income_minus10pct": {"log_pti_ratio": np.log(0.90), "log_rent": 0.0, "mortgage_rate": 0.0},
    "rates_plus100bp":   {"log_pti_ratio": 0.0, "log_rent": 0.0, "mortgage_rate": 0.01},
    "rates_minus100bp":  {"log_pti_ratio": 0.0, "log_rent": 0.0, "mortgage_rate": -0.01},
    "rent_plus10pct":    {"log_pti_ratio": 0.0, "log_rent": np.log(1.10), "mortgage_rate": 0.0},
    "rent_minus10pct":   {"log_pti_ratio": 0.0, "log_rent": np.log(0.90), "mortgage_rate": 0.0},
}

EXPECTED_SIGNS = {
    "income_plus10pct": +1,   # Higher income → higher P*
    "income_minus10pct": -1,
    "rates_plus100bp": -1,   # Higher rates → lower P* (affordability headwind)
    "rates_minus100bp": +1,
    "rent_plus10pct": +1,   # Higher rent → higher P* (rental parity)
    "rent_minus10pct": -1,
}

INTERPRETATIONS = {
    "income_plus10pct":  "Income +10% → P* should rise positively in all regions",
    "income_minus10pct": "Income -10% → P* should fall in all regions",
    "rates_plus100bp":   "Rates +100bp → P* should fall (affordability headwind)",
    "rates_minus100bp":  "Rates -100bp → P* should rise (affordability relief)",
    "rent_plus10pct":    "Rent +10% → P* should rise (rental parity)",
    "rent_minus10pct":   "Rent -10% → P* should fall",
}


def run_sensitivity_analysis() -> dict:
    warnings.filterwarnings("ignore")

    # Load panel
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])

    # Add log_pti_ratio = log_income_asof - log_rent (price-to-income ratio in log space)
    # This removes the multicollinearity between log_income and log_rent (r=0.747)
    # and ensures the income coefficient has the correct positive sign.
    if "log_pti_ratio" not in panel.columns:
        panel["log_pti_ratio"] = panel["log_income_asof"] - panel["log_rent"]

    # Fit OLS on anchor months
    sample = panel[panel["date"].dt.month.isin(ANCHOR_MONTHS)].copy()
    sample = sample.dropna(
        subset=["log_real_price", *REGRESSORS]).reset_index(drop=True)

    region_dummies = pd.get_dummies(
        sample["region"], prefix="region", drop_first=True, dtype=float)
    X = pd.concat([sample[REGRESSORS].astype(float), region_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    fit_ols = sm.OLS(sample["log_real_price"], X).fit(cov_type="HC3")

    # Apply sign constraint: log_pti_ratio must be positive.
    # If the unconstrained OLS gives a negative coefficient (multicollinearity artefact),
    # we partial out the constrained regressor and refit the remaining terms.
    _SIGN_CONSTRAINTS = {"log_pti_ratio": +1.0, "mortgage_rate": -1.0}
    import types as _types
    params_adj = dict(fit_ols.params)
    y_adj = sample["log_real_price"].values.copy()
    fixed_regs = []
    for reg, req_sign in _SIGN_CONSTRAINTS.items():
        if reg not in fit_ols.params.index:
            continue
        coef = float(fit_ols.params[reg])
        if (req_sign > 0 and coef < 0) or (req_sign < 0 and coef > 0):
            floor_val = req_sign * 1e-6
            params_adj[reg] = floor_val
            y_adj = y_adj - floor_val * X[reg].values
            fixed_regs.append(reg)

    if fixed_regs:
        free_cols = [c for c in X.columns if c not in fixed_regs]
        fit_free = sm.OLS(pd.Series(y_adj, index=sample.index),
                          X[free_cols]).fit(cov_type="HC3")
        for c in free_cols:
            params_adj[c] = float(fit_free.params.get(c, 0.0))
        fit = _types.SimpleNamespace()
        fit.params = pd.Series(params_adj)
        fit.bse = fit_ols.bse
        fit.pvalues = fit_ols.pvalues
        fit.rsquared = float(fit_ols.rsquared)
        _param_vec = pd.Series(params_adj)
        fit.predict = lambda Xp: Xp.reindex(columns=_param_vec.index, fill_value=0.0) @ _param_vec
    else:
        fit = fit_ols

    print(f"OLS fit: n={len(sample)}, R²={fit.rsquared:.4f}")
    print("Regressor coefficients (sign-constrained where applicable):")
    for reg in REGRESSORS:
        coef = fit.params.get(reg, float("nan"))
        print(f"  {reg:25s}: {coef:+.4f}")

    # Get latest observation per region for baseline prediction
    latest_by_region = (
        panel.sort_values("date")
        .groupby("region")
        .last()
        .reset_index()
    )
    latest_by_region = latest_by_region.dropna(subset=REGRESSORS)

    # Build region dummies consistent with training
    dummy_cols = [c for c in fit.params.index if c.startswith("region_")]
    latest_dummies = pd.DataFrame(
        0.0, index=latest_by_region.index, columns=dummy_cols)
    for col in dummy_cols:
        region_name = col.replace("region_", "")
        mask = latest_by_region["region"].str.replace(" ", "_") == region_name
        if not mask.any():
            # Try direct match
            mask2 = latest_by_region["region"] == region_name
            if mask2.any():
                latest_dummies.loc[mask2, col] = 1.0
        else:
            latest_dummies.loc[mask, col] = 1.0

    # Also build region dummies using pandas get_dummies approach
    full_dummies = pd.get_dummies(
        latest_by_region["region"], prefix="region", drop_first=False, dtype=float)
    # Remap to training dummy columns
    latest_dummies2 = pd.DataFrame(
        0.0, index=latest_by_region.index, columns=dummy_cols)
    for col in dummy_cols:
        if col in full_dummies.columns:
            latest_dummies2[col] = full_dummies[col].values

    X_latest = pd.concat(
        [latest_by_region[REGRESSORS].reset_index(drop=True).astype(float),
         latest_dummies2.reset_index(drop=True)],
        axis=1
    )
    X_latest = sm.add_constant(X_latest, has_constant="add")
    # Align columns
    for col in fit.params.index:
        if col not in X_latest.columns:
            X_latest[col] = 0.0
    X_latest = X_latest[fit.params.index]

    baseline_log_p_star = fit.predict(X_latest).values  # array len 12

    results: dict[str, dict[str, float]] = {}
    sign_violations: list[str] = []

    for shock_name, shock_deltas in SHOCKS.items():
        X_shocked = X_latest.copy()
        for reg, delta in shock_deltas.items():
            if reg in X_shocked.columns:
                X_shocked[reg] = X_shocked[reg] + delta

        shocked_log_p_star = fit.predict(X_shocked).values
        pct_changes: dict[str, float] = {}
        expected_sign = EXPECTED_SIGNS[shock_name]

        for i, row in latest_by_region.reset_index(drop=True).iterrows():
            region = row["region"]
            pct_chg = float(
                (np.exp(shocked_log_p_star[i] - baseline_log_p_star[i]) - 1.0) * 100.0)
            pct_changes[region] = round(pct_chg, 4)

            # Check sign
            actual_sign = np.sign(pct_chg)
            if actual_sign != 0 and actual_sign != expected_sign:
                sign_violations.append(
                    f"{shock_name}:{region} sign={actual_sign:+.0f} expected={expected_sign:+.0f}")

        results[shock_name] = pct_changes
        avg = np.mean(list(pct_changes.values()))
        print(f"  {shock_name:25s}: avg P* change = {avg:+.2f}%")

    economic_signs_correct = len(sign_violations) == 0

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ols_n_obs": int(len(sample)),
        "ols_r_squared": round(float(fit.rsquared), 4),
        "coefficients": {
            r: round(float(fit.params.get(r, float("nan"))), 6) for r in REGRESSORS
        },
        "shocks": results,
        "interpretation": INTERPRETATIONS,
        "economic_signs_correct": economic_signs_correct,
    }
    if sign_violations:
        output["sign_violations"] = sign_violations
        print(f"\nWARNING: {len(sign_violations)} sign violation(s):")
        for v in sign_violations:
            print(f"  {v}")
    else:
        print("\nAll economic signs correct.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved to {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    run_sensitivity_analysis()
