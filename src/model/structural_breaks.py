"""
Chow-test structural break diagnostics for the P* fair-value regressors.

Tests whether OLS coefficients for log_income_asof, log_rent, and mortgage_rate
are stable across the 2008 financial crisis and 2020 pandemic break points,
addressing R005 in MODEL_RISK_REGISTER.csv.

The test is a joint 3-period Chow test (pre-GFC / post-GFC pre-COVID / post-COVID).
A significant result (p < 0.05) suggests the full-sample coefficient may not hold
in the current regime and warrants disclosure or subsample re-estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f as f_dist

# Canonical P* regressors — matches config/parameters.json fair_value.regressors
# and the column names confirmed in data/processed/master_dataset_canonical.csv.
_P_STAR_REGRESSORS = ["log_income_asof", "log_rent", "mortgage_rate"]

# Sub-period boundaries (inclusive)
_PRE_GFC_END      = pd.Timestamp("2008-08-01")
_POST_GFC_START   = pd.Timestamp("2008-09-01")
_PRE_COVID_END    = pd.Timestamp("2020-01-01")
_POST_COVID_START = pd.Timestamp("2020-02-01")

_BREAK_LABEL = "2008-09 & 2020-02"


def _ols_rss(y: np.ndarray, X: np.ndarray) -> float:
    """Return OLS residual sum of squares, or NaN if sample is too small."""
    n_params = X.shape[1] + 1  # +1 for constant
    if len(y) <= n_params:
        return np.nan
    result = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()
    return float(result.ssr)


def run_chow_tests(panel_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Run a joint 3-period Chow test for each P* regressor.

    The panel is split into:
      - Pre-GFC:           2005-01 to 2008-08
      - Post-GFC/Pre-COVID: 2008-09 to 2020-01
      - Post-COVID:        2020-02 to end of panel

    For each regressor the test runs pooled OLS (no region dummies) of
    log_real_price on [const, regressor] and applies the Chow F-statistic:

        F = ((RSS_full - (RSS_1 + RSS_2 + RSS_3)) / k)
            / ((RSS_1 + RSS_2 + RSS_3) / (n - 3*k))

    where k = 2 (constant + one regressor), n = total valid observations.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Must contain 'date', 'log_real_price', and one or more P* regressors.

    Returns
    -------
    results_df : pd.DataFrame
        Columns: regressor, break_point, f_stat, p_value, significant_at_5pct
    summary : dict
        Keys: any_breaks_detected (bool), break_points_detected (list[str]),
        interpretation (str)
    """
    df = panel_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    dep_var = "log_real_price"
    if dep_var not in df.columns:
        raise ValueError(f"panel_df must contain '{dep_var}'")

    available = [r for r in _P_STAR_REGRESSORS if r in df.columns]
    if not available:
        raise ValueError(
            f"None of the expected P* regressors {_P_STAR_REGRESSORS} found in panel_df"
        )

    records: list[dict] = []

    for regressor in available:
        sub = df[["date", dep_var, regressor]].dropna().reset_index(drop=True)

        mask1 = sub["date"] <= _PRE_GFC_END
        mask2 = (sub["date"] >= _POST_GFC_START) & (sub["date"] <= _PRE_COVID_END)
        mask3 = sub["date"] >= _POST_COVID_START

        y_full = sub[dep_var].to_numpy()
        X_full = sub[[regressor]].to_numpy()
        n = len(y_full)
        k = 2  # constant + 1 regressor

        rss_full = _ols_rss(y_full, X_full)
        rss_1    = _ols_rss(sub.loc[mask1, dep_var].to_numpy(),
                            sub.loc[mask1, [regressor]].to_numpy())
        rss_2    = _ols_rss(sub.loc[mask2, dep_var].to_numpy(),
                            sub.loc[mask2, [regressor]].to_numpy())
        rss_3    = _ols_rss(sub.loc[mask3, dep_var].to_numpy(),
                            sub.loc[mask3, [regressor]].to_numpy())

        if any(np.isnan(v) for v in [rss_full, rss_1, rss_2, rss_3]):
            records.append({
                "regressor": regressor,
                "break_point": _BREAK_LABEL,
                "f_stat": np.nan,
                "p_value": np.nan,
                "significant_at_5pct": False,
            })
            continue

        rss_sub = rss_1 + rss_2 + rss_3
        df_num  = k
        df_den  = n - 3 * k

        if df_den <= 0 or rss_sub <= 0:
            f_stat  = np.nan
            p_value = np.nan
        else:
            f_stat  = ((rss_full - rss_sub) / df_num) / (rss_sub / df_den)
            p_value = float(f_dist.sf(max(f_stat, 0.0), df_num, df_den))

        records.append({
            "regressor":         regressor,
            "break_point":       _BREAK_LABEL,
            "f_stat":            round(float(f_stat), 4) if not np.isnan(f_stat) else np.nan,
            "p_value":           round(float(p_value), 4) if not np.isnan(p_value) else np.nan,
            "significant_at_5pct": bool(not np.isnan(p_value) and p_value < 0.05),
        })

    results_df = pd.DataFrame(records)

    breaks_detected = results_df.loc[
        results_df["significant_at_5pct"], "regressor"
    ].tolist()
    any_breaks = len(breaks_detected) > 0

    if any_breaks:
        interpretation = (
            f"Structural instability detected in: {', '.join(breaks_detected)}. "
            "Full-sample coefficient estimates for these regressors may not hold "
            "in the current post-COVID regime. "
            "Consider disclosure or subsample re-estimation at next recalibration."
        )
    else:
        interpretation = (
            "No structural breaks detected at 5% significance across the 2008 "
            "financial crisis and 2020 pandemic break points. "
            "Full-sample coefficient estimates appear stable."
        )

    summary: dict = {
        "any_breaks_detected":   any_breaks,
        "break_points_detected": breaks_detected,
        "interpretation":        interpretation,
    }

    return results_df, summary
