"""Canonical Stage 4 OLS helpers for fair value and growth calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


@dataclass(frozen=True)
class FairValueModelResult:
    coefficients: pd.DataFrame
    fitted_panel: pd.DataFrame
    diagnostics: pd.DataFrame


def _vif_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    clean = df[columns].dropna()
    values = clean.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "feature": columns,
            "vif": [float(variance_inflation_factor(values, index)) for index in range(len(columns))],
        }
    )


def fit_structural_fair_value(
    master: pd.DataFrame,
    *,
    regressors: list[str],
    anchor_months: list[int],
) -> FairValueModelResult:
    """Fit a semi-structural fair-value model on annual anchor observations."""
    sample = master[master["date"].dt.month.isin(anchor_months)].copy()
    sample = sample.dropna(
        subset=["log_real_price", *regressors]).reset_index(drop=True)

    region_dummies = pd.get_dummies(
        sample["region"], prefix="region", drop_first=True, dtype=float)
    X = pd.concat([sample[regressors].astype(float), region_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(sample["log_real_price"], X).fit(cov_type="HC3")

    full_region_dummies = pd.get_dummies(
        master["region"], prefix="region", drop_first=True, dtype=float)
    for column in region_dummies.columns:
        if column not in full_region_dummies.columns:
            full_region_dummies[column] = 0.0
    full_region_dummies = full_region_dummies.reindex(
        columns=region_dummies.columns, fill_value=0.0)
    X_full = pd.concat([master[regressors].astype(
        float), full_region_dummies], axis=1)
    X_full = sm.add_constant(X_full, has_constant="add")

    fitted = master.copy()
    fitted["fair_value_log_real"] = fit.predict(X_full)
    fitted["fair_value_real"] = np.exp(fitted["fair_value_log_real"])
    fitted["fair_value_gap_log"] = fitted["log_real_price"] - \
        fitted["fair_value_log_real"]

    coefficients = pd.DataFrame(
        {
            "feature": fit.params.index,
            "coefficient": fit.params.values,
            "std_error": fit.bse.values,
            "p_value": fit.pvalues.values,
        }
    )
    vif = _vif_table(sample, regressors)
    diagnostics = pd.DataFrame(
        [
            {
                "n_obs": int(len(sample)),
                "r_squared": float(fit.rsquared),
                "adj_r_squared": float(fit.rsquared_adj),
                "durbin_watson": float(sm.stats.stattools.durbin_watson(fit.resid)),
                # Residual SD in log-price space; used for prediction-interval approximation.
                # sqrt(mse_resid) is the OLS RMSE on in-sample anchor observations.
                "resid_std": float(np.sqrt(fit.mse_resid)),
            }
        ]
    )
    diagnostics = diagnostics.merge(
        vif.assign(key=1),
        how="cross",
    ).drop(columns=["key"], errors="ignore")

    return FairValueModelResult(
        coefficients=coefficients,
        fitted_panel=fitted,
        diagnostics=diagnostics,
    )


def compute_p_star_confidence_interval(
    ols_result,
    X_new: pd.DataFrame | None = None,
    alpha: float = 0.10,
    *,
    resid_std: float | None = None,
) -> dict:
    """Return a prediction interval for P* (log-real-price space) at level (1-alpha).

    Two modes:
    1. Full mode   — ``ols_result`` is a fitted statsmodels OLS result and ``X_new``
                     is a single-row DataFrame of regressors (same columns as fit).
                     Uses ``get_prediction()`` for the exact per-observation SE.
    2. Approx mode — ``resid_std`` is supplied (or read from ``ols_result.mse_resid``).
                     Uses a 1.645-sigma band as an approximation of the 90% PI.
                     APPROXIMATION: ignores the hat-matrix leverage term; valid for
                     large n where residual variance dominates fit uncertainty.

    Returns
    -------
    dict with keys:
        p_star_point  : float  — predicted log P* (point)
        p_star_lower  : float  — lower bound of the prediction interval
        p_star_upper  : float  — upper bound of the prediction interval
        ci_level      : str    — e.g. "90%"
        mode          : str    — "get_prediction" | "resid_std_approx"
    """
    from scipy import stats as _sp_stats

    ci_level = f"{int(round((1.0 - alpha) * 100))}%"

    # ------------------------------------------------------------------ #
    # Full mode: statsmodels get_prediction                                #
    # ------------------------------------------------------------------ #
    if X_new is not None and hasattr(ols_result, "get_prediction"):
        try:
            pred = ols_result.get_prediction(X_new)
            summary = pred.summary_frame(alpha=alpha)
            point = float(summary["mean"].iloc[0])
            lower = float(summary["obs_ci_lower"].iloc[0])
            upper = float(summary["obs_ci_upper"].iloc[0])
            return {
                "p_star_point": point,
                "p_star_lower": lower,
                "p_star_upper": upper,
                "ci_level": ci_level,
                "mode": "get_prediction",
            }
        except Exception:
            pass  # fall through to approximation

    # ------------------------------------------------------------------ #
    # Approx mode: fixed ±z * resid_std band (90% CI → z = 1.645)        #
    # APPROXIMATION: SE_pred ≈ resid_std (leverage term omitted).         #
    # Valid when n is large relative to the number of parameters.          #
    # ------------------------------------------------------------------ #
    if resid_std is None:
        resid_std = float(np.sqrt(ols_result.mse_resid))

    # Point estimate: use the pre-computed "fair_value_log_real" if available,
    # otherwise predict from X_new.
    if X_new is not None and hasattr(ols_result, "predict"):
        try:
            point = float(ols_result.predict(X_new).iloc[0])
        except Exception:
            point = float("nan")
    else:
        point = float("nan")

    z = float(_sp_stats.norm.ppf(1.0 - alpha / 2.0))
    return {
        "p_star_point": point,
        "p_star_lower": point - z * resid_std,
        "p_star_upper": point + z * resid_std,
        "ci_level": ci_level,
        "mode": "resid_std_approx",
    }
