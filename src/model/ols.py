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
    sign_constraints: dict[str, float] | None = None,
) -> FairValueModelResult:
    """Fit a semi-structural fair-value model on annual anchor observations.

    Parameters
    ----------
    sign_constraints : dict mapping regressor name to required sign (+1 or -1).
        If supplied and an unconstrained OLS coefficient has the wrong sign,
        the model is re-estimated with a constrained approach: the offending
        regressor is fixed at a small value of the correct sign (1e-6) while
        the remaining coefficients are re-fitted via OLS on the residual.
        This preserves the linear prediction while enforcing the economic prior.
        Example: ``{"log_pti_ratio": +1, "mortgage_rate": -1}``
    """
    sample = master[master["date"].dt.month.isin(anchor_months)].copy()
    sample = sample.dropna(
        subset=["log_real_price", *regressors]).reset_index(drop=True)

    region_dummies = pd.get_dummies(
        sample["region"], prefix="region", drop_first=True, dtype=float)
    X = pd.concat([sample[regressors].astype(float), region_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(sample["log_real_price"], X).fit(cov_type="HC3")

    # Apply sign constraints: if any constrained regressor has the wrong sign,
    # fix it at a small value of the correct sign and refit the remaining
    # free regressors on the residual.
    if sign_constraints:
        import warnings as _w
        params_adj = dict(fit.params)
        y_adj = sample["log_real_price"].values.copy()
        for reg, required_sign in sign_constraints.items():
            coef = float(fit.params.get(reg, 0.0))
            if np.isfinite(coef) and (required_sign > 0 and coef < 0) or (required_sign < 0 and coef > 0):
                _w.warn(
                    f"fit_structural_fair_value: {reg} coefficient {coef:+.4f} violates "
                    f"sign constraint (required: {'+' if required_sign>0 else '-'}). "
                    "Applying floor: regressor fixed at sign-floor, remaining coefficients re-fitted.",
                    UserWarning, stacklevel=2
                )
                floor_val = required_sign * 1e-6
                params_adj[reg] = floor_val
                # Partial out the fixed regressor from y
                y_adj = y_adj - float(floor_val) * X[reg].values

        # Re-fit on residual with fixed regressors partialled out
        free_cols = [c for c in X.columns if c not in sign_constraints or
                     float(params_adj.get(c, 0.0)) != sign_constraints.get(c, None) * 1e-6]
        # Actually: re-fit ALL columns except those that were fixed (sign-violated ones)
        fixed_regs = [r for r, s in sign_constraints.items()
                      if r in fit.params.index and
                      ((s > 0 and float(fit.params[r]) < 0) or (s < 0 and float(fit.params[r]) > 0))]
        if fixed_regs:
            free_cols = [c for c in X.columns if c not in fixed_regs]
            X_free = X[free_cols]
            y_residual = pd.Series(y_adj, index=sample.index)
            fit_free = sm.OLS(y_residual, X_free).fit(cov_type="HC3")
            # Reconstruct the full params series
            full_params = {}
            for c in X.columns:
                if c in fixed_regs:
                    full_params[c] = params_adj[c]
                else:
                    full_params[c] = float(fit_free.params.get(c, 0.0))
            # Build a fake fit object wrapper to maintain downstream compatibility
            # We keep fit (the original) for diagnostics but override predictions using full_params.
            import types
            fit_constrained = types.SimpleNamespace()
            fit_constrained.params = pd.Series(full_params)
            fit_constrained.bse = pd.Series({c: float(fit.bse.get(c, np.nan)) for c in X.columns})
            fit_constrained.pvalues = pd.Series({c: float(fit.pvalues.get(c, np.nan)) for c in X.columns})
            fit_constrained.rsquared = float(fit.rsquared)
            fit_constrained.rsquared_adj = float(fit.rsquared_adj)
            fit_constrained.mse_resid = float(fit.mse_resid)
            fit_constrained.resid = fit.resid
            # Predict using constrained params
            param_vec = pd.Series(full_params)
            X_aligned = X.reindex(columns=param_vec.index, fill_value=0.0)
            fit_constrained.predict = lambda Xp: Xp.reindex(columns=param_vec.index, fill_value=0.0) @ param_vec
            fit = fit_constrained

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
