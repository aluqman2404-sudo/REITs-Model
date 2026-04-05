"""Canonical Stage 5 simulation utilities used by the app and tests."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.core.config import load_config


def _coerce_mean_path(long_run_mean: float | Iterable[float], n_months: int) -> np.ndarray:
    """Expand a scalar or iterable fair-value path to simulation length."""
    if np.isscalar(long_run_mean):
        return np.full(n_months + 1, float(long_run_mean), dtype=float)

    mean_path = np.asarray(list(long_run_mean), dtype=float)
    if len(mean_path) == n_months:
        mean_path = np.insert(mean_path, 0, mean_path[0])
    if len(mean_path) != n_months + 1:
        raise ValueError(
            f"long_run_mean must have length {n_months + 1} (or {n_months} without t0)"
        )
    return mean_path


def run_monte_carlo(
    start_price: float,
    n_months: int = 60,
    n_paths: int | None = None,
    params: dict | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Run log-OU Monte Carlo simulations for regional house prices.

    The interactive app uses a log-price OU process:
      d log(P_t) = kappa * (log(P*_t) - log(P_t)) dt + drift dt + sigma dW_t

    This preserves positivity, is numerically stable for UI use, and maps
    naturally to the Stage 4 calibrated `kappa`, `sigma`, and `mu_equilibrium`
    outputs.
    """
    config = load_config()
    n_paths = n_paths or config.controls.default_paths_interactive
    params = params or {}
    seed = config.controls.random_seed if seed is None else seed

    if start_price <= 0:
        raise ValueError("start_price must be positive")
    if n_months <= 0 or n_paths <= 0:
        raise ValueError("n_months and n_paths must be positive")

    dt = float(params.get("dt", config.model.dt))
    kappa = float(params.get(
        "kappa", config.model.mean_reversion_theta or 0.12))
    sigma = float(params.get("sigma", config.model.diffusion_sigma or 0.08))
    drift_annual = float(params.get("drift_annual", params.get(
        "gamma_annual", config.model.drift_mu or 0.0)))
    long_run_mean = params.get(
        "long_run_mean", params.get("mu_equilibrium", start_price))
    mean_path = _coerce_mean_path(long_run_mean, n_months)

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_months, n_paths))
    log_paths = np.empty((n_months + 1, n_paths), dtype=float)
    log_paths[0, :] = np.log(start_price)
    drift_step = drift_annual * dt
    sigma_step = sigma * np.sqrt(dt)

    for month in range(1, n_months + 1):
        current_log_price = log_paths[month - 1, :]
        target_log_price = np.log(np.maximum(mean_path[month], 1.0))
        log_paths[month, :] = (
            current_log_price +
            kappa * (target_log_price - current_log_price) * dt +
            drift_step +
            sigma_step * shocks[month - 1, :]
        )

    prices = np.exp(log_paths)
    columns = [f"path_{idx + 1}" for idx in range(n_paths)]
    index = pd.RangeIndex(start=0, stop=n_months + 1, step=1, name="month")
    return pd.DataFrame(prices, index=index, columns=columns)


def summarize_paths(
    paths: pd.DataFrame,
    percentiles: Iterable[int] = (10, 25, 50, 75, 90),
) -> pd.DataFrame:
    """Return per-month percentiles and mean across simulation paths."""
    pct_values = np.percentile(paths.to_numpy(), list(percentiles), axis=1).T
    summary = pd.DataFrame(pct_values, index=paths.index, columns=[
                           f"p{p}" for p in percentiles])
    summary["mean"] = paths.mean(axis=1).values
    return summary
