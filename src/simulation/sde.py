"""
Stage 5: SDE simulation using Euler-Maruyama discretisation.
Default process: Ornstein-Uhlenbeck (mean-reverting).
"""

import numpy as np
import pandas as pd
from config.settings import load_parameters


def run_monte_carlo(
    start_price: float,
    n_months: int = 60,
    n_paths: int = 10_000,
    params: dict | None = None,
) -> pd.DataFrame:
    """
    Run N Monte Carlo simulations of the OU process.

    Args:
        start_price: current house price (real terms)
        n_months:    forecast horizon in months
        n_paths:     number of simulation paths
        params:      override model params (defaults to config/parameters.json)

    Returns:
        DataFrame of shape (n_months, n_paths) — each column is one price path.
        Also returns a summary DataFrame with 10th / 50th / 90th percentiles.
    """
    # TODO (Stage 5): implement OU Euler-Maruyama simulation
    raise NotImplementedError("run_monte_carlo not yet implemented")
