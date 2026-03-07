"""
Stage 4: OLS calibration.
Fits the price growth regression and extracts drift/diffusion parameters
for use by the SDE simulation.
"""

import pandas as pd
import statsmodels.api as sm
from config.settings import save_parameters, load_parameters


def calibrate_ols(df: pd.DataFrame, dep_var: str = "d_log_price") -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS regression of house price growth on economic drivers.

    Args:
        df:       master dataset from Stage 3
        dep_var:  dependent variable column name

    Returns:
        Fitted statsmodels OLS result; also writes parameters to config/parameters.json
    """
    # TODO (Stage 4): specify regressors, fit OLS, run diagnostics, save params
    raise NotImplementedError("calibrate_ols not yet implemented")


def run_diagnostics(result) -> dict:
    """
    Run Durbin-Watson, White test, VIF, and residual plots on an OLS result.

    Returns dict of diagnostic statistics.
    """
    # TODO (Stage 4): implement diagnostics
    raise NotImplementedError("run_diagnostics not yet implemented")
