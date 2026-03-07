"""
Stage 3: Data quality reporting.
Produces missingness stats, outlier flags, and time series plots.
"""

import pandas as pd
import matplotlib.pyplot as plt


def data_quality_report(df: pd.DataFrame, outlier_sigma: float = 3.0) -> dict:
    """
    Generate a data quality report for the master dataset.

    Returns a dict with:
        - missingness: % missing per column
        - outliers: rows flagged as > outlier_sigma std devs from mean
        - summary: descriptive stats
    """
    # TODO (Stage 3): implement missingness, outlier detection, and plots
    raise NotImplementedError("data_quality_report not yet implemented")
