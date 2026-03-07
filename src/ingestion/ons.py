"""
Stage 2: ONS data ingestion.
Pulls income, CPI, and population series via the ONS API.
"""

import pandas as pd
import requests
from config.settings import DATA_RAW, ONS_API_KEY

_ONS_BASE = "https://api.ons.gov.uk/v1"


def fetch_ons_series(dataset_id: str, edition: str = "time-series", save: bool = True) -> pd.DataFrame:
    """
    Fetch a named ONS dataset via the ONS API.

    Args:
        dataset_id: ONS dataset identifier (e.g. 'EARN01', 'MM23')
        edition:    API edition string

    Returns:
        DataFrame with columns: date, value, series_id
    """
    # TODO (Stage 2): implement paginated ONS API pull
    raise NotImplementedError("fetch_ons_series not yet implemented")
