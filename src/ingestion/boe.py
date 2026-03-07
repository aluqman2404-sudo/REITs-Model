"""
Stage 2: Bank of England base rate ingestion.
"""

import pandas as pd
from config.settings import DATA_RAW


def fetch_boe_rate(save: bool = True) -> pd.DataFrame:
    """
    Download BoE official bank rate series and forward-fill to monthly frequency.

    Returns:
        DataFrame with columns: date, base_rate
    """
    # TODO (Stage 2): implement BoE rate download and monthly resampling
    raise NotImplementedError("fetch_boe_rate not yet implemented")
