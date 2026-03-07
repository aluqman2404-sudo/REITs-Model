"""
Stage 3: Merge all ingested series into a single master analytical dataset.
Aligns all sources to a common monthly date index.
"""

import pandas as pd
from config.settings import DATA_PROCESSED


def build_master_dataset() -> pd.DataFrame:
    """
    Load all cleaned raw series and merge on (date, region).

    Returns:
        master DataFrame with columns:
            date, region, house_price_real, income_real, base_rate,
            cpi, population, price_income_ratio, affordability_index
    """
    # TODO (Stage 3): load each cleaned series and merge
    raise NotImplementedError("build_master_dataset not yet implemented")
