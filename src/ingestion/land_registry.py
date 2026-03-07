"""
Stage 2: HM Land Registry data ingestion.
Downloads the UK HPI monthly price paid data by region.
"""

import pandas as pd
import requests
from config.settings import DATA_RAW, PARAMS

# HM Land Registry UK HPI — average price by region
_HPI_URL = (
    "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    "/UK-HPI-full-file-2024-08.csv"
)


def fetch_land_registry(url: str = _HPI_URL, save: bool = True) -> pd.DataFrame:
    """
    Download and parse the HM Land Registry UK HPI full dataset.

    Returns a DataFrame with columns:
        date, region, average_price, index, sales_volume
    """
    # TODO (Stage 2): implement download, parse, and regional filter
    raise NotImplementedError("fetch_land_registry not yet implemented")
