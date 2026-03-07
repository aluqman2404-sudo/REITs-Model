"""
Stage 2: HM Land Registry UK HPI ingestion.
Downloads the full monthly price paid dataset by region.

Source: HM Land Registry Open Data S3 bucket (no auth required)
Frequency: Monthly
Coverage: England & Wales from Jan 1995; Scotland/NI added later
"""

import io
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from config.settings import DATA_RAW, PARAMS

# Regions to keep (ONS standard + Scotland/NI from separate series)
_TARGET_REGIONS = set(PARAMS["project"]["regions"])

_S3_BASE = (
    "http://prod.publicdata.landregistry.gov.uk"
    ".s3-website-eu-west-1.amazonaws.com"
)


def _latest_hpi_url() -> str:
    """
    Try months backwards from today until a valid CSV URL is found.
    The Land Registry typically publishes with a ~6-week lag.
    """
    today = datetime.today()
    for months_back in range(2, 8):
        candidate = today - timedelta(days=30 * months_back)
        url = f"{_S3_BASE}/UK-HPI-full-file-{candidate.year}-{candidate.month:02d}.csv"
        try:
            r = requests.head(url, timeout=10)
            if r.status_code == 200:
                return url
        except requests.RequestException:
            continue
    raise RuntimeError("Could not find a valid Land Registry HPI file in the last 8 months.")


def fetch_land_registry(url: str | None = None, save: bool = True) -> pd.DataFrame:
    """
    Download and parse the HM Land Registry UK HPI full dataset.

    Args:
        url:  Override the auto-detected URL (optional)
        save: If True, saves raw CSV to data/raw/

    Returns:
        DataFrame with columns:
            date, region, average_price, index, sales_volume
    """
    if url is None:
        url = _latest_hpi_url()

    print(f"Downloading Land Registry HPI from:\n  {url}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text), low_memory=False)

    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Parse date
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    # Keep only region-level rows (drop national aggregate)
    region_col = _detect_region_col(df)
    df = df.rename(columns={region_col: "region"})
    df = df[df["region"].isin(_TARGET_REGIONS)].copy()

    # Select and rename key columns
    col_map = {
        "averageprice":   "average_price",
        "average_price":  "average_price",
        "index":          "index",
        "salesvolume":    "sales_volume",
        "sales_volume":   "sales_volume",
    }
    keep = ["date", "region"] + [c for c in df.columns if c in col_map]
    df = df[keep].rename(columns=col_map)
    df = df.sort_values(["region", "date"]).reset_index(drop=True)

    if save:
        out_path = DATA_RAW / f"land_registry_hpi_{datetime.today():%Y%m}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df):,} rows → {out_path}")

    return df


def _detect_region_col(df: pd.DataFrame) -> str:
    """Find the region column regardless of exact naming in the CSV."""
    for candidate in ["regionname", "region_name", "areaofresidence", "name"]:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Cannot find region column. Available: {list(df.columns)}")
