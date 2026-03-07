"""
Stage 2: HM Land Registry UK HPI ingestion.
Downloads the full monthly price paid dataset by region.

Source: HM Land Registry Open Data S3 bucket (no auth required)
Frequency: Monthly
"""

import io
import re
import requests
import pandas as pd
from datetime import datetime

from config.settings import DATA_RAW, PARAMS

_TARGET_REGIONS = set(PARAMS["project"]["regions"])

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Stable GOV.UK page listing all HPI downloads — scrape it for the current URL
_GOV_UK_HPI_PAGE = (
    "https://www.gov.uk/government/statistical-data-sets/"
    "uk-house-price-index-data-downloads-august-2023"
)

# Fallback: try to find the page via the collection index
_GOV_UK_HPI_COLLECTION = (
    "https://www.gov.uk/government/collections/uk-house-price-index-reports"
)


def _find_latest_url() -> tuple[str, str]:
    """
    Scrape the GOV.UK Land Registry HPI page to find the current full-file CSV URL.
    Returns (url, label).
    """
    from bs4 import BeautifulSoup

    # Try the direct data page first, then collection index
    for page_url in [_GOV_UK_HPI_PAGE, _GOV_UK_HPI_COLLECTION]:
        try:
            r = requests.get(page_url, headers=_HEADERS, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")

            # Find all links pointing to the S3 HPI full file
            for a in soup.find_all("a", href=True):
                href = a["href"]
                m = re.search(r"UK-HPI-full-file-(\d{4}-\d{2})\.csv", href, re.IGNORECASE)
                if m:
                    label = m.group(1)
                    # Ensure it's a full URL
                    if not href.startswith("http"):
                        href = "https://prod.publicdata.landregistry.gov.uk" \
                               ".s3-website-eu-west-1.amazonaws.com/" + href.lstrip("/")
                    return href, label
        except Exception:
            continue

    # Final fallback — the S3 path is deterministic; try the most likely recent months
    # (Land Registry publishes ~8 weeks after reference month)
    from dateutil.relativedelta import relativedelta
    from datetime import date
    today = date.today()
    for months_back in range(3, 9):
        candidate = today - relativedelta(months=months_back)
        url = (
            "https://prod.publicdata.landregistry.gov.uk"
            ".s3-website-eu-west-1.amazonaws.com"
            f"/UK-HPI-full-file-{candidate.year}-{candidate.month:02d}.csv"
        )
        try:
            r = requests.get(url, headers=_HEADERS, stream=True, timeout=10)
            if r.status_code == 200:
                r.close()
                return url, candidate.strftime("%Y-%m")
        except Exception:
            continue

    raise RuntimeError(
        "Could not locate Land Registry HPI full file automatically.\n"
        "Download manually from:\n"
        "  https://www.gov.uk/government/collections/uk-house-price-index-reports\n"
        "Save the 'UK HPI full file' CSV to:\n"
        f"  {DATA_RAW / 'land_registry_hpi_YYYYMM.csv'}"
    )


def fetch_land_registry(url: str | None = None, save: bool = True) -> pd.DataFrame:
    """
    Download and parse the HM Land Registry UK HPI full dataset.

    Returns DataFrame with columns: date, region, average_price, sales_volume
    """
    if url is None:
        url, label = _find_latest_url()
        print(f"  Latest file: {label}")
    else:
        label = "custom"

    print(f"  URL: {url}")
    response = requests.get(url, headers=_HEADERS, timeout=180)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text), low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # Find and standardise the region column
    region_col = next(
        (c for c in df.columns if c in ("regionname", "region_name", "areaofresidence")),
        None
    )
    if region_col is None:
        raise KeyError(f"Region column not found. Columns: {list(df.columns)}")

    df = df.rename(columns={region_col: "region"})
    df = df[df["region"].isin(_TARGET_REGIONS)].copy()

    # Standardise price/volume column names
    rename = {
        "averageprice": "average_price",
        "salesvolume":  "sales_volume",
    }
    df = df.rename(columns=rename)

    keep = [c for c in ["date", "region", "average_price", "index", "sales_volume"] if c in df.columns]
    df = df[keep].sort_values(["region", "date"]).reset_index(drop=True)

    if save:
        out = DATA_RAW / f"land_registry_hpi_{datetime.today():%Y%m}.csv"
        df.to_csv(out, index=False)
        print(f"  Saved {len(df):,} rows → {out.name}")

    return df
