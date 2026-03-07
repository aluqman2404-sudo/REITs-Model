"""
Stage 2: ONS data ingestion.
Pulls CPI, regional earnings, and population via the ONS API and bulk downloads.

ONS API base: https://api.ons.gov.uk/v1
Format: GET /datasets/{dataset_id}/timeseries/{series_id}/data
Returns JSON with keys: months, quarters, years
"""

import io
import requests
import pandas as pd
from datetime import datetime
from config.settings import DATA_RAW, PARAMS

_ONS_API = "https://api.ons.gov.uk/v1"
_START_YEAR = PARAMS["project"]["base_year"]   # 2000
_REGIONS    = PARAMS["project"]["regions"]

# ── ONS series identifiers ───────────────────────────────────────────────────
# CPI all items index (base 2015=100) — dataset MM23
_CPI_DATASET  = "mm23"
_CPI_SERIES   = "d7bt"

# CPIH (used as fallback if CPI series unavailable)
_CPIH_DATASET = "cpih01"
_CPIH_SERIES  = "l522"


# ── Public API ───────────────────────────────────────────────────────────────

def fetch_ons_cpi(save: bool = True) -> pd.DataFrame:
    """
    Fetch monthly CPI index (all items, base 2015=100) from ONS API.

    Returns:
        DataFrame with columns: date, cpi
    """
    print("Fetching ONS CPI (MM23 D7BT)...")
    df = _fetch_timeseries(_CPI_DATASET, _CPI_SERIES, freq="months")

    df = df.rename(columns={"value": "cpi"})
    df = df[df["date"].dt.year >= _START_YEAR].copy()

    if save:
        path = DATA_RAW / f"ons_cpi_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {len(df):,} rows → {path}")

    return df


def fetch_ons_earnings(save: bool = True) -> pd.DataFrame:
    """
    Fetch ONS median annual earnings by region (ASHE Table 7).
    Published annually — will be interpolated to monthly in Stage 3.

    The ASHE regional earnings are not in the main ONS API; this function
    downloads the published CSV from the ONS bulk download service.

    Returns:
        DataFrame with columns: date, region, median_annual_earnings
    """
    print("Fetching ONS regional earnings (ASHE Table 7)...")

    # ONS ASHE Table 7 — median gross annual earnings by region
    # Published each autumn for the previous tax year
    url = (
        "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/"
        "peopleinwork/earningsandworkinghours/datasets/"
        "regionbyoccupation4digitsoc2010ashetable7/"
        "2023provisional/table72023provisional.csv"
    )

    try:
        df = _download_ashe_regional(url)
    except Exception as e:
        print(f"ASHE bulk download failed ({e}). Falling back to national EARN01 series.")
        df = _fetch_earn01_national()

    if save:
        path = DATA_RAW / f"ons_earnings_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {len(df):,} rows → {path}")

    return df


def fetch_ons_population(save: bool = True) -> pd.DataFrame:
    """
    Fetch ONS mid-year population estimates by region (annual).
    Will be interpolated to monthly in Stage 3.

    Returns:
        DataFrame with columns: date, region, population
    """
    print("Fetching ONS regional population estimates...")

    # ONS mid-year population estimates — published annually
    url = (
        "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
        "populationandmigration/populationestimates/datasets/"
        "populationestimatesforukenglandandwalesscotlandandnorthernireland/"
        "mid2022/ukpopestimatesmid2022on2021geography.csv"
    )

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        df = _parse_population_csv(response.content)
    except Exception as e:
        print(f"Population download failed ({e}). Creating placeholder.")
        df = _population_placeholder()

    if save:
        path = DATA_RAW / f"ons_population_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {len(df):,} rows → {path}")

    return df


def fetch_ons_series(dataset_id: str, series_id: str, freq: str = "months", save: bool = True) -> pd.DataFrame:
    """
    Generic ONS API timeseries fetch. Useful for pulling any additional series.

    Args:
        dataset_id: ONS dataset ID (e.g. 'mm23')
        series_id:  ONS series ID (e.g. 'd7bt')
        freq:       'months', 'quarters', or 'years'

    Returns:
        DataFrame with columns: date, value
    """
    df = _fetch_timeseries(dataset_id, series_id, freq=freq)

    if save:
        path = DATA_RAW / f"ons_{dataset_id}_{series_id}_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {len(df):,} rows → {path}")

    return df


# ── Internal helpers ─────────────────────────────────────────────────────────

def _fetch_timeseries(dataset_id: str, series_id: str, freq: str = "months") -> pd.DataFrame:
    """Hit the ONS API timeseries endpoint and return a clean DataFrame."""
    url = f"{_ONS_API}/datasets/{dataset_id}/timeseries/{series_id}/data"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    records = data.get(freq, [])
    if not records:
        raise ValueError(f"No '{freq}' data returned for {dataset_id}/{series_id}")

    rows = []
    for obs in records:
        raw_date = obs.get("date", "")
        value    = obs.get("value", "")
        if value in ("", ".", None):
            continue
        try:
            rows.append({"date": _parse_ons_date(raw_date, freq), "value": float(value)})
        except (ValueError, TypeError):
            continue

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _parse_ons_date(raw: str, freq: str) -> str:
    """Convert ONS date strings like '2020 JAN', '2020 Q1', '2020' to ISO format."""
    raw = raw.strip()
    if freq == "months":
        return pd.to_datetime(raw, format="%Y %b").strftime("%Y-%m-%d")
    elif freq == "quarters":
        year, q = raw.split(" Q")
        month = int(q) * 3 - 2
        return f"{year}-{month:02d}-01"
    elif freq == "years":
        return f"{raw}-01-01"
    return raw


def _download_ashe_regional(url: str) -> pd.DataFrame:
    """
    Download and parse an ASHE regional earnings CSV.
    Returns long-format DataFrame: date, region, median_annual_earnings
    """
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # ASHE files are typically wide-format — region rows, year columns
    raw = pd.read_csv(io.StringIO(response.text), skiprows=4, low_memory=False)

    # Melt to long format — exact parsing depends on file structure
    # This is a best-effort parse; may need adjustment per ASHE file version
    region_col = raw.columns[0]
    year_cols  = [c for c in raw.columns if str(c).strip().isdigit()]

    df = raw[[region_col] + year_cols].copy()
    df = df.rename(columns={region_col: "region"})
    df = df[df["region"].isin(_REGIONS)]
    df = df.melt(id_vars="region", var_name="year", value_name="median_annual_earnings")
    df["date"]                   = pd.to_datetime(df["year"].astype(str) + "-04-01")  # tax year reference
    df["median_annual_earnings"] = pd.to_numeric(df["median_annual_earnings"], errors="coerce")
    df = df.dropna(subset=["median_annual_earnings"])
    df = df[df["date"].dt.year >= _START_YEAR]

    return df[["date", "region", "median_annual_earnings"]].sort_values(["region", "date"]).reset_index(drop=True)


def _fetch_earn01_national() -> pd.DataFrame:
    """
    Fallback: fetch national median weekly earnings from ONS API (EARN01).
    Returns DataFrame with a single 'United Kingdom' region entry.
    """
    # EARN01 — median weekly earnings, whole economy (series KAI7)
    df = _fetch_timeseries("earn01", "kai7", freq="quarters")
    df["region"]                  = "United Kingdom"
    df["median_annual_earnings"]  = df["value"] * 52
    df["date"]                    = df["date"].dt.to_period("Q").dt.to_timestamp()
    return df[["date", "region", "median_annual_earnings"]]


def _parse_population_csv(content: bytes) -> pd.DataFrame:
    """Parse the ONS mid-year population estimates CSV into long format."""
    raw = pd.read_csv(io.BytesIO(content), skiprows=7, low_memory=False)

    region_col  = raw.columns[0]
    year_cols   = [c for c in raw.columns if str(c).strip().isdigit()]

    df = raw[[region_col] + year_cols].copy()
    df = df.rename(columns={region_col: "region"})
    df = df[df["region"].isin(_REGIONS)]
    df = df.melt(id_vars="region", var_name="year", value_name="population")
    df["date"]       = pd.to_datetime(df["year"].astype(str) + "-06-30")  # mid-year
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna(subset=["population"])
    df = df[df["date"].dt.year >= _START_YEAR]

    return df[["date", "region", "population"]].sort_values(["region", "date"]).reset_index(drop=True)


def _population_placeholder() -> pd.DataFrame:
    """Minimal placeholder so the pipeline doesn't break if download fails."""
    import numpy as np
    dates   = pd.date_range("2000-06-30", "2024-06-30", freq="YE")
    rows    = []
    for region in _REGIONS:
        for d in dates:
            rows.append({"date": d, "region": region, "population": np.nan})
    return pd.DataFrame(rows)
