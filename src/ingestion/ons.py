"""
Stage 2: ONS data ingestion.
Uses the ONS Generator API (more stable than the timeseries API).

Generator URL format:
  https://www.ons.gov.uk/generator?format=csv&uri=/...timeseries/{series}/{dataset}

CSV format returned:
  Several header rows (title, CDid, unit, release date, notes)
  Then:  "Time","Value"
         "Jan 2015","100.0"
"""

import io
import time
import requests
import pandas as pd
from datetime import datetime
from config.settings import DATA_RAW, PARAMS

_START_YEAR = PARAMS["project"]["base_year"]   # 2000
_REGIONS    = PARAMS["project"]["regions"]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# ONS Generator URLs — more stable than the timeseries API
_ONS_GENERATOR = "https://www.ons.gov.uk/generator?format=csv&uri={uri}"

_SERIES = {
    "cpi":             "/economy/inflationandpriceindices/timeseries/d7bt/mm23",
    "cpih":            "/economy/inflationandpriceindices/timeseries/l522/cpih01",
    # AWE whole economy — seasonally adjusted total pay (£/week)
    "earnings_weekly": "/employmentandlabourmarket/peopleinwork/earningsandworkinghours/timeseries/kab9/emp",
    # AWE whole economy — not seasonally adjusted (fallback)
    "earnings_median": "/employmentandlabourmarket/peopleinwork/earningsandworkinghours/timeseries/kai7/emp",
}


# ── Public API ───────────────────────────────────────────────────────────────

def fetch_ons_cpi(save: bool = True) -> pd.DataFrame:
    """
    Fetch monthly CPI index (all items, base 2015=100).
    Falls back to CPIH if CPI series unavailable.

    Returns DataFrame with columns: date, cpi
    """
    print("  Fetching ONS CPI (D7BT)...")
    try:
        df = _fetch_generator(_SERIES["cpi"])
    except Exception as e:
        print(f"  CPI failed ({e}), trying CPIH fallback...")
        df = _fetch_generator(_SERIES["cpih"])

    df = df.rename(columns={"value": "cpi"})
    df = df[df["date"].dt.year >= _START_YEAR].reset_index(drop=True)

    if save:
        path = DATA_RAW / f"ons_cpi_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {len(df):,} rows → {path.name}")

    return df


def fetch_ons_earnings(save: bool = True) -> pd.DataFrame:
    """
    Fetch ONS median weekly earnings (national, from EARN01).
    Regional earnings require the ASHE publication — handled separately.

    Returns DataFrame with columns: date, region, median_annual_earnings
    """
    print("  Fetching ONS earnings (EARN01 KAI7)...")
    time.sleep(1)  # avoid rate-limiting after CPI call

    try:
        df = _fetch_generator(_SERIES["earnings_weekly"], freq="quarterly")
    except Exception:
        time.sleep(2)
        df = _fetch_generator(_SERIES["earnings_median"], freq="quarterly")

    df["median_annual_earnings"] = df["value"] * 52
    df["region"] = "United Kingdom"
    df = df[df["date"].dt.year >= _START_YEAR]
    df = df[["date", "region", "median_annual_earnings"]].reset_index(drop=True)

    if save:
        path = DATA_RAW / f"ons_earnings_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {len(df):,} rows → {path.name}")
        print("  Note: national series only. Regional ASHE data → fetch_ons_earnings_regional()")

    return df


def fetch_ons_earnings_regional(save: bool = True) -> pd.DataFrame:
    """
    Download ASHE Table 7 regional annual earnings CSV from ONS bulk downloads.
    This is a separate, annual publication — interpolated to monthly in Stage 3.

    Returns DataFrame with columns: date, region, median_annual_earnings
    """
    print("  Fetching ONS ASHE regional earnings (Table 7)...")
    # Try multiple known ASHE Table 7 URLs (ONS updates these each autumn)
    ashe_urls = [
        "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/"
        "earningsandworkinghours/datasets/regionbyoccupation4digitsoc2010ashetable7/"
        "2024revised/table72024revised.csv",
        "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/"
        "earningsandworkinghours/datasets/regionbyoccupation4digitsoc2010ashetable7/"
        "2023provisional/table72023provisional.csv",
    ]
    for url in ashe_urls:
        try:
            r = requests.get(url, headers=_HEADERS, timeout=60)
            r.raise_for_status()
            df = _parse_ashe_regional(r.text)
            if save:
                path = DATA_RAW / f"ons_earnings_regional_{datetime.today():%Y%m}.csv"
                df.to_csv(path, index=False)
                print(f"  Saved {len(df):,} rows → {path.name}")
            return df
        except Exception as e:
            print(f"  ASHE URL failed: {e}")
            time.sleep(1)

    raise RuntimeError(
        "All ASHE regional earnings URLs failed.\n"
        "Download manually from: https://www.ons.gov.uk/employmentandlabourmarket/"
        "peopleinwork/earningsandworkinghours/datasets/"
        "regionbyoccupation4digitsoc2010ashetable7\n"
        "Save to data/raw/ons_earnings_regional_YYYYMM.csv"
    )


def fetch_ons_population(save: bool = True) -> pd.DataFrame:
    """
    Fetch ONS mid-year population estimates by region (annual).
    Interpolated to monthly in Stage 3.

    Returns DataFrame with columns: date, region, population
    """
    print("  Fetching ONS population estimates...")
    time.sleep(1)

    # Try multiple known population estimate URLs (mid2024 is latest as of 2025)
    pop_urls = [
        "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
        "populationandmigration/populationestimates/datasets/"
        "populationestimatesforukenglandandwalesscotlandandnorthernireland/"
        "mid2024/ukpopestimatesmid2024on2021geography.csv",
        "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
        "populationandmigration/populationestimates/datasets/"
        "populationestimatesforukenglandandwalesscotlandandnorthernireland/"
        "mid2023/ukpopestimatesmid2023on2021geography.csv",
    ]

    for url in pop_urls:
        try:
            time.sleep(3)  # ONS rate limiting — wait between requests
            r = requests.get(url, headers=_HEADERS, timeout=60)
            r.raise_for_status()
            df = _parse_population_csv(r.content)
            if save:
                path = DATA_RAW / f"ons_population_{datetime.today():%Y%m}.csv"
                df.to_csv(path, index=False)
                print(f"  Saved {len(df):,} rows → {path.name}")
            return df
        except Exception as e:
            print(f"  Population URL failed: {e}")
            time.sleep(1)

    raise RuntimeError(
        "Population download failed.\n"
        "Download manually from: https://www.ons.gov.uk/peoplepopulationandcommunity/"
        "populationandmigration/populationestimates\n"
        "Save to data/raw/ons_population_YYYYMM.csv"
    )


def fetch_ons_series(dataset_id: str, series_id: str, freq: str = "months", save: bool = True) -> pd.DataFrame:
    """Generic fetch for any ONS timeseries via the generator URL."""
    uri = f"/economy/timeseries/{series_id}/{dataset_id}"
    df = _fetch_generator(uri)
    if save:
        path = DATA_RAW / f"ons_{dataset_id}_{series_id}_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
    return df


# ── Internal helpers ─────────────────────────────────────────────────────────

def _fetch_generator(uri: str, freq: str = "monthly") -> pd.DataFrame:
    """
    Fetch a series from the ONS generator CSV endpoint.

    The ONS generator returns rows in this format (no header row for data):
      Metadata rows:  "Title","..."  "CDID","..."  etc.  (first 8 rows)
      Annual rows:    "1988","49.6"
      Quarterly rows: "1988 Q1","48.6"
      Monthly rows:   "1988 JAN","48.4"

    Args:
        uri:  ONS URI path (e.g. /economy/...timeseries/d7bt/mm23)
        freq: 'monthly', 'quarterly', or 'annual' — which rows to keep
    """
    url = _ONS_GENERATOR.format(uri=uri)
    r = requests.get(url, headers=_HEADERS, timeout=30)
    r.raise_for_status()

    import csv
    rows = list(csv.reader(r.text.splitlines()))

    # Skip metadata rows (Title, CDID, Source, PreUnit, Unit, Release, Next, Notes)
    _METADATA_KEYS = {"title", "cdid", "source dataset id", "preunit", "unit",
                      "release date", "next release", "important notes"}
    data_rows = []
    for row in rows:
        if len(row) < 2:
            continue
        if row[0].strip().lower() in _METADATA_KEYS:
            continue
        data_rows.append(row)

    if not data_rows:
        raise ValueError(f"No data rows found in ONS response for {uri}")

    records = []
    _MONTHS = {"JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"}

    for row in data_rows:
        date_str = row[0].strip()
        val_str  = row[1].strip()

        # Classify row type
        parts = date_str.upper().split()
        if len(parts) == 2 and parts[1] in _MONTHS:
            row_freq = "monthly"
        elif len(parts) == 2 and parts[1].startswith("Q"):
            row_freq = "quarterly"
        elif len(parts) == 1 and parts[0].isdigit():
            row_freq = "annual"
        else:
            continue

        if row_freq != freq:
            continue

        try:
            date  = _parse_ons_date(date_str, row_freq)
            value = float(val_str)
            records.append({"date": date, "value": value})
        except (ValueError, TypeError):
            continue

    if not records:
        raise ValueError(f"No {freq} rows found in ONS response for {uri}")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)[["date", "value"]]


def _parse_ons_date(date_str: str, freq: str) -> str:
    """Convert a single ONS date string to ISO format."""
    _MONTH_MAP = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    parts = date_str.upper().split()
    if freq == "monthly":
        year, month_abbr = parts
        return f"{year}-{_MONTH_MAP[month_abbr]}-01"
    elif freq == "quarterly":
        year, q = parts
        month = int(q[1]) * 3 - 2
        return f"{year}-{month:02d}-01"
    else:  # annual
        return f"{parts[0]}-01-01"


def _parse_ashe_regional(text: str) -> pd.DataFrame:
    """Parse ASHE Table 7 regional earnings CSV into long format."""
    raw = pd.read_csv(io.StringIO(text), skiprows=4, low_memory=False)
    region_col = raw.columns[0]
    year_cols  = [c for c in raw.columns[1:] if str(c).strip().isdigit()]

    df = raw[[region_col] + year_cols].copy()
    df = df.rename(columns={region_col: "region"})
    df = df[df["region"].isin(_REGIONS)]
    df = df.melt(id_vars="region", var_name="year", value_name="median_annual_earnings")
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-04-01")
    df["median_annual_earnings"] = pd.to_numeric(
        df["median_annual_earnings"].astype(str).str.replace(",", ""), errors="coerce"
    )
    df = df.dropna(subset=["median_annual_earnings"])
    df = df[df["date"].dt.year >= _START_YEAR]
    return df[["date", "region", "median_annual_earnings"]].sort_values(["region", "date"]).reset_index(drop=True)


def _parse_population_csv(content: bytes) -> pd.DataFrame:
    """Parse ONS population estimates CSV into long format."""
    import numpy as np
    for skip in range(4, 12):
        try:
            raw = pd.read_csv(io.BytesIO(content), skiprows=skip, low_memory=False)
            region_col = raw.columns[0]
            year_cols  = [c for c in raw.columns[1:] if str(c).strip().isdigit()]
            if len(year_cols) > 5:
                break
        except Exception:
            continue

    df = raw[[region_col] + year_cols].copy()
    df = df.rename(columns={region_col: "region"})
    df = df[df["region"].isin(_REGIONS)]
    df = df.melt(id_vars="region", var_name="year", value_name="population")
    df["date"]       = pd.to_datetime(df["year"].astype(str) + "-06-30")
    df["population"] = pd.to_numeric(
        df["population"].astype(str).str.replace(",", ""), errors="coerce"
    )
    df = df.dropna(subset=["population"])
    df = df[df["date"].dt.year >= _START_YEAR]
    return df[["date", "region", "population"]].sort_values(["region", "date"]).reset_index(drop=True)
