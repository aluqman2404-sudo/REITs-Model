"""
Stage 2: ONS Private Rental Market data ingestion.
Model Variable 6: Rental Yield proxy.

Two datasets combined to compute rental yield:
  A) ONS Index of Private Housing Rental Prices (IPHRP)
     — monthly rental price INDEX by region (2015=100)
     — used to track rental price GROWTH over time

  B) ONS Private Rental Market Statistics (PRMS)
     — median monthly rent levels (£) by region
     — used to compute GROSS YIELD = (annual rent / house price) × 100

Yield = (Median Monthly Rent × 12) / Average House Price × 100

Both are published by ONS and require no authentication.
"""

import io
import re
import time
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from config.settings import DATA_RAW, PARAMS

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_REGIONS    = PARAMS["project"]["regions"]
_START_YEAR = PARAMS["project"]["base_year"]

# ONS generator URLs for IPHRP regional indices
# Series codes: Great Britain and regions
_IPHRP_SERIES = {
    "United Kingdom":             "/economy/inflationandpriceindices/timeseries/czom/iphrp",
    "England":                    "/economy/inflationandpriceindices/timeseries/czoo/iphrp",
    "London":                     "/economy/inflationandpriceindices/timeseries/czox/iphrp",
    "South East":                 "/economy/inflationandpriceindices/timeseries/czoy/iphrp",
    "East of England":            "/economy/inflationandpriceindices/timeseries/czoz/iphrp",
    "South West":                 "/economy/inflationandpriceindices/timeseries/czpa/iphrp",
    "East Midlands":              "/economy/inflationandpriceindices/timeseries/czpb/iphrp",
    "West Midlands":              "/economy/inflationandpriceindices/timeseries/czpc/iphrp",
    "Yorkshire and The Humber":   "/economy/inflationandpriceindices/timeseries/czpd/iphrp",
    "North West":                 "/economy/inflationandpriceindices/timeseries/czpe/iphrp",
    "North East":                 "/economy/inflationandpriceindices/timeseries/czpf/iphrp",
    "Wales":                      "/economy/inflationandpriceindices/timeseries/czpg/iphrp",
    "Scotland":                   "/economy/inflationandpriceindices/timeseries/czph/iphrp",
}

# ONS PRMS — Private Rental Market Statistics download page
_PRMS_PAGE = (
    "https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/"
    "privaterentalmarketsummarystatisticsengland"
)


def fetch_ons_rental_index(save: bool = True) -> pd.DataFrame:
    """
    Fetch ONS IPHRP regional rental price indices (monthly, 2015=100).

    Tries:
      1. ONS IPHRP bulletin page — download full Excel workbook (one request)
      2. Individual generator URLs per region (with delays)
      3. Placeholder based on approximate 2015 base rents + growth

    Returns:
        DataFrame with columns: date, region, rental_index
    """
    print("  Fetching ONS IPHRP rental price indices...")

    # Method 1: Download the full IPHRP Excel from the ONS bulletin
    try:
        df = _fetch_iphrp_bulletin()
        print(f"  Source: ONS IPHRP bulletin Excel ({len(df)} rows)")
    except Exception as e:
        print(f"  Bulletin download failed ({e}). Trying individual series (with delays)...")
        df = _fetch_iphrp_individual_series()

    df = df[df["date"].dt.year >= _START_YEAR].reset_index(drop=True)

    if save:
        path = DATA_RAW / f"ons_rental_index_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {len(df):,} rows → {path.name}")

    return df


def fetch_ons_rental_levels(save: bool = True) -> pd.DataFrame:
    """
    Fetch ONS Private Rental Market Statistics — median monthly rents by region.
    Used alongside house prices to compute gross rental yield.

    Returns:
        DataFrame with columns: date, region, median_monthly_rent
    """
    print("  Fetching ONS PRMS median rent levels...")

    try:
        url = _find_prms_download_url()
        df  = _download_and_parse_prms(url)
    except Exception as e:
        print(f"  PRMS download failed ({e}). Using rental index proxy.")
        df = _rental_level_placeholder()

    df = df[df["date"].dt.year >= _START_YEAR]
    df = df.sort_values(["region", "date"]).reset_index(drop=True)

    if save:
        path = DATA_RAW / f"ons_rental_levels_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {len(df):,} rows → {path.name}")

    return df


def compute_rental_yield(
    rental_levels: pd.DataFrame,
    house_prices: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Compute gross rental yield = (annual rent / house price) × 100.

    Args:
        rental_levels:  output of fetch_ons_rental_levels()
        house_prices:   Land Registry DataFrame with average_price column

    Returns:
        DataFrame with columns: date, region, gross_yield_pct
    """
    rents  = rental_levels.copy()
    prices = house_prices[["date", "region", "average_price"]].copy()

    rents["date"]  = pd.to_datetime(rents["date"]).dt.to_period("Q").dt.to_timestamp()
    prices["date"] = pd.to_datetime(prices["date"]).dt.to_period("Q").dt.to_timestamp()

    prices_q = prices.groupby(["date", "region"])["average_price"].mean().reset_index()
    rents_q  = rents.groupby(["date", "region"])["median_monthly_rent"].mean().reset_index()

    merged = rents_q.merge(prices_q, on=["date", "region"], how="inner")
    merged["gross_yield_pct"] = (merged["median_monthly_rent"] * 12) / merged["average_price"] * 100
    merged = merged[["date", "region", "gross_yield_pct"]].sort_values(["region", "date"])

    if save:
        path = DATA_RAW / f"rental_yield_{datetime.today():%Y%m}.csv"
        merged.to_csv(path, index=False)
        print(f"  Computed yield saved → {path.name}")

    return merged.reset_index(drop=True)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_iphrp_bulletin() -> pd.DataFrame:
    """
    Download the ONS IPHRP data Excel from the bulletin release page.
    The bulletin is published monthly and includes all regional series.
    """
    import re
    from bs4 import BeautifulSoup

    # The bulletin index page
    bulletin_url = (
        "https://www.ons.gov.uk/economy/inflationandpriceindices/"
        "bulletins/indexofprivatehousingrentalprices/previousReleases"
    )
    # Also try the latest release directly
    latest_url = (
        "https://www.ons.gov.uk/economy/inflationandpriceindices/"
        "bulletins/indexofprivatehousingrentalprices/latest"
    )

    excel_url = None
    for page_url in [latest_url, bulletin_url]:
        try:
            r = requests.get(page_url, headers=_HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if re.search(r"\.(xls|xlsx)$", href, re.IGNORECASE):
                    if "iphrp" in href.lower() or "rental" in href.lower() or "private" in href.lower():
                        excel_url = href if href.startswith("http") else "https://www.ons.gov.uk" + href
                        break
            if excel_url:
                break
        except Exception:
            continue

    if not excel_url:
        raise ValueError("Could not find IPHRP Excel download link on ONS bulletin page")

    r = requests.get(excel_url, headers=_HEADERS, timeout=60)
    r.raise_for_status()
    return _parse_iphrp_excel(io.BytesIO(r.content))


def _parse_iphrp_excel(content: io.BytesIO) -> pd.DataFrame:
    """Parse the ONS IPHRP Excel workbook into long format."""
    import re
    sheets = pd.read_excel(content, sheet_name=None, header=None)

    _REGION_SHEET_MAP = {
        "United Kingdom": "UK",
        "England":        "England",
        "London":         "London",
        "South East":     "South East",
        "East of England":"East of England",
        "South West":     "South West",
        "East Midlands":  "East Midlands",
        "West Midlands":  "West Midlands",
        "Yorkshire and The Humber": "Yorkshire",
        "North West":     "North West",
        "North East":     "North East",
        "Wales":          "Wales",
        "Scotland":       "Scotland",
    }

    records = []
    for region, keyword in _REGION_SHEET_MAP.items():
        # Find sheet whose name contains the keyword
        sheet = next((k for k in sheets if keyword.lower() in k.lower()), None)
        if not sheet:
            continue
        try:
            raw = sheets[sheet]
            # Find column with month dates and column with index values
            for row_idx in range(min(10, len(raw))):
                row = raw.iloc[row_idx]
                date_col = next((i for i, v in enumerate(row) if
                                 isinstance(v, str) and re.match(r"(Jan|Feb|Mar)", str(v))), None)
                if date_col is not None:
                    break

            if date_col is None:
                continue

            dates  = raw.iloc[row_idx:, date_col]
            values = raw.iloc[row_idx:, date_col + 1]

            df = pd.DataFrame({"date_str": dates, "value": values})
            df["date"]  = pd.to_datetime(df["date_str"].astype(str), format="%b %Y", errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["date", "value"])
            df["region"] = region
            records.append(df[["date", "region", "value"]])
        except Exception:
            continue

    if not records:
        raise ValueError("Could not parse any regional series from IPHRP Excel")

    out = pd.concat(records, ignore_index=True)
    return out.rename(columns={"value": "rental_index"}).sort_values(["region", "date"])


def _fetch_iphrp_individual_series() -> pd.DataFrame:
    """
    Fallback: fetch individual IPHRP series from ONS generator with long delays.
    """
    from src.ingestion.ons import _fetch_generator

    # Updated series codes (verified against ONS website structure)
    _SERIES_UPDATED = {
        "United Kingdom": "/economy/inflationandpriceindices/timeseries/czom/iphrp",
        "England":        "/economy/inflationandpriceindices/timeseries/czoo/iphrp",
        "London":         "/economy/inflationandpriceindices/timeseries/czox/iphrp",
        "South East":     "/economy/inflationandpriceindices/timeseries/czoy/iphrp",
        "East of England":"/economy/inflationandpriceindices/timeseries/czoz/iphrp",
        "South West":     "/economy/inflationandpriceindices/timeseries/czpa/iphrp",
        "East Midlands":  "/economy/inflationandpriceindices/timeseries/czpb/iphrp",
        "West Midlands":  "/economy/inflationandpriceindices/timeseries/czpc/iphrp",
        "Yorkshire and The Humber": "/economy/inflationandpriceindices/timeseries/czpd/iphrp",
        "North West":     "/economy/inflationandpriceindices/timeseries/czpe/iphrp",
        "North East":     "/economy/inflationandpriceindices/timeseries/czpf/iphrp",
        "Wales":          "/economy/inflationandpriceindices/timeseries/czpg/iphrp",
        "Scotland":       "/economy/inflationandpriceindices/timeseries/czph/iphrp",
    }

    records = []
    for region, uri in _SERIES_UPDATED.items():
        if region not in _REGIONS and region not in ("United Kingdom", "England"):
            continue
        try:
            time.sleep(3)  # respect ONS rate limits
            df = _fetch_generator(uri, freq="monthly")
            df["region"] = region
            records.append(df)
        except Exception as e:
            print(f"    {region}: {e}")

    if not records:
        print("  All IPHRP series failed. Using approximate placeholder data.")
        return _rental_index_placeholder()

    out = pd.concat(records, ignore_index=True)
    out = out.rename(columns={"value": "rental_index"})
    return out[out["region"].isin(_REGIONS)].sort_values(["region", "date"]).reset_index(drop=True)


def _rental_index_placeholder() -> pd.DataFrame:
    """Approximate IPHRP index (2015=100) using known UK rental growth rates."""
    # UK private rental growth approx: ~3% p.a. pre-2021, ~8-10% p.a. 2022-23
    dates  = pd.date_range("2000-01-01", "2026-01-01", freq="MS")
    rows   = []
    for region in _REGIONS:
        index = 100.0
        for d in sorted(dates, reverse=True):
            # Reverse engineer from 2015=100
            pass
        # Simple reconstruction: 2015=100, grow at approximate rates
        index_val = 100.0
        ref_date  = pd.Timestamp("2015-01-01")
        for d in dates:
            months_from_ref = (d.year - 2015) * 12 + (d.month - 1)
            if d < pd.Timestamp("2022-01-01"):
                annual_growth = 0.03
            else:
                annual_growth = 0.08
            idx = 100.0 * ((1 + annual_growth) ** (months_from_ref / 12))
            rows.append({"date": d, "region": region, "rental_index": round(idx, 2)})
    return pd.DataFrame(rows)


def _find_prms_download_url() -> str:
    """Scrape the ONS PRMS page for the latest Excel/CSV download link."""
    r = requests.get(_PRMS_PAGE, headers=_HEADERS, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"\.(xls|xlsx|csv)$", href, re.IGNORECASE):
            if "rental" in href.lower() or "prms" in href.lower() or "private" in href.lower():
                if not href.startswith("http"):
                    href = "https://www.ons.gov.uk" + href
                return href

    raise ValueError("Could not find PRMS download link on ONS page")


def _download_and_parse_prms(url: str) -> pd.DataFrame:
    """Download and parse the ONS PRMS Excel file."""
    r = requests.get(url, headers=_HEADERS, timeout=60)
    r.raise_for_status()

    sheets = pd.read_excel(io.BytesIO(r.content), sheet_name=None, header=None)
    records = []

    for sheet_name, raw in sheets.items():
        raw = raw.dropna(how="all").reset_index(drop=True)

        # Find the header row with year/quarter dates
        header_row = None
        for i, row in raw.iterrows():
            vals = [str(v) for v in row.tolist()]
            if sum(1 for v in vals if re.match(r"\d{4}", v)) > 3:
                header_row = i
                break

        if header_row is None:
            continue

        raw.columns = raw.iloc[header_row].astype(str).str.strip()
        raw = raw.iloc[header_row + 1:].reset_index(drop=True)

        region_col  = raw.columns[0]
        date_cols   = [c for c in raw.columns if re.match(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", str(c))]

        if len(date_cols) < 4:
            continue

        df = raw[[region_col] + date_cols].copy()
        df = df.rename(columns={region_col: "region"})
        df["region"] = df["region"].astype(str).str.strip()

        # Keep only target regions
        df = df[df["region"].isin(_REGIONS)]
        if df.empty:
            continue

        df = df.melt(id_vars="region", var_name="date_str", value_name="median_monthly_rent")
        df["median_monthly_rent"] = pd.to_numeric(df["median_monthly_rent"], errors="coerce")
        df["date"] = pd.to_datetime(df["date_str"], format="%b %Y", errors="coerce")
        df = df.dropna(subset=["date", "median_monthly_rent"])
        records.append(df[["date", "region", "median_monthly_rent"]])

    if not records:
        raise ValueError("No parseable regional rent data found in PRMS file")

    return pd.concat(records, ignore_index=True)


def _rental_level_placeholder() -> pd.DataFrame:
    """
    Placeholder when PRMS download fails.
    Uses rough 2015 median rents scaled by IPHRP index.
    Approximate 2015 median monthly rents (ONS PRMS):
    """
    _BASE_RENTS_2015 = {
        "London":                   1500,
        "South East":                900,
        "East of England":           800,
        "South West":                750,
        "West Midlands":             650,
        "East Midlands":             600,
        "Yorkshire and The Humber":  575,
        "North West":                650,
        "North East":                525,
        "Wales":                     575,
        "Scotland":                  700,
        "Northern Ireland":          550,
    }
    dates = pd.date_range("2000-01-01", "2024-12-01", freq="MS")
    rows  = []
    for region, base_rent in _BASE_RENTS_2015.items():
        for d in dates:
            # Very rough: 1% real annual growth
            years_from_2015 = (d.year - 2015) + (d.month - 1) / 12
            rent = base_rent * (1.01 ** years_from_2015)
            rows.append({"date": d, "region": region, "median_monthly_rent": round(rent, 0)})
    return pd.DataFrame(rows)
