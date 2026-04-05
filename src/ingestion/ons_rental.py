"""
Stage 2: ONS Private Rental Market data ingestion.
Model Variable 6: Rental Yield proxy.

Two datasets combined to compute rental yield:
  A) ONS Price Index of Private Rents (PIPR) — historical series
     — monthly rental price INDEX by region (Jan 2015=100)
     — replaces the discontinued IPHRP (discontinued Feb 2024)
     — available from Jan 2005 to latest month, all UK regions

  B) ONS Private Rental Market Statistics (PRMS)
     — median monthly rent levels (£) by region
     — used to compute GROSS YIELD = (annual rent / house price) × 100

Yield = (Median Monthly Rent × 12) / Average House Price × 100

Sources:
  PIPR: https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/
        priceindexofprivaterentsukhistoricalseries
  PRMS: https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/
        privaterentalmarketsummarystatisticsengland
"""

import io
import re
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

_REGIONS = PARAMS["project"]["regions"]
_START_YEAR = PARAMS["project"]["base_year"]

# ONS PIPR historical series dataset page — scraped for latest Excel URL
_PIPR_DATASET_PAGE = (
    "https://www.ons.gov.uk/economy/inflationandpriceindices/"
    "datasets/priceindexofprivaterentsukhistoricalseries"
)

# Known direct URL for PIPR historical series (March 2025 release)
# Updated from dataset page — check annually
_PIPR_DIRECT_URL = (
    "https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/"
    "datasets/priceindexofprivaterentsukhistoricalseries/"
    "26march2025/priceindexofprivaterentsukhistoricalseries.xlsx"
)

# ONS PRMS — Private Rental Market Statistics download page
_PRMS_PAGE = (
    "https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/"
    "privaterentalmarketsummarystatisticsinengland"
)

# PIPR region names in Excel → ONS standard names used by this model
_PIPR_REGION_MAP = {
    "United Kingdom":              None,          # aggregate — not used
    "Great Britain":               None,          # aggregate — not used
    "England":                     None,          # aggregate — not used
    "ENGLAND":                     None,
    "Wales":                       "Wales",
    "Scotland":                    "Scotland",
    "Northern Ireland":            "Northern Ireland",
    "North East":                  "North East",
    "North West":                  "North West",
    "Yorkshire and The Humber":    "Yorkshire and The Humber",
    "Yorkshire and the Humber":    "Yorkshire and The Humber",
    "East Midlands":               "East Midlands",
    "West Midlands":               "West Midlands",
    "East of England":             "East of England",
    # Excel sometimes truncates this
    "East":                        "East of England",
    "London":                      "London",
    "South East":                  "South East",
    "South West":                  "South West",
    # PRMS uses UPPER CASE region names
    "WALES":                       "Wales",
    "SCOTLAND":                    "Scotland",
    "NORTHERN IRELAND":            "Northern Ireland",
    "NORTH EAST":                  "North East",
    "NORTH WEST":                  "North West",
    "YORKSHIRE AND THE HUMBER":    "Yorkshire and The Humber",
    "EAST MIDLANDS":               "East Midlands",
    "WEST MIDLANDS":               "West Midlands",
    "EAST OF ENGLAND":             "East of England",
    "EAST":                        "East of England",
    "LONDON":                      "London",
    "SOUTH EAST":                  "South East",
    "SOUTH WEST":                  "South West",
}


def fetch_ons_rental_index(save: bool = True) -> pd.DataFrame:
    """
    Fetch ONS PIPR regional rental price indices (monthly, Jan 2015=100).

    Downloads the ONS Price Index of Private Rents (PIPR) historical series
    Excel file which replaced the discontinued IPHRP in February 2024.

    Data spans January 2005 to latest available month for all UK regions.

    Returns:
        DataFrame with columns: date, region, rental_index
    """
    print("  Fetching ONS PIPR rental price indices (PIPR historical series)...")

    df = None

    # Method 1: Scrape ONS dataset page for latest Excel URL
    try:
        url = _find_latest_pipr_url()
        df = _download_and_parse_pipr(url)
        print(f"  Source: ONS PIPR historical series Excel ({len(df)} rows)")
    except Exception as e:
        print(f"  Latest URL scrape failed ({e}). Trying known direct URL...")

    # Method 2: Fallback to known March 2025 direct URL
    if df is None:
        try:
            df = _download_and_parse_pipr(_PIPR_DIRECT_URL)
            print(f"  Source: ONS PIPR direct URL ({len(df)} rows)")
        except Exception as e:
            print(f"  Direct URL failed ({e}). Using placeholder data.")
            df = _rental_index_placeholder()

    df = df[df["date"].dt.year >= _START_YEAR].reset_index(drop=True)

    if save:
        path = DATA_RAW / f"ons_rental_index_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {len(df):,} rows → {path.name}")
        from src.ingestion.release_metadata import write_release_metadata
        write_release_metadata(
            series="ons_rental_index",
            raw_file_path=path,
            source_url=_PIPR_DATASET_PAGE,
            publication_lag_months_estimated=1,
            df=df,
        )

    return df


def fetch_ons_rental_levels(save: bool = True) -> pd.DataFrame:
    """
    Construct a monthly regional rent level series (£/month) for 2005–present.

    Method:
      1. Scrape the ONS PRMS page to get the latest cross-sectional median rent
         by English region (one observation per semi-annual release, ~Oct 2022-Sep 2023).
      2. Use this as an anchor rent level at a known reference date.
      3. Backcast / project using the PIPR rental price INDEX:
            rent(t) = anchor_rent × (PIPR_index(t) / PIPR_index(anchor_date))
      This produces a methodologically sound monthly time series from 2005 to present.

    PRMS covers England regions only. Scotland, Wales, NI use ONS PIPR-based backcasts
    anchored to approximate 2023 rent levels from respective national statistics.

    Returns:
        DataFrame with columns: date, region, median_monthly_rent
    """
    print("  Fetching ONS PRMS median rent levels (PIPR-anchored time series)...")

    # Step 1: Load the PIPR index (already fetched — or fetch it now)
    pipr = _load_pipr_index()
    if pipr is None or pipr.empty:
        print("  PIPR index unavailable. Using placeholder.")
        df = _rental_level_placeholder()
    else:
        # Step 2: Get anchor rents from the latest PRMS release
        anchor_rents = _fetch_prms_anchor_rents()
        if not anchor_rents:
            print("  PRMS anchor fetch failed. Using placeholder.")
            df = _rental_level_placeholder()
        else:
            # Step 3: Backcast via PIPR index
            df = _backcast_rent_levels(pipr, anchor_rents)
            print(f"  PIPR-anchored time series: {len(df)} rows "
                  f"({df['date'].min():%Y-%m} → {df['date'].max():%Y-%m})")

    df = df[df["date"].dt.year >= _START_YEAR]
    df = df.sort_values(["region", "date"]).reset_index(drop=True)

    if save:
        path = DATA_RAW / f"ons_rental_levels_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {len(df):,} rows → {path.name}")
        from src.ingestion.release_metadata import write_release_metadata
        write_release_metadata(
            series="ons_rental_levels",
            raw_file_path=path,
            source_url=_PRMS_PAGE,
            publication_lag_months_estimated=6,
            df=df,
        )

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
    rents = rental_levels.copy()
    prices = house_prices[["date", "region", "average_price"]].copy()

    rents["date"] = pd.to_datetime(
        rents["date"]).dt.to_period("Q").dt.to_timestamp()
    prices["date"] = pd.to_datetime(
        prices["date"]).dt.to_period("Q").dt.to_timestamp()

    prices_q = prices.groupby(["date", "region"])[
                              "average_price"].mean().reset_index()
    rents_q = rents.groupby(["date", "region"])[
                            "median_monthly_rent"].mean().reset_index()

    merged = rents_q.merge(prices_q, on=["date", "region"], how="inner")
    merged["gross_yield_pct"] = (
        merged["median_monthly_rent"] * 12) / merged["average_price"] * 100
    merged = merged[["date", "region", "gross_yield_pct"]
                    ].sort_values(["region", "date"])

    if save:
        path = DATA_RAW / f"rental_yield_{datetime.today():%Y%m}.csv"
        merged.to_csv(path, index=False)
        print(f"  Computed yield saved → {path.name}")
        from src.ingestion.release_metadata import write_release_metadata
        write_release_metadata(
            series="rental_yield",
            raw_file_path=path,
            source_url="derived:ons_rental_levels+land_registry_hpi",
            publication_lag_months_estimated=6,
            df=merged,
        )

    return merged.reset_index(drop=True)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_latest_pipr_url() -> str:
    """
    Scrape the ONS PIPR dataset page for the most recent Excel download link.
    The first /file?uri= link on the page is always the most recent release.
    """
    r = requests.get(_PIPR_DATASET_PAGE, headers=_HEADERS, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "file?uri" in href and "priceindexofprivaterents" in href.lower():
            return "https://www.ons.gov.uk" + href if not href.startswith("http") else href

    raise ValueError("No PIPR Excel link found on ONS dataset page")


def _download_and_parse_pipr(url: str) -> pd.DataFrame:
    """
    Download the PIPR historical series Excel and parse Table 1.

    Table 1 structure (March 2025 release):
      Row 0: title
      Row 1: accessibility note
      Row 2: region names (United Kingdom, Great Britain, England, Wales, ...)
      Row 3: region codes
      Row 4+: monthly data — col 0 = datetime, cols 1+ = index values or [x]
    """
    r = requests.get(url, headers=_HEADERS, timeout=90)
    r.raise_for_status()

    raw = pd.read_excel(io.BytesIO(r.content),
                        sheet_name="Table 1", header=None)

    # Find the region-name header row: first row where col 1 contains a known region
    header_row_idx = None
    for i in range(min(10, len(raw))):
        vals = raw.iloc[i].astype(str).str.strip().tolist()
        if any(v in _PIPR_REGION_MAP for v in vals):
            header_row_idx = i
            break

    if header_row_idx is None:
        raise ValueError(
            f"No region header row found in PIPR Excel. "
            f"First row: {raw.iloc[0].tolist()[:5]}"
        )

    header_row = raw.iloc[header_row_idx].astype(str).str.strip().tolist()

    # Build column index → model region name mapping
    # Also handle truncated names (e.g. "East" → "East of England")
    col_region = {}
    for col_idx, name in enumerate(header_row):
        mapped = _PIPR_REGION_MAP.get(name)
        # Prefix match removed — explicit "East" entry in _PIPR_REGION_MAP handles truncation
        if mapped and mapped in _REGIONS:
            col_region[col_idx] = mapped

    if not col_region:
        raise ValueError(
            f"No recognised region columns found in PIPR Excel. "
            f"Header row: {header_row[:10]}"
        )

    # Data starts two rows after the header (skip header + codes row)
    data_start = header_row_idx + 2
    records = []
    for _, row in raw.iloc[data_start:].iterrows():
        date_val = row.iloc[0]
        if pd.isna(date_val):
            continue

        # Dates in PIPR Excel are already datetime objects
        try:
            date = pd.Timestamp(date_val)
        except Exception:
            date = pd.to_datetime(str(date_val), errors="coerce")
        if pd.isna(date):
            continue

        for col_idx, region in col_region.items():
            val_raw = row.iloc[col_idx]
            # Skip "[x]" (not available) and NaN entries
            if pd.isna(val_raw) or str(val_raw).strip() in ("[x]", ""):
                continue
            try:
                val = float(str(val_raw).replace(",", ""))
                records.append(
                    {"date": date, "region": region, "rental_index": val})
            except (ValueError, TypeError):
                continue

    if not records:
        raise ValueError(
            "No data rows parsed from PIPR historical series Excel")

    df = pd.DataFrame(records)
    return df.sort_values(["region", "date"]).reset_index(drop=True)


def _rental_index_placeholder() -> pd.DataFrame:
    """
    Approximate PIPR index (Jan 2015=100) using known UK rental growth rates.
    Only used when both ONS download methods fail.
    """
    dates = pd.date_range("2000-01-01", "2026-01-01", freq="MS")
    rows = []
    for region in _REGIONS:
        for d in dates:
            months_from_ref = (d.year - 2015) * 12 + (d.month - 1)
            if d < pd.Timestamp("2022-01-01"):
                annual_growth = 0.03
            else:
                annual_growth = 0.08
            idx = 100.0 * ((1 + annual_growth) ** (months_from_ref / 12))
            rows.append({"date": d, "region": region,
                        "rental_index": round(idx, 2)})
    return pd.DataFrame(rows)


def _load_pipr_index() -> pd.DataFrame:
    """Load the PIPR rental index from disk (saved by fetch_ons_rental_index)."""
    try:
        candidates = sorted(DATA_RAW.glob(
            "ons_rental_index_*.csv"), reverse=True)
        if candidates:
            df = pd.read_csv(candidates[0], parse_dates=["date"])
            return df
    except Exception:
        pass
    return None


def _fetch_prms_anchor_rents() -> dict:
    """
    Download the latest ONS PRMS release and extract median monthly rent (£)
    by English region. Returns {region: (anchor_date, median_rent)}.

    PRMS structure (Table 1.1): cross-sectional snapshot for one period.
    Row 6 = header (Area Code, Region, Count, Mean, LQ, Median, UQ).
    Rows 7+ = regions (ENGLAND, NORTH EAST, ...).
    Region names are UPPER CASE.
    Anchor date = midpoint of the published period (scrape from filename/page).
    """
    try:
        r = requests.get(_PRMS_PAGE, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Get the first (most recent) .xls/.xlsx link
        prms_url = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if isinstance(href, list):
                href = href[0]
            href = str(href)
            if "file?uri" in href and re.search(r"\.xlsx?$", href, re.IGNORECASE):
                if "privaterentalmarket" in href.lower():
                    prms_url = "https://www.ons.gov.uk" + href
                    break

        if not prms_url:
            raise ValueError("No PRMS Excel link found")

        # Extract period from URL to set anchor date
        # e.g. "october2022toseptember2023" → anchor = April 2023 (midpoint)
        period_match = re.search(
            r"(\w+)(\d{4})to(\w+)(\d{4})", prms_url, re.IGNORECASE
        )
        if period_match:
            end_month_str = period_match.group(3)[:3].capitalize()
            end_year = int(period_match.group(4))
            anchor_date = pd.to_datetime(
                f"{end_month_str} {end_year}", format="%b %Y")
        else:
            anchor_date = pd.Timestamp("2023-03-01")   # safe fallback

        r2 = requests.get(prms_url, headers=_HEADERS, timeout=60)
        r2.raise_for_status()

        raw = pd.read_excel(io.BytesIO(r2.content),
                            sheet_name="Table 1.7", header=None)

        # Find header row: contains "Median" or "Region"
        header_row = None
        for i in range(min(15, len(raw))):
            vals = raw.iloc[i].astype(str).str.strip().tolist()
            if "Median" in vals or "Region" in vals:
                header_row = i
                break

        if header_row is None:
            raise ValueError("Header row not found in PRMS Table 1.1")

        headers = [str(v).strip() if pd.notna(
            v) else "" for v in raw.iloc[header_row].tolist()]
        # Prefer exact "Region" column over "Area Code" columns
        region_c = next((i for i, h in enumerate(
            headers) if h == "Region"), None)
        if region_c is None:
            region_c = next(i for i, h in enumerate(headers)
                            if "Region" in h or "Area" in h)
        median_c = next(i for i, h in enumerate(headers) if "Median" in h)

        anchors = {}
        for _, row in raw.iloc[header_row + 1:].iterrows():
            region_raw = str(row.iloc[region_c]).strip()
            mapped = _PIPR_REGION_MAP.get(
                region_raw) or _PIPR_REGION_MAP.get(region_raw.title())
            if not mapped or mapped not in _REGIONS:
                # Try upper-case lookup
                mapped = _PIPR_REGION_MAP.get(region_raw.upper())
                if not mapped:
                    continue
            val_raw = str(row.iloc[median_c]).replace(",", "").strip()
            try:
                rent = float(val_raw)
                if rent > 0:
                    anchors[mapped] = (anchor_date, rent)
            except (ValueError, TypeError):
                continue

        return anchors

    except Exception as e:
        print(f"  PRMS anchor fetch failed ({e})")
        return {}


def _backcast_rent_levels(pipr: pd.DataFrame, anchor_rents: dict) -> pd.DataFrame:
    """
    Construct a monthly rent level series per region using:
        rent(t) = anchor_rent × (PIPR_index(t) / PIPR_index(anchor_date))

    For regions not in anchor_rents (Scotland, Wales, NI), approximate 2023
    anchor rents are used from national housing statistics.
    """
    # Approximate 2023 median monthly rents for nations not in PRMS (England-only)
    _NATION_ANCHORS_2023 = {
        "Scotland":         pd.Timestamp("2023-03-01"),
        "Wales":            pd.Timestamp("2023-03-01"),
        "Northern Ireland": pd.Timestamp("2023-03-01"),
    }
    _NATION_RENTS_2023 = {
        "Scotland":         925,   # ONS PIPR / Scottish Government 2022-23
        "Wales":            750,   # StatsWales 2022-23
        "Northern Ireland": 750,   # NIHE Private Rental Panel 2022-23
    }

    records = []
    for region in _REGIONS:
        region_pipr = pipr[pipr["region"] == region].set_index("date")[
                                                               "rental_index"]
        if region_pipr.empty:
            continue

        if region in anchor_rents:
            anchor_date, anchor_rent = anchor_rents[region]
        elif region in _NATION_ANCHORS_2023:
            anchor_date = _NATION_ANCHORS_2023[region]
            anchor_rent = _NATION_RENTS_2023[region]
        else:
            continue

        # Snap anchor_date to nearest available PIPR date
        anchor_date = pd.Timestamp(anchor_date).to_period("M").to_timestamp()
        if anchor_date not in region_pipr.index:
            nearest = region_pipr.index[
                abs(region_pipr.index - anchor_date).argmin()
            ]
            anchor_date = nearest

        anchor_index = region_pipr.get(anchor_date)
        if anchor_index is None:
            continue
        # Handle duplicate dates returning a Series
        if hasattr(anchor_index, "__len__"):
            anchor_index = float(anchor_index.iloc[0])
        else:
            anchor_index = float(anchor_index)
        if anchor_index == 0:
            continue

        for date, idx in region_pipr.items():
            rent = anchor_rent * (idx / anchor_index)
            records.append({
                "date": date,
                "region": region,
                "median_monthly_rent": round(rent, 2),
            })

    return pd.DataFrame(records)


def _rental_level_placeholder() -> pd.DataFrame:
    """
    Placeholder when PRMS download fails.
    Uses approximate 2015 median rents (ONS PRMS) with 1% p.a. real growth.
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
    rows = []
    for region, base_rent in _BASE_RENTS_2015.items():
        for d in dates:
            years_from_2015 = (d.year - 2015) + (d.month - 1) / 12
            rent = base_rent * (1.01 ** years_from_2015)
            rows.append({"date": d, "region": region,
                        "median_monthly_rent": round(rent, 0)})
    return pd.DataFrame(rows)
