"""
Stage 2: MHCLG housing supply ingestion.
Model Variable 4: Housing Supply (net additional dwellings by region).

Source: MHCLG Live Table 118 — Annual net additional dwellings and components,
        England and the regions, 2000-01 to latest year.

Published at:
  https://www.gov.uk/government/statistical-data-sets/live-tables-on-net-supply-of-housing

Table 118 is the primary measure of housing supply — it captures not just
new builds but also conversions, changes of use, and demolitions. It is
published annually for financial years (April–March).

Data is annual (FY) → interpolated to quarterly in Stage 3.
"""

import io
import requests
import pandas as pd
from datetime import datetime
from config.settings import DATA_RAW, PARAMS

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_REGIONS    = PARAMS["project"]["regions"]
_START_YEAR = PARAMS["project"]["base_year"]

# Direct URL for Table 118 (net additional dwellings by region)
# Updated from GOV.UK net supply page — check annually
_TABLE_118_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "691de48bd140bbbaa59a2a77/Live_Table_118.ods"
)

_GOV_UK_NET_SUPPLY = (
    "https://www.gov.uk/government/statistical-data-sets/"
    "live-tables-on-net-supply-of-housing"
)

# MHCLG region names → ONS standard names
_REGION_MAP = {
    "North East":              "North East",
    "North West":              "North West",
    "Yorkshire and The Humber":"Yorkshire and The Humber",
    "East Midlands":           "East Midlands",
    "West Midlands":           "West Midlands",
    "East of England":         "East of England",
    "London":                  "London",
    "South East":              "South East",
    "South West":              "South West",
}
# Note: Wales, Scotland, Northern Ireland are not in Table 118 (England only)


def fetch_mhclg_supply(save: bool = True) -> pd.DataFrame:
    """
    Download MHCLG Table 118 — net additional dwellings by English region.

    Returns:
        DataFrame with columns:
            date        (April 1 of the financial year start, annual)
            region      (ONS standard region name)
            net_additions (integer, annual new dwellings)
    """
    print("  Fetching MHCLG Table 118 — net additional dwellings by region...")

    df = None

    # Try direct URL first, then scrape GOV.UK for updated link
    for url in [_TABLE_118_URL, _find_latest_table118_url()]:
        try:
            df = _download_table118(url)
            print(f"  Source: MHCLG Table 118 ({len(df)} rows)")
            break
        except Exception as e:
            print(f"  URL failed: {e}")

    if df is None:
        raise RuntimeError(
            "Could not download MHCLG Table 118.\n"
            "Download manually from:\n"
            f"  {_GOV_UK_NET_SUPPLY}\n"
            "Save 'Live_Table_118.ods' and re-run."
        )

    df = df[df["date"].dt.year >= _START_YEAR].reset_index(drop=True)

    if save:
        path = DATA_RAW / f"mhclg_supply_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {len(df):,} rows → {path.name}")
        print("  Note: England regions only (Wales/Scotland/NI published separately)")

    return df


# ── Internal helpers ──────────────────────────────────────────────────────────

def _download_table118(url: str) -> pd.DataFrame:
    """Download and parse Table 118 ODS file."""
    r = requests.get(url, headers=_HEADERS, timeout=90)
    r.raise_for_status()

    sheets = pd.read_excel(io.BytesIO(r.content), engine="odf", sheet_name=None, header=None)

    # Use the rounded sheet (cleaner for modelling)
    target_sheet = next(
        (k for k in sheets if "rounded" in k.lower() and "un" not in k.lower()),
        next((k for k in sheets if k not in ("Cover", "Contents", "Notes")), None)
    )
    if not target_sheet:
        raise ValueError(f"No data sheet found. Sheets: {list(sheets.keys())}")

    raw = sheets[target_sheet]

    # Find header row — the one containing region column names
    header_row = None
    for i, row in raw.iterrows():
        vals = row.astype(str).tolist()
        if "North East" in vals and "London" in vals:
            header_row = i
            break

    if header_row is None:
        raise ValueError("Could not find region header row in Table 118")

    raw.columns = raw.iloc[header_row].astype(str).str.strip()
    raw         = raw.iloc[header_row + 1:].reset_index(drop=True)

    # Keep only "Net additions" rows (first column value)
    component_col = raw.columns[0]
    year_col      = raw.columns[1]
    region_cols   = [c for c in raw.columns if c in _REGION_MAP]

    data = raw[raw[component_col].astype(str).str.strip() == "Net additions"].copy()
    if data.empty:
        # Some files have the label in every row — just take all numeric rows
        data = raw.copy()

    rows = []
    for _, row in data.iterrows():
        year_str = str(row[year_col]).strip()
        if not year_str or year_str == "nan":
            continue
        # Parse "2000-01" → April 2000 (start of UK financial year)
        try:
            start_year = int(year_str.split("-")[0])
            date = pd.Timestamp(f"{start_year}-04-01")
        except (ValueError, IndexError):
            continue

        for region_name in region_cols:
            ons_region = _REGION_MAP[region_name]
            try:
                val = float(str(row[region_name]).replace(",", "").strip())
                rows.append({"date": date, "region": ons_region, "net_additions": val})
            except (ValueError, TypeError):
                continue

    if not rows:
        raise ValueError("No net additions rows parsed from Table 118")

    df = pd.DataFrame(rows)
    return df.sort_values(["region", "date"]).reset_index(drop=True)


def _find_latest_table118_url() -> str:
    """Scrape GOV.UK net supply page for the current Table 118 download URL."""
    from bs4 import BeautifulSoup
    import re

    try:
        r = requests.get(_GOV_UK_NET_SUPPLY, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            if "118" in text and href.endswith(".ods"):
                return href if href.startswith("http") else "https://www.gov.uk" + href
    except Exception:
        pass
    return _TABLE_118_URL  # return known URL as fallback
