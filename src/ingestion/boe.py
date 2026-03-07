"""
Stage 2: Bank of England base rate ingestion.

The BoE publishes its official bank rate history through their
Interactive Database (IADB). This module downloads the full history
and resamples to a monthly series using last-observation-carried-forward.

Source: https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp
"""

import io
import requests
import pandas as pd
from datetime import datetime
from config.settings import DATA_RAW

# BoE IADB download endpoint for Bank Rate (series code: IUDBEDR)
_BOE_URL = (
    "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
    "?Travel=NIxRSxSUx&FromSeries=1&ToSeries=50&DAT=ALL"
    "&VFD=Y&html.x=66&html.y=26&C=IUDBEDR&Filter=N"
)

# Fallback: BoE statistics CSV (published separately)
_BOE_CSV_FALLBACK = (
    "https://www.bankofengland.co.uk/-/media/boe/files/monetary-policy/"
    "bank-rate-data.csv"
)


def fetch_boe_rate(save: bool = True) -> pd.DataFrame:
    """
    Download BoE official bank rate history and resample to monthly frequency
    using last-observation-carried-forward (LOCF).

    Returns:
        DataFrame with columns: date (month-start), base_rate (%)
    """
    print("Fetching Bank of England base rate...")

    df = None

    # Try primary IADB endpoint
    try:
        df = _fetch_from_iadb()
        print("  Source: BoE IADB")
    except Exception as e:
        print(f"  IADB failed ({e}), trying CSV fallback...")

    # Try CSV fallback
    if df is None:
        try:
            df = _fetch_from_csv()
            print("  Source: BoE CSV file")
        except Exception as e:
            print(f"  CSV fallback also failed ({e})")
            raise RuntimeError("Could not fetch BoE base rate from any source.") from e

    # Resample irregular rate changes to monthly (LOCF)
    df = _resample_to_monthly(df)

    if save:
        path = DATA_RAW / f"boe_base_rate_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {len(df):,} rows → {path}")

    return df


# ── Internal helpers ─────────────────────────────────────────────────────────

def _fetch_from_iadb() -> pd.DataFrame:
    """Download from the BoE Interactive Database."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(_BOE_URL, headers=headers, timeout=30)
    response.raise_for_status()

    # The IADB returns an HTML page with a data table
    tables = pd.read_html(io.StringIO(response.text))
    if not tables:
        raise ValueError("No tables found in BoE IADB response")

    # Find the table that contains the rate data
    for tbl in tables:
        tbl.columns = [str(c).strip().lower() for c in tbl.columns]
        date_col = _find_date_col(tbl)
        rate_col = _find_rate_col(tbl)
        if date_col and rate_col:
            df = tbl[[date_col, rate_col]].copy()
            df.columns = ["date", "base_rate"]
            df["date"]      = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            df["base_rate"] = pd.to_numeric(df["base_rate"], errors="coerce")
            df = df.dropna()
            return df.sort_values("date").reset_index(drop=True)

    raise ValueError("Could not identify rate table in IADB response")


def _fetch_from_csv() -> pd.DataFrame:
    """Download the BoE base rate CSV file."""
    response = requests.get(_BOE_CSV_FALLBACK, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(
        io.StringIO(response.text),
        skiprows=3,
        header=None,
        names=["date", "base_rate"],
        skip_blank_lines=True,
    )
    df["date"]      = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["base_rate"] = pd.to_numeric(df["base_rate"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


def _resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert irregular rate change dates to a regular monthly series.
    The BoE rate is constant between MPC decisions, so we use LOCF.
    """
    df = df.set_index("date")

    # Create a complete monthly date range from first rate change to today
    monthly_idx = pd.date_range(
        start=df.index.min().replace(day=1),
        end=pd.Timestamp.today().replace(day=1),
        freq="MS",  # month-start
    )

    # Reindex and forward-fill
    df = df.reindex(df.index.union(monthly_idx))
    df["base_rate"] = df["base_rate"].ffill()
    df = df.reindex(monthly_idx)

    df = df.reset_index().rename(columns={"index": "date"})
    df = df[df["date"].dt.year >= 2000].reset_index(drop=True)
    return df


def _find_date_col(df: pd.DataFrame) -> str | None:
    """Find the date column in a table by scanning column names."""
    for col in df.columns:
        if any(kw in str(col) for kw in ["date", "day", "period", "effective"]):
            return col
    return None


def _find_rate_col(df: pd.DataFrame) -> str | None:
    """Find the rate column in a table by scanning column names."""
    for col in df.columns:
        if any(kw in str(col) for kw in ["rate", "bank", "%", "percent", "value"]):
            return col
    return None
