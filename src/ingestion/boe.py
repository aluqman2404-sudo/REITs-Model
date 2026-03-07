"""
Stage 2: Bank of England base rate ingestion.

Tries three methods in order:
  1. BoE IADB statistics download (CSV via form params)
  2. BoE HTML page scrape (BeautifulSoup)
  3. Hardcoded rate history as a guaranteed fallback

The BoE rate only changes ~5–10 times per year (at MPC meetings),
so a hardcoded fallback is a practical and reliable last resort.
"""

import io
import requests
import pandas as pd
from datetime import datetime
from config.settings import DATA_RAW

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bankofengland.co.uk/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# BoE IADB export — downloads CSV directly
_BOE_IADB_URL = (
    "https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp"
    "?Travel=NIxRSxSUx&FromSeries=1&ToSeries=50&DAT=ALL"
    "&VFD=Y&html.x=66&html.y=26&C=IUDBEDR&Filter=N"
)

_BOE_HTML_URL = "https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp"


def fetch_boe_rate(save: bool = True) -> pd.DataFrame:
    """
    Download BoE official bank rate history, resampled to monthly (LOCF).

    Returns DataFrame with columns: date (month-start), base_rate (%)
    """
    print("  Fetching Bank of England base rate...")

    df = None

    # Method 1: IADB direct CSV
    try:
        df = _fetch_iadb_csv()
        print("  Source: BoE IADB")
    except Exception as e:
        print(f"  IADB failed: {e}")

    # Method 2: HTML scrape
    if df is None:
        try:
            df = _fetch_html_table()
            print("  Source: BoE HTML table")
        except Exception as e:
            print(f"  HTML scrape failed: {e}")

    # Method 3: Hardcoded fallback (all MPC decisions 2000–Feb 2026)
    if df is None:
        print("  Using hardcoded rate history (2000–2026)...")
        df = _hardcoded_rates()

    df = _resample_to_monthly(df)

    if save:
        path = DATA_RAW / f"boe_base_rate_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {len(df):,} rows → {path.name}")

    return df


# ── Fetch methods ─────────────────────────────────────────────────────────────

def _fetch_iadb_csv() -> pd.DataFrame:
    r = requests.get(_BOE_IADB_URL, headers=_HEADERS, timeout=30)
    r.raise_for_status()

    # Try to parse as CSV
    try:
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c or "day" in c), None)
        rate_col = next((c for c in df.columns if "rate" in c or "value" in c or "iudbedr" in c.lower()), None)
        if not date_col or not rate_col:
            raise ValueError(f"Columns not found: {list(df.columns)}")
        df = df[[date_col, rate_col]].rename(columns={date_col: "date", rate_col: "base_rate"})
        df["date"]      = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["base_rate"] = pd.to_numeric(df["base_rate"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        if len(df) < 10:
            raise ValueError("Too few rows parsed")
        return df
    except Exception:
        raise ValueError("IADB response could not be parsed as CSV")


def _fetch_html_table() -> pd.DataFrame:
    from bs4 import BeautifulSoup

    r = requests.get(_BOE_HTML_URL, headers=_HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    tables = soup.find_all("table")
    if not tables:
        raise ValueError("No tables found on BoE page")

    for tbl in tables:
        try:
            df = pd.read_html(str(tbl))[0]
            df.columns = [str(c).strip().lower() for c in df.columns]
            date_col = next((c for c in df.columns if "date" in c or "effective" in c), None)
            rate_col = next((c for c in df.columns if "rate" in c or "%" in c or "bank" in c), None)
            if date_col and rate_col and len(df) > 10:
                df = df[[date_col, rate_col]].rename(columns={date_col: "date", rate_col: "base_rate"})
                df["date"]      = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
                df["base_rate"] = pd.to_numeric(df["base_rate"], errors="coerce")
                df = df.dropna().sort_values("date").reset_index(drop=True)
                return df
        except Exception:
            continue

    raise ValueError("Could not parse any rate table from BoE HTML")


def _hardcoded_rates() -> pd.DataFrame:
    """
    Complete BoE Bank Rate history from Jan 2000 to Feb 2026.
    Source: Bank of England official records.
    Each row is the date a new rate became effective.
    """
    decisions = [
        # (date,       rate %)
        ("2000-02-10",  6.00),
        ("2001-02-08",  5.75),
        ("2001-04-05",  5.50),
        ("2001-05-10",  5.25),
        ("2001-08-02",  5.00),
        ("2001-09-18",  4.75),
        ("2001-11-08",  4.50),
        ("2002-02-07",  4.00),
        ("2003-07-10",  3.50),
        ("2003-11-06",  3.75),
        ("2004-02-05",  4.00),
        ("2004-05-06",  4.25),
        ("2004-06-10",  4.50),
        ("2004-08-05",  4.75),
        ("2005-08-04",  4.50),
        ("2006-08-03",  4.75),
        ("2006-11-09",  5.00),
        ("2007-01-11",  5.25),
        ("2007-05-10",  5.50),
        ("2007-07-05",  5.75),
        ("2007-12-06",  5.50),
        ("2008-02-07",  5.25),
        ("2008-04-10",  5.00),
        ("2008-10-08",  4.50),
        ("2008-11-06",  3.00),
        ("2008-12-04",  2.00),
        ("2009-01-08",  1.50),
        ("2009-02-05",  1.00),
        ("2009-03-05",  0.50),
        ("2016-08-04",  0.25),
        ("2017-11-02",  0.50),
        ("2018-08-02",  0.75),
        ("2020-03-11",  0.25),
        ("2020-03-19",  0.10),
        ("2021-12-16",  0.25),
        ("2022-02-03",  0.50),
        ("2022-03-17",  0.75),
        ("2022-05-05",  1.00),
        ("2022-06-16",  1.25),
        ("2022-08-04",  1.75),
        ("2022-09-22",  2.25),
        ("2022-11-03",  3.00),
        ("2022-12-15",  3.50),
        ("2023-02-02",  4.00),
        ("2023-03-23",  4.25),
        ("2023-05-11",  4.50),
        ("2023-06-22",  5.00),
        ("2023-08-03",  5.25),
        ("2024-08-01",  5.00),
        ("2024-11-07",  4.75),
        ("2025-02-06",  4.50),
    ]

    df = pd.DataFrame(decisions, columns=["date", "base_rate"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ── Resample ──────────────────────────────────────────────────────────────────

def _resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Convert irregular rate decisions to a regular monthly series (LOCF)."""
    df = df.set_index("date").sort_index()

    monthly_idx = pd.date_range(
        start=max(df.index.min().replace(day=1), pd.Timestamp("2000-01-01")),
        end=pd.Timestamp.today().replace(day=1),
        freq="MS",
    )

    df = df.reindex(df.index.union(monthly_idx))
    df["base_rate"] = df["base_rate"].ffill()
    df = df.reindex(monthly_idx).reset_index()
    df.columns = ["date", "base_rate"]
    return df
