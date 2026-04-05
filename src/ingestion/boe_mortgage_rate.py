"""
Stage 2: Bank of England mortgage rate ingestion.

Fetches the quoted 2-year fixed mortgage rate (75% LTV, new advances).
BoE series: IUMBV34 (2yr Fixed, 75% LTV, Owner Occupied quoted rate)

This is the model's Variable 3: Mortgage Rate — the initial quoted rate
that buyers face, far more relevant than the base rate alone as it captures
lender margins and credit conditions.

Sources tried in order:
  1. BoE Bankstats ZIP → TabG1.3.xls → IUMBV34 (recent 24 months, real data)
  2. Computed proxy: BoE base rate + historical spread (full history)
     The proxy and real data are spliced together for the full series.

Note: The BoE IADB (online database) is JavaScript-driven and cannot be
      scraped programmatically. The Bankstats ZIP provides real recent data.
"""

import io
import zipfile
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
}

# BoE Bankstats monthly tables — ZIP of all tables (2-3 MB)
_BANKSTATS_ZIP_URL = (
    "https://www.bankofengland.co.uk/-/media/boe/files/statistics/"
    "bankstats-latest-tables.zip"
)

# Within the ZIP: TabG1.3.xls contains "Average quoted household interest rates"
# IUMBV34 = 2yr Fixed, 75% LTV, Owner Occupied (column index 5, 0-based)
_BANKSTATS_FILE = "Latest Tables/TabG1.3.xls"
_IUMBV34_COL_IDX = 5    # column in TabG1.3 for 2yr Fixed 75% LTV rate
_DATE_COL_IDX = 1    # column in TabG1.3 for the date


def fetch_boe_mortgage_rate(save: bool = True) -> pd.DataFrame:
    """
    Fetch the 2-year fixed mortgage rate (75% LTV, IUMBV34).

    Strategy:
      - Real data from BoE Bankstats ZIP for available history (~24 months)
      - Base rate + spread proxy for older history
      - The two are spliced: proxy fills all months, real data overwrites where
        available to ensure a clean, gap-free series.

    Returns:
        DataFrame with columns: date (month-start), mortgage_rate (%)
    """
    print("  Fetching BoE quoted mortgage rate (IUMBV34, 2yr Fixed 75% LTV)...")

    # Build full proxy series first (guaranteed to cover 2000–present)
    proxy = _base_rate_proxy()
    proxy["source"] = "proxy"

    # Attempt to get real BoE data from Bankstats ZIP
    real = None
    try:
        real = _fetch_bankstats_mortgage_rate()
        print(f"  Real data from BoE Bankstats: {len(real)} months "
              f"({real['date'].min():%Y-%m} to {real['date'].max():%Y-%m})")
        real["source"] = "bankstats"
    except Exception as e:
        print(f"  Bankstats download failed ({e}). Using proxy only.")

    # Splice: proxy for full history, real data overwrites where available
    if real is not None and not real.empty:
        proxy = proxy[~proxy["date"].isin(real["date"])]
        df = pd.concat(
            [proxy, real[["date", "mortgage_rate", "source"]]], ignore_index=True)
    else:
        df = proxy

    df = df.sort_values("date").reset_index(drop=True)
    df = df[df["date"].dt.year >= 2000].reset_index(drop=True)

    n_real = (df["source"] == "bankstats").sum()
    n_proxy = (df["source"] == "proxy").sum()
    print(f"  Series: {n_real} real + {n_proxy} proxy months")

    df = df[["date", "mortgage_rate"]]

    if save:
        path = DATA_RAW / f"boe_mortgage_rate_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {len(df):,} rows → {path.name}")
        from src.ingestion.release_metadata import write_release_metadata
        write_release_metadata(
            series="boe_mortgage_rate",
            raw_file_path=path,
            source_url=_BANKSTATS_ZIP_URL,
            publication_lag_months_estimated=2,
            df=df,
        )

    return df


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_bankstats_mortgage_rate() -> pd.DataFrame:
    """
    Download BoE Bankstats ZIP and extract IUMBV34 from TabG1.3.xls.

    TabG1.3 = "Average quoted household interest rates"
    Column 1 (0-based): date (as Excel datetime)
    Column 5 (0-based): IUMBV34 — 2yr Fixed, 75% LTV, Owner Occupied (%)
    """
    r = requests.get(_BANKSTATS_ZIP_URL, headers=_HEADERS, timeout=90)
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open(_BANKSTATS_FILE) as f:
        raw = pd.read_excel(io.BytesIO(f.read()), sheet_name=0, header=None)

    # Verify column 5 contains IUMBV34
    code_row = raw.iloc[10]    # row 10 (0-indexed) has series codes
    code_suffix = str(code_row.iloc[_IUMBV34_COL_IDX]).strip()
    if code_suffix not in ("BV34", "nan"):
        # Search for BV34 dynamically if column shifts
        bv34_cols = [i for i, v in enumerate(code_row) if "BV34" in str(v)]
        if not bv34_cols:
            raise ValueError(
                f"BV34 column not found in row 10. Got: {code_row.tolist()[:10]}")
        col_idx = bv34_cols[0]
    else:
        col_idx = _IUMBV34_COL_IDX

    records = []
    for _, row in raw.iloc[11:].iterrows():    # data starts at row 11
        date_val = row.iloc[_DATE_COL_IDX]
        rate_val = row.iloc[col_idx]

        if pd.isna(date_val) or isinstance(date_val, str):
            continue
        try:
            ts = pd.Timestamp(date_val)
            rate = float(str(rate_val).replace(",", "").strip())
            if not pd.isna(rate):
                records.append({"date": ts, "mortgage_rate": rate})
        except (ValueError, TypeError):
            continue

    if not records:
        raise ValueError("No data rows parsed from TabG1.3.xls")

    df = pd.DataFrame(records)
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.sort_values("date").reset_index(drop=True)


def _base_rate_proxy() -> pd.DataFrame:
    """
    Compute mortgage rate proxy = BoE base rate + typical spread.

    Historical UK quoted mortgage spread over base rate (approximate):
      - 2000–2007: ~1.2%  (benign credit conditions)
      - 2008–2012: ~3.0%  (credit crunch — elevated risk premium)
      - 2013–2021: ~1.8%  (post-crisis normalisation)
      - 2022+:     ~1.3%  (rising rates compressed spreads slightly)
    """
    try:
        base_file = next(
            f for f in sorted(DATA_RAW.iterdir())
            if "boe_base_rate" in f.name
        )
        base = pd.read_csv(base_file)
    except Exception:
        from src.ingestion.boe import _hardcoded_rates, _resample_to_monthly
        base = _resample_to_monthly(_hardcoded_rates())

    base["date"] = pd.to_datetime(base["date"])

    def _spread(date: pd.Timestamp) -> float:
        year = date.year
        if year <= 2007:
            return 1.2
        if year <= 2012:
            return 3.0
        if year <= 2021:
            return 1.8
        return 1.3

    base["mortgage_rate"] = base["base_rate"] + base["date"].map(_spread)
    return base[["date", "mortgage_rate"]]
