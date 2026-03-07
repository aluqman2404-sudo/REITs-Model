"""
Stage 2: Bank of England mortgage rate ingestion.

Fetches the effective 2-year fixed mortgage rate (75% LTV, new advances).
BoE series: IUMTLMV

This is the model's Variable 3: Mortgage Rate — far more relevant to buyer
demand than the base rate alone, as it captures lender margins and credit
conditions on top of the policy rate.

Sources tried in order:
  1. BoE IADB HTML page for IUMTLMV series
  2. BoE statistics Excel download (Table A5.2)
  3. Computed proxy: BoE base rate + historical spread
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
}

# BoE IADB page for mortgage rate series (IUMTLMV = 2yr fixed 75% LTV)
_BOE_MORTGAGE_URL = (
    "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
    "?Travel=NIxRSxSUx&FromSeries=1&ToSeries=50&DAT=ALL"
    "&VFD=Y&html.x=66&html.y=26&C=IUMTLMV&Filter=N"
)

# BoE quoted mortgage rate (broader series, available monthly from 1995)
_BOE_QUOTED_URL = (
    "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
    "?Travel=NIxRSxSUx&FromSeries=1&ToSeries=50&DAT=ALL"
    "&VFD=Y&html.x=66&html.y=26&C=IUMZICQ&Filter=N"
)


def fetch_boe_mortgage_rate(save: bool = True) -> pd.DataFrame:
    """
    Fetch the effective 2-year fixed mortgage rate (BoE IUMTLMV).
    Falls back to average quoted rate (IUMZICQ) then to a base rate proxy.

    Returns:
        DataFrame with columns: date (month-start), mortgage_rate (%)
    """
    print("  Fetching BoE effective mortgage rate (IUMTLMV)...")

    df = None

    for url, label in [(_BOE_MORTGAGE_URL, "IUMTLMV"), (_BOE_QUOTED_URL, "IUMZICQ")]:
        try:
            df = _fetch_boe_html_table(url)
            df = df.rename(columns={"value": "mortgage_rate"})
            print(f"  Source: BoE IADB ({label}) — {len(df)} rows")
            break
        except Exception as e:
            print(f"  {label} failed: {e}")

    # Fallback: base rate + historical spread (base rate data is already downloaded)
    if df is None:
        print("  Using base rate + spread proxy...")
        df = _base_rate_proxy()

    df = df[df["date"].dt.year >= 2000].reset_index(drop=True)

    if save:
        path = DATA_RAW / f"boe_mortgage_rate_{datetime.today():%Y%m}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {len(df):,} rows → {path.name}")

    return df


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_boe_html_table(url: str) -> pd.DataFrame:
    """Fetch and parse a BoE IADB HTML table."""
    from bs4 import BeautifulSoup

    r = requests.get(url, headers=_HEADERS, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    for tbl in soup.find_all("table"):
        try:
            df = pd.read_html(io.StringIO(str(tbl)))[0]
            df.columns = [str(c).strip().lower() for c in df.columns]
            date_col = next((c for c in df.columns if "date" in c or "period" in c), None)
            val_col  = next((c for c in df.columns if "rate" in c or "value" in c or "%" in c), None)
            if date_col and val_col and len(df) > 12:
                df = df[[date_col, val_col]].rename(columns={date_col: "date", val_col: "value"})
                df["date"]  = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.dropna().sort_values("date").reset_index(drop=True)
                # Resample to monthly start
                df = df.set_index("date").resample("MS").last().reset_index()
                return df
        except Exception:
            continue

    raise ValueError("No parseable mortgage rate table found in BoE IADB response")


def _base_rate_proxy() -> pd.DataFrame:
    """
    Compute mortgage rate proxy = BoE base rate + typical spread.
    Historical UK mortgage spread over base rate is roughly:
      - 2000–2007: ~1.0–1.5%
      - 2008–2012: ~2.5–3.5% (credit crunch premium)
      - 2013–2021: ~1.5–2.0%
      - 2022+:     ~1.0–1.5% (normalisation)
    We use a simple time-varying spread.
    """
    try:
        base = pd.read_csv(DATA_RAW / [f for f in sorted(DATA_RAW.iterdir())
                           if "boe_base_rate" in f.name][-1].name
                           if False else DATA_RAW / next(
                               f for f in sorted(DATA_RAW.iterdir())
                               if "boe_base_rate" in f.name
                           ))
    except Exception:
        from src.ingestion.boe import _hardcoded_rates, _resample_to_monthly
        base = _resample_to_monthly(_hardcoded_rates())

    base["date"] = pd.to_datetime(base["date"])

    def _spread(date: pd.Timestamp) -> float:
        year = date.year
        if year <= 2007: return 1.2
        if year <= 2012: return 3.0
        if year <= 2021: return 1.8
        return 1.3

    base["mortgage_rate"] = base["base_rate"] + base["date"].map(_spread)
    return base[["date", "mortgage_rate"]]
