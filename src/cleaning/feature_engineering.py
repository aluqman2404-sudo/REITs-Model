"""
Stage 3b: Feature engineering for the UK housing SDE model.

Loads data/processed/master_dataset.csv and produces
data/processed/master_dataset_engineered.csv with:

  Transformation 1 — Log variables
  Transformation 2 — Price growth (1m, 3m, 12m), within-region
  Transformation 3 — Lagged macro variables (lags 1, 3, 6, 12), within-region
  Transformation 4 — Monthly frequency consistency check

All diff / shift operations are applied per region to avoid
cross-region contamination at region boundaries.

Output:
  data/processed/master_dataset_engineered.csv  — 2,676 rows × 34 columns
  data/processed/feature_engineering_log.txt    — timestamped transformation log

Run:
    cd housing_model/
    python -m src.cleaning.feature_engineering
"""

from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

INPUT_FILE  = PROC / "master_dataset.csv"
OUTPUT_FILE = PROC / "master_dataset_engineered.csv"
LOG_FILE    = PROC / "feature_engineering_log.txt"

# ── Model window ──────────────────────────────────────────────────────────────
MODEL_START = pd.Timestamp("2005-01-01")
MODEL_END   = pd.Timestamp("2025-10-01")

REGIONS = [
    "East Midlands", "East of England", "London", "North East",
    "North West", "Northern Ireland", "Scotland", "South East",
    "South West", "Wales", "West Midlands", "Yorkshire and The Humber",
]

# ── Log buffer (written to file at end) ───────────────────────────────────────
_LOG: list[str] = []


def _log(msg: str = "") -> None:
    """Append a timestamped line to the log buffer and print it."""
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}" if msg else ""
    _LOG.append(line)
    print(line)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_log(series: pd.Series, label: str) -> pd.Series:
    """Log of a series; warn if any non-positive values are present."""
    n_bad = (series <= 0).sum()
    if n_bad > 0:
        _log(f"  WARNING: {n_bad} non-positive values in '{label}' — will produce NaN")
    return np.log(series)


def _per_region_diff(df: pd.DataFrame, col: str, periods: int) -> pd.Series:
    """First-difference of `col` within each region (never across boundaries)."""
    return df.groupby("region", sort=False)[col].transform(lambda s: s.diff(periods))


def _per_region_shift(df: pd.DataFrame, col: str, lag: int) -> pd.Series:
    """Lag (shift) of `col` within each region."""
    return df.groupby("region", sort=False)[col].transform(lambda s: s.shift(lag))


# ── Transformations ───────────────────────────────────────────────────────────

def transformation1_log_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformation 1: Log-transform 6 variables.

    log_real_price         = log(real_house_price)
    log_income             = log(real_annual_earnings)
    log_population         = log(population)
    log_rent               = log(real_monthly_rent)      [NaN where rent missing]
    log_supply             = log(net_additions_monthly)
    log_price_to_income    = log(price_to_income_ratio_derived)
    """
    _log("=" * 60)
    _log("Transformation 1: Log variables")
    _log("=" * 60)

    df["log_real_price"]      = _safe_log(df["real_house_price"],                     "real_house_price")
    df["log_income"]          = _safe_log(df["real_annual_earnings"],                  "real_annual_earnings")
    df["log_population"]      = _safe_log(df["population"],                            "population")
    df["log_rent"]            = _safe_log(df["real_monthly_rent"].clip(lower=1e-6),    "real_monthly_rent")
    df["log_supply"]          = _safe_log(df["net_additions_monthly"].clip(lower=1),   "net_additions_monthly")
    df["log_price_to_income"] = _safe_log(df["price_to_income_ratio_derived"],         "price_to_income_ratio_derived")

    for col in ["log_real_price", "log_income", "log_population",
                "log_supply", "log_price_to_income"]:
        n = df[col].isna().sum()
        _log(f"  {col}: {df[col].notna().sum()} valid, {n} NaN")

    _log(f"  log_rent: {df['log_rent'].notna().sum()} valid, "
         f"{df['log_rent'].isna().sum()} NaN (structural — devolved nations pre-2009)")

    return df


def transformation2_price_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformation 2: Price growth variables (within-region diffs of log price).

    price_growth     = Δ log_real_price (1-month continuously compounded return)
    price_growth_3m  = log_real_price.diff(3)  / 3   (per-month avg over 3m window)
    price_growth_12m = log_real_price.diff(12) / 12  (per-month avg over 12m window)

    Note: diff(3)/3 and diff(12)/12 express the per-month average rate over the
    window — consistent units with price_growth for regression use.
    """
    _log("")
    _log("=" * 60)
    _log("Transformation 2: Price growth variables (within-region)")
    _log("=" * 60)

    df = df.sort_values(["region", "date"]).reset_index(drop=True)

    df["price_growth"]     = _per_region_diff(df, "log_real_price", 1)
    df["price_growth_3m"]  = _per_region_diff(df, "log_real_price", 3)  / 3
    df["price_growth_12m"] = _per_region_diff(df, "log_real_price", 12) / 12

    for col in ["price_growth", "price_growth_3m", "price_growth_12m"]:
        n_nan = df[col].isna().sum()
        _log(f"  {col}: {df[col].notna().sum()} valid, {n_nan} NaN "
             f"(first obs per region — expected)")

    return df


def transformation3_lagged_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformation 3: Lagged macro / fundamental variables (within-region shifts).

    Lag 1  : mortgage_rate_lag1, base_rate_lag1, earnings_lag1, yield_lag1
    Lag 3  : mortgage_rate_lag3, supply_lag3
    Lag 6  : supply_lag6
    Lag 12 : supply_lag12, earnings_lag12
    """
    _log("")
    _log("=" * 60)
    _log("Transformation 3: Lagged variables (within-region)")
    _log("=" * 60)

    lag_spec = [
        # (source_column,              lag, output_name)
        ("mortgage_rate",              1,  "mortgage_rate_lag1"),
        ("base_rate",                  1,  "base_rate_lag1"),
        ("real_annual_earnings",       1,  "earnings_lag1"),
        ("gross_yield_pct",            1,  "yield_lag1"),
        ("mortgage_rate",              3,  "mortgage_rate_lag3"),
        ("net_additions_monthly",      3,  "supply_lag3"),
        ("net_additions_monthly",      6,  "supply_lag6"),
        ("net_additions_monthly",      12, "supply_lag12"),
        ("real_annual_earnings",       12, "earnings_lag12"),
    ]

    for src, lag, out in lag_spec:
        df[out] = _per_region_shift(df, src, lag)
        n_nan = df[out].isna().sum()
        _log(f"  {out} (lag={lag}): {df[out].notna().sum()} valid, {n_nan} NaN")

    return df


def transformation4_frequency_check(df: pd.DataFrame) -> None:
    """
    Transformation 4: Verify monthly frequency consistency.

    Checks:
      1. No duplicate (date, region) pairs
      2. All expected 2,676 grid points present (223 months × 12 regions)
      3. Per-region date gaps are all exactly 1 month
      4. Date range matches model window
    """
    _log("")
    _log("=" * 60)
    _log("Transformation 4: Frequency consistency checks")
    _log("=" * 60)

    expected_months = len(pd.date_range(MODEL_START, MODEL_END, freq="MS"))
    expected_rows   = expected_months * len(REGIONS)

    # 1. Duplicates
    n_dup  = df.duplicated(subset=["date", "region"]).sum()
    status = "PASS" if n_dup == 0 else "FAIL"
    _log(f"  Duplicate (date, region) pairs: {n_dup}  [{status}]")

    # 2. Grid completeness
    n_rows = len(df)
    status = "PASS" if n_rows == expected_rows else "FAIL"
    _log(f"  Total rows: {n_rows} (expected {expected_rows})  [{status}]")

    # 3. Per-region month gaps
    max_gap_days    = 0
    problem_regions = []
    for region, grp in df.groupby("region"):
        dates = grp["date"].sort_values().reset_index(drop=True)
        gaps  = dates.diff().dropna().dt.days
        max_g = gaps.max()
        if max_g > 31:   # >31 days → gap larger than one month
            problem_regions.append((region, max_g))
        max_gap_days = max(max_gap_days, max_g)

    if not problem_regions:
        _log(f"  Max date gap across all regions: {max_gap_days:.0f} days  [PASS]")
    else:
        _log("  Date gaps > 1 month detected:  [FAIL]")
        for r, g in problem_regions:
            _log(f"    {r}: {g:.0f} days")

    # 4. Date range
    actual_start = df["date"].min()
    actual_end   = df["date"].max()
    ok     = (actual_start == MODEL_START) and (actual_end == MODEL_END)
    status = "PASS" if ok else "FAIL"
    _log(f"  Date range: {actual_start:%Y-%m-%d} → {actual_end:%Y-%m-%d}  [{status}]")


# ── Correlation summary ───────────────────────────────────────────────────────

def log_lag_correlation_table(df: pd.DataFrame) -> None:
    """Log correlation of lagged predictors vs price_growth."""
    _log("")
    _log("=" * 60)
    _log("Correlation with price_growth (lagged predictors)")
    _log("=" * 60)

    lag_cols = [
        "mortgage_rate_lag1", "base_rate_lag1", "earnings_lag1", "yield_lag1",
        "mortgage_rate_lag3", "supply_lag3", "supply_lag6", "supply_lag12",
        "earnings_lag12",
    ]
    target = df["price_growth"].dropna()
    rows   = []
    for col in lag_cols:
        aligned = df.loc[target.index, col].dropna()
        idx     = target.index.intersection(aligned.index)
        if len(idx) > 50:
            r = target.loc[idx].corr(aligned.loc[idx])
            rows.append((col, f"{r:+.4f}", len(idx)))
        else:
            rows.append((col, "  n/a  ", len(idx)))

    col_w = max(len(r[0]) for r in rows) + 2
    _log(f"  {'Predictor':<{col_w}}  {'r':>8}  {'n':>6}")
    _log(f"  {'-'*col_w}  {'--------'}  {'------'}")
    for name, r, n in rows:
        _log(f"  {name:<{col_w}}  {r:>8}  {n:>6,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_feature_engineering() -> pd.DataFrame:
    """
    Main entry point. Runs all 4 transformations, logs results,
    saves the engineered dataset and the transformation log.
    """
    _log("UK Housing Market Model — Stage 3b: Feature Engineering")
    _log(f"Run time:    {datetime.now():%Y-%m-%d %H:%M:%S}")
    _log(f"Input:       {INPUT_FILE}")
    _log(f"Output:      {OUTPUT_FILE}")
    _log(f"Log:         {LOG_FILE}")

    # Load master dataset
    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    _log(f"\nLoaded {len(df):,} rows from {INPUT_FILE.name}")
    _log(f"Source columns ({len(df.columns)}): {list(df.columns)}")

    # Sort by region + date before all per-region operations
    df = df.sort_values(["region", "date"]).reset_index(drop=True)

    # Apply transformations
    df = transformation1_log_variables(df)
    df = transformation2_price_growth(df)
    df = transformation3_lagged_variables(df)

    # Quality check
    transformation4_frequency_check(df)

    # Correlation summary
    log_lag_correlation_table(df)

    # Column summary
    _log("")
    _log("=" * 60)
    _log("FINAL SUMMARY")
    _log("=" * 60)
    _log(f"  Final dataset shape: {df.shape}")

    engineered_cols = [c for c in df.columns if c not in [
        "date", "region", "nominal_house_price", "real_house_price",
        "hpi_index", "price_to_income_ratio_derived",
        "price_to_income_ratio_official", "mortgage_rate", "base_rate",
        "net_additions_monthly", "real_annual_earnings", "gross_yield_pct",
        "rental_index", "real_monthly_rent", "population", "cpi",
    ]]
    _log(f"  New engineered columns ({len(engineered_cols)}): {engineered_cols}")

    _log("")
    _log("  Missing values in engineered columns:")
    any_missing = False
    for col in engineered_cols:
        n   = df[col].isna().sum()
        pct = 100 * n / len(df)
        if n > 0:
            _log(f"    {col}: {n} ({pct:.1f}%)")
            any_missing = True
    if not any_missing:
        _log("    None — all engineered columns complete")

    # Save dataset
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    _log(f"\n  Saved {len(df):,} rows × {len(df.columns)} columns → {OUTPUT_FILE.name}")

    # Save log
    LOG_FILE.write_text("\n".join(_LOG), encoding="utf-8")
    _log(f"  Saved log → {LOG_FILE.name}")

    _log("")
    _log("  Stage 3b complete.")
    _log("  Next → Stage 4: OLS calibration and SDE parameter estimation.")
    _log("=" * 60)

    return df


if __name__ == "__main__":
    run_feature_engineering()
