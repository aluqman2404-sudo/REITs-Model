"""
Stage 3: Data Cleaning and Merging Pipeline
UK Housing Market Model — Dissertation

Reads 12 raw CSV files from data/raw/, applies the following transformations,
and produces a single master dataset ready for OLS calibration and SDE modelling:

  Step 1  — Load all raw files, standardise date columns to month-start YYYY-MM-DD
  Step 2  — Merge 9-region MHCLG supply with Scotland/Wales/NI supply
  Step 3  — Interpolate annual series (supply, earnings, population) to monthly
  Step 4  — CPI-deflate house prices, earnings, and rents (base = 2015-01-01)
  Step 5  — Derive price-to-income ratio (real house price / real annual income)
  Step 6  — Left-join all variables into one master DataFrame
  Step 7  — Run data quality checks and print a full report
  Step 8  — Save master_dataset.csv, supply_combined.csv, quality report, cleaning log

Output:
  data/processed/master_dataset.csv     — 2,892 rows × 16 columns
  data/processed/supply_combined.csv    — 300 rows  × 3 columns
  data/processed/data_quality_report.txt
  data/processed/cleaning_log.txt

Run:
    cd housing_model/
    python -m src.cleaning.clean_and_merge
"""

import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# ── Model parameters ──────────────────────────────────────────────────────────
MODEL_START = pd.Timestamp("2005-01-01")
# Extended 2026-03-31: LR HPI covers 2026-01-01. Transactions/approvals/
# unemployment forward-filled 3 months from 2025-10-01. Governance: 6-month forward-fill
# limit per INTERPOLATED_FEATURE_POLICY.md. Triggers test_data_freshness threshold reset.
MODEL_END = pd.Timestamp("2026-01-01")
CPI_BASE_DATE = pd.Timestamp("2015-01-01")

# 253 monthly periods: Jan 2005 → Jan 2026
MONTHLY_IDX = pd.date_range(MODEL_START, MODEL_END, freq="MS")

# Governance limit for forward-filling flow series (transactions, approvals, unemployment)
FLOW_SERIES_FFILL_LIMIT = 6  # months

REGIONS = [
    "East Midlands", "East of England", "London", "North East", "North West",
    "Northern Ireland", "Scotland", "South East", "South West", "Wales",
    "West Midlands", "Yorkshire and The Humber",
]

# ── Cleaning log (written to file in Step 8) ──────────────────────────────────
_LOG: list[str] = []


def _log(msg: str = "") -> None:
    """Append timestamped message to the cleaning log and print it."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}" if msg else ""
    _LOG.append(line)
    print(line)


def _find_raw(pattern: str) -> Path:
    """
    Return the most recent file in data/raw/ whose name matches the glob pattern.
    Raises FileNotFoundError with a helpful message if nothing matches.
    """
    matches = sorted(RAW.glob(pattern), reverse=True)
    if not matches:
        present = [f.name for f in sorted(RAW.glob("*.csv"))]
        raise FileNotFoundError(
            f"No file matching '{pattern}' in {RAW}\n"
            f"Files present: {present}"
        )
    return matches[0]


def _check_flow_series_staleness() -> None:
    """Check forward-fill staleness for flow series not handled in this pipeline.

    transactions, mortgage_approvals, and unemployment are loaded by
    src/data/integrate_downloads.py, not here.  This function reads their raw CSV
    files, computes how many months MODEL_END extends beyond the last observation,
    logs a WARNING for each affected series, and raises ValueError if any series
    would be forward-filled beyond the 6-month governance limit defined in
    INTERPOLATED_FEATURE_POLICY.md.

    Called from main() before the pipeline steps run.
    """
    flow_specs = [
        ("transactions",       "hmrc_transactions*.csv"),
        # exact name avoids matching _raw.csv
        ("mortgage_approvals", "boe_mortgage_approvals.csv"),
        ("unemployment",       "ons_regional_unemployment*.csv"),
    ]
    for series_name, pattern in flow_specs:
        try:
            path = _find_raw(pattern)
            raw = pd.read_csv(path)
            date_col = next(
                (c for c in raw.columns if c.lower() == "date"), None)
            if date_col is None:
                _log(
                    f"  WARNING: no date column in {path.name} — skipping staleness check for {series_name}")
                continue
            raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
            last_obs = raw[date_col].max()
            n_months = (
                (MODEL_END.year - last_obs.year) * 12 +
                (MODEL_END.month - last_obs.month)
            )
            if n_months > 0:
                _log(
                    f"  WARNING: {series_name} forward-filled {n_months} months beyond "
                    f"last observation {last_obs.date()}. Governance limit: "
                    f"{FLOW_SERIES_FFILL_LIMIT} months."
                )
                if n_months > FLOW_SERIES_FFILL_LIMIT:
                    raise ValueError(
                        f"Flow series '{series_name}' would be forward-filled {n_months} months "
                        f"(last obs: {last_obs.date()}, MODEL_END: {MODEL_END.date()}). "
                        f"Governance limit is {FLOW_SERIES_FFILL_LIMIT} months per "
                        "INTERPOLATED_FEATURE_POLICY.md. Rebuild requires a data refresh "
                        "or explicit governance approval."
                    )
        except FileNotFoundError:
            _log(
                f"  WARNING: raw file for '{series_name}' not found "
                f"(pattern '{pattern}') — skipping staleness check"
            )


def _to_month_start(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """
    Coerce a date column to month-start (1st of month) Timestamps.
    Handles strings, datetime objects, and mixed formats.
    """
    df = df.copy()
    df[col] = (
        pd.to_datetime(df[col], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load and standardise
# ══════════════════════════════════════════════════════════════════════════════

def step1_load_and_standardise() -> dict[str, pd.DataFrame]:
    """
    Load all 12 raw CSV files and standardise date columns.

    Each file is identified by a glob pattern so that timestamp-suffixed
    filenames (e.g. land_registry_hpi_202603.csv) are found automatically.
    Only the columns needed downstream are retained.

    Returns a dict: short_name → cleaned DataFrame.
    """
    _log("=" * 60)
    _log("STEP 1 — Loading and standardising raw files")
    _log("=" * 60)

    # Map: short key → (glob pattern, columns to keep)
    file_specs = {
        "hpi":           ("land_registry_hpi*.csv",
                          ["date", "region", "average_price", "index"]),
        "mortgage":      ("boe_mortgage_rate*.csv",
                          ["date", "mortgage_rate"]),
        "base_rate":     ("boe_base_rate*.csv",
                          ["date", "base_rate"]),
        "supply_eng":    ("mhclg_supply_2*.csv",            # avoids matching _missing_
                          ["date", "region", "net_additions"]),
        "supply_ext":    ("mhclg_supply_missing*.csv",
                          ["date", "region", "net_additions"]),
        "earnings":      ("ons_earnings_regional*.csv",
                          ["date", "region", "median_annual_earnings"]),
        "cpi":           ("ons_cpi*.csv",
                          ["date", "cpi"]),
        "rent_levels":   ("ons_rental_levels*.csv",
                          ["date", "region", "median_monthly_rent"]),
        "yield":         ("rental_yield*.csv",
                          ["date", "region", "gross_yield_pct"]),
        "rental_index":  ("ons_rental_index*.csv",
                          ["date", "region", "rental_index"]),
        "population":    ("ons_population*.csv",
                          ["date", "region", "population"]),
        "affordability": ("ons_affordability_ratio*.csv",
                          ["date", "region", "price_to_income_ratio"]),
    }

    dfs = {}
    for key, (pattern, want_cols) in file_specs.items():
        path = _find_raw(pattern)
        raw = pd.read_csv(path)

        _log(f"  Loaded {path.name}  ({len(raw):,} rows)")
        _log(f"    Columns in file: {list(raw.columns)}")

        # Keep only required columns that actually exist in this file
        keep = [c for c in want_cols if c in raw.columns]
        raw = raw[keep].copy()

        # Standardise date
        raw = _to_month_start(raw, "date")
        n_bad = raw["date"].isna().sum()
        if n_bad:
            _log(f"    WARNING: {n_bad} unparseable date(s) dropped")
            raw = raw.dropna(subset=["date"])

        dfs[key] = raw
        _log(f"    → {key}: {len(raw):,} rows | "
             f"{raw['date'].min().date()} → {raw['date'].max().date()}")

    return dfs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Merge housing supply
# ══════════════════════════════════════════════════════════════════════════════

def step2_merge_supply(dfs: dict) -> pd.DataFrame:
    """
    Concatenate the two supply files vertically:
      mhclg_supply        → 9 English regions (225 rows)
      mhclg_supply_missing → Scotland, Wales, NI (75 rows)

    Expected result: 300 rows (12 regions × 25 financial years).
    Saves supply_combined.csv to data/processed/.
    """
    _log("")
    _log("=" * 60)
    _log("STEP 2 — Merging housing supply files")
    _log("=" * 60)

    eng = dfs["supply_eng"]
    ext = dfs["supply_ext"]

    _log(f"  English supply:  {len(eng):>3} rows | "
         f"regions: {sorted(eng['region'].unique())}")
    _log(f"  Extended supply: {len(ext):>3} rows | "
         f"regions: {sorted(ext['region'].unique())}")

    # Safety: remove any overlapping regions from ext
    overlap = set(eng["region"].unique()) & set(ext["region"].unique())
    if overlap:
        _log(f"  WARNING: overlapping regions {overlap} — removing from ext")
        ext = ext[~ext["region"].isin(overlap)]

    combined = (
        pd.concat([eng, ext], ignore_index=True)
        .sort_values(["region", "date"])
        .reset_index(drop=True)
    )

    n_reg = combined["region"].nunique()
    n_yr = combined["date"].nunique()
    _log(
        f"  Combined: {len(combined)} rows | {n_reg} regions | {n_yr} annual dates")

    # Warn if expected 300 rows not met
    if len(combined) != 300:
        _log(f"  NOTE: expected 300 rows (12 × 25), got {len(combined)}")

    path = PROC / "supply_combined.csv"
    combined.to_csv(path, index=False)
    _log("  Saved → supply_combined.csv")

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Interpolate annual series to monthly
# ══════════════════════════════════════════════════════════════════════════════

def _interpolate_annual_to_monthly(
    df: pd.DataFrame,
    value_col: str,
    label: str,
    ffill_limit: int = 3,
) -> pd.DataFrame:
    """
    Interpolate an annual-frequency regional series to MONTHLY_IDX using
    time-based linear interpolation between the known annual anchor dates.

    Method:
      1. For each region, set the known annual values as anchors.
      2. Reindex to the union of anchor dates + MONTHLY_IDX.
      3. Apply pandas time-based interpolation (linear in calendar time).
      4. Forward/back-fill at edges (max ffill_limit months).
      5. Select only MONTHLY_IDX dates.

    This preserves the shape of the annual series exactly at its anchors
    and fills in between with straight-line interpolation.
    """
    frames = []
    for region in REGIONS:
        sub = df[df["region"] == region][["date", value_col]].copy()

        if sub.empty:
            _log(
                f"    WARNING: no '{label}' data for region '{region}' — filling NaN")
            frames.append(pd.DataFrame({
                "date":    MONTHLY_IDX,
                "region":  region,
                value_col: np.nan,
            }))
            continue

        # Deduplicate and sort by date
        series = (
            sub.drop_duplicates("date")
            .set_index("date")
            .sort_index()[value_col]
        )

        # Union of known anchor dates and monthly target index
        all_dates = series.index.union(MONTHLY_IDX)
        series = series.reindex(all_dates)

        # Linear interpolation across calendar time between anchors
        series = series.interpolate(method="time")

        # Fill any remaining edge NaNs (e.g. model window starts before first obs)
        series = series.ffill(limit=ffill_limit).bfill(limit=ffill_limit)

        # Restrict to model monthly index only
        series = series.reindex(MONTHLY_IDX)

        frames.append(pd.DataFrame({
            "date":    MONTHLY_IDX,
            "region":  region,
            value_col: series.values,
        }))

    return pd.concat(frames, ignore_index=True)


def step3_interpolate_annual(dfs: dict, supply_combined: pd.DataFrame) -> dict:
    """
    Interpolate three annual series to monthly frequency:
      - Housing supply    (April anchor dates → monthly net_additions_monthly)
      - Regional earnings (December anchor dates → monthly median_annual_earnings)
      - Population        (June anchor dates → monthly population)

    Each produces 12 regions × 223 months = 2,676 rows.
    """
    _log("")
    _log("=" * 60)
    _log("STEP 3 — Interpolating annual series to monthly")
    _log("=" * 60)

    # --- Housing supply ---
    _log("  Housing supply (annual April → monthly)...")
    before = len(supply_combined)
    supply_m = _interpolate_annual_to_monthly(
        supply_combined, "net_additions", "supply"
    )
    # Rename to final column name
    supply_m = supply_m.rename(
        columns={"net_additions": "net_additions_monthly"})
    n_null = supply_m["net_additions_monthly"].isna().sum()
    _log(f"    {before} annual rows → {len(supply_m):,} monthly rows | {n_null} NaN")
    dfs["supply_monthly"] = supply_m

    # --- Regional earnings ---
    _log("  Regional earnings (annual Dec → monthly)...")
    before = len(dfs["earnings"])
    earn_m = _interpolate_annual_to_monthly(
        dfs["earnings"], "median_annual_earnings", "earnings"
    )
    n_null = earn_m["median_annual_earnings"].isna().sum()
    _log(f"    {before} annual rows → {len(earn_m):,} monthly rows | {n_null} NaN")
    dfs["earnings_monthly"] = earn_m

    # --- Population ---
    _log("  Population (annual June → monthly)...")
    before = len(dfs["population"])
    pop_m = _interpolate_annual_to_monthly(
        dfs["population"], "population", "population"
    )
    n_null = pop_m["population"].isna().sum()
    _log(f"    {before} annual rows → {len(pop_m):,} monthly rows | {n_null} NaN")
    dfs["population_monthly"] = pop_m

    return dfs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CPI deflation
# ══════════════════════════════════════════════════════════════════════════════

def step4_cpi_deflation(dfs: dict) -> dict:
    """
    Deflate nominal monetary series to real terms using ONS CPI.
    Base month: 2015-01-01 (CPI = cpi_base at that date).

    Formula:  real_value = nominal_value / (cpi_t / cpi_base)

    Series deflated:
      average_price          → real_house_price      (land registry HPI)
      median_annual_earnings → real_annual_earnings  (earnings_monthly)
      median_monthly_rent    → real_monthly_rent      (rental levels)

    Also filters the HPI to the model window [2005-01-01, 2023-07-01] and
    renames columns for the master dataset.
    """
    _log("")
    _log("=" * 60)
    _log("STEP 4 — CPI deflation (base = 2015-01-01)")
    _log("=" * 60)

    # Build CPI lookup: date → cpi value
    cpi_series = dfs["cpi"].drop_duplicates("date").set_index("date")["cpi"]

    # Get base CPI at 2015-01-01
    if CPI_BASE_DATE in cpi_series.index:
        cpi_base = float(cpi_series.loc[CPI_BASE_DATE])
    else:
        # Fall back to nearest available date
        nearest = cpi_series.index[abs(
            cpi_series.index - CPI_BASE_DATE).argmin()]
        cpi_base = float(cpi_series.loc[nearest])
        _log(
            f"  WARNING: 2015-01-01 not in CPI. Using nearest: {nearest.date()}")

    _log(f"  CPI base value at 2015-01-01 = {cpi_base:.4f}")
    _log(f"  >>> Record in dissertation: CPI_base = {cpi_base:.4f} <<<")

    def _deflate(df: pd.DataFrame, nominal_col: str, real_col: str) -> pd.DataFrame:
        """Join CPI on date and compute real column."""
        df = df.copy()
        df["_cpi_t"] = df["date"].map(cpi_series)
        n_miss = df["_cpi_t"].isna().sum()
        if n_miss:
            _log(
                f"    WARNING: {n_miss} rows missing CPI match — forward-filling")
            df["_cpi_t"] = df["_cpi_t"].ffill().bfill()
        # Deflation: nominal / (cpi_t / cpi_base)
        df[real_col] = df[nominal_col] / (df["_cpi_t"] / cpi_base)
        return df.drop(columns=["_cpi_t"])

    # --- House prices (filter to model window first) ---
    hpi = dfs["hpi"].copy()
    n_before = len(hpi)
    hpi = hpi[(hpi["date"] >= MODEL_START) & (hpi["date"] <= MODEL_END)].copy()
    _log(f"  HPI: {n_before:,} rows → {len(hpi):,} after model-window filter")

    hpi = _deflate(hpi, "average_price", "real_house_price")
    hpi = hpi.rename(columns={
        "average_price": "nominal_house_price",
        "index":         "hpi_index",
    })
    _log(f"  Nominal house price range: "
         f"£{hpi['nominal_house_price'].min():,.0f} – £{hpi['nominal_house_price'].max():,.0f}")
    _log(f"  Real house price range:    "
         f"£{hpi['real_house_price'].min():,.0f} – £{hpi['real_house_price'].max():,.0f}")
    dfs["hpi_clean"] = hpi

    # --- Earnings (already monthly-interpolated) ---
    earn = dfs["earnings_monthly"].copy()
    earn = _deflate(earn, "median_annual_earnings", "real_annual_earnings")
    _log(f"  Earnings deflated: {len(earn):,} rows")
    dfs["earnings_real"] = earn

    # --- Monthly rents (filter to model window) ---
    rents = dfs["rent_levels"].copy()
    rents = rents[(rents["date"] >= MODEL_START) &
                  (rents["date"] <= MODEL_END)].copy()
    rents = _deflate(rents, "median_monthly_rent", "real_monthly_rent")
    _log(f"  Rents deflated: {len(rents):,} rows")
    dfs["rents_real"] = rents

    return dfs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Derive price-to-income ratio
# ══════════════════════════════════════════════════════════════════════════════

def step5_derive_pti(dfs: dict) -> dict:
    """
    Derive the price-to-income ratio (PTI) in annual income multiples.

    Formula:
        price_to_income_ratio_derived = real_house_price / real_annual_earnings

    This is equivalent to:
        (real_house_price / (real_annual_earnings / 12)) / 12
    i.e. house price as a multiple of annual real income — the standard
    UK convention used by the ONS Housing Affordability bulletin.

    Merged on [date, region] with an inner join so both series must have
    observations for the same date-region combinations.
    """
    _log("")
    _log("=" * 60)
    _log("STEP 5 — Deriving price-to-income ratio")
    _log("=" * 60)

    hpi = dfs["hpi_clean"][["date", "region", "real_house_price"]].copy()
    earn = dfs["earnings_real"][[
        "date", "region", "real_annual_earnings"]].copy()

    merged = hpi.merge(earn, on=["date", "region"], how="inner")

    # PTI = house price / annual income (annual income multiples)
    merged["price_to_income_ratio_derived"] = (
        merged["real_house_price"] / merged["real_annual_earnings"]
    )

    n_nan = merged["price_to_income_ratio_derived"].isna().sum()
    n_inf = np.isinf(merged["price_to_income_ratio_derived"]).sum()
    _log(f"  PTI derived: {len(merged):,} rows | {n_nan} NaN | {n_inf} inf")

    # Print min/max per region so any outliers are visible immediately
    _log("  PTI range by region (annual income multiples):")
    stats = merged.groupby("region")["price_to_income_ratio_derived"].agg(
        ["min", "max", "mean"])
    for region, row in stats.iterrows():
        _log(f"    {region:<35}  "
             f"min={row['min']:.1f}  max={row['max']:.1f}  mean={row['mean']:.1f}")

    dfs["pti_derived"] = merged[["date", "region", "price_to_income_ratio_derived"]]
    return dfs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Build master dataset
# ══════════════════════════════════════════════════════════════════════════════

def _expand_to_monthly(
    df: pd.DataFrame,
    value_col: str,
    method: str = "ffill",
) -> pd.DataFrame:
    """
    Expand a sub-monthly (annual or quarterly) regional DataFrame to MONTHLY_IDX
    using forward-fill within each region.

    Used for:
      - Official affordability ratio (annual Dec values → monthly)
      - Rental yield (quarterly Jan/Apr/Jul/Oct values → monthly)

    For each region: reindex to the union of known dates + MONTHLY_IDX,
    forward-fill, then select only MONTHLY_IDX.
    """
    frames = []
    for region in REGIONS:
        sub = df[df["region"] == region][["date", value_col]].copy()
        if sub.empty:
            frames.append(pd.DataFrame({
                "date":    MONTHLY_IDX,
                "region":  region,
                value_col: np.nan,
            }))
            continue

        series = sub.drop_duplicates("date").set_index(
            "date").sort_index()[value_col]
        all_dates = series.index.union(MONTHLY_IDX)
        series = series.reindex(all_dates).ffill().bfill()
        series = series.reindex(MONTHLY_IDX)

        frames.append(pd.DataFrame({
            "date":    MONTHLY_IDX,
            "region":  region,
            value_col: series.values,
        }))
    return pd.concat(frames, ignore_index=True)


def step6_build_master(dfs: dict) -> pd.DataFrame:
    """
    Left-join all cleaned variables onto the Land Registry HPI base.

    Join rules:
      - Regional variables (house prices, supply, earnings, yield, rents,
        population, PTI) → join on [date, region]
      - National variables (mortgage_rate, base_rate, cpi) → join on [date] only;
        the same national value is broadcast to all 12 regions.

    Final column order (16 columns):
      date, region,
      nominal_house_price, real_house_price, hpi_index,
      price_to_income_ratio_derived, price_to_income_ratio_official,
      mortgage_rate, base_rate,
      net_additions_monthly,
      real_annual_earnings,
      gross_yield_pct, rental_index, real_monthly_rent,
      population, cpi
    """
    _log("")
    _log("=" * 60)
    _log("STEP 6 — Building master dataset")
    _log("=" * 60)

    # ── Base: HPI already filtered to model window in Step 4 ──────────────────
    _base_cols = ["date", "region", "nominal_house_price",
                  "real_house_price", "hpi_index"]
    _avail = [c for c in _base_cols if c in dfs["hpi_clean"].columns]
    master = dfs["hpi_clean"][_avail].copy()
    _log(f"  Base (LR HPI):  {len(master):,} rows | "
         f"{master['date'].nunique()} months | {master['region'].nunique()} regions")

    missing_reg = [r for r in REGIONS if r not in master["region"].values]
    if missing_reg:
        _log(f"  WARNING: regions missing from HPI base: {missing_reg}")

    def _join(right: pd.DataFrame, keys: list[str], label: str) -> pd.DataFrame:
        """Left-join right onto master and log null count for new column(s)."""
        nonlocal master
        new_cols = [c for c in right.columns if c not in keys]
        master = master.merge(right.drop_duplicates(keys), on=keys, how="left")
        for col in new_cols:
            n = master[col].isna().sum()
            _log(f"    {label:<32}  {col}: {n} NaN")
        return master

    # ── National variables (broadcast to all regions via date-only join) ──────
    _log("  Joining national variables...")
    _join(dfs["mortgage"].drop_duplicates("date"),   ["date"], "Mortgage rate")
    _join(dfs["base_rate"].drop_duplicates("date"),  ["date"], "Base rate")
    _join(dfs["cpi"].drop_duplicates("date"),        ["date"], "CPI")

    # ── Regional: price-to-income ratio (derived) ─────────────────────────────
    _log("  Joining regional variables...")
    _join(dfs["pti_derived"], ["date", "region"], "PTI derived")

    # ── Official affordability ratio: annual → forward-filled monthly ─────────
    aff = dfs["affordability"].rename(
        columns={"price_to_income_ratio": "price_to_income_ratio_official"}
    )
    aff_monthly = _expand_to_monthly(aff, "price_to_income_ratio_official")
    _join(aff_monthly, ["date", "region"], "PTI official")

    # ── Supply (monthly-interpolated in Step 3) ───────────────────────────────
    _join(
        dfs["supply_monthly"][["date", "region", "net_additions_monthly"]],
        ["date", "region"], "Supply"
    )

    # ── Real earnings (monthly-interpolated + deflated) ───────────────────────
    _join(
        dfs["earnings_real"][["date", "region", "real_annual_earnings"]],
        ["date", "region"], "Real earnings"
    )

    # ── Rental yield: quarterly → forward-filled monthly ─────────────────────
    yield_monthly = _expand_to_monthly(dfs["yield"], "gross_yield_pct")
    yield_monthly = yield_monthly[
        (yield_monthly["date"] >= MODEL_START) & (
            yield_monthly["date"] <= MODEL_END)
    ]
    _join(yield_monthly, ["date", "region"], "Gross yield")

    # ── Rental index (monthly, already model-window aligned) ─────────────────
    ri = dfs["rental_index"].copy()
    ri = ri[(ri["date"] >= MODEL_START) & (ri["date"] <= MODEL_END)]
    _join(ri, ["date", "region"], "Rental index")

    # ── Real monthly rent (monthly, model window) ─────────────────────────────
    _join(
        dfs["rents_real"][["date", "region", "real_monthly_rent"]],
        ["date", "region"], "Real monthly rent"
    )

    # ── Population (monthly-interpolated in Step 3) ───────────────────────────
    _join(
        dfs["population_monthly"][["date", "region", "population"]],
        ["date", "region"], "Population"
    )

    # ── Final column ordering (16 columns as specified) ───────────────────────
    col_order = [
        "date", "region",
        "nominal_house_price", "real_house_price", "hpi_index",
        "price_to_income_ratio_derived", "price_to_income_ratio_official",
        "mortgage_rate", "base_rate",
        "net_additions_monthly",
        "real_annual_earnings",
        "gross_yield_pct", "rental_index", "real_monthly_rent",
        "population", "cpi",
    ]
    # Safety: add any columns missing from master as NaN
    for col in col_order:
        if col not in master.columns:
            _log(f"  WARNING: column '{col}' missing — adding as NaN")
            master[col] = np.nan

    master = (
        master[col_order]
        .sort_values(["region", "date"])
        .reset_index(drop=True)
    )

    _log(f"  Master dataset shape: {master.shape}")
    return master


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Data quality checks
# ══════════════════════════════════════════════════════════════════════════════

def step7_quality_checks(master: pd.DataFrame) -> str:
    """
    Run comprehensive data quality checks on the master dataset.
    Prints all results and returns the full report as a string for saving.
    """
    _log("")
    _log("=" * 60)
    _log("STEP 7 — Data quality checks")
    _log("=" * 60)

    lines: list[str] = []

    def p(s: str = "") -> None:
        """Append line to report and print it."""
        lines.append(s)
        _log("  " + s if s else "")

    p("=" * 60)
    p("DATA QUALITY REPORT")
    p(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    p(f"Model window: {MODEL_START.date()} → {MODEL_END.date()}")
    p("=" * 60)

    # ── Shape check ───────────────────────────────────────────────────────────
    p()
    p("SHAPE CHECK")
    p(f"  Actual:   {master.shape}")
    p("  Expected: (2892, 16)  [12 regions × 241 months × 16 columns]")
    if master.shape == (2892, 16):
        p("  Result:   PASS")
    else:
        rows_diff = master.shape[0] - 2892
        cols_diff = master.shape[1] - 16
        p(f"  Result:   NOTE  (rows diff={rows_diff:+d}, cols diff={cols_diff:+d})")

    # ── Date range ────────────────────────────────────────────────────────────
    p()
    p("DATE RANGE CHECK")
    actual_start = master["date"].min()
    actual_end = master["date"].max()
    p(f"  Actual:   {actual_start.date()} → {actual_end.date()}")
    p(f"  Expected: {MODEL_START.date()} → {MODEL_END.date()}")
    if actual_start == MODEL_START and actual_end == MODEL_END:
        p("  Result:   PASS")
    else:
        p("  Result:   NOTE")

    # ── Region check ──────────────────────────────────────────────────────────
    p()
    p("REGION CHECK")
    present = sorted(master["region"].unique())
    missing = [r for r in REGIONS if r not in present]
    p(f"  Regions present ({len(present)}): {present}")
    if missing:
        p(f"  FAIL — missing: {missing}")
    else:
        p("  Result: PASS — all 12 regions present")

    # Equal row counts per region
    row_counts = master.groupby("region").size()
    if row_counts.nunique() == 1:
        p(f"  Row counts: PASS — {row_counts.iloc[0]} rows per region")
    else:
        p(f"  Row counts: UNEQUAL — {row_counts.to_dict()}")

    # ── Missing values ────────────────────────────────────────────────────────
    p()
    p("MISSING VALUES (per column)")
    p(f"  {'Column':<44} {'Count':>7}  {'%':>6}")
    p(f"  {'-'*44} {'-'*7}  {'-'*6}")
    total = len(master)
    any_warn = False
    for col in master.columns:
        n = master[col].isna().sum()
        pct = n / total * 100
        warn = "  *** WARNING > 5% missing ***" if pct > 5 else ""
        if pct > 5:
            any_warn = True
        p(f"  {col:<44} {n:>7,}  {pct:>5.1f}%{warn}")
    if not any_warn:
        p("  All columns within acceptable missing threshold (<5%).")

    # ── Derived vs official PTI correlation ──────────────────────────────────
    p()
    p("DERIVED vs OFFICIAL PRICE-TO-INCOME RATIO")
    valid = master.dropna(
        subset=["price_to_income_ratio_derived",
                "price_to_income_ratio_official"]
    )
    if len(valid) >= 10:
        corr = valid["price_to_income_ratio_derived"].corr(
            valid["price_to_income_ratio_official"]
        )
        mad = (
            valid["price_to_income_ratio_derived"] -
            valid["price_to_income_ratio_official"]
        ).abs().mean()
        bias = (
            valid["price_to_income_ratio_derived"] -
            valid["price_to_income_ratio_official"]
        ).mean()
        p(f"  Valid paired observations:  {len(valid):,}")
        p(f"  Pearson correlation:        {corr:.4f}")
        p(f"  Mean absolute difference:   {mad:.4f}")
        p(f"  Mean bias (derived−official): {bias:+.4f}")
        if corr > 0.95:
            p("  Result: PASS — very high correlation (r > 0.95)")
        elif corr > 0.90:
            p("  Result: PASS — strong correlation (r > 0.90)")
        else:
            p(f"  Result: NOTE — r = {corr:.3f}. Review methodology.")
    else:
        p("  Insufficient overlapping non-null values for comparison.")

    # ── Sanity range checks ───────────────────────────────────────────────────
    p()
    p("SANITY RANGE CHECKS")
    p(f"  {'Column':<44} {'Min':>12}  {'Max':>12}  Result")
    p(f"  {'-'*44} {'-'*12}  {'-'*12}  ------")

    checks = [
        ("real_house_price",               50_000,  800_000, "£",  True),
        ("nominal_house_price",            50_000, 1_200_000, "£", False),
        ("mortgage_rate",                   1.0,      10.0,   "%", True),
        ("base_rate",                       0.0,      10.0,   "%", False),
        ("gross_yield_pct",                 2.0,      12.0,   "%", True),
        ("price_to_income_ratio_derived",   3.0,      20.0,   "x", True),
        ("net_additions_monthly",           0.0,   10_000.0,  "",  True),
        ("real_annual_earnings",        10_000,  100_000,     "£", True),
        ("real_monthly_rent",              200,    5_000,     "£", True),
    ]
    for col, lo, hi, unit, strict in checks:
        if col not in master.columns:
            p(f"  {col:<44} {'COLUMN MISSING':>28}")
            continue
        col_data = master[col].dropna()
        if col_data.empty:
            p(f"  {col:<44} {'ALL NaN':>28}")
            continue
        mn = col_data.min()
        mx = col_data.max()
        fail = strict and (mn < lo or mx > hi)
        status = "NOTE" if fail else "PASS"
        p(f"  {col:<44} {mn:>11.1f}{unit}  {mx:>11.1f}{unit}  {status}")
        if fail:
            if mn < lo:
                bad = master[master[col] < lo][["region", "date", col]].head(3)
                p(f"    Below {lo}{unit}: {bad.to_string(index=False)}")
            if mx > hi:
                n_bad = (master[col] > hi).sum()
                p(f"    Above {hi}{unit}: {n_bad} rows")

    # ── Summary statistics per region ─────────────────────────────────────────
    p()
    p("SUMMARY STATISTICS BY REGION (mean across model window)")
    stat_cols = [
        "real_house_price", "price_to_income_ratio_derived",
        "mortgage_rate", "gross_yield_pct",
        "net_additions_monthly", "real_annual_earnings",
        "real_monthly_rent", "population",
    ]
    for col in stat_cols:
        if col not in master.columns:
            continue
        p()
        p(f"  {col}:")
        p(f"    {'Region':<35}  {'Min':>10}  {'Max':>10}  {'Mean':>10}  {'Std':>8}")
        p(f"    {'-'*35}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
        stats = master.groupby("region")[col].agg(
            ["min", "max", "mean", "std"])
        for region, row in stats.iterrows():
            p(f"    {region:<35}  {row['min']:>10.2f}  {row['max']:>10.2f}  "
              f"{row['mean']:>10.2f}  {row['std']:>8.2f}")

    p()
    p("=" * 60)
    p("END OF QUALITY REPORT")
    p("=" * 60)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Save outputs
# ══════════════════════════════════════════════════════════════════════════════

def step8_save_outputs(master: pd.DataFrame, quality_report: str) -> None:
    """
    Save all outputs to data/processed/:
      master_dataset.csv       — the full merged dataset
      data_quality_report.txt  — output of all Step 7 checks
      cleaning_log.txt         — timestamped log of every transformation
    """
    _log("")
    _log("=" * 60)
    _log("STEP 8 — Saving outputs")
    _log("=" * 60)

    # Master dataset
    path_master = PROC / "master_dataset.csv"
    master.to_csv(path_master, index=False)
    _log(
        f"  Saved master_dataset.csv  ({len(master):,} rows × {len(master.columns)} cols)")

    # Data quality report
    path_qr = PROC / "data_quality_report.txt"
    path_qr.write_text(quality_report, encoding="utf-8")
    _log("  Saved data_quality_report.txt")

    # Cleaning log (append the final save entries too before writing)
    path_log = PROC / "cleaning_log.txt"
    path_log.write_text("\n".join(_LOG), encoding="utf-8")
    _log("  Saved cleaning_log.txt")

    # ── Final printed summary ─────────────────────────────────────────────────
    _log("")
    _log("=" * 60)
    _log("FINAL SUMMARY")
    _log("=" * 60)
    _log(f"  Rows:        {len(master):,}")
    _log(f"  Columns:     {len(master.columns)}  {list(master.columns)}")
    _log(
        f"  Date range:  {master['date'].min().date()} → {master['date'].max().date()}")
    _log(f"  Regions:     {sorted(master['region'].unique())}")

    null_counts = master.isna().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        _log("  Missing:     None — dataset is complete")
    else:
        _log("  Missing values:")
        for col, n in null_counts.items():
            _log(f"    {col}: {n} ({n / len(master) * 100:.1f}%)")

    _log("")
    _log("  Stage 3 complete.")
    _log("  Output → data/processed/master_dataset.csv")
    _log("  Next   → Stage 4: OLS calibration and SDE parameter estimation.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _log("UK Housing Market Model — Stage 3: Data Cleaning & Merging")
    _log(f"Run time:     {datetime.now():%Y-%m-%d %H:%M:%S}")
    _log(f"Model window: {MODEL_START.date()} → {MODEL_END.date()} "
         f"({len(MONTHLY_IDX)} months)")
    _log(f"Regions:      {len(REGIONS)}")
    _log(f"RAW data dir: {RAW}")
    _log(f"OUTPUT dir:   {PROC}")

    # Pre-flight: check forward-fill governance for flow series
    _check_flow_series_staleness()

    # Step 1: Load and standardise all raw files
    dfs = step1_load_and_standardise()

    # Step 2: Merge English + devolved supply → supply_combined.csv
    supply_combined = step2_merge_supply(dfs)
    dfs["supply_combined"] = supply_combined

    # Step 3: Interpolate annual series (supply, earnings, population) → monthly
    dfs = step3_interpolate_annual(dfs, supply_combined)

    # Step 4: CPI-deflate house prices, earnings, rents → real terms
    dfs = step4_cpi_deflation(dfs)

    # Step 5: Derive price-to-income ratio
    dfs = step5_derive_pti(dfs)

    # Step 6: Merge everything into one master DataFrame
    master = step6_build_master(dfs)

    # Step 7: Data quality checks and report
    quality_report = step7_quality_checks(master)

    # Step 8: Save all outputs
    step8_save_outputs(master, quality_report)


if __name__ == "__main__":
    main()
