"""
Pre-OLS Validation Checks — UK Housing Market SDE Model
Stage 3c: Four validation checks confirming the engineered dataset
          is suitable for OLS regression before Stage 4.

Checks:
  1. Multicollinearity — Pearson correlation matrix + VIF
  2. Stationarity      — ADF + KPSS per region, averaged across 12 regions
  3. Missing values    — NaN audit after lagging; produce ols_ready_dataset.csv
  4. Outliers          — 3-sigma rule; OLS comparison with / without outliers

Inputs:
  data/processed/master_dataset_engineered.csv

Outputs (data/outputs/):
  correlation_matrix.png
  stationarity_report.txt
  outlier_report.txt
  price_growth_outliers.png
  pre_ols_validation_report.txt

Outputs (data/processed/):
  ols_ready_dataset.csv   ← used by Stage 4 OLS

Run:
    cd housing_model/
    python -m src.cleaning.pre_ols_checks
"""

from __future__ import annotations
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — no display required

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROC / "master_dataset_engineered.csv"
OLS_READY_FILE = PROC / "ols_ready_dataset.csv"
CORR_PLOT = OUT / "correlation_matrix.png"
STATIONARITY_TXT = OUT / "stationarity_report.txt"
OUTLIER_TXT = OUT / "outlier_report.txt"
OUTLIER_PLOT = OUT / "price_growth_outliers.png"
SUMMARY_TXT = OUT / "pre_ols_validation_report.txt"

# ── Variable sets ──────────────────────────────────────────────────────────────
OLS_INDEP = [
    "mortgage_rate_lag3",
    "earnings_lag1",
    "supply_lag6",
    "gross_yield_pct",
    "log_price_to_income",
    "price_growth_lag1",
]
OLS_DEP = "price_growth"

CORR_VARS = OLS_INDEP + [
    OLS_DEP,
    "base_rate",
    "mortgage_rate",
    "mortgage_rate_lag1",
    "log_income",
    "log_real_price",
]

STATIONARITY_VARS = [
    "price_growth",
    "mortgage_rate_lag3",
    "earnings_lag1",
    "supply_lag6",
    "gross_yield_pct",
    "log_price_to_income",
    "price_growth_lag1",
    "log_real_price",          # reference — expected non-stationary
]

REGIONS = [
    "East Midlands", "East of England", "London", "North East",
    "North West", "Northern Ireland", "Scotland", "South East",
    "South West", "Wales", "West Midlands", "Yorkshire and The Humber",
]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _hdr(title: str) -> None:
    """Print a section header."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _subhdr(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


def _load_data() -> pd.DataFrame:
    """
    Load the engineered dataset and derive price_growth_lag1 in-memory.
    The source file is never modified.
    """
    _hdr("LOADING DATA")
    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    df = df.sort_values(["region", "date"]).reset_index(drop=True)
    print(
        f"  Loaded {len(df):,} rows × {len(df.columns)} cols from {INPUT_FILE.name}")

    # price_growth_lag1 is not in the engineered dataset — derive it here
    df["price_growth_lag1"] = (
        df.groupby("region", sort=False)["price_growth"]
        .transform(lambda s: s.shift(1))
    )
    n_nan = df["price_growth_lag1"].isna().sum()
    print(f"  Derived price_growth_lag1: {df['price_growth_lag1'].notna().sum():,} valid, "
          f"{n_nan} NaN (1 per region — expected)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Multicollinearity
# ══════════════════════════════════════════════════════════════════════════════

def check1_multicollinearity(df: pd.DataFrame) -> tuple[str, list[str]]:
    """
    1a. Pearson correlation matrix for OLS variables + supporting variables.
    1b. VIF for planned OLS independent variables.

    Returns
    -------
    status : 'PASS' | 'WARNING' | 'FAIL'
    drop_recommendations : list of variable names flagged by VIF
    """
    _hdr("CHECK 1 — MULTICOLLINEARITY")

    # ── 1a  Correlation matrix ─────────────────────────────────────────────
    _subhdr("1a  Pearson Correlation Matrix")

    present = [v for v in CORR_VARS if v in df.columns]
    missing = [v for v in CORR_VARS if v not in df.columns]
    if missing:
        print(
            f"  WARNING: these columns not found and will be skipped: {missing}")

    corr_df = df[present].dropna().corr(method="pearson").round(2)

    # Print full matrix
    print("\n" + corr_df.to_string())

    # Flag high / severe pairs (upper triangle only to avoid duplicates)
    severe_pairs = []
    high_pairs = []
    n = len(present)
    for i in range(n):
        for j in range(i + 1, n):
            r = abs(corr_df.iloc[i, j])
            v1, v2 = present[i], present[j]
            if r > 0.90:
                severe_pairs.append((v1, v2, corr_df.iloc[i, j]))
            elif r > 0.80:
                high_pairs.append((v1, v2, corr_df.iloc[i, j]))

    print()
    if severe_pairs:
        print("  ⚠️  SEVERE multicollinearity (|r| > 0.90):")
        for v1, v2, r in severe_pairs:
            print(f"    {v1} ↔ {v2}: r = {r:+.2f}")
            print("    → Using both in OLS will make coefficients highly unstable.")
            print("      Consider dropping the less theoretically important variable.")
    else:
        print("  No SEVERE pairs (|r| > 0.90) found.")

    print()
    if high_pairs:
        print("  ⚠️  HIGH multicollinearity (|r| > 0.80):")
        for v1, v2, r in high_pairs:
            print(f"    {v1} ↔ {v2}: r = {r:+.2f}")
            print("    → Coefficients may be imprecise; monitor VIF before dropping.")
    else:
        print("  No HIGH pairs (|r| > 0.80) found.")

    # Heatmap
    fig, ax = plt.subplots(
        figsize=(max(10, len(present)), max(8, len(present) - 1)))
    sns.heatmap(
        corr_df,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Pearson Correlation Matrix — OLS Variables",
                 fontsize=13, pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(CORR_PLOT, dpi=150)
    plt.close(fig)
    print(f"\n  Heatmap saved → {CORR_PLOT.name}")

    # ── 1b  VIF ────────────────────────────────────────────────────────────
    _subhdr("1b  Variance Inflation Factors (VIF)")

    vif_vars = [v for v in OLS_INDEP if v in df.columns]
    vif_data = df[vif_vars].dropna()
    vif_matrix = sm.add_constant(vif_data)

    print(f"\n  {'Variable':<28}  {'VIF':>8}  Status")
    print(f"  {'─'*28}  {'─'*8}  ──────")

    drop_recs: list[str] = []
    vif_rows: list[tuple] = []
    for i, col in enumerate(vif_vars):
        # statsmodels VIF expects the full matrix including constant at col 0
        col_idx = list(vif_matrix.columns).index(col)
        v = variance_inflation_factor(vif_matrix.values, col_idx)
        if v > 10:
            status = "SEVERE ⚠️  — recommend DROP"
            drop_recs.append(col)
        elif v > 5:
            status = "MODERATE ⚠️  — monitor"
        else:
            status = "OK ✅"
        print(f"  {col:<28}  {v:>8.2f}  {status}")
        vif_rows.append((col, v))

    print()
    if drop_recs:
        print(
            f"  VIF RECOMMENDATION: consider dropping {drop_recs} before OLS.")
    else:
        print("  VIF RECOMMENDATION: no variables need to be dropped (all VIF ≤ 10).")

    # Determine status
    if severe_pairs or drop_recs:
        status = "WARNING"
    elif high_pairs:
        status = "WARNING"
    else:
        status = "PASS"

    print(f"\n  CHECK 1 STATUS: {status}")
    return status, drop_recs


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Stationarity
# ══════════════════════════════════════════════════════════════════════════════

def _run_adf(series: pd.Series) -> tuple[float, float]:
    """Return (ADF statistic, p-value) using AIC lag selection."""
    result = adfuller(series.dropna(), autolag="AIC")
    return float(result[0]), float(result[1])


def _run_kpss(series: pd.Series) -> tuple[float, float]:
    """Return (KPSS statistic, p-value) using constant regression."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(series.dropna(), regression="c", nlags="auto")
    return float(result[0]), float(result[1])


def _stationarity_verdict(adf_p: float, kpss_p: float) -> str:
    """Map ADF + KPSS p-values to a combined verdict string."""
    adf_stat = adf_p < 0.05
    kpss_stat = kpss_p > 0.05
    if adf_stat and kpss_stat:
        return "PASS ✅ — Both stationary — safe to use in OLS"
    elif adf_stat and not kpss_stat:
        return "LIKELY STATIONARY — ADF only — use with caution"
    elif not adf_stat and kpss_stat:
        return "LIKELY STATIONARY — KPSS only — use with caution"
    else:
        return "FAIL ⚠️ — Neither test stationary — needs differencing"


def check2_stationarity(df: pd.DataFrame) -> tuple[str, list[str]]:
    """
    Run ADF and KPSS stationarity tests on each OLS variable.
    Tests are applied per region; results are averaged across 12 regions.
    If a variable fails, first-differencing is applied and tests are re-run.

    Returns
    -------
    status : 'PASS' | 'WARNING' | 'FAIL'
    failed_vars : list of variable names that failed (even after differencing)
    """
    _hdr("CHECK 2 — STATIONARITY")

    lines: list[str] = [
        "=" * 60,
        "STATIONARITY REPORT — UK Housing Market Model",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 60,
        "",
        "Method: ADF (autolag=AIC) + KPSS (regression='c')",
        "Tests run per region; results averaged across 12 regions.",
        "",
    ]

    failed_vars:  list[str] = []
    warning_vars: list[str] = []

    vars_to_test = [v for v in STATIONARITY_VARS if v in df.columns]

    for var in vars_to_test:
        _subhdr(f"Variable: {var}")
        lines.append(f"{'─'*60}")
        lines.append(f"Variable: {var}")

        # Collect per-region ADF and KPSS results
        adf_stats, adf_pvals = [], []
        kpss_stats, kpss_pvals = [], []

        for region in REGIONS:
            series = df.loc[df["region"] == region, var].dropna()
            if len(series) < 20:
                continue
            try:
                a_stat, a_p = _run_adf(series)
                adf_stats.append(a_stat)
                adf_pvals.append(a_p)
            except Exception:
                pass
            try:
                k_stat, k_p = _run_kpss(series)
                kpss_stats.append(k_stat)
                kpss_pvals.append(k_p)
            except Exception:
                pass

        # Average across regions
        avg_adf_stat = float(np.mean(adf_stats)) if adf_stats else np.nan
        avg_adf_p = float(np.mean(adf_pvals)) if adf_pvals else np.nan
        avg_kpss_stat = float(np.mean(kpss_stats)) if kpss_stats else np.nan
        avg_kpss_p = float(np.mean(kpss_pvals)) if kpss_pvals else np.nan

        adf_verdict = "STATIONARY ✅" if avg_adf_p < 0.05 else "NON-STATIONARY ⚠️"
        kpss_verdict = "STATIONARY ✅" if avg_kpss_p > 0.05 else "NON-STATIONARY ⚠️"
        combined = _stationarity_verdict(avg_adf_p, avg_kpss_p)

        adf_line = (
            f"  ADF  stat={avg_adf_stat:>8.4f}  p={avg_adf_p:.4f}  → {adf_verdict}")
        kpss_line = (
            f"  KPSS stat={avg_kpss_stat:>8.4f}  p={avg_kpss_p:.4f}  → {kpss_verdict}")
        comb_line = (f"  Combined: {combined}")

        for ln in [adf_line, kpss_line, comb_line]:
            print(ln)
            lines.append(ln)

        # Handle failures
        if "FAIL" in combined:
            failed_vars.append(var)
            diff_msg = f"\n  → Applying first-differencing to '{var}'..."
            print(diff_msg)
            lines.append(diff_msg)

            # First difference per region
            diff_col = f"{var}_diff"
            df[diff_col] = (
                df.groupby("region", sort=False)[var]
                .transform(lambda s: s.diff(1))
            )

            # Re-test on differenced series
            d_adf_p_list, d_kpss_p_list = [], []
            for region in REGIONS:
                series = df.loc[df["region"] == region, diff_col].dropna()
                if len(series) < 20:
                    continue
                try:
                    _, dp = _run_adf(series)
                    d_adf_p_list.append(dp)
                except Exception:
                    pass
                try:
                    _, kp = _run_kpss(series)
                    d_kpss_p_list.append(kp)
                except Exception:
                    pass

            d_adf_p = float(np.mean(d_adf_p_list)) if d_adf_p_list else np.nan
            d_kpss_p = float(np.mean(d_kpss_p_list)
                             ) if d_kpss_p_list else np.nan
            d_verdict = _stationarity_verdict(d_adf_p, d_kpss_p)

            resolved = "PASS" in d_verdict
            res_msg = (
                f"  After differencing → {d_verdict}\n"
                f"  {'✅ Differencing resolved non-stationarity.' if resolved else '⚠️  Still non-stationary after differencing — review carefully.'}"
            )
            print(res_msg)
            lines.append(res_msg)

            if resolved:
                failed_vars.remove(var)
                warning_vars.append(var)

            print(f"  Saved differenced column: '{diff_col}'")
            lines.append(f"  Saved differenced column: '{diff_col}'")

        elif "caution" in combined.lower():
            warning_vars.append(var)

        lines.append("")

    # Status
    if failed_vars:
        stat_status = "FAIL"
    elif warning_vars:
        stat_status = "WARNING"
    else:
        stat_status = "PASS"

    summary_lines = [
        "",
        "=" * 60,
        "STATIONARITY SUMMARY",
        "=" * 60,
        f"  Failed (non-stationary, not resolved): {failed_vars if failed_vars else 'None'}",
        f"  Warnings (use with caution):           {warning_vars if warning_vars else 'None'}",
        f"  Status: {stat_status}",
    ]
    for ln in summary_lines:
        print(ln)
        lines.append(ln)

    # Save report
    STATIONARITY_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Stationarity report saved → {STATIONARITY_TXT.name}")

    return stat_status, failed_vars


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Missing values
# ══════════════════════════════════════════════════════════════════════════════

def check3_missing_values(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Audit NaN counts for every OLS variable after lagging.
    Drop all rows with any NaN in the OLS variable set.
    Save the clean OLS-ready dataset.

    Returns
    -------
    status     : 'PASS' | 'WARNING' | 'FAIL'
    ols_df     : cleaned DataFrame ready for OLS
    """
    _hdr("CHECK 3 — MISSING VALUES FROM LAGGING")

    ols_vars = [OLS_DEP] + OLS_INDEP

    # ── NaN audit ─────────────────────────────────────────────────────────
    _subhdr("Missing value count per OLS variable")
    print(f"\n  {'Variable':<28}  {'NaN count':>10}  {'%':>6}  Expected source")
    print(f"  {'─'*28}  {'─'*10}  {'─'*6}  ─────────────────────────")

    expected = {
        "price_growth":       ("1 per region (first obs)",      12),
        "price_growth_lag1":  ("2 per region (first 2 obs)",    24),
        "mortgage_rate_lag3": ("3 per region (first 3 months)", 36),
        "earnings_lag1":      ("1 per region (first obs)",      12),
        "supply_lag6":        ("6 per region (first 6 months)", 72),
        "gross_yield_pct":    ("data coverage gaps",            None),
        "log_price_to_income": ("derived — should be 0",         0),
    }

    any_unexpected = False
    for var in ols_vars:
        if var not in df.columns:
            print(f"  {var:<28}  {'COLUMN MISSING':>17}")
            continue
        n = df[var].isna().sum()
        pct = 100 * n / len(df)
        exp_str, exp_n = expected.get(var, ("—", None))
        flag = ""
        if exp_n is not None and n > exp_n * 1.1:
            flag = "  ← UNEXPECTED HIGH"
            any_unexpected = True
        print(f"  {var:<28}  {n:>10,}  {pct:>5.1f}%  {exp_str}{flag}")

    # ── Drop rows with any NaN in OLS variable set ─────────────────────────
    _subhdr("Dropping NaN rows to create OLS-ready dataset")

    avail_vars = [v for v in ols_vars if v in df.columns]
    rows_before = len(df)
    ols_df = df.dropna(subset=avail_vars).copy()
    rows_after = len(ols_df)
    dropped = rows_before - rows_after

    print(f"\n  Rows before drop:  {rows_before:,}")
    print(f"  Rows after drop:   {rows_after:,}")
    print(
        f"  Rows dropped:      {dropped}  (expected ~72 = 6 per region × 12 regions)")

    # Flag if dropped count is very different from expected 72
    if dropped > 120:
        print(
            f"  ⚠️  WARNING: dropped {dropped} rows — substantially more than expected 72.")
        any_unexpected = True
    elif dropped < 60:
        print(
            f"  ⚠️  NOTE: dropped only {dropped} rows — fewer than expected 72.")
    else:
        print("  ✅ Drop count is within expected range.")

    # ── Per-region row counts after drop ──────────────────────────────────
    _subhdr("Rows per region after dropping NaNs")
    region_counts = ols_df.groupby("region").size()
    print(f"\n  {'Region':<35}  {'Rows':>6}")
    print(f"  {'─'*35}  {'─'*6}")
    for region, count in region_counts.items():
        flag = "  ← UNEQUAL" if count != region_counts.iloc[0] else ""
        print(f"  {region:<35}  {count:>6}{flag}")

    unequal = region_counts.nunique() > 1
    if unequal:
        print("\n  ⚠️  WARNING: unequal row counts across regions.")
        any_unexpected = True
    else:
        print(
            f"\n  ✅ All regions have {region_counts.iloc[0]} rows — balanced panel.")

    # ── Save OLS-ready dataset ─────────────────────────────────────────────
    ols_df.to_csv(OLS_READY_FILE, index=False)
    print(f"\n  OLS-ready dataset saved → {OLS_READY_FILE.name}")
    print(f"  Shape: {ols_df.shape}  ({ols_df['region'].nunique()} regions × "
          f"{ols_df.groupby('region').size().iloc[0]} months)")

    status = "WARNING" if any_unexpected else "PASS"
    print(f"\n  CHECK 3 STATUS: {status}")
    return status, ols_df


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Outliers
# ══════════════════════════════════════════════════════════════════════════════

def _zscore(series: pd.Series) -> pd.Series:
    """Return z-scores for a series."""
    return (series - series.mean()) / series.std()


def check4_outliers(df: pd.DataFrame, ols_df: pd.DataFrame) -> str:
    """
    4a. Price growth outliers (3-sigma rule).
    4b. Supply shock outliers (3-sigma rule).
    4c. OLS comparison with / without outliers.

    Parameters
    ----------
    df     : full engineered DataFrame (for context)
    ols_df : OLS-ready DataFrame (NaNs already dropped)

    Returns
    -------
    status : 'PASS' | 'WARNING' | 'FAIL'
    """
    _hdr("CHECK 4 — OUTLIERS")

    lines: list[str] = [
        "=" * 60,
        "OUTLIER REPORT — UK Housing Market Model",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 60,
    ]

    overall_status = "PASS"

    # ── 4a  Price growth outliers ──────────────────────────────────────────
    _subhdr("4a  Price Growth Outliers (3-sigma rule)")

    pg_mean = ols_df["price_growth"].mean()
    pg_std = ols_df["price_growth"].std()
    threshold = 3 * pg_std

    ols_df = ols_df.copy()
    ols_df["pg_zscore"] = _zscore(ols_df["price_growth"])
    pg_outliers = ols_df[ols_df["pg_zscore"].abs() > 3].copy()

    lines.append(f"\n  price_growth: mean={pg_mean:.6f}  std={pg_std:.6f}  "
                 f"3σ threshold=±{threshold:.6f}")
    lines.append(f"  Outliers found: {len(pg_outliers)}")

    if len(pg_outliers) == 0:
        msg = "  ✅ No price growth outliers at 3-sigma threshold."
        print(msg)
        lines.append(msg)
    else:
        print(f"\n  Found {len(pg_outliers)} price growth outliers:")
        print(
            f"\n  {'Date':<12}  {'Region':<35}  {'price_growth':>13}  {'Z-score':>8}")
        print(f"  {'─'*12}  {'─'*35}  {'─'*13}  {'─'*8}")
        lines.append(
            f"\n  {'Date':<12}  {'Region':<35}  {'price_growth':>13}  {'Z-score':>8}")
        lines.append(f"  {'─'*12}  {'─'*35}  {'─'*13}  {'─'*8}")

        for _, row in pg_outliers.sort_values("pg_zscore", key=abs, ascending=False).iterrows():
            ln = (f"  {str(row['date'])[:10]:<12}  {row['region']:<35}  "
                  f"{row['price_growth']:>13.6f}  {row['pg_zscore']:>+8.2f}")
            print(ln)
            lines.append(ln)

        # Cluster by date
        _subhdr("Outlier clustering by date")
        date_counts = pg_outliers.groupby(
            pd.to_datetime(pg_outliers["date"]).dt.to_period("Q")
        ).size().sort_index()
        print(f"\n  {'Quarter':<10}  {'Count':>6}")
        print(f"  {'─'*10}  {'─'*6}")
        lines.append("\n  Outlier clustering by quarter:")
        for period, cnt in date_counts.items():
            ln = f"  {str(period):<10}  {cnt:>6}"
            print(ln)
            lines.append(ln)

        # Known events commentary
        lines.append("\n  Known economic events (for dissertation):")
        event_periods = {
            "2008": "2008–2009 Global Financial Crisis",
            "2009": "2008–2009 Global Financial Crisis",
            "2020": "COVID-19 pandemic disruption",
            "2022": "BoE rate shock / cost-of-living crisis",
        }
        pg_outliers_copy = pg_outliers.copy()
        pg_outliers_copy["year"] = pd.to_datetime(
            pg_outliers_copy["date"]).dt.year.astype(str)
        for yr, label in event_periods.items():
            n = (pg_outliers_copy["year"] == yr).sum()
            if n > 0:
                msg = f"    {yr}: {n} outlier(s) — likely {label}"
                print(msg)
                lines.append(msg)

        # Per-region count
        _subhdr("Outliers per region")
        region_out = pg_outliers.groupby("region").size()
        print(f"\n  {'Region':<35}  {'Outliers':>8}")
        lines.append("\n  Outliers per region:")
        for region in REGIONS:
            cnt = region_out.get(region, 0)
            ln = f"  {region:<35}  {cnt:>8}"
            print(ln)
            lines.append(ln)

        overall_status = "WARNING"

    # ── 4b  Supply shock outliers ──────────────────────────────────────────
    _subhdr("4b  Supply Shock Outliers (net_additions_monthly, 3-sigma)")

    sup_col = "net_additions_monthly"
    if sup_col in ols_df.columns:
        sup_mean = ols_df[sup_col].mean()
        sup_std = ols_df[sup_col].std()
        ols_df["sup_zscore"] = _zscore(ols_df[sup_col])
        sup_outliers = ols_df[ols_df["sup_zscore"].abs() > 3]

        lines.append(f"\n  {sup_col}: mean={sup_mean:.1f}  std={sup_std:.1f}  "
                     f"3σ threshold=±{3*sup_std:.1f}")
        lines.append(f"  Supply outliers found: {len(sup_outliers)}")

        if len(sup_outliers) == 0:
            msg = "  ✅ No supply outliers at 3-sigma threshold."
            print(msg)
            lines.append(msg)
        else:
            print(f"  Found {len(sup_outliers)} supply outliers.")
            lines.append("  NOTE: supply outliers are less critical for the SDE "
                         "but should be documented.")
            for _, row in sup_outliers.head(10).iterrows():
                ln = (f"  {str(row['date'])[:10]}  {row['region']:<35}  "
                      f"{row[sup_col]:>10.1f}  z={row['sup_zscore']:>+6.2f}")
                print(ln)
                lines.append(ln)
    else:
        print("  net_additions_monthly not available in OLS dataset.")

    # ── 4c  OLS comparison with / without outliers ─────────────────────────
    _subhdr("4c  OLS Comparison: With vs Without Outliers")

    avail_indep = [v for v in OLS_INDEP if v in ols_df.columns]
    y = ols_df[OLS_DEP]
    X_raw = sm.add_constant(ols_df[avail_indep])

    # With outliers
    model_with = sm.OLS(y, X_raw, missing="drop").fit()

    # Without outliers (drop rows where |pg_zscore| > 3)
    clean_mask = ols_df["pg_zscore"].abs() <= 3
    ols_clean = ols_df[clean_mask]
    y_clean = ols_clean[OLS_DEP]
    X_clean = sm.add_constant(ols_clean[avail_indep])
    model_without = sm.OLS(y_clean, X_clean, missing="drop").fit()

    # Build comparison table
    label_map = {
        "mortgage_rate_lag3":  "beta_rate",
        "earnings_lag1":       "beta_income",
        "supply_lag6":         "beta_supply",
        "gross_yield_pct":     "beta_yield",
        "log_price_to_income": "beta_pti",
        "price_growth_lag1":   "beta_lag",
    }

    print(
        f"\n  Outliers dropped in 'without' version: {len(ols_df) - len(ols_clean)}")
    print(
        f"\n  {'Metric':<28}  {'With outliers':>15}  {'Without outliers':>16}  {'Δ%':>6}")
    print(f"  {'─'*28}  {'─'*15}  {'─'*16}  {'─'*6}")

    comparison_lines = [
        "\n  OLS Comparison: With vs Without Outliers",
        f"  {'Metric':<28}  {'With outliers':>15}  {'Without outliers':>16}  {'Δ%':>6}",
        f"  {'─'*28}  {'─'*15}  {'─'*16}  {'─'*6}",
    ]

    # R-squared
    r2_w = model_with.rsquared
    r2_wo = model_without.rsquared
    delta_str = f"{100 * (r2_wo - r2_w) / abs(r2_w):+.1f}%" if r2_w != 0 else "—"
    ln = f"  {'R-squared':<28}  {r2_w:>15.4f}  {r2_wo:>16.4f}  {delta_str:>6}"
    print(ln)
    comparison_lines.append(ln)

    # Coefficients
    coef_changes_large = False
    for var, lbl in label_map.items():
        if var not in avail_indep:
            continue
        c_w = model_with.params.get(var, np.nan)
        c_wo = model_without.params.get(var, np.nan)
        if abs(c_w) > 1e-10:
            pct_change = 100 * abs(c_wo - c_w) / abs(c_w)
            delta_str = f"{pct_change:+.1f}%"
            if pct_change > 20:
                coef_changes_large = True
                delta_str += " ⚠️"
        else:
            delta_str = "—"
        ln = f"  {lbl:<28}  {c_w:>15.6f}  {c_wo:>16.6f}  {delta_str:>6}"
        print(ln)
        comparison_lines.append(ln)

    # Recommendation
    print()
    if coef_changes_large:
        rec = ("  RECOMMENDATION: Coefficients change >20% when outliers are removed.\n"
               "  → Remove outliers before OLS. Document the removed observations\n"
               "    in the dissertation methodology (cite economic events).")
        overall_status = "WARNING"
    else:
        rec = ("  RECOMMENDATION: Coefficients are stable (<20% change).\n"
               "  → Keep outliers in the main OLS specification.\n"
               "    Report the comparison table in a robustness footnote.")
    print(rec)
    lines.extend(comparison_lines)
    lines.append(rec)

    # ── Plot: price_growth time series per region, outliers in red ─────────
    _subhdr("Saving outlier plot")

    pg_regions = [r for r in REGIONS if r in ols_df["region"].values]
    ncols = 3
    nrows = int(np.ceil(len(pg_regions) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(18, nrows * 3), sharey=True)
    axes = axes.flatten()

    for idx, region in enumerate(pg_regions):
        ax = axes[idx]
        sub = ols_df[ols_df["region"] == region].sort_values("date")
        out = sub[sub["pg_zscore"].abs() > 3]

        ax.plot(pd.to_datetime(sub["date"]), sub["price_growth"],
                color="#4472C4", linewidth=0.9, alpha=0.85)
        if len(out) > 0:
            ax.scatter(pd.to_datetime(out["date"]), out["price_growth"],
                       color="red", s=30, zorder=5, label="Outlier (|z|>3)")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(region, fontsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(axis="both", labelsize=6)
        ax.yaxis.set_major_formatter(
            mtick.PercentFormatter(xmax=1, decimals=1))

    # Hide unused subplots
    for idx in range(len(pg_regions), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Price Growth: Monthly (2005–2023)  |  Red = Outlier (|z| > 3)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(OUTLIER_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Outlier plot saved → {OUTLIER_PLOT.name}")

    # Save report
    OUTLIER_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Outlier report saved → {OUTLIER_TXT.name}")
    print(f"\n  CHECK 4 STATUS: {overall_status}")
    return overall_status


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def _final_summary(
    status1: str,
    status2: str,
    status3: str,
    status4: str,
    ols_df: pd.DataFrame,
    drop_recs: list[str],
    failed_stats: list[str],
) -> None:
    """Print and save the GO / NO-GO summary."""

    _hdr("PRE-OLS VALIDATION SUMMARY")

    all_statuses = [status1, status2, status3, status4]
    if "FAIL" in all_statuses:
        recommendation = "RESOLVE ISSUES FIRST — do not proceed to OLS"
    elif "WARNING" in all_statuses:
        recommendation = "PROCEED WITH CAUTION — address warnings where possible"
    else:
        recommendation = "PROCEED TO OLS ✅"

    to_drop = list(set(drop_recs + failed_stats))

    lines = [
        "=" * 60,
        "PRE-OLS VALIDATION SUMMARY",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 60,
        "",
        f"CHECK 1 — Multicollinearity:   {status1}",
        f"CHECK 2 — Stationarity:        {status2}",
        f"CHECK 3 — Missing values:      {status3}",
        f"CHECK 4 — Outliers:            {status4}",
        "",
        f"OLS-READY DATASET:  {OLS_READY_FILE}",
        f"ROWS:               {len(ols_df):,}",
        f"VARIABLES TO DROP:  {to_drop if to_drop else 'None — all variables cleared'}",
        "",
        f"RECOMMENDATION:     {recommendation}",
        "=" * 60,
    ]

    for ln in lines:
        print(ln)

    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Summary saved → {SUMMARY_TXT.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  UK Housing Market Model — Pre-OLS Validation Checks")
    print(f"  Run time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    # Load
    df = _load_data()

    # Check 1
    status1, drop_recs = check1_multicollinearity(df)

    # Check 2
    status2, failed_stats = check2_stationarity(df)

    # Check 3
    status3, ols_df = check3_missing_values(df)

    # Check 4
    status4 = check4_outliers(df, ols_df)

    # Summary
    _final_summary(status1, status2, status3, status4,
                   ols_df, drop_recs, failed_stats)


if __name__ == "__main__":
    main()
