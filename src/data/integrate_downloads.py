"""
Integrate manually-downloaded UK housing data into master_dataset_v2.

Processes five downloaded files:
  1. BoE LPMVTVX  → approvals_growth (national, monthly)
  2. MHCLG Table 213 → starts_lag6, starts_lag12 (national quarterly → monthly)
  3. ONS HI00     → unemployment_rate per region (quarterly rolling → monthly)
  4. HMRC SDLT    → transaction_growth per nation (monthly)
  5. MHCLG P153   → planning speed (diagnostic only — not a supply-volume proxy)

Selection rule: add to v2 spec if |t| > 2.0 and correct sign.
Saves updated master_dataset_v2.csv and re-runs OLS comparison.

Run:
    cd housing_model/
    python -m src.data.integrate_downloads
"""

from linearmodels.panel import PanelOLS
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from pathlib import Path
import scipy.stats as sp_stats
import statsmodels.api as sm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "outputs"
VIMP = OUT / "variable_improvement"
FIN_DIR = OUT / "stage4_final"

REGIONS = [
    "East Midlands", "East of England", "London", "North East",
    "North West", "Northern Ireland", "Scotland", "South East",
    "South West", "Wales", "West Midlands", "Yorkshire and The Humber",
]
MONTH_COLS = [f"month_{m}" for m in range(2, 13)]
REGIME_COLS = ["regime_gfc", "regime_stamp_duty"]
TRAIN_END = pd.Timestamp("2015-12-01")
TEST_START = pd.Timestamp("2016-01-01")
MODEL_START = pd.Timestamp("2005-01-01")
MODEL_END = pd.Timestamp("2025-10-01")

OLD_V2_RMSE = 0.01509   # Spec B from build_improved_variables.py


# ── Helpers ───────────────────────────────────────────────────────────────────

def _h(t): print(f"\n{'='*60}\n{t}\n{'='*60}")


def _adf_panel(df, col):
    pvals = []
    for _, grp in df.groupby(level="region"):
        s = grp[col].dropna()
        if len(s) >= 20:
            try:
                pvals.append(adfuller(s.values, autolag="AIC")[1])
            except Exception:
                pass
    if not pvals:
        return {"median_p": np.nan, "stationary": False}
    med = float(np.median(pvals))
    return {"median_p": med, "stationary": med < 0.10}


def _stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _univariate_test(df, var, dep="price_growth"):
    clean = df.dropna(subset=[dep, var])
    if len(clean) < 50:
        return {"coef": np.nan, "t_stat": np.nan, "p_val": np.nan,
                "r2_within": np.nan, "n_obs": 0}
    y = clean[dep]
    X = sm.add_constant(clean[[var]], has_constant="add")
    try:
        res = PanelOLS(y, X, entity_effects=True).fit(
            cov_type="clustered", cluster_entity=True)
        c = res.params.get(var, np.nan)
        se = res.std_errors.get(var, np.nan)
        t = c / se if (not np.isnan(se) and se > 0) else np.nan
        p = float(res.pvalues.get(var, np.nan))
        return {"coef": c, "t_stat": t, "p_val": p,
                "r2_within": float(res.rsquared), "n_obs": len(clean)}
    except Exception as e:
        return {"coef": np.nan, "t_stat": np.nan, "p_val": np.nan,
                "r2_within": np.nan, "n_obs": 0, "error": str(e)}


def _driscoll_kraay_se(model_result, X, bandwidth=12):
    resids = model_result.resids
    shared = X.index.intersection(resids.index)
    X_ = X.loc[shared].values.astype(float)
    e_ = resids.loc[shared].values.astype(float)
    dates = X.loc[shared].index.get_level_values("date")
    dates_u = sorted(set(dates))
    T, k = len(dates_u), X_.shape[1]
    d_map = {d: i for i, d in enumerate(dates_u)}
    S = np.zeros((T, k))
    for i in range(len(X_)):
        S[d_map[dates[i]]] += X_[i] * e_[i]
    Omega = np.zeros((k, k))
    for t in range(T):
        Omega += np.outer(S[t], S[t])
    for lag in range(1, bandwidth + 1):
        w = 1 - lag / (bandwidth + 1)
        gl = np.zeros((k, k))
        for t in range(lag, T):
            gl += np.outer(S[t], S[t - lag])
        Omega += w * (gl + gl.T)
    Omega /= T
    XX = X_.T @ X_ / T
    try:
        XX_inv = np.linalg.inv(XX)
    except Exception:
        XX_inv = np.linalg.pinv(XX)
    return np.sqrt(np.diag((XX_inv @ Omega @ XX_inv) / T))


def _extrapolate_terminal_rent(df: pd.DataFrame) -> pd.DataFrame:
    """
    For any region where real_monthly_rent is NaN at MODEL_END (or was
    flagged as extrapolated in a prior run), extrapolate using the trailing
    6-month average YoY growth rate applied to the same calendar month in
    the prior year.

    Idempotent: if rent_extrapolated=True was preserved from a prior run
    (because main() no longer resets the flag), this function is a no-op.
    If the column was absent (fresh base file), it is initialised to False.

    Writes 'rent_extrapolated' flag column (True for extrapolated rows).
    Logs each extrapolation to stdout.
    """
    if "rent_extrapolated" not in df.columns:
        df["rent_extrapolated"] = False

    for region, grp in df.groupby(level="region"):
        if MODEL_END not in grp.index.get_level_values("date"):
            continue
        already_flagged = bool(df.loc[(region, MODEL_END), "rent_extrapolated"]) \
            if (region, MODEL_END) in df.index else False
        val_at_end = df.loc[(region, MODEL_END), "real_monthly_rent"] \
            if (region, MODEL_END) in df.index else np.nan
        needs_extrap = pd.isna(val_at_end) or already_flagged
        if not needs_extrap:
            continue
        if pd.notna(val_at_end):
            # Value already filled from prior run — just ensure flag is set
            df.loc[(region, MODEL_END), "rent_extrapolated"] = True
        else:
            # Trailing 6-month avg YoY
            valid = grp["real_monthly_rent"].dropna()
            yoy = valid.pct_change(12).dropna().tail(6)
            if len(yoy) < 3:
                print(
                    f"  WARN: {region} — fewer than 3 valid YoY obs; skipping extrapolation")
                continue
            avg_yoy = float(yoy.mean())
            # Anchor: same calendar month one year prior
            prior_date = MODEL_END - pd.DateOffset(years=1)
            prior_date_ts = pd.Timestamp(prior_date).replace(day=1)
            if (region, prior_date_ts) not in df.index:
                print(
                    f"  WARN: {region} — prior-year anchor {prior_date_ts:%Y-%m-%d} missing; skipping")
                continue
            prior_val = df.loc[(region, prior_date_ts), "real_monthly_rent"]
            if pd.isna(prior_val):
                print(
                    f"  WARN: {region} — prior-year rent is NaN; skipping extrapolation")
                continue
            extrap_val = float(prior_val) * (1.0 + avg_yoy)
            df.loc[(region, MODEL_END), "real_monthly_rent"] = extrap_val
            df.loc[(region, MODEL_END), "rent_extrapolated"] = True
            # Recompute derived columns so they are not NaN downstream
            if "log_rent" in df.columns:
                df.loc[(region, MODEL_END), "log_rent"] = np.log(extrap_val)
            if "price_to_rent" in df.columns:
                # Formula matches build_improved_variables: real_house_price / (real_monthly_rent * 12)
                real_hp = df.loc[(region, MODEL_END), "real_house_price"] \
                    if "real_house_price" in df.columns else np.nan
                if np.isfinite(real_hp) and extrap_val > 0:
                    df.loc[(region, MODEL_END), "price_to_rent"] = \
                        float(real_hp) / (extrap_val * 12.0)
                if "log_ptr" in df.columns and np.isfinite(df.loc[(region, MODEL_END), "price_to_rent"]):
                    import math
                    df.loc[(region, MODEL_END), "log_ptr"] = \
                        math.log(
                            max(df.loc[(region, MODEL_END), "price_to_rent"], 1e-6))
            print(
                f"  Extrapolated from 6m trailing YoY: {region}, "
                f"{MODEL_END:%Y-%m-%d}, value={extrap_val:.4f}, "
                f"method=trailing_6m_yoy_growth"
            )

    return df


def _entity_effects(params, train_df, feat_cols):
    y_m = train_df["price_growth"].groupby(level="region").mean()
    X_m = train_df[feat_cols].groupby(level="region").mean()
    const = params.get("const", 0.0)
    beta = params.drop("const", errors="ignore")
    avail = [c for c in feat_cols if c in beta.index]
    return {r: float(y_m[r] - const - beta[avail] @ X_m.loc[r, avail])
            for r in REGIONS if r in y_m.index and r in X_m.index}


def _predict_fe(params, test_df, feat_cols, alpha):
    const = params.get("const", 0.0)
    beta = params.drop("const", errors="ignore")
    avail = [c for c in feat_cols if c in beta.index and c in test_df.columns]
    preds = np.full(len(test_df), const)
    preds += test_df[avail].values @ beta[avail].values
    for i, reg in enumerate(test_df.index.get_level_values("region")):
        preds[i] += alpha.get(reg, 0.0)
    return preds


def _walk_forward(df, exog_list, stamp_coef=0.0):
    dates = df.index.get_level_values("date")
    exog_wf = [c for c in exog_list if c != "regime_stamp_duty"]
    train_df = df[dates <= TRAIN_END].dropna(subset=exog_wf + ["price_growth"])
    test_df = df[dates >= TEST_START].dropna(subset=exog_wf + ["price_growth"])
    actual = test_df["price_growth"].values
    res_wf = PanelOLS(
        train_df["price_growth"],
        sm.add_constant(train_df[exog_wf], has_constant="add"),
        entity_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)
    alpha = _entity_effects(res_wf.params, train_df, exog_wf)
    preds = _predict_fe(res_wf.params, test_df, exog_wf, alpha)
    preds += stamp_coef * test_df["regime_stamp_duty"].values
    err = actual - preds
    return (float(np.sqrt(np.nanmean(err**2))),
            float(np.nanmean(np.abs(err))),
            float(np.nanmean(np.sign(actual) == np.sign(preds))) * 100)


# ══════════════════════════════════════════════════════════════════════════════
# 1. BoE MORTGAGE APPROVALS
# ══════════════════════════════════════════════════════════════════════════════

def parse_boe_approvals(df: pd.DataFrame) -> pd.DataFrame:
    _h("1. BoE MORTGAGE APPROVALS (LPMVTVX)")

    raw = (RAW / "boe_mortgage_approvals_raw.csv").read_text()
    lines = raw.strip().splitlines()

    # Find the line that starts with "DATE"
    data_start = next(i for i, l in enumerate(lines) if l.startswith("DATE"))
    data = pd.read_csv(
        __import__("io").StringIO("\n".join(lines[data_start:])),
        parse_dates=["DATE"], dayfirst=True
    )
    data.columns = ["date", "approvals"]
    data["date"] = pd.to_datetime(data["date"], dayfirst=True)
    # Align to month-start
    data["date"] = data["date"].dt.to_period("M").dt.to_timestamp()
    data = data.set_index("date").sort_index()
    data = data.loc[MODEL_START:MODEL_END]

    print(f"  BoE approvals: {len(data)} monthly obs "
          f"({data.index.min():%Y-%m} → {data.index.max():%Y-%m})")
    print(
        f"  Range: {data.approvals.min():,.0f} – {data.approvals.max():,.0f}")

    # Save clean version
    data.reset_index().to_csv(RAW / "boe_mortgage_approvals.csv", index=False)

    # Drop any pre-existing approvals columns to prevent _x/_y suffix conflicts
    # when integrate_downloads is re-run against an already-augmented v2 file.
    _drop = [c for c in ["approvals", "approvals_lag1",
                         "approvals_growth"] if c in df.columns]
    if _drop:
        df = df.drop(columns=_drop)

    # Broadcast to all regions
    df = df.reset_index()
    df = df.merge(data.reset_index().rename(columns={"date": "date"}),
                  on="date", how="left")
    df = df.set_index(["region", "date"]).sort_index()

    df["approvals_lag1"] = df.groupby(level="region")["approvals"].transform(
        lambda s: s.shift(1))
    df["approvals_growth"] = df.groupby(level="region")["approvals"].transform(
        lambda s: s.pct_change(12))

    # Test
    adf = _adf_panel(df, "approvals_growth")
    test = _univariate_test(df, "approvals_growth")
    corr = df[["approvals_growth", "price_growth"]].dropna().corr().iloc[0, 1]
    sign = "✅ correct (+)" if test["coef"] > 0 else "⚠️ wrong (-)"
    print(f"  approvals_growth: Coef={test['coef']:+.6f}  "
          f"t={test['t_stat']:+.2f}{_stars(test['p_val'])}  "
          f"R²={test['r2_within']:.4f}  corr={corr:.3f}  {sign}")
    print(
        f"  ADF median p={adf['median_p']:.4f}  stationary={adf['stationary']}")

    return df, test


# ══════════════════════════════════════════════════════════════════════════════
# 2. MHCLG HOUSING STARTS TABLE 213
# ══════════════════════════════════════════════════════════════════════════════

def parse_housing_starts(df: pd.DataFrame) -> pd.DataFrame:
    _h("2. MHCLG HOUSING STARTS TABLE 213")

    raw = pd.read_excel(
        RAW / "mhclg_housing_starts_213.ods", engine="odf",
        sheet_name="LT_213_quarterly", header=None
    )
    # Row 4 = header, data from row 5
    raw.columns = raw.iloc[4]
    raw = raw.iloc[5:].reset_index(drop=True)
    raw = raw.rename(columns={raw.columns[0]: "period",
                              "All Starts": "all_starts"})
    raw = raw[["period", "all_starts"]].dropna(subset=["period"])

    # Parse "1978 Q1" → quarter-start date
    def parse_quarter(s):
        s = str(s).strip()
        if " Q" not in s:
            return pd.NaT
        yr, q = s.split(" Q")
        m = {"1": 1, "2": 4, "3": 7, "4": 10}[q.strip()]
        return pd.Timestamp(f"{yr.strip()}-{m:02d}-01")

    raw["date_q"] = raw["period"].apply(parse_quarter)
    raw = raw.dropna(subset=["date_q"])
    raw["all_starts"] = pd.to_numeric(raw["all_starts"], errors="coerce")
    raw = raw.dropna(subset=["all_starts"])
    raw = raw.set_index("date_q").sort_index()[["all_starts"]]

    # Trim to model window + buffer for lags
    raw = raw.loc["2004-01-01":"2023-10-01"]

    # Interpolate quarterly → monthly (linear)
    # Reindex to monthly, forward-fill quarter values then interpolate
    monthly_idx = pd.date_range("2004-01-01", "2023-10-01", freq="MS")
    starts_m = raw.reindex(monthly_idx).interpolate(method="linear")

    print(f"  Housing starts: {len(raw)} quarters → {len(starts_m)} months")
    print(f"  Range: {starts_m.all_starts.min():,.0f} – "
          f"{starts_m.all_starts.max():,.0f} (England quarterly total)")

    # Save
    starts_m.reset_index().rename(columns={"index": "date",
                                           "all_starts": "housing_starts"}).to_csv(
        RAW / "mhclg_housing_starts.csv", index=False
    )

    # Broadcast to regions
    starts_series = starts_m.rename(columns={"all_starts": "housing_starts"})
    _drop = [c for c in ["housing_starts", "starts_lag6",
                         "starts_lag12"] if c in df.columns]
    if _drop:
        df = df.drop(columns=_drop)
    df = df.reset_index()
    df = df.merge(starts_series.reset_index().rename(columns={"index": "date"}),
                  on="date", how="left")
    df = df.set_index(["region", "date"]).sort_index()

    df["starts_lag6"] = df.groupby(level="region")["housing_starts"].transform(
        lambda s: s.shift(6))
    df["starts_lag12"] = df.groupby(level="region")["housing_starts"].transform(
        lambda s: s.shift(12))

    # Test
    for var in ["starts_lag6", "starts_lag12"]:
        test = _univariate_test(df, var)
        sign = "✅ correct (-)" if test["coef"] < 0 else "⚠️ wrong (+)"
        print(f"  {var}: Coef={test['coef']:+.4e}  "
              f"t={test['t_stat']:+.2f}{_stars(test['p_val'])}  "
              f"R²={test['r2_within']:.4f}  {sign}")

    test_s6 = _univariate_test(df, "starts_lag6")
    test_s12 = _univariate_test(df, "starts_lag12")
    return df, test_s6, test_s12


# ══════════════════════════════════════════════════════════════════════════════
# 3. ONS HI00 REGIONAL UNEMPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════

# Sheet name → our region name mapping
UNEMP_SHEETS = {
    "neast_p":  "North East",
    "nwest_p":  "North West",
    "ykhu_p":   "Yorkshire and The Humber",
    "emids_p":  "East Midlands",
    "wmids_p":  "West Midlands",
    "east_p":   "East of England",
    "lon_p":    "London",
    "seast_p":  "South East",
    "swest_p":  "South West",
    "wal_p":    "Wales",
    "scot_p":   "Scotland",
    "nire_p":   "Northern Ireland",
}


def _parse_rolling_quarter(s: str) -> pd.Timestamp | None:
    """
    Convert "Mar-May 1992" to the month-start of the LAST month in the quarter.
    Mar-May → May 1992 → 1992-05-01
    """
    try:
        parts = str(s).strip().split()
        if len(parts) != 2:
            return None
        months_str, year = parts
        end_month_str = months_str.split("-")[-1]
        dt = pd.to_datetime(f"{end_month_str} {year}", format="%b %Y")
        return dt.replace(day=1)
    except Exception:
        return None


def parse_regional_unemployment(df: pd.DataFrame) -> pd.DataFrame:
    _h("3. ONS HI00 REGIONAL UNEMPLOYMENT")

    xl_path = RAW / "ons_regional_unemployment_hi00.xlsx"
    records = []

    for sheet, region in UNEMP_SHEETS.items():
        raw = pd.read_excel(xl_path, sheet_name=sheet, header=None)
        # Row 7 = column labels, row 10+ = data
        # Unemployment rate (16+) is column 8 (0-indexed)
        # Unemployment rate (16-64) is column 17
        data_rows = raw.iloc[10:].copy()
        data_rows.columns = range(raw.shape[1])

        dates = data_rows[0].apply(_parse_rolling_quarter)
        unemp_r = pd.to_numeric(data_rows[17], errors="coerce")   # 16-64 rate

        region_df = pd.DataFrame({
            "date":              dates.values,
            "unemployment_rate": unemp_r.values,
            "region":            region,
        }).dropna(subset=["date", "unemployment_rate"])

        region_df["date"] = pd.to_datetime(region_df["date"])
        region_df = region_df[
            (region_df["date"] >= MODEL_START) &
            (region_df["date"] <= MODEL_END)
        ]
        records.append(region_df)
        print(f"  {region:<35}  {len(region_df):>3} quarterly obs")

    unemp_all = pd.concat(records, ignore_index=True)

    # Interpolate each region from quarterly to monthly
    monthly_records = []
    for region, grp in unemp_all.groupby("region"):
        grp = grp.set_index("date").sort_index()[["unemployment_rate"]]
        grp = grp[~grp.index.duplicated(keep="last")]
        monthly_idx = pd.date_range(MODEL_START, MODEL_END, freq="MS")
        grp_m = grp.reindex(monthly_idx).interpolate(method="linear")
        grp_m["region"] = region
        monthly_records.append(grp_m.reset_index().rename(
            columns={"index": "date"}))

    unemp_monthly = pd.concat(monthly_records, ignore_index=True)

    # Save
    unemp_monthly.to_csv(RAW / "ons_regional_unemployment.csv", index=False)

    # Merge into df
    _drop = [c for c in ["unemployment_rate", "unemployment_diff",
                         "unemployment_lag1"] if c in df.columns]
    if _drop:
        df = df.drop(columns=_drop)
    df = df.reset_index()
    df = df.merge(unemp_monthly[["region", "date", "unemployment_rate"]],
                  on=["region", "date"], how="left")
    df = df.set_index(["region", "date"]).sort_index()

    df["unemployment_diff"] = df.groupby(level="region")["unemployment_rate"].transform(
        lambda s: s.diff(1))
    df["unemployment_lag1"] = df.groupby(level="region")["unemployment_rate"].transform(
        lambda s: s.shift(1))

    # Tests
    for var in ["unemployment_diff", "unemployment_lag1"]:
        test = _univariate_test(df, var)
        sign_ok = test["coef"] < 0
        sign = "✅ correct (-)" if sign_ok else "⚠️ wrong (+)"
        adf = _adf_panel(df, var)
        print(f"  {var}: Coef={test['coef']:+.6f}  "
              f"t={test['t_stat']:+.2f}{_stars(test['p_val'])}  "
              f"R²={test['r2_within']:.4f}  ADF p={adf['median_p']:.3f}  {sign}")

    test_diff = _univariate_test(df, "unemployment_diff")
    test_lag1 = _univariate_test(df, "unemployment_lag1")
    return df, test_diff, test_lag1


# ══════════════════════════════════════════════════════════════════════════════
# 4. HMRC PROPERTY TRANSACTIONS
# ══════════════════════════════════════════════════════════════════════════════

NATION_REGION_MAP = {
    "England":          ["East Midlands", "East of England", "London",
                         "North East", "North West", "South East",
                         "South West", "West Midlands", "Yorkshire and The Humber"],
    "Scotland":         ["Scotland"],
    "Wales":            ["Wales"],
    "Northern Ireland": ["Northern Ireland"],
}


def parse_transactions(df: pd.DataFrame) -> pd.DataFrame:
    _h("4. HMRC PROPERTY TRANSACTIONS")

    raw = pd.read_excel(
        RAW / "hmrc_transactions.ods", engine="odf",
        sheet_name="Residential_monthly", header=None
    )

    # Row 5 = header
    raw.columns = raw.iloc[5]
    raw = raw.iloc[6:].reset_index(drop=True)
    raw = raw.rename(columns={
        raw.columns[0]: "date_str",
        "England":          "England",
        "Scotland":         "Scotland",
        "Wales":            "Wales",
        "Northern Ireland": "Northern Ireland",
        "UK":               "UK",
    })

    raw["date"] = pd.to_datetime(
        raw["date_str"], format="%B %Y", errors="coerce")
    raw = raw.dropna(subset=["date"])
    for col in ["England", "Scotland", "Wales", "Northern Ireland", "UK"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw[["date", "England", "Scotland",
               "Wales", "Northern Ireland", "UK"]]
    raw = raw.sort_values("date").set_index("date")
    raw = raw.loc[MODEL_START:MODEL_END]

    print(f"  HMRC transactions: {len(raw)} monthly obs "
          f"({raw.index.min():%Y-%m} → {raw.index.max():%Y-%m})")

    # Save
    raw.reset_index().to_csv(RAW / "hmrc_transactions.csv", index=False)

    # Build region-level mapping: each region gets its nation's transaction count
    txn_records = []
    for nation, regions in NATION_REGION_MAP.items():
        nation_s = raw[nation].dropna()
        for region in regions:
            tmp = nation_s.reset_index()
            tmp["region"] = region
            tmp = tmp.rename(columns={nation: "transactions"})
            txn_records.append(tmp)

    txn_df = pd.concat(txn_records, ignore_index=True)

    _drop = [c for c in ["transactions", "transaction_growth",
                         "transactions_lag1"] if c in df.columns]
    if _drop:
        df = df.drop(columns=_drop)
    df = df.reset_index()
    df = df.merge(txn_df[["date", "region", "transactions"]],
                  on=["date", "region"], how="left")
    df = df.set_index(["region", "date"]).sort_index()

    df["transaction_growth"] = df.groupby(level="region")["transactions"].transform(
        lambda s: s.pct_change(12))
    df["transactions_lag1"] = df.groupby(level="region")["transactions"].transform(
        lambda s: s.shift(1))

    # Tests
    for var in ["transaction_growth", "transactions_lag1"]:
        test = _univariate_test(df, var)
        sign = "✅ correct (+)" if test["coef"] > 0 else "⚠️ wrong (-)"
        adf = _adf_panel(df, var)
        print(f"  {var}: Coef={test['coef']:+.6f}  "
              f"t={test['t_stat']:+.2f}{_stars(test['p_val'])}  "
              f"R²={test['r2_within']:.4f}  ADF p={adf['median_p']:.3f}  {sign}")

    test_growth = _univariate_test(df, "transaction_growth")
    return df, test_growth


# ══════════════════════════════════════════════════════════════════════════════
# 5. MHCLG PLANNING P153 — diagnostic only
# ══════════════════════════════════════════════════════════════════════════════

def inspect_p153():
    _h("5. MHCLG PLANNING P153 — diagnostic inspection")
    xl = pd.ExcelFile(RAW / "mhclg_planning_p153.ods", engine="odf")
    print(f"  Sheets: {xl.sheet_names}")
    for sh in xl.sheet_names[:3]:
        raw = pd.read_excel(RAW / "mhclg_planning_p153.ods",
                            engine="odf", sheet_name=sh, header=None)
        print(f"\n  Sheet '{sh}': {raw.shape}")
        # Find what column 0 looks like
        non_null = raw.iloc[:, 0].dropna().head(6)
        print(f"  Col 0 sample: {non_null.tolist()}")
        if raw.shape[1] > 1:
            print(f"  Col headers row 4: {raw.iloc[4, :6].tolist()}")

    print("\n  NOTE: P153 measures planning decision speed (weeks), not volume.")
    print("  It is NOT a forward-looking supply volume proxy.")
    print("  EXCLUDED from variable set — not structurally motivated.")


# ══════════════════════════════════════════════════════════════════════════════
# RE-RUN OLS WITH NEW VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

def run_updated_ols(df: pd.DataFrame, add_vars: list[str]) -> None:
    _h("RE-RUN V2 OLS WITH DOWNLOADED VARIABLES")

    # Base Spec B variables
    SPEC_B_BASE = ["payment_burden_gap", "earnings_lag1_diff", "supply_gap_lag12",
                   "financial_stress_lag3", "price_growth_lag1", "time_trend"]
    EXOG = SPEC_B_BASE + add_vars + REGIME_COLS + MONTH_COLS
    EXOG = [c for c in EXOG if c in df.columns]

    clean = df.dropna(subset=EXOG + ["price_growth"])
    print(
        f"  Estimation sample: {len(clean):,} obs  |  EXOG ({len(EXOG)}): {add_vars}")

    y = clean["price_growth"]
    X = sm.add_constant(clean[EXOG], has_constant="add")

    model = PanelOLS(y, X, entity_effects=True)
    res = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=12)
    dk_se = _driscoll_kraay_se(res, X)
    dk_map = dict(zip(X.columns, dk_se))

    print(f"\n  {'Variable':<30} {'Coef':>12} {'DK SE':>10} {'t':>7} {'':>5}")
    print(f"  {'─'*30} {'─'*12} {'─'*10} {'─'*7} {'─'*5}")
    for var in SPEC_B_BASE + add_vars:
        if var not in df.columns:
            continue
        c = res.params.get(var, np.nan)
        se = dk_map.get(var, np.nan)
        t = c / se if (not np.isnan(se) and se > 0) else np.nan
        p = 2 * (1 - sp_stats.norm.cdf(abs(t))) if not np.isnan(t) else np.nan
        sig = _stars(p) if not np.isnan(p) else ""
        print(f"  {var:<30} {c:>+12.6f} {se:>10.6f} {t:>+7.2f} {sig:>5}")

    print(f"\n  R² within: {res.rsquared:.4f}")

    # Walk-forward
    stamp_c = float(res.params.get("regime_stamp_duty", 0.0))
    rmse, mae, da = _walk_forward(df, EXOG, stamp_c)
    print(
        f"  Walk-forward RMSE = {rmse:.5f}  MAE = {mae:.5f}  DirAcc = {da:.1f}%")
    print(f"  Old Spec B RMSE   = {OLD_V2_RMSE:.5f}")

    if rmse < OLD_V2_RMSE:
        delta = OLD_V2_RMSE - rmse
        print(f"\n  ✅ Improvement from downloaded data: "
              f"RMSE {OLD_V2_RMSE:.5f} → {rmse:.5f}  (Δ = {delta:.5f})")
    else:
        delta = rmse - OLD_V2_RMSE
        print(f"\n  ⚠️  No RMSE improvement: {rmse:.5f} vs {OLD_V2_RMSE:.5f} "
              f"(Δ = +{delta:.5f})")

    # Print summary line for the report
    print(f"\n  Improvement from manual downloads: "
          f"RMSE {OLD_V2_RMSE:.5f} → {rmse:.5f}")
    added = add_vars if add_vars else ["none"]
    print(f"  New significant variables: {added}")

    return res, dk_map, rmse


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    _h("Integrating Downloaded Data into Master Dataset V2")
    print(f"Run: {t0:%Y-%m-%d %H:%M:%S}")

    # Load v2 dataset
    df = pd.read_csv(PROC / "master_dataset_v2.csv", parse_dates=["date"])
    df = df.set_index(["region", "date"]).sort_index()
    print(f"  Loaded master_dataset_v2: {df.shape}")

    # ── Initialise extrapolation flag (preserve if already set from prior run) ─
    if "rent_extrapolated" not in df.columns:
        df["rent_extrapolated"] = False

    # ── Parse each file ───────────────────────────────────────────────────────
    df, test_appr = parse_boe_approvals(df)
    df, test_s6, test_s12 = parse_housing_starts(df)
    df, test_udiff, test_ulag1 = parse_regional_unemployment(df)
    df, test_txn = parse_transactions(df)
    inspect_p153()

    # ── Extrapolate terminal-month rent for any region with NaN at MODEL_END ──
    _h("TERMINAL RENT EXTRAPOLATION")
    df = _extrapolate_terminal_rent(df)

    # ── Selection: |t| > 2.0 and correct sign ────────────────────────────────
    _h("VARIABLE SELECTION FROM DOWNLOADED DATA")

    candidates = {
        # pos expected
        "approvals_growth":    (test_appr,  True,   "positive"),
        "starts_lag6":         (test_s6,    False,  "negative"),
        "starts_lag12":        (test_s12,   False,  "negative"),
        "unemployment_diff":   (test_udiff, False,  "negative"),
        "unemployment_lag1":   (test_ulag1, False,  "negative"),
        "transaction_growth":  (test_txn,   True,   "positive"),
    }

    add_vars = []
    reject_vars = []

    print(
        f"  {'Variable':<25} {'Coef':>10} {'t':>7} {'|t|>2':>7} {'Sign':>7} {'Include':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")

    for var, (test, pos_expected, sign_label) in candidates.items():
        if var not in df.columns:
            continue
        c = test["coef"]
        t = test["t_stat"]
        if np.isnan(c) or np.isnan(t):
            reject_vars.append((var, "no valid estimate"))
            continue
        t_ok = abs(t) > 2.0
        sign_ok = (c > 0) if pos_expected else (c < 0)
        include = t_ok and sign_ok
        t_flag = "✅" if t_ok else "❌"
        s_flag = "✅" if sign_ok else "❌"
        i_flag = "✅ YES" if include else "❌ NO"
        print(
            f"  {var:<25} {c:>+10.4f} {t:>+7.2f} {t_flag:>7} {s_flag:>7} {i_flag:>8}")
        if include:
            add_vars.append(var)
        else:
            reason = []
            if not t_ok:
                reason.append(f"|t|={abs(t):.2f} < 2.0")
            if not sign_ok:
                reason.append(f"wrong sign (coef={c:+.4f})")
            reject_vars.append((var, ", ".join(reason)))

    print(
        f"\n  INCLUDED: {add_vars if add_vars else ['none — all below threshold']}")
    for var, reason in reject_vars:
        print(f"  REJECTED: {var} — {reason}")

    # ── Save updated v2 dataset ───────────────────────────────────────────────
    df.to_csv(PROC / "master_dataset_v2.csv")
    print(f"\n  master_dataset_v2.csv updated: {df.shape}")

    # ── Re-run OLS ────────────────────────────────────────────────────────────
    if add_vars:
        res, dk_map, new_rmse = run_updated_ols(df, add_vars)

        # Update sde_parameters_final.csv with any new significant variables
        sde = pd.read_csv(FIN_DIR / "sde_parameters_final.csv")
        for var in add_vars:
            c = float(res.params.get(var, np.nan))
            se = float(dk_map.get(var, np.nan))
            sde[f"beta_{var}"] = c
            sde[f"beta_{var}_se"] = se
        sde["rmse_wf_with_downloads"] = new_rmse
        sde.to_csv(FIN_DIR / "sde_parameters_final.csv", index=False)
        sde.to_csv(OUT / "sde_parameters.csv", index=False)
        print(f"\n  sde_parameters_final.csv updated with: {add_vars}")
    else:
        _h("NO NEW VARIABLES ADDED")
        print("  All downloaded variables failed |t| > 2.0 or sign check.")
        print("  Spec B remains the official v2 specification.")
        # Still run OLS with base spec for RMSE reporting
        run_updated_ols(df, [])

    elapsed = (datetime.now() - t0).total_seconds()
    _h(f"COMPLETE — {elapsed:.0f}s")


if __name__ == "__main__":
    main()
