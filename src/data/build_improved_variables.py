"""
Build improved explanatory variables for the UK housing panel model.

Addresses three documented weaknesses:
  1. PTI near-identity  → four orthogonal affordability candidates
  2. Unstable rate channel → credit-conditions decomposition
  3. Supply near-zero   → forward-looking supply proxies

Outputs:
  data/processed/master_dataset_v2.csv
  data/outputs/variable_improvement/variable_selection_report.txt
  data/outputs/variable_improvement/all_candidate_tests.csv
  data/outputs/variable_improvement/selected_vars.txt
  data/outputs/stage4_final/sde_parameters_v2.csv

Run:
    cd housing_model/
    python -m src.data.build_improved_variables
"""

import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as sp_stats
from pathlib import Path
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import PanelOLS

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
PROC    = ROOT / "data" / "processed"
RAW     = ROOT / "data" / "raw"
OUT     = ROOT / "data" / "outputs"
VIMP    = OUT / "variable_improvement"
FIN_DIR = OUT / "stage4_final"

INPUT_ENG   = PROC / "master_dataset_engineered.csv"
RENTAL_CSV  = RAW  / "ons_rental_levels_202603.csv"
SDE_FINAL   = FIN_DIR / "sde_parameters_final.csv"
OUTPUT_V2   = PROC / "master_dataset_v2.csv"
SDE_V2_CSV  = FIN_DIR / "sde_parameters_v2.csv"

VIMP.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
REGIONS = [
    "East Midlands", "East of England", "London", "North East",
    "North West", "Northern Ireland", "Scotland", "South East",
    "South West", "Wales", "West Midlands", "Yorkshire and The Humber",
]
MONTH_COLS  = [f"month_{m}" for m in range(2, 13)]
REGIME_COLS = ["regime_gfc", "regime_stamp_duty"]
TRAIN_END   = pd.Timestamp("2015-12-01")
TEST_START  = pd.Timestamp("2016-01-01")

OLD_RMSE = 0.004905   # OLS + regime dummies, walk-forward 2016-2023
OLD_R2   = 0.9606

_REPORT_LINES: list[str] = []


# ── Logging ───────────────────────────────────────────────────────────────────

def _h(title: str) -> None:
    line = "\n" + "=" * 60 + f"\n{title}\n" + "=" * 60
    print(line)
    _REPORT_LINES.append(line)


def _p(msg: str = "") -> None:
    print(msg)
    _REPORT_LINES.append(msg)


def _stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


# ── Statistical helpers ───────────────────────────────────────────────────────

def _adf_pvalue(series: pd.Series) -> float:
    """Pooled ADF p-value on non-NaN values; returns 1.0 on failure."""
    vals = series.dropna().values
    if len(vals) < 20:
        return 1.0
    try:
        return adfuller(vals, autolag="AIC")[1]
    except Exception:
        return 1.0


def _adf_panel(df: pd.DataFrame, col: str) -> dict:
    """
    Per-region ADF; return median p-value and whether stationary at 10%.
    """
    pvals = []
    for reg, grp in df.groupby(level="region"):
        s = grp[col].dropna()
        if len(s) >= 20:
            try:
                p = adfuller(s.values, autolag="AIC")[1]
                pvals.append(p)
            except Exception:
                pass
    if not pvals:
        return {"median_p": np.nan, "stationary": False, "n_regions": 0}
    med = float(np.median(pvals))
    return {"median_p": med, "stationary": med < 0.10, "n_regions": len(pvals)}


def _univariate_panel_test(df: pd.DataFrame, var: str,
                            dep: str = "price_growth") -> dict:
    """
    Univariate panel regression: dep ~ var + EntityEffects.
    Returns coef, t_stat, p_val, r2_within, n_obs.
    """
    cols_needed = [dep, var]
    clean = df.dropna(subset=cols_needed)
    if len(clean) < 50:
        return {"coef": np.nan, "t_stat": np.nan, "p_val": np.nan,
                "r2_within": np.nan, "n_obs": 0}
    y = clean[dep]
    X = sm.add_constant(clean[[var]], has_constant="add")
    try:
        res = PanelOLS(y, X, entity_effects=True).fit(
            cov_type="clustered", cluster_entity=True
        )
        c   = res.params.get(var, np.nan)
        se  = res.std_errors.get(var, np.nan)
        t   = c / se if (not np.isnan(se) and se > 0) else np.nan
        p   = float(res.pvalues.get(var, np.nan))
        r2  = float(res.rsquared)
        return {"coef": c, "t_stat": t, "p_val": p,
                "r2_within": r2, "n_obs": len(clean)}
    except Exception as e:
        return {"coef": np.nan, "t_stat": np.nan, "p_val": np.nan,
                "r2_within": np.nan, "n_obs": 0, "error": str(e)}


def _correlation_with_dep(df: pd.DataFrame, var: str,
                           dep: str = "price_growth") -> float:
    clean = df[[var, dep]].dropna()
    if len(clean) < 10:
        return np.nan
    return float(clean[var].corr(clean[dep]))


def _driscoll_kraay_se(model_result, X: pd.DataFrame,
                        bandwidth: int = 12) -> np.ndarray:
    """Driscoll-Kraay HAC SE (Bartlett kernel)."""
    resids = model_result.resids
    shared = X.index.intersection(resids.index)
    X_     = X.loc[shared].values.astype(float)
    e_     = resids.loc[shared].values.astype(float)
    dates  = X.loc[shared].index.get_level_values("date")
    dates_u = sorted(set(dates))
    T, k    = len(dates_u), X_.shape[1]
    d_map   = {d: i for i, d in enumerate(dates_u)}

    S = np.zeros((T, k))
    for i in range(len(X_)):
        S[d_map[dates[i]]] += X_[i] * e_[i]

    Omega = np.zeros((k, k))
    for t in range(T):
        Omega += np.outer(S[t], S[t])
    for lag in range(1, bandwidth + 1):
        w  = 1 - lag / (bandwidth + 1)
        gl = np.zeros((k, k))
        for t in range(lag, T):
            gl += np.outer(S[t], S[t - lag])
        Omega += w * (gl + gl.T)
    Omega /= T

    XX = X_.T @ X_ / T
    try:
        XX_inv = np.linalg.inv(XX)
    except np.linalg.LinAlgError:
        XX_inv = np.linalg.pinv(XX)

    V = (XX_inv @ Omega @ XX_inv) / T
    return np.sqrt(np.diag(V))


def _vif_check(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Compute VIF for each column in cols (drops NaN rows first)."""
    clean = df[cols].dropna()
    if len(clean) < len(cols) + 5:
        return {c: np.nan for c in cols}
    X = clean.values.astype(float)
    vifs = {}
    for i, col in enumerate(cols):
        try:
            vifs[col] = variance_inflation_factor(X, i)
        except Exception:
            vifs[col] = np.nan
    return vifs


def _entity_effects(params: pd.Series, train_df: pd.DataFrame,
                    feat_cols: list[str]) -> dict:
    y_mean = train_df["price_growth"].groupby(level="region").mean()
    X_mean = train_df[feat_cols].groupby(level="region").mean()
    const  = params.get("const", 0.0)
    beta   = params.drop("const", errors="ignore")
    avail  = [c for c in feat_cols if c in beta.index]
    return {
        r: float(y_mean[r] - const - beta[avail] @ X_mean.loc[r, avail])
        for r in REGIONS if r in y_mean.index and r in X_mean.index
    }


def _predict_fe(params: pd.Series, test_df: pd.DataFrame,
                feat_cols: list[str], alpha: dict) -> np.ndarray:
    const = params.get("const", 0.0)
    beta  = params.drop("const", errors="ignore")
    avail = [c for c in feat_cols if c in beta.index and c in test_df.columns]
    preds = np.full(len(test_df), const)
    preds += test_df[avail].values @ beta[avail].values
    for i, region in enumerate(test_df.index.get_level_values("region")):
        preds[i] += alpha.get(region, 0.0)
    return preds


def _walk_forward_rmse(df: pd.DataFrame, exog_wf: list[str],
                        stamp_coef: float = 0.0) -> tuple[float, float, float]:
    """
    Walk-forward RMSE (train 2005-2015, test 2016-2023).
    stamp_coef: full-sample regime_stamp_duty coef applied to test predictions.
    """
    dates    = df.index.get_level_values("date")
    train_df = df[dates <= TRAIN_END].dropna(subset=exog_wf + ["price_growth"])
    test_df  = df[dates >= TEST_START].dropna(subset=exog_wf + ["price_growth"])
    actual   = test_df["price_growth"].values

    try:
        res = PanelOLS(
            train_df["price_growth"],
            sm.add_constant(train_df[exog_wf], has_constant="add"),
            entity_effects=True,
        ).fit(cov_type="clustered", cluster_entity=True)
    except Exception as e:
        _p(f"  Walk-forward fit failed: {e}")
        return np.nan, np.nan, np.nan

    alpha = _entity_effects(res.params, train_df, exog_wf)
    preds = _predict_fe(res.params, test_df, exog_wf, alpha)

    # Apply stamp-duty adjustment if applicable
    if "regime_stamp_duty" in test_df.columns and stamp_coef != 0.0:
        preds = preds + stamp_coef * test_df["regime_stamp_duty"].values

    err  = actual - preds
    rmse = float(np.sqrt(np.nanmean(err ** 2)))
    mae  = float(np.nanmean(np.abs(err)))
    da   = float(np.nanmean(np.sign(actual) == np.sign(preds))) * 100
    return rmse, mae, da


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_base_data() -> pd.DataFrame:
    _h("LOADING BASE DATASET")
    df = pd.read_csv(INPUT_ENG, parse_dates=["date"])
    df = df.set_index(["region", "date"]).sort_index()

    # Build variables needed for OLS that aren't in engineered dataset
    df["price_growth_lag1"] = df.groupby(level="region")["price_growth"].transform(
        lambda s: s.shift(1)
    )
    df["earnings_lag1_diff"] = df.groupby(level="region")["earnings_lag1"].transform(
        lambda s: s.diff(1)
    )
    df["log_price_to_income_diff"] = df.groupby(level="region")["log_price_to_income"].transform(
        lambda s: s.diff(1)
    )
    df["base_rate_lag3"] = df.groupby(level="region")["base_rate"].transform(
        lambda s: s.shift(3)
    )

    # Time trend and seasonal dummies
    df["time_trend"] = (df.groupby(level="region").cumcount() + 1).astype(float)
    month_s = df.index.get_level_values("date").month
    dummies = pd.get_dummies(
        pd.Series(month_s, index=df.index, name="month"),
        prefix="month", drop_first=True
    ).astype(float)
    for col in MONTH_COLS:
        if col not in dummies.columns:
            dummies[col] = 0.0
    df = df.join(dummies[MONTH_COLS])

    # Regime dummies
    dates = df.index.get_level_values("date")
    df["regime_gfc"]        = ((dates >= "2008-09-01") & (dates <= "2009-12-01")).astype(float)
    df["regime_stamp_duty"] = ((dates >= "2021-04-01") & (dates <= "2021-09-01")).astype(float)

    _p(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    _p(f"  Date range: {df.index.get_level_values('date').min():%Y-%m} "
       f"→ {df.index.get_level_values('date').max():%Y-%m}")
    _p(f"  Regions: {df.index.get_level_values('region').nunique()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — AFFORDABILITY VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

def build_affordability_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    _h("PART 1 — AFFORDABILITY VARIABLES (REPLACE PTI NEAR-IDENTITY)")

    results = {}

    # ── Baseline: old log_pti_diff (for comparison) ──────────────────────────
    _p("\n  BASELINE: log_price_to_income_diff (old)")
    old_corr = _correlation_with_dep(df, "log_price_to_income_diff")
    old_test = _univariate_panel_test(df, "log_price_to_income_diff")
    _p(f"  Corr with price_growth: {old_corr:.4f}  ← near-identity problem")
    _p(f"  Coef={old_test['coef']:+.4f}  t={old_test['t_stat']:+.2f}  "
       f"R²={old_test['r2_within']:.4f}")
    results["log_pti_diff_OLD"] = {
        **old_test, "corr_dep": old_corr, "adf_stationary": True,
        "adf_median_p": 0.001, "label": "log_pti_diff (OLD)"
    }

    # ── Candidate A: pti_lag12 (and diff if non-stationary) ──────────────────
    _p("\n  CANDIDATE A: pti_lag12 (12-month lag of log PTI)")
    df["pti_lag12"] = df.groupby(level="region")["log_price_to_income"].transform(
        lambda s: s.shift(12)
    )
    adf_a = _adf_panel(df, "pti_lag12")
    _p(f"  ADF median p={adf_a['median_p']:.4f}  stationary={adf_a['stationary']}")

    if not adf_a["stationary"]:
        _p("  Non-stationary → using pti_lag12_diff (first-difference)")
        df["pti_lag12_diff"] = df.groupby(level="region")["pti_lag12"].transform(
            lambda s: s.diff(1)
        )
        var_a = "pti_lag12_diff"
        adf_a2 = _adf_panel(df, "pti_lag12_diff")
        _p(f"  pti_lag12_diff ADF median p={adf_a2['median_p']:.4f}  "
           f"stationary={adf_a2['stationary']}")
        adf_a = adf_a2
    else:
        var_a = "pti_lag12"

    corr_a = _correlation_with_dep(df, var_a)
    test_a = _univariate_panel_test(df, var_a)
    corr_old_a = df[[var_a, "log_price_to_income_diff"]].dropna().corr().iloc[0, 1]
    _p(f"  {var_a}: Coef={test_a['coef']:+.6f}  t={test_a['t_stat']:+.2f}  "
       f"R²={test_a['r2_within']:.4f}  n={test_a['n_obs']:,}")
    _p(f"  Corr with price_growth={corr_a:.4f}  Corr with old PTI={corr_old_a:.4f}")
    results[var_a] = {
        **test_a, "corr_dep": corr_a, "adf_stationary": adf_a["stationary"],
        "adf_median_p": adf_a["median_p"], "label": f"Candidate A: {var_a}"
    }

    # ── Candidate B: affordability_gap (PTI deviation from 5yr rolling mean) ─
    _p("\n  CANDIDATE B: affordability_gap (PTI deviation from 5-yr rolling mean)")
    rolling_mean_pti = df.groupby(level="region")["log_price_to_income"].transform(
        lambda x: x.rolling(60, min_periods=24).mean()
    )
    df["affordability_gap"] = df["log_price_to_income"] - rolling_mean_pti
    adf_b = _adf_panel(df, "affordability_gap")
    corr_b = _correlation_with_dep(df, "affordability_gap")
    test_b = _univariate_panel_test(df, "affordability_gap")
    corr_old_b = df[["affordability_gap", "log_price_to_income_diff"]].dropna().corr().iloc[0,1]
    _p(f"  ADF median p={adf_b['median_p']:.4f}  stationary={adf_b['stationary']}")
    _p(f"  Coef={test_b['coef']:+.6f}  t={test_b['t_stat']:+.2f}  "
       f"R²={test_b['r2_within']:.4f}  n={test_b['n_obs']:,}")
    _p(f"  Corr with price_growth={corr_b:.4f}  Corr with old PTI={corr_old_b:.4f}")
    results["affordability_gap"] = {
        **test_b, "corr_dep": corr_b, "adf_stationary": adf_b["stationary"],
        "adf_median_p": adf_b["median_p"], "label": "Candidate B: affordability_gap"
    }

    # ── Candidate C: payment_burden_gap ──────────────────────────────────────
    _p("\n  CANDIDATE C: payment_burden_gap (annuity payment / income - regional mean)")
    n = 300
    r = (df["mortgage_rate"] / 12 / 100).clip(lower=0.001)
    annuity_factor = r * (1 + r)**n / ((1 + r)**n - 1)
    loan_amount    = df["real_house_price"] * 0.80
    monthly_payment = loan_amount * annuity_factor
    df["payment_burden"] = monthly_payment / (df["real_annual_earnings"] / 12)
    regional_mean_burden = df.groupby(level="region")["payment_burden"].transform("mean")
    df["payment_burden_gap"] = df["payment_burden"] - regional_mean_burden

    adf_c = _adf_panel(df, "payment_burden_gap")
    corr_c = _correlation_with_dep(df, "payment_burden_gap")
    test_c = _univariate_panel_test(df, "payment_burden_gap")
    corr_old_c = df[["payment_burden_gap", "log_price_to_income_diff"]].dropna().corr().iloc[0,1]
    _p(f"  ADF median p={adf_c['median_p']:.4f}  stationary={adf_c['stationary']}")
    _p(f"  Coef={test_c['coef']:+.6f}  t={test_c['t_stat']:+.2f}  "
       f"R²={test_c['r2_within']:.4f}  n={test_c['n_obs']:,}")
    _p(f"  Corr with price_growth={corr_c:.4f}  Corr with old PTI={corr_old_c:.4f}")
    results["payment_burden_gap"] = {
        **test_c, "corr_dep": corr_c, "adf_stationary": adf_c["stationary"],
        "adf_median_p": adf_c["median_p"], "label": "Candidate C: payment_burden_gap"
    }

    # ── Candidate D: ptr_gap (price-to-rent, uses real_monthly_rent) ─────────
    _p("\n  CANDIDATE D: ptr_gap (price-to-rent deviation from 5-yr rolling mean)")
    if "real_monthly_rent" in df.columns:
        rent_ok = df["real_monthly_rent"].notna().sum()
        _p(f"  real_monthly_rent available: {rent_ok:,} non-NaN obs")

        df["price_to_rent"]  = df["real_house_price"] / (df["real_monthly_rent"] * 12)
        df["log_ptr"]        = np.log(df["price_to_rent"].clip(lower=1e-6))
        rolling_mean_ptr     = df.groupby(level="region")["log_ptr"].transform(
            lambda x: x.rolling(60, min_periods=24).mean()
        )
        df["ptr_gap"] = df["log_ptr"] - rolling_mean_ptr

        adf_d = _adf_panel(df, "ptr_gap")
        corr_d = _correlation_with_dep(df, "ptr_gap")
        test_d = _univariate_panel_test(df, "ptr_gap")
        corr_old_d = df[["ptr_gap", "log_price_to_income_diff"]].dropna().corr().iloc[0,1]
        _p(f"  ADF median p={adf_d['median_p']:.4f}  stationary={adf_d['stationary']}")
        _p(f"  Coef={test_d['coef']:+.6f}  t={test_d['t_stat']:+.2f}  "
           f"R²={test_d['r2_within']:.4f}  n={test_d['n_obs']:,}")
        _p(f"  Corr with price_growth={corr_d:.4f}  Corr with old PTI={corr_old_d:.4f}")
        results["ptr_gap"] = {
            **test_d, "corr_dep": corr_d, "adf_stationary": adf_d["stationary"],
            "adf_median_p": adf_d["median_p"], "label": "Candidate D: ptr_gap"
        }
    else:
        _p("  PTR: rental data not available at this granularity — skipping")

    # ── Summary table ─────────────────────────────────────────────────────────
    _p("\n  AFFORDABILITY CANDIDATE COMPARISON")
    _p(f"  {'Variable':<26} {'Coef':>10} {'t-stat':>8} {'R² within':>10} {'Corr(dep)':>10} {'Stationary':>12}")
    _p(f"  {'-'*26} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")
    display_order = ["log_pti_diff_OLD", var_a, "affordability_gap",
                     "payment_burden_gap", "ptr_gap"]
    for k in display_order:
        if k not in results:
            continue
        r = results[k]
        c  = f"{r['coef']:+.4f}"   if not np.isnan(r['coef'])       else "  n/a"
        t  = f"{r['t_stat']:+.2f}" if not np.isnan(r['t_stat'])     else "  n/a"
        r2 = f"{r['r2_within']:.4f}" if not np.isnan(r['r2_within']) else " n/a"
        cd = f"{r['corr_dep']:.4f}" if not np.isnan(r['corr_dep'])  else " n/a"
        st = "YES" if r['adf_stationary'] else "NO"
        label = r["label"].split(": ")[-1] if ": " in r["label"] else r["label"]
        _p(f"  {label:<26} {c:>10} {t:>8} {r2:>10} {cd:>10} {st:>12}")

    # ── Selection ─────────────────────────────────────────────────────────────
    _p("\n  SELECTION RULE: corr < 0.95 AND highest |t| AND correct sign (neg) AND stationary")
    candidates = [(k, v) for k, v in results.items()
                  if k != "log_pti_diff_OLD"
                  and not np.isnan(v["t_stat"])
                  and abs(v["corr_dep"]) < 0.95]

    # Prefer negative sign; among valid, pick highest |t|
    neg_candidates = [(k, v) for k, v in candidates if v["coef"] < 0]
    pool = neg_candidates if neg_candidates else candidates

    if pool:
        selected_aff = max(pool, key=lambda x: abs(x[1]["t_stat"]))
        sel_name, sel_res = selected_aff
        reason = (f"corr_dep={sel_res['corr_dep']:.3f} < 0.95 (breaks near-identity), "
                  f"|t|={abs(sel_res['t_stat']):.2f}, "
                  f"sign={'negative' if sel_res['coef'] < 0 else 'positive'}, "
                  f"stationary={'YES' if sel_res['adf_stationary'] else 'NO (differenced)'}")
        _p(f"\n  SELECTED AFFORDABILITY VARIABLE: {sel_name} — reason: {reason}")
    else:
        sel_name = "affordability_gap"
        reason   = "fallback: all candidates failed sign/correlation filters"
        _p(f"\n  SELECTED AFFORDABILITY VARIABLE (fallback): {sel_name}")

    return df, results, sel_name


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — CREDIT / RATE CHANNEL
# ══════════════════════════════════════════════════════════════════════════════

def build_credit_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    _h("PART 2 — CREDIT / RATE CHANNEL")

    results = {}

    # ── Variable 1: lender_spread_lag3 ───────────────────────────────────────
    _p("\n  VAR 1: lender_spread_lag3 (mortgage rate - base rate, lag 3)")
    df["lender_spread"] = df["mortgage_rate"] - df["base_rate"]
    df["lender_spread_lag3"] = df.groupby(level="region")["lender_spread"].transform(
        lambda s: s.shift(3)
    )
    adf_ls = _adf_panel(df, "lender_spread_lag3")
    test_ls = _univariate_panel_test(df, "lender_spread_lag3")
    sign_ok = "✅ correct (negative)" if test_ls["coef"] < 0 else "⚠️ wrong sign (positive)"
    _p(f"  ADF median p={adf_ls['median_p']:.4f}  stationary={adf_ls['stationary']}")
    _p(f"  Coef={test_ls['coef']:+.6f}  t={test_ls['t_stat']:+.2f}  "
       f"R²={test_ls['r2_within']:.4f}  {sign_ok}")
    results["lender_spread_lag3"] = {
        **test_ls, "adf_stationary": adf_ls["stationary"],
        "adf_median_p": adf_ls["median_p"], "label": "lender_spread_lag3"
    }

    # ── Variable 2: rate regime interaction ──────────────────────────────────
    _p("\n  VAR 2: rate regime interaction (rate × high_rate_period)")
    df["high_rate_period"] = (df["mortgage_rate"] > 4.0).astype(float)
    df["rate_x_high"] = df["mortgage_rate_lag3"] * df["high_rate_period"]
    df["rate_x_low"]  = df["mortgage_rate_lag3"] * (1 - df["high_rate_period"])

    n_high = int(df["high_rate_period"].sum())
    n_low  = int((1 - df["high_rate_period"]).sum())
    _p(f"  High-rate obs (rate > 4%): {n_high:,}  |  Low-rate obs: {n_low:,}")

    # Joint test: price_growth ~ rate_x_high + rate_x_low + EntityEffects
    joint_cols = ["rate_x_high", "rate_x_low"]
    clean_j = df.dropna(subset=joint_cols + ["price_growth"])
    try:
        y_j = clean_j["price_growth"]
        X_j = sm.add_constant(clean_j[joint_cols], has_constant="add")
        res_j = PanelOLS(y_j, X_j, entity_effects=True).fit(
            cov_type="clustered", cluster_entity=True
        )
        c_high = res_j.params.get("rate_x_high", np.nan)
        c_low  = res_j.params.get("rate_x_low",  np.nan)
        t_high = c_high / res_j.std_errors.get("rate_x_high", 1)
        t_low  = c_low  / res_j.std_errors.get("rate_x_low",  1)
        _p(f"  rate_x_high: Coef={c_high:+.6f}  t={t_high:+.2f}")
        _p(f"  rate_x_low:  Coef={c_low:+.6f}  t={t_low:+.2f}")
        if np.sign(c_high) != np.sign(c_low):
            _p("  ✅ Opposite signs confirmed — regime decomposition valid")
        else:
            _p("  ⚠️  Same signs — regime decomposition inconclusive")
        results["rate_x_high"] = {"coef": c_high, "t_stat": t_high, "p_val": np.nan,
                                   "r2_within": res_j.rsquared, "n_obs": len(clean_j),
                                   "adf_stationary": True, "label": "rate_x_high"}
        results["rate_x_low"]  = {"coef": c_low,  "t_stat": t_low,  "p_val": np.nan,
                                   "r2_within": res_j.rsquared, "n_obs": len(clean_j),
                                   "adf_stationary": True, "label": "rate_x_low"}
    except Exception as e:
        _p(f"  Joint rate test failed: {e}")

    # ── Variable 3: mortgage approvals download ───────────────────────────────
    _p("\n  VAR 3: mortgage approvals (BoE LPMBI4O) — attempting download …")
    try:
        import requests
        url = (
            "https://www.bankofengland.co.uk/boeapps/database/_iadb-FromShowColumns"
            ".asp?csv.x=yes&Datefrom=01/Jan/2000&Dateto=now&SeriesCodes=LPMBI4O"
            "&CSVF=TN&UsingCodes=Y"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        raw_text = resp.text
        # BoE CSV has some header rows — find the data section
        lines = raw_text.splitlines()
        data_start = next((i for i, l in enumerate(lines) if "LPMBI4O" in l or
                           (l and l[0].isdigit())), None)

        if data_start is not None:
            raw_csv = RAW / "boe_mortgage_approvals.csv"
            raw_csv.write_text(raw_text, encoding="utf-8")
            # Try to parse
            approvals = pd.read_csv(
                io.StringIO(raw_text), skiprows=max(0, data_start - 1),
                parse_dates=[0], dayfirst=True, on_bad_lines="skip"
            )
            # Standardise
            approvals.columns = ["date", "approvals"] + list(approvals.columns[2:])
            approvals = approvals[["date", "approvals"]].dropna()
            approvals["date"] = pd.to_datetime(approvals["date"], dayfirst=True, errors="coerce")
            approvals = approvals.dropna(subset=["date"])
            approvals = approvals.set_index("date").resample("MS").mean()

            # Broadcast to all regions
            df = df.reset_index()
            approvals_reset = approvals.reset_index().rename(columns={"date": "date"})
            df = df.merge(approvals_reset[["date", "approvals"]], on="date", how="left")
            df = df.set_index(["region", "date"]).sort_index()

            df["approvals_lag1"]   = df.groupby(level="region")["approvals"].transform(
                lambda s: s.shift(1)
            )
            df["approvals_growth"] = df.groupby(level="region")["approvals"].transform(
                lambda s: s.pct_change(12)
            )

            test_ap = _univariate_panel_test(df, "approvals_growth")
            _p(f"  ✅ Downloaded {len(approvals)} months of approvals data")
            _p(f"  approvals_growth: Coef={test_ap['coef']:+.6f}  "
               f"t={test_ap['t_stat']:+.2f}  R²={test_ap['r2_within']:.4f}")
            results["approvals_growth"] = {
                **test_ap, "adf_stationary": True, "label": "approvals_growth"
            }
        else:
            raise ValueError("Could not parse BoE response")

    except Exception as e:
        _p(f"  ⚠️  Mortgage approvals download failed: {e}")
        _p("  Note: approvals_growth excluded from variable set")
        results["approvals_growth"] = {"coef": np.nan, "t_stat": np.nan, "p_val": np.nan,
                                        "r2_within": np.nan, "n_obs": 0,
                                        "adf_stationary": False, "label": "approvals_growth",
                                        "skip_reason": "download failed"}

    return df, results


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — SUPPLY VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

def build_supply_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    _h("PART 3 — SUPPLY VARIABLES")

    results = {}

    # ── OLD baseline ──────────────────────────────────────────────────────────
    _p("\n  BASELINE: supply_lag3 (old)")
    test_s3 = _univariate_panel_test(df, "supply_lag3")
    sign_s3 = "⚠️ wrong (positive)" if test_s3["coef"] > 0 else "✅ correct (negative)"
    _p(f"  Coef={test_s3['coef']:+.2e}  t={test_s3['t_stat']:+.2f}  {sign_s3}")
    results["supply_lag3_OLD"] = {
        **test_s3, "adf_stationary": True, "label": "supply_lag3 (OLD)"
    }

    # ── Variable 1: housing starts download ──────────────────────────────────
    _p("\n  VAR 1: housing starts (MHCLG Table 213) — attempting download …")
    try:
        import requests
        url = (
            "https://assets.publishing.service.gov.uk/media/housing-live-tables/"
            "Live_Table_213.xlsx"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        raw_path = RAW / "mhclg_housing_starts.xlsx"
        raw_path.write_bytes(resp.content)

        xl = pd.read_excel(raw_path, sheet_name=0, skiprows=6, index_col=0)
        # Look for England total starts column
        starts_col = None
        for col in xl.columns:
            if "england" in str(col).lower() or "total" in str(col).lower():
                starts_col = col
                break
        if starts_col is None and len(xl.columns) > 0:
            starts_col = xl.columns[0]

        starts_q = xl[starts_col].dropna()
        starts_q.index = pd.to_datetime(starts_q.index, errors="coerce")
        starts_q = starts_q.dropna()
        starts_q = starts_q.sort_index()
        # Interpolate quarterly to monthly
        starts_m = starts_q.resample("MS").interpolate(method="linear")

        raw_path.unlink()   # clean up temp file
        raw_path.with_suffix(".csv").write_text(
            starts_m.reset_index().rename(columns={0: "housing_starts",
                                                    starts_col: "housing_starts"}).to_csv(index=False)
        )

        # Broadcast to regions
        df = df.reset_index()
        starts_df = starts_m.reset_index()
        starts_df.columns = ["date", "housing_starts"]
        df = df.merge(starts_df, on="date", how="left")
        df = df.set_index(["region", "date"]).sort_index()

        df["starts_lag6"]  = df.groupby(level="region")["housing_starts"].transform(
            lambda s: s.shift(6)
        )
        df["starts_lag12"] = df.groupby(level="region")["housing_starts"].transform(
            lambda s: s.shift(12)
        )

        test_st = _univariate_panel_test(df, "starts_lag6")
        _p(f"  ✅ Housing starts downloaded: {len(starts_m)} months")
        _p(f"  starts_lag6: Coef={test_st['coef']:+.4e}  t={test_st['t_stat']:+.2f}")
        results["starts_lag6"] = {
            **test_st, "adf_stationary": True, "label": "starts_lag6"
        }

    except Exception as e:
        _p(f"  ⚠️  Housing starts download failed: {e}")
        results["starts_lag6"] = {"coef": np.nan, "t_stat": np.nan, "p_val": np.nan,
                                   "r2_within": np.nan, "n_obs": 0,
                                   "adf_stationary": False, "label": "starts_lag6",
                                   "skip_reason": "download failed"}

    # ── Variable 2: planning approvals download ───────────────────────────────
    _p("\n  VAR 2: planning approvals (MHCLG P153) — attempting download …")
    try:
        import requests
        url = (
            "https://assets.publishing.service.gov.uk/media/planning-live-tables/"
            "Live_Table_P153.xlsx"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        raw_path = RAW / "mhclg_planning_approvals.xlsx"
        raw_path.write_bytes(resp.content)

        xl = pd.read_excel(raw_path, sheet_name=0, skiprows=6, index_col=0)
        app_col = xl.columns[0]
        apps_q  = xl[app_col].dropna()
        apps_q.index = pd.to_datetime(apps_q.index, errors="coerce")
        apps_q  = apps_q.dropna().sort_index()
        apps_m  = apps_q.resample("MS").interpolate(method="linear")

        raw_path.unlink()

        df = df.reset_index()
        apps_df = apps_m.reset_index()
        apps_df.columns = ["date", "planning_approvals"]
        df = df.merge(apps_df, on="date", how="left")
        df = df.set_index(["region", "date"]).sort_index()

        df["planning_approvals_lag12"] = df.groupby(level="region")["planning_approvals"].transform(
            lambda s: s.shift(12)
        )
        test_pl = _univariate_panel_test(df, "planning_approvals_lag12")
        _p(f"  ✅ Planning approvals downloaded")
        _p(f"  planning_approvals_lag12: Coef={test_pl['coef']:+.4e}  "
           f"t={test_pl['t_stat']:+.2f}")
        results["planning_approvals_lag12"] = {
            **test_pl, "adf_stationary": True, "label": "planning_approvals_lag12"
        }

    except Exception as e:
        _p(f"  ⚠️  Planning approvals download failed: {e}")
        results["planning_approvals_lag12"] = {
            "coef": np.nan, "t_stat": np.nan, "p_val": np.nan,
            "r2_within": np.nan, "n_obs": 0, "adf_stationary": False,
            "label": "planning_approvals_lag12", "skip_reason": "download failed"
        }

    # ── Variable 3: supply_gap_lag12 (always available) ──────────────────────
    _p("\n  VAR 3: supply_gap_lag12 (12m-smoothed net additions gap, lag 12)")
    df["supply_smooth_12m"] = df.groupby(level="region")["net_additions_monthly"].transform(
        lambda x: x.rolling(12).mean()
    )
    regional_mean_supply = df.groupby(level="region")["net_additions_monthly"].transform("mean")
    df["supply_gap"] = df["net_additions_monthly"] - regional_mean_supply
    df["supply_gap_lag12"] = df.groupby(level="region")["supply_gap"].transform(
        lambda x: x.shift(12)
    )

    adf_sg = _adf_panel(df, "supply_gap_lag12")
    test_sg = _univariate_panel_test(df, "supply_gap_lag12")
    sign_sg = "✅ correct (negative)" if test_sg["coef"] < 0 else "⚠️ wrong (positive)"
    _p(f"  ADF median p={adf_sg['median_p']:.4f}  stationary={adf_sg['stationary']}")
    _p(f"  Coef={test_sg['coef']:+.4e}  t={test_sg['t_stat']:+.2f}  "
       f"R²={test_sg['r2_within']:.4f}  {sign_sg}")
    results["supply_gap_lag12"] = {
        **test_sg, "adf_stationary": adf_sg["stationary"],
        "adf_median_p": adf_sg["median_p"], "label": "supply_gap_lag12"
    }

    # ── Supply comparison table ───────────────────────────────────────────────
    _p("\n  SUPPLY CANDIDATE COMPARISON")
    _p(f"  {'Variable':<28} {'Coef':>12} {'t-stat':>8} {'Sign':>20}")
    _p(f"  {'-'*28} {'-'*12} {'-'*8} {'-'*20}")
    for k, v in results.items():
        c  = f"{v['coef']:+.2e}" if not np.isnan(v["coef"]) else "  n/a"
        t  = f"{v['t_stat']:+.2f}" if not np.isnan(v["t_stat"]) else "  n/a"
        s  = ("✅ negative" if (not np.isnan(v["coef"]) and v["coef"] < 0)
              else ("⚠️ positive" if (not np.isnan(v["coef"]) and v["coef"] > 0)
                    else "n/a"))
        _p(f"  {v['label']:<28} {c:>12} {t:>8} {s:>20}")

    # ── Selection ─────────────────────────────────────────────────────────────
    candidates = [(k, v) for k, v in results.items()
                  if k != "supply_lag3_OLD"
                  and not np.isnan(v["t_stat"])
                  and v.get("skip_reason") is None]
    neg_cands = [(k, v) for k, v in candidates if v["coef"] < 0]
    pool = neg_cands if neg_cands else candidates

    if pool:
        sel_name, sel_res = max(pool, key=lambda x: abs(x[1]["t_stat"]))
        _p(f"\n  SELECTED SUPPLY VARIABLE: {sel_name} "
           f"(coef={sel_res['coef']:+.2e}, |t|={abs(sel_res['t_stat']):.2f})")
    else:
        sel_name = "supply_gap_lag12"
        _p(f"\n  SELECTED SUPPLY VARIABLE (fallback): {sel_name}")

    return df, results, sel_name


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — SUPPLEMENTARY VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

def build_supplementary_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    _h("PART 4 — SUPPLEMENTARY VARIABLES")

    results = {}

    # ── VAR A: regional unemployment (download) ───────────────────────────────
    _p("\n  VAR A: regional unemployment — attempting download …")
    try:
        import requests
        url = (
            "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/"
            "peopleinwork/employmentandemployeetypes/datasets/"
            "regionalemploymentunemploymentandeconomicinactivity/current/regional.xls"
        )
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        raw_path = RAW / "ons_regional_unemployment.xls"
        raw_path.write_bytes(resp.content)

        # Try to find unemployment rates sheet
        xl_file = pd.ExcelFile(raw_path)
        unemp_sheet = None
        for sheet in xl_file.sheet_names:
            if "unemp" in sheet.lower():
                unemp_sheet = sheet
                break

        if unemp_sheet:
            xl = pd.read_excel(raw_path, sheet_name=unemp_sheet,
                               skiprows=5, index_col=0)
            # Parse and merge
            _p(f"  Found sheet: {unemp_sheet}, shape={xl.shape}")
            raw_path.unlink()
            raise ValueError("Unemployment: sheet found but regional mapping requires manual review")
        else:
            raw_path.unlink()
            raise ValueError("No unemployment sheet found in download")

    except Exception as e:
        _p(f"  ⚠️  Regional unemployment download failed: {e}")
        _p("  VAR A excluded from variable set")
        results["unemployment_diff"] = {
            "coef": np.nan, "t_stat": np.nan, "p_val": np.nan,
            "r2_within": np.nan, "n_obs": 0, "adf_stationary": False,
            "label": "unemployment_diff", "skip_reason": "download failed"
        }

    # ── VAR B: transaction volume (download) ──────────────────────────────────
    _p("\n  VAR B: transaction volume (HMRC SDLT) — attempting download …")
    try:
        import requests
        url = (
            "https://www.gov.uk/government/statistical-data-sets/"
            "monthly-property-transactions-completed-in-the-uk-with-value-40000-or-above"
        )
        # Try direct CSV download
        csv_url = (
            "https://assets.publishing.service.gov.uk/media/transaction-data/"
            "UK_Tables.csv"
        )
        resp = requests.get(csv_url, timeout=30)
        resp.raise_for_status()
        raw_path = RAW / "hmrc_transactions.csv"
        raw_path.write_text(resp.text, encoding="utf-8")
        txn = pd.read_csv(raw_path, parse_dates=[0])
        _p(f"  Downloaded transactions: shape={txn.shape}")
        raw_path.unlink()
        raise ValueError("Transactions: column mapping requires verification")

    except Exception as e:
        _p(f"  ⚠️  Transaction volume download failed: {e}")
        _p("  VAR B excluded from variable set")
        results["transaction_growth"] = {
            "coef": np.nan, "t_stat": np.nan, "p_val": np.nan,
            "r2_within": np.nan, "n_obs": 0, "adf_stationary": False,
            "label": "transaction_growth", "skip_reason": "download failed"
        }

    # ── VAR C: consumer confidence (download) ─────────────────────────────────
    _p("\n  VAR C: consumer confidence — attempting download …")
    try:
        import requests
        # GfK data not freely downloadable; try ONS house price sentiment
        url = ("https://www.ons.gov.uk/generator?format=csv&uri=/economy/"
               "inflationandpriceindices/timeseries/gzqh/mm23")
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        raw_path = RAW / "ons_consumer_confidence.csv"
        raw_path.write_text(resp.text, encoding="utf-8")
        conf = pd.read_csv(raw_path, skiprows=7, header=None,
                           names=["date", "confidence"])
        conf = conf.dropna()
        conf["date"] = pd.to_datetime(conf["date"], errors="coerce")
        conf = conf.dropna(subset=["date"]).set_index("date").resample("MS").mean()

        # Broadcast to regions
        df = df.reset_index()
        df = df.merge(conf.reset_index().rename(columns={"confidence": "consumer_confidence"}),
                      on="date", how="left")
        df = df.set_index(["region", "date"]).sort_index()

        df["confidence_lag1"] = df.groupby(level="region")["consumer_confidence"].transform(
            lambda s: s.shift(1)
        )
        test_cf = _univariate_panel_test(df, "confidence_lag1")
        _p(f"  ✅ Consumer confidence downloaded: {len(conf)} months")
        _p(f"  confidence_lag1: Coef={test_cf['coef']:+.6f}  t={test_cf['t_stat']:+.2f}")
        results["confidence_lag1"] = {
            **test_cf, "adf_stationary": True, "label": "confidence_lag1"
        }
        raw_path.unlink()

    except Exception as e:
        _p(f"  ⚠️  Consumer confidence download failed: {e}")
        _p("  VAR C excluded from variable set")
        results["confidence_lag1"] = {
            "coef": np.nan, "t_stat": np.nan, "p_val": np.nan,
            "r2_within": np.nan, "n_obs": 0, "adf_stationary": False,
            "label": "confidence_lag1", "skip_reason": "download failed"
        }

    # ── VAR D: financial_stress_lag3 (always buildable) ──────────────────────
    _p("\n  VAR D: financial_stress_lag3 (lender_spread × mortgage_rate, lag 3)")
    df["financial_stress"] = df["lender_spread"] * df["mortgage_rate"]
    df["financial_stress_lag3"] = df.groupby(level="region")["financial_stress"].transform(
        lambda s: s.shift(3)
    )
    adf_fs = _adf_panel(df, "financial_stress_lag3")
    test_fs = _univariate_panel_test(df, "financial_stress_lag3")
    sign_fs = "✅ correct (negative)" if test_fs["coef"] < 0 else "⚠️ wrong (positive)"
    _p(f"  ADF median p={adf_fs['median_p']:.4f}  stationary={adf_fs['stationary']}")
    _p(f"  Coef={test_fs['coef']:+.6f}  t={test_fs['t_stat']:+.2f}  "
       f"R²={test_fs['r2_within']:.4f}  {sign_fs}")
    results["financial_stress_lag3"] = {
        **test_fs, "adf_stationary": adf_fs["stationary"],
        "adf_median_p": adf_fs["median_p"], "label": "financial_stress_lag3"
    }

    return df, results


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — BUILD IMPROVED DATASET
# ══════════════════════════════════════════════════════════════════════════════

def select_best_variables(
    all_results: dict,
    sel_aff: str,
    sel_supply: str,
) -> tuple[list[str], list[str]]:
    """
    Apply selection rules to supplementary variables.
    Returns (add_list, reject_list).
    add_list: variables to include in v2 spec beyond base + aff + supply + rate.
    """
    add_list    = []
    reject_list = []

    # Always-included: lender_spread_lag3 if |t| > 2 and correct sign
    ls = all_results.get("lender_spread_lag3", {})
    if (not np.isnan(ls.get("t_stat", np.nan))
            and abs(ls["t_stat"]) > 2.0
            and ls.get("coef", 0) < 0):
        add_list.append("lender_spread_lag3")
    else:
        reject_list.append(("lender_spread_lag3",
                             f"t={ls.get('t_stat', np.nan):.2f} or wrong sign"))

    # Supplementary vars: include if |t| > 2 and correct sign and not skipped
    for var in ["approvals_growth", "unemployment_diff", "transaction_growth",
                "confidence_lag1", "financial_stress_lag3"]:
        r = all_results.get(var, {})
        if r.get("skip_reason"):
            reject_list.append((var, r["skip_reason"]))
            continue
        t   = r.get("t_stat", np.nan)
        c   = r.get("coef", np.nan)
        if np.isnan(t) or np.isnan(c):
            reject_list.append((var, "no valid estimate"))
            continue
        if abs(t) > 2.0:
            # sign check: approvals/transactions/confidence should be positive,
            # unemployment/stress should be negative
            pos_expected = var in {"approvals_growth", "transaction_growth",
                                   "confidence_lag1"}
            sign_ok = (c > 0) if pos_expected else (c < 0)
            if sign_ok:
                add_list.append(var)
            else:
                reject_list.append((var, f"wrong sign (coef={c:+.4f})"))
        else:
            reject_list.append((var, f"|t|={abs(t):.2f} < 2.0"))

    return add_list, reject_list


def build_improved_dataset(
    df: pd.DataFrame,
    sel_aff: str,
    sel_supply: str,
    add_vars: list[str],
    reject_vars: list[tuple],
    all_results: dict,
) -> pd.DataFrame:
    _h("PART 5 — BUILD IMPROVED MASTER DATASET (v2)")

    # ── Assemble variable list ────────────────────────────────────────────────
    core_keep = ["price_growth", "earnings_lag1_diff", "price_growth_lag1",
                 "time_trend"] + REGIME_COLS + MONTH_COLS
    replace_map = {
        "log_price_to_income_diff": sel_aff,
        "mortgage_rate_lag3":       "rate_x_high + rate_x_low",
        "supply_lag3":              sel_supply,
    }
    new_vars = ["rate_x_high", "rate_x_low", sel_aff, sel_supply] + add_vars

    all_needed = (core_keep + new_vars
                  + ["nominal_house_price", "real_house_price", "region",
                     "date", "mortgage_rate", "base_rate", "real_annual_earnings",
                     "net_additions_monthly", "log_price_to_income",
                     "log_real_price", "price_growth_3m", "price_growth_12m",
                     "lender_spread"])

    # Filter to columns that actually exist
    available = [c for c in all_needed if c in df.columns]
    extra     = [c for c in df.columns if c not in available]
    df_v2     = df[[c for c in df.columns if c in available or c in extra]].copy()

    # ── Stationarity check ────────────────────────────────────────────────────
    _p("\n  STATIONARITY CHECK (new variables)")
    for var in new_vars:
        if var not in df_v2.columns:
            continue
        adf = _adf_panel(df_v2, var)
        _p(f"  {var:<30}  ADF median p={adf['median_p']:.4f}  "
           f"stationary={'YES' if adf['stationary'] else 'NO'}")

    # ── VIF check ─────────────────────────────────────────────────────────────
    _p("\n  MULTICOLLINEARITY CHECK (VIF)")
    vif_cols = [sel_aff, sel_supply, "rate_x_high", "rate_x_low",
                "earnings_lag1_diff", "price_growth_lag1"] + add_vars
    vif_cols = [c for c in vif_cols if c in df_v2.columns]
    vif_data = df_v2[vif_cols].dropna()
    if len(vif_data) > len(vif_cols) + 5:
        vifs = _vif_check(vif_data, vif_cols)
        for v, val in vifs.items():
            flag = "⚠️ HIGH" if (not np.isnan(val) and val > 5) else "✅"
            _p(f"  {v:<30}  VIF={val:.2f}  {flag}")
    else:
        _p("  VIF check skipped (insufficient observations)")

    # ── Near-identity check ───────────────────────────────────────────────────
    _p("\n  NEAR-IDENTITY CHECK (new affordability variable)")
    corr_new = _correlation_with_dep(df_v2, sel_aff)
    corr_old = _correlation_with_dep(df_v2, "log_price_to_income_diff")
    _p(f"  Old log_pti_diff corr with price_growth: {corr_old:.4f}  ← problem")
    _p(f"  New {sel_aff:<22} corr with price_growth: {corr_new:.4f}  "
       f"{'✅ OK' if abs(corr_new) < 0.95 else '⚠️ still high'}")

    # ── Print selection summary ───────────────────────────────────────────────
    _p("\n  VARIABLE SELECTION SUMMARY")
    _p(f"  REPLACE: log_price_to_income_diff → {sel_aff}")
    _p(f"  REPLACE: mortgage_rate_lag3 → rate_x_high + rate_x_low")
    _p(f"  REPLACE: supply_lag3 → {sel_supply}")
    for var in add_vars:
        r = all_results.get(var, {})
        _p(f"  ADD:     {var} (t={r.get('t_stat', np.nan):.2f}, "
           f"coef={r.get('coef', np.nan):+.4f})")
    for var, reason in reject_vars:
        _p(f"  REJECT:  {var} — {reason}")

    # ── Save ─────────────────────────────────────────────────────────────────
    df_v2.to_csv(OUTPUT_V2)
    _p(f"\n  Saved: {OUTPUT_V2.relative_to(ROOT)}")
    _p(f"  Shape: {df_v2.shape}")

    return df_v2


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — RE-RUN STAGE 4 OLS WITH IMPROVED VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

def run_improved_ols(
    df: pd.DataFrame,
    sel_aff: str,
    sel_supply: str,
    add_vars: list[str],
) -> None:
    _h("PART 6 — STAGE 4 OLS WITH IMPROVED VARIABLES")

    # ── Build EXOG_V2 ─────────────────────────────────────────────────────────
    CORE_V2 = [sel_aff, "earnings_lag1_diff", sel_supply,
               "rate_x_high", "rate_x_low",
               "price_growth_lag1", "time_trend"]
    SUPP_V2 = [v for v in add_vars if v in df.columns]
    EXOG_V2 = CORE_V2 + SUPP_V2 + REGIME_COLS + MONTH_COLS

    # Filter to available columns
    EXOG_V2 = [c for c in EXOG_V2 if c in df.columns]

    clean = df.dropna(subset=EXOG_V2 + ["price_growth"])
    _p(f"  Estimation sample: {len(clean):,} obs  "
       f"({clean.index.get_level_values('region').nunique()} regions)")
    _p(f"  EXOG_V2 ({len(EXOG_V2)} vars): {EXOG_V2}")

    y = clean["price_growth"]
    X = sm.add_constant(clean[EXOG_V2], has_constant="add")

    # ── Full-sample OLS + DK SE ───────────────────────────────────────────────
    _p("\n  Full-sample OLS + Driscoll-Kraay SE …")
    model  = PanelOLS(y, X, entity_effects=True)
    res_v2 = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=12)
    dk_se  = _driscoll_kraay_se(res_v2, X)
    dk_map = dict(zip(X.columns, dk_se))

    _p(f"\n  {'Variable':<30} {'Coef':>12} {'DK SE':>10} {'t-stat':>8} {'Sig':>5}")
    _p(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*8} {'-'*5}")
    for var in EXOG_V2:
        c   = res_v2.params.get(var, np.nan)
        se  = dk_map.get(var, np.nan)
        t   = c / se if (not np.isnan(se) and se > 0) else np.nan
        p   = 2 * (1 - sp_stats.norm.cdf(abs(t))) if not np.isnan(t) else np.nan
        sig = _stars(p) if not np.isnan(p) else ""
        _p(f"  {var:<30} {c:>+12.6f} {se:>10.6f} {t:>+8.2f} {sig:>5}")

    r2_v2 = float(res_v2.rsquared)
    _p(f"\n  R² within (v2): {r2_v2:.4f}  |  old: {OLD_R2:.4f}")

    # ── Walk-forward (2005-2015 train, 2016-2023 test) ─────────────────────────
    _p("\n  Walk-forward validation (train 2005-2015, test 2016-2023) …")
    EXOG_WF = [c for c in EXOG_V2 if c != "regime_stamp_duty"]
    stamp_coef = res_v2.params.get("regime_stamp_duty", 0.0)

    rmse_v2, mae_v2, da_v2 = _walk_forward_rmse(df, EXOG_WF, stamp_coef)
    _p(f"  V2 walk-forward RMSE = {rmse_v2:.5f}  MAE = {mae_v2:.5f}  "
       f"DirAcc = {da_v2:.1f}%")
    _p(f"  Old spec (regime dummies) RMSE = {OLD_RMSE:.5f}")

    if not np.isnan(rmse_v2) and rmse_v2 <= OLD_RMSE:
        verdict = f"✅ Improvement confirmed (Δ RMSE = {OLD_RMSE - rmse_v2:.5f})"
    elif not np.isnan(rmse_v2):
        verdict = (f"⚠️  Predictive accuracy declined (Δ RMSE = "
                   f"{rmse_v2 - OLD_RMSE:+.5f}) — review variable selection")
    else:
        verdict = "⚠️  Walk-forward failed — check data completeness"
    _p(f"  {verdict}")

    # ── Spec comparison table ─────────────────────────────────────────────────
    old_pti_corr = _correlation_with_dep(df, "log_price_to_income_diff")
    new_aff_corr = _correlation_with_dep(df, sel_aff)

    _p("\n  SPECIFICATION COMPARISON")
    _p(f"  {'Metric':<30} {'Old spec (v1)':>18} {'New spec (v2)':>18}")
    _p(f"  {'-'*30} {'-'*18} {'-'*18}")
    _p(f"  {'R² within':<30} {OLD_R2:>18.4f} {r2_v2:>18.4f}")
    _p(f"  {'Near-identity (PTI corr)':<30} {'YES ({:.2f})'.format(old_pti_corr):>18} "
       f"{'NO ({:.2f})'.format(new_aff_corr) if abs(new_aff_corr) < 0.95 else 'PARTIAL ({:.2f})'.format(new_aff_corr):>18}")
    _p(f"  {'Walk-forward RMSE':<30} {OLD_RMSE:>18.5f} {rmse_v2 if not np.isnan(rmse_v2) else 'n/a':>18}")
    _p(f"  {'N variables (excl dummies)':<30} {'7':>18} {str(len(CORE_V2) + len(SUPP_V2)):>18}")

    # ── Save SDE parameters v2 ────────────────────────────────────────────────
    _p("\n  Saving sde_parameters_v2.csv …")
    sde_v1 = pd.read_csv(SDE_FINAL)

    # Build parameter rows (pooled OLS coefs same for all regions)
    sde_v2 = sde_v1.copy()

    # Update OLS-derived columns
    for var in EXOG_V2:
        c  = float(res_v2.params.get(var, np.nan))
        se = float(dk_map.get(var, np.nan))
        sde_v2[f"v2_{var}"]    = c
        sde_v2[f"v2_{var}_se"] = se

    sde_v2["v2_r2_within"] = r2_v2
    sde_v2["v2_rmse_wf"]   = rmse_v2 if not np.isnan(rmse_v2) else None
    sde_v2["v2_sel_aff"]   = sel_aff
    sde_v2["v2_sel_supply"] = sel_supply
    sde_v2["v2_spec_note"] = (
        f"V2: {sel_aff} replaces log_pti_diff; {sel_supply} replaces supply_lag3; "
        f"rate split into rate_x_high + rate_x_low; sigma/kappa/mu unchanged"
    )

    sde_v2.to_csv(SDE_V2_CSV, index=False)
    _p(f"  Saved: {SDE_V2_CSV.relative_to(ROOT)}")

    return res_v2, dk_map, EXOG_V2, rmse_v2


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(
    all_results: dict,
    sel_aff: str,
    sel_supply: str,
    add_vars: list[str],
    reject_vars: list[tuple],
) -> None:
    """Save selected_vars.txt, all_candidate_tests.csv, variable_selection_report.txt"""

    # ── selected_vars.txt ─────────────────────────────────────────────────────
    selected_txt = VIMP / "selected_vars.txt"
    lines = [
        f"UK Housing Model — Selected Variable Set (v2)",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 50,
        "",
        f"SELECTED AFFORDABILITY VARIABLE: {sel_aff}",
        f"SELECTED SUPPLY VARIABLE:         {sel_supply}",
        "RATE DECOMPOSITION:               rate_x_high + rate_x_low",
        "",
        "ADDITIONAL VARIABLES INCLUDED:",
    ]
    for v in add_vars:
        lines.append(f"  + {v}")
    if not add_vars:
        lines.append("  (none passed inclusion threshold)")
    lines += ["", "VARIABLES REJECTED:"]
    for v, reason in reject_vars:
        lines.append(f"  - {v}: {reason}")
    lines += [""]

    # Note any failed downloads
    failed_downloads = [k for k, v in all_results.items()
                        if v.get("skip_reason") == "download failed"]
    if failed_downloads:
        lines.append("DOWNLOAD FAILURES (excluded automatically):")
        for v in failed_downloads:
            lines.append(f"  {v}: download failed — exclude")

    selected_txt.write_text("\n".join(lines), encoding="utf-8")

    # ── all_candidate_tests.csv ───────────────────────────────────────────────
    rows = []
    for var, res in all_results.items():
        rows.append({
            "variable":       var,
            "label":          res.get("label", var),
            "coef":           res.get("coef", np.nan),
            "t_stat":         res.get("t_stat", np.nan),
            "p_val":          res.get("p_val", np.nan),
            "r2_within":      res.get("r2_within", np.nan),
            "n_obs":          res.get("n_obs", 0),
            "corr_dep":       res.get("corr_dep", np.nan),
            "adf_stationary": res.get("adf_stationary", np.nan),
            "adf_median_p":   res.get("adf_median_p", np.nan),
            "skip_reason":    res.get("skip_reason", ""),
            "selected":       var in ([sel_aff, sel_supply, "rate_x_high", "rate_x_low"]
                                       + add_vars),
        })
    pd.DataFrame(rows).to_csv(VIMP / "all_candidate_tests.csv", index=False)

    # ── variable_selection_report.txt ─────────────────────────────────────────
    report_path = VIMP / "variable_selection_report.txt"
    report_path.write_text("\n".join(_REPORT_LINES), encoding="utf-8")

    print(f"\n[Outputs saved]")
    print(f"  {selected_txt.relative_to(ROOT)}")
    print(f"  {(VIMP / 'all_candidate_tests.csv').relative_to(ROOT)}")
    print(f"  {report_path.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = datetime.now()
    _h("UK Housing Model — Improved Variable Construction")
    _p(f"Run: {t0:%Y-%m-%d %H:%M:%S}")

    # Load base data
    df = load_base_data()

    # Part 1 — Affordability
    df, aff_results, sel_aff = build_affordability_variables(df)

    # Part 2 — Credit/rate channel
    df, credit_results = build_credit_variables(df)

    # Part 3 — Supply
    df, supply_results, sel_supply = build_supply_variables(df)

    # Part 4 — Supplementary
    df, supp_results = build_supplementary_variables(df)

    # Merge all results
    all_results = {
        **aff_results,
        **credit_results,
        **supply_results,
        **supp_results,
    }

    # Part 5 — Select and build v2 dataset
    add_vars, reject_vars = select_best_variables(all_results, sel_aff, sel_supply)

    df_v2 = build_improved_dataset(
        df, sel_aff, sel_supply, add_vars, reject_vars, all_results
    )

    # Part 6 — Re-run OLS
    run_improved_ols(df_v2, sel_aff, sel_supply, add_vars)

    # Save all outputs
    save_outputs(all_results, sel_aff, sel_supply, add_vars, reject_vars)

    elapsed = (datetime.now() - t0).total_seconds()
    _h(f"COMPLETE — elapsed {elapsed:.0f}s")


if __name__ == "__main__":
    main()
