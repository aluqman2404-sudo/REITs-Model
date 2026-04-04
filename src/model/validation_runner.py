"""
Independent validation runner for the UK housing market model.

IMPORT COMPLIANCE
-----------------
Reads only artifact files. No imports from src/model/, src/scoring/,
src/simulation/, or any estimation code.
Imports: pathlib, sys (stdlib) — numpy, pandas only.

FEATURE COVERAGE LIMITATION
----------------------------
sde_parameters_bankgrade.csv contains five channel betas:
  beta_rate         → mortgage_rate_lag3             (present in panel ✓)
  beta_financial_stress → financial_stress_excess_lag3 (present in panel ✓)
  beta_income       → earnings_growth_12m_lag1       (present in panel ✓)
  beta_transaction_growth → transaction_growth        (present in panel ✓)
  beta_supply       → approvals_growth               (ABSENT from panel ✗)
  beta_affordability → affordability_gap_lag1        (ABSENT from panel ✗)

beta_supply and beta_affordability are omitted from predicted_growth.
All forecast errors are therefore computed on a partial-feature prediction.
This is documented in the output files.

INPUT FILES
-----------
  data/outputs/stage4_final/sde_parameters_bankgrade.csv
  data/outputs/stage4_final/fair_value_panel_bankgrade.csv
  data/outputs/stage6/stage7_handoff_bankgrade.csv

OUTPUT FILES
------------
  data/outputs/validation/independent_validation_results.csv
  data/outputs/validation/independent_validation_summary.csv
  data/outputs/validation/score_ir_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (no src/ imports — paths resolved from file location only)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

STAGE4_DIR = ROOT / "data" / "outputs" / "stage4_final"
STAGE6_DIR = ROOT / "data" / "outputs" / "stage6"
VALIDATION_DIR = ROOT / "data" / "outputs" / "validation"

SDE_PARAMS_FILE   = STAGE4_DIR / "sde_parameters_bankgrade.csv"
PANEL_FILE        = STAGE4_DIR / "fair_value_panel_bankgrade.csv"
HANDOFF_FILE      = STAGE6_DIR / "stage7_handoff_bankgrade.csv"

RESULTS_FILE      = VALIDATION_DIR / "independent_validation_results.csv"
SUMMARY_FILE      = VALIDATION_DIR / "independent_validation_summary.csv"
SCORE_IR_FILE     = VALIDATION_DIR / "score_ir_summary.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZONS             = [1, 3, 6, 12]    # months ahead
VALIDATION_MONTHS    = 24               # last N months of panel used for evaluation
TRAILING_MEAN_WINDOW = 12               # months for the trailing-mean benchmark

# Beta name in sde_parameters → feature column in fair_value_panel
# beta_supply (approvals_growth) and beta_affordability (affordability_gap_lag1)
# are not in the panel and are therefore excluded.
FEATURE_MAP: dict[str, str] = {
    "beta_rate":               "mortgage_rate_lag3",
    "beta_financial_stress":   "financial_stress_excess_lag3",
    "beta_income":             "earnings_growth_12m_lag1",
    "beta_transaction_growth": "transaction_growth",
}

SUPPORTIVE_THRESHOLD = 55.0  # consumer_score >= this → Supportive
CAUTIOUS_THRESHOLD   = 45.0  # consumer_score <  this → Cautious
IR_HORIZONS          = [12, 36]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params  = pd.read_csv(SDE_PARAMS_FILE)
    panel   = pd.read_csv(PANEL_FILE, parse_dates=["date"])
    handoff = pd.read_csv(HANDOFF_FILE)
    return params, panel, handoff


# ---------------------------------------------------------------------------
# 1. Expanding-window forecast validation
# ---------------------------------------------------------------------------

def _trailing_mean(growth: np.ndarray, i: int, window: int) -> float:
    """Mean of the `window` monthly returns ending at i-1 (no look-ahead)."""
    if i < window:
        return np.nan
    segment = growth[i - window : i]
    if np.all(np.isnan(segment)):
        return np.nan
    return float(np.nanmean(segment))


def build_forecast_results(
    params: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    """
    For each region and each month in the last VALIDATION_MONTHS of the panel,
    apply stored Stage 4 coefficients to compute predicted_growth, then compare
    to actual price_growth at horizons 1, 3, 6, 12 months ahead.

    Also records naive benchmarks: zero-growth, AR1 (price_growth_lag1),
    and 12-month trailing mean.
    """
    params_idx = params.set_index("region")
    max_date = panel["date"].max()
    val_start = max_date - pd.DateOffset(months=VALIDATION_MONTHS - 1)

    records: list[dict] = []

    for region, grp in panel.groupby("region"):
        if region not in params_idx.index:
            continue

        coef = params_idx.loc[region]
        grp  = grp.sort_values("date").reset_index(drop=True)
        n    = len(grp)

        growth = grp["price_growth"].to_numpy(dtype=float)

        # Precompute feature arrays (length n, NaN where unavailable)
        feat_arrays: dict[str, np.ndarray] = {}
        for beta_col, feat_col in FEATURE_MAP.items():
            if feat_col in grp.columns:
                feat_arrays[beta_col] = grp[feat_col].to_numpy(dtype=float)

        # price_growth_lag1[i] = growth[i-1], NaN at i=0
        growth_lag1 = np.empty(n)
        growth_lag1[:] = np.nan
        growth_lag1[1:] = growth[:-1]

        # Validation window: rows where date >= val_start
        val_indices = grp.index[grp["date"] >= val_start].tolist()

        for i in val_indices:
            date_t = grp.loc[i, "date"]

            # --- Predicted growth (partial-feature; see module docstring) ---
            predicted = 0.0
            if not np.isnan(growth_lag1[i]):
                predicted += float(coef["rho"]) * growth_lag1[i]
            for beta_col, arr in feat_arrays.items():
                val = arr[i]
                if not np.isnan(val):
                    predicted += float(coef[beta_col]) * val

            # --- Benchmarks at time t ---
            bm_ar1   = growth_lag1[i]          # NaN if i == 0
            bm_trail = _trailing_mean(growth, i, TRAILING_MEAN_WINDOW)

            # --- Compare to actual at each horizon ---
            for h in HORIZONS:
                j = i + h
                if j >= n:
                    continue
                actual = growth[j]
                if np.isnan(actual):
                    continue

                records.append({
                    "region":                  region,
                    "date_t":                  date_t,
                    "horizon":                 h,
                    "predicted_growth":        round(predicted, 6),
                    "actual_growth":           round(actual, 6),
                    "error":                   round(predicted - actual, 6),
                    "benchmark_zero":          0.0,
                    "benchmark_ar1":           round(bm_ar1, 6)   if not np.isnan(bm_ar1)   else np.nan,
                    "benchmark_trailing_mean": round(bm_trail, 6) if not np.isnan(bm_trail) else np.nan,
                    "features_omitted":        "beta_supply,beta_affordability",
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Benchmark comparison (RMSE / MAE) and 3. Directional accuracy
# ---------------------------------------------------------------------------

def _rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def _mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))


def _dir_acc(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.sign(pred) == np.sign(actual)))


def compute_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregated RMSE, MAE, and directional accuracy by horizon.
    Per-region directional accuracy is derivable from results CSV.
    """
    overall_rows: list[dict] = []
    region_dir_rows: list[dict] = []

    for h in HORIZONS:
        sub = results[results["horizon"] == h].copy()
        sub = sub.dropna(subset=["actual_growth", "predicted_growth",
                                  "benchmark_ar1", "benchmark_trailing_mean"])
        if sub.empty:
            continue

        actual    = sub["actual_growth"].values
        predicted = sub["predicted_growth"].values
        zero      = np.zeros_like(actual)
        ar1       = sub["benchmark_ar1"].values
        trail     = sub["benchmark_trailing_mean"].values

        overall_rows.append({
            "report_type":           "overall",
            "horizon":               h,
            "n_obs":                 len(sub),
            "model_rmse":            round(_rmse(predicted, actual), 6),
            "model_mae":             round(_mae(predicted, actual), 6),
            "zero_rmse":             round(_rmse(zero, actual), 6),
            "zero_mae":              round(_mae(zero, actual), 6),
            "ar1_rmse":              round(_rmse(ar1, actual), 6),
            "ar1_mae":               round(_mae(ar1, actual), 6),
            "trailing_mean_rmse":    round(_rmse(trail, actual), 6),
            "trailing_mean_mae":     round(_mae(trail, actual), 6),
            "directional_accuracy":  round(_dir_acc(predicted, actual), 4),
            "region":                "all",
        })

        # Per-region directional accuracy
        for region, rgrp in sub.groupby("region"):
            if len(rgrp) < 2:
                continue
            region_dir_rows.append({
                "report_type":          "directional_accuracy_by_region",
                "horizon":              h,
                "n_obs":                len(rgrp),
                "model_rmse":           np.nan,
                "model_mae":            np.nan,
                "zero_rmse":            np.nan,
                "zero_mae":             np.nan,
                "ar1_rmse":             np.nan,
                "ar1_mae":              np.nan,
                "trailing_mean_rmse":   np.nan,
                "trailing_mean_mae":    np.nan,
                "directional_accuracy": round(_dir_acc(
                    rgrp["predicted_growth"].values,
                    rgrp["actual_growth"].values,
                ), 4),
                "region": region,
            })

    return pd.DataFrame(overall_rows + region_dir_rows)


# ---------------------------------------------------------------------------
# 4. Score information ratio
# ---------------------------------------------------------------------------

def _compound_forward_return(growth: np.ndarray, i: int, h: int) -> float:
    window = growth[i + 1 : i + 1 + h]
    if len(window) < h or np.any(np.isnan(window)):
        return np.nan
    return float(np.prod(1.0 + window) - 1.0)


def compute_score_ir(panel: pd.DataFrame, handoff: pd.DataFrame) -> pd.DataFrame:
    """
    Splits regions into Supportive (consumer_score >= 55) and
    Cautious (consumer_score < 45) at the scoring_date, then computes
    mean and IR of historical forward returns across all panel months
    for each group.

    Mixed regions (45 <= score < 55) are excluded from IR computation.
    If no Cautious regions exist under current thresholds, IR = NaN.
    """
    score_map = handoff.set_index("region")["consumer_score"].to_dict()
    supportive = {r for r, s in score_map.items() if s >= SUPPORTIVE_THRESHOLD}
    cautious   = {r for r, s in score_map.items() if s <  CAUTIOUS_THRESHOLD}

    rows: list[dict] = []

    for h in IR_HORIZONS:
        sup_returns: list[float] = []
        cau_returns: list[float] = []

        for region, grp in panel.groupby("region"):
            grp    = grp.sort_values("date").reset_index(drop=True)
            growth = grp["price_growth"].to_numpy(dtype=float)
            n      = len(grp)

            for i in range(n):
                fwd = _compound_forward_return(growth, i, h)
                if np.isnan(fwd):
                    continue
                if region in supportive:
                    sup_returns.append(fwd)
                elif region in cautious:
                    cau_returns.append(fwd)

        sup_arr = np.array(sup_returns)
        cau_arr = np.array(cau_returns)

        mean_sup = float(np.mean(sup_arr)) if len(sup_arr) > 0 else np.nan
        mean_cau = float(np.mean(cau_arr)) if len(cau_arr) > 0 else np.nan

        if len(sup_arr) > 0 and len(cau_arr) > 0:
            pooled_std = float(np.std(np.concatenate([sup_arr, cau_arr])))
            ir = (mean_sup - mean_cau) / pooled_std if pooled_std > 0 else np.nan
        else:
            pooled_std = np.nan
            ir = np.nan

        rows.append({
            "horizon_months":         h,
            "n_supportive_regions":   len(supportive),
            "n_cautious_regions":     len(cautious),
            "n_supportive_obs":       len(sup_arr),
            "n_cautious_obs":         len(cau_arr),
            "mean_return_supportive": round(mean_sup, 6)    if not np.isnan(mean_sup)    else np.nan,
            "mean_return_cautious":   round(mean_cau, 6)    if not np.isnan(mean_cau)    else np.nan,
            "pooled_std":             round(pooled_std, 6)  if not np.isnan(pooled_std)  else np.nan,
            "ir":                     round(ir, 4)          if not np.isnan(ir)          else np.nan,
            "note": (
                "No cautious regions at current thresholds (supportive>=55, cautious<45); "
                "IR not computable."
                if len(cautious) == 0 else ""
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stdout summary
# ---------------------------------------------------------------------------

def print_summary(summary: pd.DataFrame, score_ir: pd.DataFrame) -> None:
    overall = summary[summary["report_type"] == "overall"].copy()
    by_region = summary[summary["report_type"] == "directional_accuracy_by_region"].copy()

    width = 76
    print("\n" + "=" * width)
    print("INDEPENDENT VALIDATION SUMMARY")
    print("=" * width)

    # RMSE table
    print("\nRMSE by horizon — model vs benchmarks (partial-feature prediction):")
    header = f"{'H':>4} {'N':>5} {'Model':>9} {'Zero':>9} {'AR1':>9} {'Trail.Mean':>11} {'Dir.Acc':>8}"
    print(header)
    print("-" * len(header))
    for _, r in overall.iterrows():
        print(
            f"{int(r['horizon']):>4} {int(r['n_obs']):>5}"
            f" {r['model_rmse']:>9.5f} {r['zero_rmse']:>9.5f}"
            f" {r['ar1_rmse']:>9.5f} {r['trailing_mean_rmse']:>11.5f}"
            f" {r['directional_accuracy']:>8.3f}"
        )

    # Per-region directional accuracy at h=1 (most interpretable)
    h1 = by_region[by_region["horizon"] == 1].sort_values("directional_accuracy", ascending=False)
    if not h1.empty:
        print(f"\nDirectional accuracy by region (h=1):")
        for _, r in h1.iterrows():
            print(f"  {r['region']:<30} {r['directional_accuracy']:.3f}  (n={int(r['n_obs'])})")

    # Score IR
    print(f"\nScore IR (consumer_score ≥{SUPPORTIVE_THRESHOLD:.0f} vs <{CAUTIOUS_THRESHOLD:.0f}):")
    h_col = f"{'H':>6}"
    print(f"{h_col} {'N_Sup_Reg':>10} {'N_Cau_Reg':>10} {'Mean_Sup':>10} {'Mean_Cau':>10} {'Pooled_Std':>12} {'IR':>8}")
    print("-" * 72)
    for _, r in score_ir.iterrows():
        mean_sup = f"{r['mean_return_supportive']:.4f}" if not pd.isna(r["mean_return_supportive"]) else "   N/A"
        mean_cau = f"{r['mean_return_cautious']:.4f}"   if not pd.isna(r["mean_return_cautious"])   else "   N/A"
        std_s    = f"{r['pooled_std']:.4f}"             if not pd.isna(r["pooled_std"])             else "   N/A"
        ir_s     = f"{r['ir']:.4f}"                     if not pd.isna(r["ir"])                     else "   N/A"
        print(
            f"{int(r['horizon_months']):>6} {int(r['n_supportive_regions']):>10}"
            f" {int(r['n_cautious_regions']):>10} {mean_sup:>10}"
            f" {mean_cau:>10} {std_s:>12} {ir_s:>8}"
        )
        if r.get("note"):
            print(f"         NOTE: {r['note']}")

    print("=" * width)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading artifacts (no src/ imports)…")
    params, panel, handoff = load_artifacts()
    print(f"  sde_parameters : {len(params)} regions")
    print(f"  fair_value_panel: {len(panel)} rows  {panel['date'].min().date()} – {panel['date'].max().date()}")
    print(f"  handoff        : {len(handoff)} regions  scoring_date={handoff['scoring_date'].iloc[0]}")
    print(
        f"  Omitted betas  : beta_supply (approvals_growth), "
        f"beta_affordability (affordability_gap_lag1) — not in panel"
    )

    print("\n[1/4] Building forecast results (last 24 months × 12 regions × 4 horizons)…")
    results = build_forecast_results(params, panel)
    results.to_csv(RESULTS_FILE, index=False)
    print(f"       {len(results)} rows → {RESULTS_FILE.name}")

    print("[2/4] Computing RMSE / MAE vs benchmarks…")
    print("[3/4] Computing directional accuracy per region and overall…")
    summary = compute_summary(results)
    summary.to_csv(SUMMARY_FILE, index=False)
    print(f"       {len(summary)} rows → {SUMMARY_FILE.name}")

    print("[4/4] Computing score information ratio…")
    score_ir = compute_score_ir(panel, handoff)
    score_ir.to_csv(SCORE_IR_FILE, index=False)
    print(f"       {len(score_ir)} rows → {SCORE_IR_FILE.name}")

    print_summary(summary, score_ir)
    return results, summary, score_ir


if __name__ == "__main__":
    run()
