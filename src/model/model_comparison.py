"""Model horse race: Log-OU vs GBM vs zero-growth vs random walk.

Evaluates four stochastic process models on the holdout panel
(2018-01-01 to 2026-01-01) for 1-month-ahead and 12-month-ahead
log-price forecasts.

Metrics computed per region and pooled across all 12 regions:
  - RMSE of log-price forecast
  - Directional accuracy (correct sign of price change)

Usage:
    python -m src.model.model_comparison

Output:
    data/outputs/validation/model_comparison.json
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "processed" / "master_dataset_canonical.csv"
FAIR_VALUE_PATH = ROOT / "data" / "outputs" / \
    "stage4_final" / "fair_value_panel_bankgrade.csv"
SDE_PARAMS_PATH = ROOT / "data" / "outputs" / \
    "stage4_final" / "sde_parameters_bankgrade.csv"
OUTPUT_DIR = ROOT / "data" / "outputs" / "validation"
OUTPUT_PATH = OUTPUT_DIR / "model_comparison.json"

HOLDOUT_START = pd.Timestamp("2018-01-01")
HOLDOUT_END = pd.Timestamp("2026-01-01")
DT = 1.0 / 12.0  # monthly step (years)
N_GBM_PATHS = 500
RNG_SEED = 42

HORIZON_1M = 1
HORIZON_12M = 12


def _ou_point_forecast(log_p_current: float, log_p_star: float,
                       kappa: float, gamma_annual: float, h: int) -> float:
    """Log-OU median forecast at horizon h months.

    E[log P_{t+h}] = log P* + (log P_t - log P*) * exp(-kappa * h * DT)
                   + gamma_annual * h * DT
    """
    x0 = log_p_current - log_p_star
    x_h = x0 * np.exp(-kappa * h * DT)
    return log_p_star + x_h + gamma_annual * h * DT


def _gbm_median_forecast(log_p_current: float, mu_monthly: float,
                         sigma_annual: float, h: int,
                         rng: np.random.Generator) -> float:
    """GBM median forecast via 500 simulated paths.

    dlog P = (mu_monthly - 0.5*sigma_monthly^2) * dt + sigma_monthly * dW
    """
    sigma_monthly = sigma_annual / np.sqrt(12.0)
    drift_monthly = mu_monthly - 0.5 * sigma_monthly ** 2
    noise = rng.standard_normal((N_GBM_PATHS, h)) * sigma_monthly
    cumulative = drift_monthly * h + noise.sum(axis=1)
    return float(log_p_current + np.median(cumulative))


def _rw_median_forecast(log_p_current: float, sigma_annual: float,
                        h: int, rng: np.random.Generator) -> float:
    """Random walk median forecast.

    P_{t+h} = P_t * exp(sigma * sqrt(h*DT) * z), z ~ N(0,1)
    Median of log-normal = log P_t (since median of z=0)
    """
    sigma_monthly = sigma_annual / np.sqrt(12.0)
    z = rng.standard_normal(N_GBM_PATHS)
    log_returns = sigma_monthly * np.sqrt(float(h)) * z
    return float(log_p_current + np.median(log_returns))


def _metrics(actuals: np.ndarray, forecasts: np.ndarray,
             actual_changes: np.ndarray) -> dict:
    """Compute RMSE and directional accuracy."""
    errors = actuals - forecasts
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    # Directional accuracy: forecast change vs actual change
    forecast_changes = forecasts - np.concatenate([[actuals[0]], actuals[:-1]])
    # Use sign of change from previous period
    # actual_changes[i] = actuals[i] - actuals[i-1] (pre-computed)
    correct_dir = np.sign(actual_changes) == np.sign(forecast_changes)
    dir_acc = float(np.mean(correct_dir[1:]))  # skip first (no prior)
    return {"rmse": round(rmse, 6), "dir_acc": round(dir_acc, 4)}


def run_model_comparison() -> dict:
    warnings.filterwarnings("ignore")

    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    fv_panel = pd.read_csv(FAIR_VALUE_PATH, parse_dates=["date"])
    sde = pd.read_csv(SDE_PARAMS_PATH).set_index("region")

    rng = np.random.default_rng(RNG_SEED)

    holdout_mask = (panel["date"] >= HOLDOUT_START) & (
        panel["date"] < HOLDOUT_END)
    regions = sde.index.tolist()

    # Accumulators for pooled results
    pooled: dict[str, dict[str, list]] = {
        "log_ou": {"rmse_1m": [], "rmse_12m": [], "dir_1m": [], "dir_12m": []},
        "gbm":    {"rmse_1m": [], "rmse_12m": [], "dir_1m": [], "dir_12m": []},
        "zero_growth": {"rmse_1m": [], "rmse_12m": [], "dir_1m": [], "dir_12m": []},
        "random_walk": {"rmse_1m": [], "rmse_12m": [], "dir_1m": [], "dir_12m": []},
    }

    regional_results: dict[str, dict] = {}

    for region in regions:
        reg = panel[panel["region"] == region].sort_values(
            "date").reset_index(drop=True)
        if len(reg) < 30:
            continue

        row = sde.loc[region]
        kappa = float(row["kappa"])
        sigma = float(row["sigma"])
        gamma_annual = float(row["gamma_annual_pp"]) / 100.0

        # Time-varying P* from fair value panel
        fv_reg = fv_panel[fv_panel["region"] == region].sort_values(
            "date").reset_index(drop=True)
        fv_dict = dict(zip(fv_reg["date"], np.log(
            fv_reg["fair_value_nominal"].values)))

        # Historical mean monthly log-return for GBM (from full panel pre-holdout)
        pre_holdout = panel[(panel["region"] == region) & (
            panel["date"] < HOLDOUT_START)].copy()
        if len(pre_holdout) > 12:
            log_prices_pre = np.log(pre_holdout["nominal_house_price"].values)
            monthly_returns_pre = np.diff(log_prices_pre)
            mu_monthly = float(np.mean(monthly_returns_pre))
        else:
            mu_monthly = gamma_annual * DT

        # Holdout series (include one observation before holdout start for lagging)
        reg_hold = panel[(panel["region"] == region) & holdout_mask].sort_values(
            "date").reset_index(drop=True)

        if len(reg_hold) < 15:
            print(
                f"  {region}: insufficient holdout data ({len(reg_hold)} obs) — skipping")
            continue

        log_prices = np.log(reg_hold["nominal_house_price"].values)
        # Time-varying log P* aligned to holdout dates
        dates = reg_hold["date"].values
        log_p_star_series = np.array([fv_dict.get(pd.Timestamp(d), np.log(float(row["mu_equilibrium"])))
                                      for d in dates])
        n = len(log_prices)

        # --- 1-month-ahead forecasts ---
        ou_1m = np.full(n, np.nan)
        gbm_1m = np.full(n, np.nan)
        zg_1m = np.full(n, np.nan)
        rw_1m = np.full(n, np.nan)

        for i in range(1, n):
            lp = log_prices[i - 1]
            lp_star = log_p_star_series[i]  # use P* at forecast target date
            ou_1m[i] = _ou_point_forecast(
                lp, lp_star, kappa, gamma_annual, HORIZON_1M)
            gbm_1m[i] = _gbm_median_forecast(
                lp, mu_monthly, sigma, HORIZON_1M, rng)
            zg_1m[i] = lp
            rw_1m[i] = _rw_median_forecast(lp, sigma, HORIZON_1M, rng)

        # --- 12-month-ahead forecasts ---
        ou_12m = np.full(n, np.nan)
        gbm_12m = np.full(n, np.nan)
        zg_12m = np.full(n, np.nan)
        rw_12m = np.full(n, np.nan)

        for i in range(HORIZON_12M, n):
            lp = log_prices[i - HORIZON_12M]
            lp_star = log_p_star_series[i]  # use P* at forecast target date
            ou_12m[i] = _ou_point_forecast(
                lp, lp_star, kappa, gamma_annual, HORIZON_12M)
            gbm_12m[i] = _gbm_median_forecast(
                lp, mu_monthly, sigma, HORIZON_12M, rng)
            zg_12m[i] = lp
            rw_12m[i] = _rw_median_forecast(lp, sigma, HORIZON_12M, rng)

        valid_1m = ~(np.isnan(ou_1m) | np.isnan(log_prices))
        valid_12m = ~(np.isnan(ou_12m) | np.isnan(log_prices))
        actual_changes = np.diff(log_prices, prepend=log_prices[0])

        def _reg_metrics_1m(fc: np.ndarray) -> dict:
            mask = valid_1m
            a, f = log_prices[mask], fc[mask]
            rmse = float(np.sqrt(np.mean((a - f) ** 2)))
            chg_a = actual_changes[mask]
            chg_f = f - \
                np.concatenate([[log_prices[0]], log_prices])[:-1][mask]
            dir_acc = float(np.mean(np.sign(chg_a[1:]) == np.sign(chg_f[1:])))
            return {"rmse": round(rmse, 6), "dir_acc": round(dir_acc, 4)}

        def _reg_metrics_12m(fc: np.ndarray) -> dict:
            mask = valid_12m
            a, f = log_prices[mask], fc[mask]
            rmse = float(np.sqrt(np.mean((a - f) ** 2)))
            # Directional accuracy: forecast vs actual change over 12m horizon
            idx = np.where(mask)[0]
            directions_correct = []
            for ii in idx:
                if ii >= HORIZON_12M:
                    actual_chg = log_prices[ii] - log_prices[ii - HORIZON_12M]
                    fc_chg = fc[ii] - log_prices[ii - HORIZON_12M]
                    if not np.isnan(fc_chg):
                        directions_correct.append(
                            np.sign(actual_chg) == np.sign(fc_chg))
            dir_acc = float(np.mean(directions_correct)
                            ) if directions_correct else float("nan")
            return {"rmse": round(rmse, 6), "dir_acc": round(dir_acc, 4) if not np.isnan(dir_acc) else None}

        m_ou_1m = _reg_metrics_1m(ou_1m)
        m_gbm_1m = _reg_metrics_1m(gbm_1m)
        m_zg_1m = _reg_metrics_1m(zg_1m)
        m_rw_1m = _reg_metrics_1m(rw_1m)
        m_ou_12m = _reg_metrics_12m(ou_12m)
        m_gbm_12m = _reg_metrics_12m(gbm_12m)
        m_zg_12m = _reg_metrics_12m(zg_12m)
        m_rw_12m = _reg_metrics_12m(rw_12m)

        regional_results[region] = {
            "n_holdout": int(n),
            "log_ou": {"rmse_1m": m_ou_1m["rmse"], "rmse_12m": m_ou_12m["rmse"],
                       "dir_1m": m_ou_1m["dir_acc"], "dir_12m": m_ou_12m["dir_acc"]},
            "gbm":    {"rmse_1m": m_gbm_1m["rmse"], "rmse_12m": m_gbm_12m["rmse"],
                       "dir_1m": m_gbm_1m["dir_acc"], "dir_12m": m_gbm_12m["dir_acc"]},
            "zero_growth": {"rmse_1m": m_zg_1m["rmse"], "rmse_12m": m_zg_12m["rmse"],
                            "dir_1m": m_zg_1m["dir_acc"], "dir_12m": m_zg_12m["dir_acc"]},
            "random_walk": {"rmse_1m": m_rw_1m["rmse"], "rmse_12m": m_rw_12m["rmse"],
                            "dir_1m": m_rw_1m["dir_acc"], "dir_12m": m_rw_12m["dir_acc"]},
        }

        for model, m1, m12 in [("log_ou", m_ou_1m, m_ou_12m), ("gbm", m_gbm_1m, m_gbm_12m),
                               ("zero_growth", m_zg_1m, m_zg_12m), ("random_walk", m_rw_1m, m_rw_12m)]:
            pooled[model]["rmse_1m"].append(m1["rmse"])
            pooled[model]["rmse_12m"].append(m12["rmse"])
            pooled[model]["dir_1m"].append(m1["dir_acc"])
            if m12["dir_acc"] is not None:
                pooled[model]["dir_12m"].append(m12["dir_acc"])

        print(f"  {region:35s}: OU-12m RMSE={m_ou_12m['rmse']:.5f}  GBM={m_gbm_12m['rmse']:.5f}  "
              f"ZG={m_zg_12m['rmse']:.5f}")

    # Pooled (mean across regions)
    def _pool(d: dict) -> dict:
        return {
            "rmse_1m": round(float(np.mean(d["rmse_1m"])), 6),
            "rmse_12m": round(float(np.mean(d["rmse_12m"])), 6),
            "dir_1m": round(float(np.mean(d["dir_1m"])), 4),
            "dir_12m": round(float(np.mean(d["dir_12m"])), 4) if d["dir_12m"] else None,
        }

    pooled_summary = {k: _pool(v) for k, v in pooled.items()}

    ou_12m = pooled_summary["log_ou"]["rmse_12m"]
    gbm_12m = pooled_summary["gbm"]["rmse_12m"]
    zg_12m = pooled_summary["zero_growth"]["rmse_12m"]
    ou_beats_gbm_12m = ou_12m < gbm_12m
    ou_beats_zg_12m = ou_12m < zg_12m

    # 1-month comparison
    ou_1m_rmse = pooled_summary["log_ou"]["rmse_1m"]
    zg_1m_rmse = pooled_summary["zero_growth"]["rmse_1m"]
    ou_beats_zg_1m = ou_1m_rmse < zg_1m_rmse

    finding = (
        f"Pooled holdout RMSE (12m): Log-OU={ou_12m:.5f}, GBM={gbm_12m:.5f}, "
        f"Zero-growth={zg_12m:.5f}. "
    )
    if not ou_beats_zg_1m:
        finding += (
            "NOTE: Log-OU does NOT beat zero-growth at 1m horizon (expected — documented in "
            "subsample_ols_validation.json; mean-reversion adds value at 12m+ horizons). "
        )
    if ou_beats_gbm_12m:
        finding += f"Log-OU beats GBM at 12m horizon (OU RMSE {ou_12m:.5f} < GBM {gbm_12m:.5f}). "
    else:
        finding += f"Log-OU does not beat GBM at 12m (OU RMSE {ou_12m:.5f} >= GBM {gbm_12m:.5f}). "
    if ou_beats_zg_12m:
        finding += f"Log-OU beats zero-growth at 12m (OU {ou_12m:.5f} < ZG {zg_12m:.5f})."
    else:
        finding += f"Log-OU does not beat zero-growth at 12m (OU {ou_12m:.5f} >= ZG {zg_12m:.5f})."

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "holdout": f"{HOLDOUT_START.date()} to {HOLDOUT_END.date()}",
        "n_paths_gbm_rw": N_GBM_PATHS,
        "models": {
            model: {
                "rmse_1m":  pooled_summary[model]["rmse_1m"],
                "rmse_12m": pooled_summary[model]["rmse_12m"],
                "dir_1m":   pooled_summary[model]["dir_1m"],
                "dir_12m":  pooled_summary[model]["dir_12m"],
            }
            for model in ["log_ou", "gbm", "zero_growth", "random_walk"]
        },
        "pooled_across_regions": pooled_summary,
        "regional_results": regional_results,
        "log_ou_beats_gbm_12m": ou_beats_gbm_12m,
        "log_ou_beats_zero_growth_12m": ou_beats_zg_12m,
        "log_ou_beats_zero_growth_1m": ou_beats_zg_1m,
        "finding": finding,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        f"\nPooled 12m RMSE: OU={ou_12m:.5f}  GBM={gbm_12m:.5f}  ZG={zg_12m:.5f}")
    print(f"OU beats GBM at 12m: {ou_beats_gbm_12m}")
    print(f"OU beats ZG  at 12m: {ou_beats_zg_12m}")
    print(f"Saved to {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    run_model_comparison()
