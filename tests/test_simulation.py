"""Tests for the canonical Stage 5 simulation helper."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.sde import run_monte_carlo, summarize_paths

ROOT = Path(__file__).resolve().parents[1]


def test_run_monte_carlo_shape_and_summary():
    paths = run_monte_carlo(
        start_price=250000,
        n_months=24,
        n_paths=100,
        params={"kappa": 0.15, "sigma": 0.08, "drift_annual": 0.01, "long_run_mean": 260000},
        seed=42,
    )
    summary = summarize_paths(paths)

    assert paths.shape == (25, 100)
    assert summary.shape[0] == 25
    assert {"p10", "p25", "p50", "p75", "p90", "mean"}.issubset(summary.columns)


def test_run_monte_carlo_is_reproducible_with_seed():
    params = {"kappa": 0.12, "sigma": 0.06, "drift_annual": 0.005, "long_run_mean": 240000}
    paths_a = run_monte_carlo(225000, n_months=12, n_paths=25, params=params, seed=7)
    paths_b = run_monte_carlo(225000, n_months=12, n_paths=25, params=params, seed=7)

    assert paths_a.equals(paths_b)


def test_iqr_minimum_coverage():
    """Assert that at least 9/12 regions have sim_IQR / hist_IQR >= 0.30.

    Ensures the IQR prior implemented in canonical_rebuild.py has prevented
    universal IQR compression across the regional panel.  Individual-region
    minimums are not enforced here — up to 3 regions may remain compressed.
    """
    sde_path = ROOT / "data/outputs/stage4_final/sde_parameters_bankgrade.csv"
    panel_path = ROOT / "data/processed/master_dataset_canonical.csv"

    sde_params = pd.read_csv(sde_path)
    panel = pd.read_csv(panel_path)

    N_SIM = 5000
    HORIZON = 60
    MIN_RATIO = 0.30
    MIN_PASSING_REGIONS = 9

    rows = []
    for _, pr in sde_params.iterrows():
        region = str(pr["region"])
        sigma = float(pr["sigma"])
        kappa = float(pr["kappa"])

        # Historical 5-year IQR from 60-month compounded log returns
        grp = panel[panel["region"] == region].sort_values("date").reset_index(drop=True)
        prices = grp["nominal_house_price"].values
        hist_returns = [
            np.log(prices[i + HORIZON] / prices[i]) * 100.0
            for i in range(len(grp) - HORIZON)
        ]
        hist_iqr = float(np.percentile(hist_returns, 75) - np.percentile(hist_returns, 25))

        # Simulated 5-year IQR (mean-zero OU, X0=0)
        rng = np.random.default_rng(42)
        eps = rng.standard_normal((N_SIM, HORIZON))
        kappa_m = kappa / 12.0
        sigma_m = sigma / np.sqrt(12.0)
        X = np.zeros(N_SIM)
        for t in range(HORIZON):
            X = X * (1.0 - kappa_m) + sigma_m * eps[:, t]
        sim_iqr = float(np.percentile(X * 100.0, 75) - np.percentile(X * 100.0, 25))

        ratio = sim_iqr / hist_iqr if hist_iqr > 0 else 0.0
        rows.append({
            "region": region,
            "sigma": round(sigma, 5),
            "kappa": round(kappa, 4),
            "hist_iqr_pp": round(hist_iqr, 2),
            "sim_iqr_pp": round(sim_iqr, 2),
            "ratio": round(ratio, 3),
            "passes": ratio >= MIN_RATIO,
        })

    result_df = pd.DataFrame(rows)
    print("\n--- IQR coverage per region ---")
    print(result_df.to_string(index=False))

    n_passing = int(result_df["passes"].sum())
    failing = result_df[~result_df["passes"]]["region"].tolist()
    assert n_passing >= MIN_PASSING_REGIONS, (
        f"Only {n_passing}/12 regions have sim_IQR/hist_IQR >= {MIN_RATIO}. "
        f"Failing regions: {failing}. "
        f"IQR prior in canonical_rebuild.py may need recalibration."
    )
