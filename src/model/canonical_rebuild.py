"""Canonical Stage 4-6 rebuild on the canonical historical research panel."""

from __future__ import annotations
from src.simulation.sde import run_monte_carlo
from src.scoring.engine import label_score
from src.model.ols import fit_structural_fair_value
from src.data.export_stage4_ready_panel import export_stage4_ready_panel
from src.data.export_stage4_ready_panel import OUTPUT_PATH as STAGE4_PANEL_PATH
from src.data.build_canonical_panel import build_canonical_panel
from src.core.paths import (
    CANONICAL_PANEL_METADATA_PATH,
    CANONICAL_PANEL_PATH,
    OUTPUT_DATA_DIR,
    STAGE4_OUTPUT_DIR,
    STAGE5_OUTPUT_DIR,
    STAGE6_OUTPUT_DIR,
    ensure_directory,
)
from src.core.config import load_config
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from scipy import stats as sp_stats
from linearmodels.panel import PanelOLS
import statsmodels.graphics.tsaplots as tsaplots
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import log

import matplotlib
matplotlib.use("Agg")


RATE_FEATURE = "mortgage_rate_lag3"
STRESS_FEATURE = "financial_stress_excess_lag3"
EARNINGS_FEATURE = "earnings_growth_12m_lag1"
SUPPLY_FEATURE = "approvals_growth"
AFFORDABILITY_FEATURE = "affordability_gap_lag1"
FEATURE_COLUMNS = [
    RATE_FEATURE,
    STRESS_FEATURE,
    "transaction_growth",
    EARNINGS_FEATURE,
    "price_growth_lag1",
    SUPPLY_FEATURE,
    AFFORDABILITY_FEATURE,
]
MONTH_COLUMNS = [f"month_{month}" for month in range(2, 13)]
RATE_CANDIDATES = list(
    dict.fromkeys(
        [
            RATE_FEATURE,
            "mortgage_rate_gap_12m_lag1",
            "real_mortgage_rate_lag1",
            "payment_burden_gap_lag1",
        ]
    )
)
SHRINKAGE_MONTHS = 60.0

STAGE4_OUTPUT_PATH = STAGE4_OUTPUT_DIR / "sde_parameters_bankgrade.csv"
STAGE4_FAIR_VALUE_PANEL_PATH = STAGE4_OUTPUT_DIR / "fair_value_panel_bankgrade.csv"
STAGE4_DIAGNOSTICS_PATH = STAGE4_OUTPUT_DIR / "canonical_stage4_diagnostics.csv"
STAGE4_POOLED_COEFFS_PATH = STAGE4_OUTPUT_DIR / \
    "canonical_stage4_pooled_coefficients.csv"
RATE_DIAGNOSTICS_PATH = STAGE4_OUTPUT_DIR / "rate_channel_candidates.csv"
RESIDUAL_ACF_DIR = ensure_directory(STAGE4_OUTPUT_DIR / "residual_acf")
STRUCTURAL_BREAK_DIAGNOSTICS_PATH = STAGE4_OUTPUT_DIR / \
    "structural_break_diagnostics.csv"
POST2015_PARAMS_PATH = STAGE4_OUTPUT_DIR / "sde_parameters_post2015.csv"
POST2015_START = pd.Timestamp("2015-01-01")
CHOW_BREAK_DATES = ["2008-09-01", "2020-03-01", "2022-06-01"]
STAGE5_OUTPUT_PATH = STAGE5_OUTPUT_DIR / "simulation_summary_bankgrade.csv"
SIGMA_MAPPING_DIAGNOSTICS_PATH = STAGE5_OUTPUT_DIR / "sigma_mapping_diagnostics.csv"
SIGMA_CALIBRATION_PATH = STAGE5_OUTPUT_DIR / "sigma_calibration.csv"
REGION_PLAUSIBILITY_PATH = STAGE5_OUTPUT_DIR / \
    "region_plausibility_diagnostics.csv"
BASELINE_FIT_PATH = STAGE5_OUTPUT_DIR / "baseline_return_model_coefficients.csv"
STAGE6_OUTPUT_PATH = STAGE6_OUTPUT_DIR / "stage7_handoff_bankgrade.csv"
PIPELINE_METADATA_PATH = STAGE6_OUTPUT_DIR / "canonical_rebuild_metadata.json"
VALIDATION_DIR = ensure_directory(OUTPUT_DATA_DIR / "validation")
SIMULATION_PLAUSIBILITY_REVISED_PATH = VALIDATION_DIR / \
    "simulation_plausibility_revised.csv"


@dataclass(frozen=True)
class CanonicalBuildResult:
    stage4_path: str
    stage5_path: str
    stage6_path: str
    generated_at_utc: str
    regions: int
    scenarios: int
    max_publication_lag_months: int | None
    effective_data_as_of: str | None


def _clip(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _ensure_month_columns(df: pd.DataFrame) -> pd.DataFrame:
    for month in range(2, 13):
        column = f"month_{month}"
        if column not in df.columns:
            df[column] = (pd.to_datetime(df["date"]).dt.month ==
                          month).astype(float)
    return df


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    config = load_config()
    grp = df.groupby("region", group_keys=False)
    if "mortgage_rate_change_3m" not in df.columns:
        df["mortgage_rate_change_3m"] = grp["mortgage_rate"].diff(3)
    if RATE_FEATURE not in df.columns:
        df[RATE_FEATURE] = grp["mortgage_rate_change_3m"].shift(1)
    if "mortgage_rate_gap_12m" not in df.columns:
        df["mortgage_rate_gap_12m"] = df["mortgage_rate"] - grp["mortgage_rate"].transform(
            lambda series: series.rolling(12, min_periods=6).mean()
        )
    if "mortgage_rate_gap_12m_lag1" not in df.columns:
        df["mortgage_rate_gap_12m_lag1"] = grp["mortgage_rate_gap_12m"].shift(
            1)
    if "real_mortgage_rate" not in df.columns:
        inflation_yoy = grp["cpi"].pct_change(
            12) * 100.0 if "cpi" in df.columns else 0.0
        df["real_mortgage_rate"] = df["mortgage_rate"] - inflation_yoy
    if "real_mortgage_rate_lag1" not in df.columns:
        df["real_mortgage_rate_lag1"] = grp["real_mortgage_rate"].shift(1)
    if EARNINGS_FEATURE not in df.columns:
        df[EARNINGS_FEATURE] = grp["real_annual_earnings"].pct_change(
            12).shift(1)
    if "payment_burden_gap_lag1" not in df.columns and "payment_burden_gap" in df.columns:
        df["payment_burden_gap_lag1"] = grp["payment_burden_gap"].shift(1)
    if "financial_stress_lag3" in df.columns and STRESS_FEATURE not in df.columns:
        threshold = float(
            config.fair_value.financial_stress_activation_threshold)
        # Treat financial stress as an episodic overlay, not a continuous structural elasticity.
        df[STRESS_FEATURE] = (
            df["financial_stress_lag3"] - threshold).clip(lower=0.0)
    if "nominal_house_price" in df.columns and "real_annual_earnings" in df.columns:
        if "loan_to_income" not in df.columns:
            # LTI at 4x conventional income multiple anchor: measures how many
            # multiples of (4× regional earnings) the current price represents.
            df["loan_to_income"] = df["nominal_house_price"] / (
                4.0 * df["real_annual_earnings"].replace(0.0, np.nan)
            )
        if "lti_rolling_mean_10yr" not in df.columns:
            df["lti_rolling_mean_10yr"] = (
                df.groupby("region")["loan_to_income"]
                .transform(lambda s: s.rolling(120, min_periods=60).mean())
            )
        if "affordability_gap" not in df.columns:
            df["affordability_gap"] = df["loan_to_income"] - \
                df["lti_rolling_mean_10yr"]
        if AFFORDABILITY_FEATURE not in df.columns:
            df[AFFORDABILITY_FEATURE] = (
                df.groupby("region")["affordability_gap"].shift(1)
            )
    return _ensure_month_columns(df)


def _load_master_dataset() -> pd.DataFrame:
    if not CANONICAL_PANEL_PATH.exists():
        build_canonical_panel()
    df = pd.read_csv(CANONICAL_PANEL_PATH, parse_dates=["date"])
    df = df.sort_values(["region", "date"]).reset_index(drop=True)
    return _add_engineered_features(df)


def load_model_panel() -> pd.DataFrame:
    """Load the enriched Stage 4-ready panel with canonical features."""
    if not STAGE4_PANEL_PATH.exists():
        export_stage4_ready_panel()
    panel = pd.read_csv(STAGE4_PANEL_PATH, parse_dates=["date"]).sort_values(
        ["region", "date"]).reset_index(drop=True)
    return _add_engineered_features(panel)


def _rate_channel_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    indexed = panel.set_index(["region", "date"]).sort_index()
    diagnostics = []
    # Keep the candidate check aligned with the deployed Stage 4 growth equation.
    base_features = ["transaction_growth", EARNINGS_FEATURE,
                     "price_growth_lag1", STRESS_FEATURE]

    for feature in RATE_CANDIDATES:
        if feature not in indexed.columns:
            continue
        row = {"feature": feature}
        signs: list[float] = []
        for sample_name, bounds in {
            "full": None,
            "early": ("2005-01-01", "2014-12-01"),
            "late": ("2015-01-01", "2023-07-01"),
        }.items():
            subset = indexed if bounds is None else indexed[
                (indexed.index.get_level_values("date") >= bounds[0]) &
                (indexed.index.get_level_values("date") <= bounds[1])
            ]
            clean = subset.dropna(
                subset=["price_growth", feature, *base_features, *MONTH_COLUMNS])
            if clean.empty:
                row[f"{sample_name}_coef"] = np.nan
                row[f"{sample_name}_r2"] = np.nan
                continue
            try:
                result = PanelOLS(
                    clean["price_growth"],
                    sm.add_constant(
                        clean[[feature, *base_features, *MONTH_COLUMNS]], has_constant="add"),
                    entity_effects=True,
                    drop_absorbed=True,
                    check_rank=False,
                ).fit(cov_type="clustered", cluster_entity=True)
                coef = float(result.params.get(feature, np.nan))
                row[f"{sample_name}_coef"] = coef
                row[f"{sample_name}_r2"] = float(result.rsquared)
                if np.isfinite(coef):
                    signs.append(np.sign(coef))
            except Exception:
                row[f"{sample_name}_coef"] = np.nan
                row[f"{sample_name}_r2"] = np.nan
        row["stable_sign"] = bool(
            len(signs) == 3 and signs[0] == signs[1] == signs[2])
        diagnostics.append(row)

    result = pd.DataFrame(diagnostics).sort_values(
        ["stable_sign", "feature"], ascending=[False, True]).reset_index(drop=True)
    result.to_csv(RATE_DIAGNOSTICS_PATH, index=False)
    return result


def _estimate_growth_coefficients(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indexed = panel.set_index(["region", "date"]).sort_index()

    # Compute affordability_gap_lag1 if _add_engineered_features did not run on the
    # fair_panel (which may be a subset of master).  This is the only feature that
    # requires a fallback because it depends on a rolling 10-year window that is not
    # always carried through by fit_structural_fair_value.
    if AFFORDABILITY_FEATURE not in indexed.columns:
        if "nominal_house_price" in indexed.columns and "real_annual_earnings" in indexed.columns:
            lti = indexed["nominal_house_price"] / (
                4.0 * indexed["real_annual_earnings"].replace(0.0, np.nan)
            )
            lti_mean = lti.groupby(level="region").transform(
                lambda s: s.rolling(120, min_periods=60).mean()
            )
            gap = lti - lti_mean
            indexed = indexed.copy()
            indexed[AFFORDABILITY_FEATURE] = gap.groupby(
                level="region").shift(1)

    # Determine the active feature set.  Optional features (SUPPLY, AFFORDABILITY) are
    # included only when they are present with ≥50% non-null coverage.
    supply_available = (
        SUPPLY_FEATURE in indexed.columns and
        float(indexed[SUPPLY_FEATURE].notna().mean()) >= 0.50
    )
    affordability_available = (
        AFFORDABILITY_FEATURE in indexed.columns and
        float(indexed[AFFORDABILITY_FEATURE].notna().mean()) >= 0.50
    )
    inactive = (
        (set() if supply_available else {SUPPLY_FEATURE}) |
        (set() if affordability_available else {AFFORDABILITY_FEATURE})
    )
    active_features = [f for f in FEATURE_COLUMNS if f not in inactive]

    clean = indexed.dropna(subset=["price_growth", *active_features, *MONTH_COLUMNS,
                           "real_annual_earnings", "real_monthly_rent", "nominal_house_price"])

    pooled_features = active_features + MONTH_COLUMNS
    pooled_model = PanelOLS(
        clean["price_growth"],
        sm.add_constant(clean[pooled_features], has_constant="add"),
        entity_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    pooled_coeffs = pd.DataFrame(
        {
            "feature": pooled_model.params.index,
            "coefficient": pooled_model.params.values,
            "std_error": pooled_model.std_errors.values,
            "p_value": pooled_model.pvalues.values,
        }
    ).sort_values("feature").reset_index(drop=True)

    config = load_config()
    guardrails = config.guardrails
    region_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    pooled_lookup = pooled_model.params.to_dict()

    # Pre-compute per-region full-history mean of nominal monthly price growth,
    # annualised.  Uses log-differences of nominal_house_price so the anchor
    # captures the full nominal drift including CPI inflation.
    # BUG FIX (2026-03-22): previously used price_growth (= Δ log real_price,
    # CPI-deflated), which produced near-zero or negative anchors for regions
    # with flat real returns (Wales −0.06%/yr, Yorkshire +0.07%/yr,
    # West Midlands +0.14%/yr) despite clearly positive nominal growth
    # (~2.8–3.0%/yr over 2005–2025).  Nominal log-differences give the
    # correct anchor (~3.0%/yr for all 12 regions).
    long_run_nominal_lookup: dict[str, float] = {}
    for _region, _rgrp in indexed.groupby(level="region"):
        _nom_price = _rgrp["nominal_house_price"].replace(0.0, np.nan)
        _nom_growth = np.log(_nom_price).diff().dropna()
        long_run_nominal_lookup[_region] = float(_nom_growth.mean() * 12.0)

    # Collect raw and guardrailed betas for post-loop distribution reports.
    supply_raw_vals: list[tuple[str, float, float]] = []
    affordability_raw_vals: list[tuple[str, float]] = []

    for region, group in clean.groupby(level="region"):
        grp = group.reset_index()
        X = sm.add_constant(grp[pooled_features], has_constant="add")
        fit = sm.OLS(grp["price_growth"], X).fit(
            cov_type="HAC", cov_kwds={"maxlags": 24, "use_correction": True}
        )
        # maxlags=24 (2 years): UK rate cycles run 3-5 years; 24 monthly lags better absorbs
        # the longer-cycle persistence in mortgage-rate and earnings series than maxlags=12.
        shrink = len(grp) / (len(grp) + SHRINKAGE_MONTHS)

        def shrunk(feature: str) -> float:
            regional = float(fit.params.get(
                feature, pooled_lookup.get(feature, 0.0)))
            pooled = float(pooled_lookup.get(feature, regional))
            return shrink * regional + (1.0 - shrink) * pooled

        beta_rate = min(shrunk(RATE_FEATURE), guardrails.beta_rate_max)
        beta_financial_stress = min(
            shrunk(STRESS_FEATURE), guardrails.beta_financial_stress_max)
        beta_income = max(shrunk(EARNINGS_FEATURE), guardrails.beta_income_min)
        beta_transaction = max(shrunk("transaction_growth"),
                               guardrails.beta_transaction_min)
        rho = min(shrunk("price_growth_lag1"), guardrails.rho_max)

        if supply_available:
            beta_supply_raw = shrunk(SUPPLY_FEATURE)
            # non-positive supply prior
            beta_supply = min(beta_supply_raw, 0.0)
            beta_supply_se = float(fit.bse.get(SUPPLY_FEATURE, np.nan))
        else:
            beta_supply_raw = np.nan
            beta_supply = np.nan
            beta_supply_se = np.nan
        supply_raw_vals.append((region, beta_supply_raw, beta_supply))

        if affordability_available:
            # No sign guardrail: the direction is data-driven (high LTI gap is
            # expected to be negative for price growth but regions differ).
            beta_affordability = shrunk(AFFORDABILITY_FEATURE)
            beta_affordability_se = float(
                fit.bse.get(AFFORDABILITY_FEATURE, np.nan))
        else:
            beta_affordability = np.nan
            beta_affordability_se = np.nan
        affordability_raw_vals.append((region, beta_affordability))

        # Flag when the regional estimate contradicts the pooled sign AND the SE is
        # wide enough that the estimate is unreliable.  The estimate is left unchanged.
        _pooled_aff_sign = np.sign(
            pooled_lookup.get(AFFORDABILITY_FEATURE, 0.0))
        beta_affordability_low_confidence = bool(
            affordability_available and
            np.isfinite(beta_affordability) and
            np.isfinite(beta_affordability_se) and
            np.sign(beta_affordability) != _pooled_aff_sign and
            beta_affordability_se > 0.04
        )

        price_growth = grp["price_growth"].dropna()
        winsor_lo = float(price_growth.quantile(0.05))
        winsor_hi = float(price_growth.quantile(0.95))
        sigma_raw_annual = float(price_growth.std() * np.sqrt(12.0))
        sigma_winsor_annual = float(price_growth.clip(
            winsor_lo, winsor_hi).std() * np.sqrt(12.0))
        sigma_annual = _clip(
            guardrails.sigma_raw_weight * sigma_raw_annual +
            guardrails.sigma_winsor_weight * sigma_winsor_annual,
            config.fair_value.sigma_floor_annual,
            config.fair_value.sigma_cap_annual,
        )

        earnings_growth = float(
            grp["real_annual_earnings"].pct_change(12).tail(24).mean())
        rent_growth = float(
            grp["real_monthly_rent"].pct_change(12).tail(24).mean())
        transaction_trend = float(grp["transaction_growth"].tail(24).mean())
        gamma_current_real = 0.55 * earnings_growth + \
            0.25 * rent_growth + 0.20 * transaction_trend
        gamma_long_run_nominal = long_run_nominal_lookup.get(
            region, gamma_current_real)
        gamma_annual = _clip(
            0.40 * gamma_current_real + 0.60 * gamma_long_run_nominal,
            guardrails.gamma_min_annual,
            guardrails.gamma_max_annual,
        )

        # --- Residual diagnostics ---
        dw_stat = float(sm.stats.stattools.durbin_watson(fit.resid))

        try:
            _bg_lm, _bg_pval, _bg_f, _bg_fpval = acorr_breusch_godfrey(
                fit, nlags=3)
            bg_stat = float(_bg_lm)
            bg_pvalue = float(_bg_pval)
        except Exception:
            bg_stat = np.nan
            bg_pvalue = np.nan

        try:
            _bp_lm, _bp_pval, _bp_f, _bp_fpval = het_breuschpagan(
                fit.resid, fit.model.exog)
            bp_stat = float(_bp_lm)
            bp_pvalue = float(_bp_pval)
        except Exception:
            bp_stat = np.nan
            bp_pvalue = np.nan

        # ACF plot saved as PNG (non-interactive — Agg backend)
        try:
            fig, ax = plt.subplots(figsize=(8, 3))
            tsaplots.plot_acf(fit.resid, lags=24, ax=ax,
                              title=f"{region} — residual ACF (lags 1-24)")
            ax.set_xlabel("Lag (months)")
            ax.set_ylabel("ACF")
            fig.tight_layout()
            safe_name = region.lower().replace(" ", "_").replace(",", "")
            fig.savefig(RESIDUAL_ACF_DIR /
                        f"{safe_name}_residual_acf.png", dpi=120)
            plt.close(fig)
        except Exception:
            plt.close("all")

        latest = grp.sort_values("date").iloc[-1]
        region_rows.append(
            {
                "region": region,
                "beta_rate": beta_rate,
                "beta_rate_se": float(fit.bse.get(RATE_FEATURE, np.nan)),
                "beta_financial_stress": beta_financial_stress,
                "beta_financial_stress_se": float(fit.bse.get(STRESS_FEATURE, np.nan)),
                "beta_income": beta_income,
                "beta_income_se": float(fit.bse.get(EARNINGS_FEATURE, np.nan)),
                "beta_transaction_growth": beta_transaction,
                "beta_transaction_growth_se": float(fit.bse.get("transaction_growth", np.nan)),
                "beta_supply": beta_supply,
                "beta_supply_se": beta_supply_se,
                "beta_affordability": beta_affordability,
                "beta_affordability_se": beta_affordability_se,
                "beta_affordability_low_confidence": beta_affordability_low_confidence,
                "rho": rho,
                "rho_se": float(fit.bse.get("price_growth_lag1", np.nan)),
                "sigma": sigma_annual,
                "sigma_raw_annual": sigma_raw_annual,
                "sigma_winsor_annual": sigma_winsor_annual,
                "sigma_winsor_lo": winsor_lo,
                "sigma_winsor_hi": winsor_hi,
                "gamma_current_real_pp": gamma_current_real * 100.0,
                "gamma_long_run_nominal_pp": gamma_long_run_nominal * 100.0,
                "gamma_blend_weight_longrun": 0.60,
                "gamma_annual_pp": gamma_annual * 100.0,
                "r_squared_within": float(fit.rsquared),
                "n_obs": int(len(grp)),
                "sigma_frequency": "annual",
                "last_observation_date": str(pd.to_datetime(latest["date"]).date()),
                "rate_channel_feature": RATE_FEATURE,
                "stress_channel_feature": STRESS_FEATURE,
                "income_channel_feature": EARNINGS_FEATURE,
                "supply_channel_feature": SUPPLY_FEATURE if supply_available else "unavailable",
                "affordability_channel_feature": AFFORDABILITY_FEATURE if affordability_available else "unavailable",
            }
        )
        diagnostics_rows.append(
            {
                "region": region,
                "n_obs": int(len(grp)),
                "r_squared": float(fit.rsquared),
                "adj_r_squared": float(fit.rsquared_adj),
                "dw_stat": dw_stat,
                "durbin_watson": dw_stat,
                "bg_stat": bg_stat,
                "bg_pvalue": bg_pvalue,
                "bp_stat": bp_stat,
                "bp_pvalue": bp_pvalue,
                "residual_std_monthly": float(fit.resid.std()),
                "sigma_raw_annual": sigma_raw_annual,
                "sigma_winsor_annual": sigma_winsor_annual,
                "sigma_final_annual": sigma_annual,
                "shrinkage_weight": shrink,
                "beta_rate_raw": float(fit.params.get(RATE_FEATURE, np.nan)),
                "beta_rate_final": beta_rate,
                "beta_financial_stress_raw": float(fit.params.get(STRESS_FEATURE, np.nan)),
                "beta_financial_stress_final": beta_financial_stress,
                "beta_transaction_raw": float(fit.params.get("transaction_growth", np.nan)),
                "beta_transaction_final": beta_transaction,
                "beta_supply_raw": beta_supply_raw,
                "beta_supply_final": beta_supply,
                "beta_affordability_raw": float(fit.params.get(AFFORDABILITY_FEATURE, np.nan)) if affordability_available else np.nan,
                "beta_affordability_final": beta_affordability,
                "beta_affordability_low_confidence": beta_affordability_low_confidence,
                "beta_affordability_note": (
                    "Estimate has opposite sign to pooled and SE > 0.04. "
                    "Northern Ireland experienced a localised housing bust 2008-2013 "
                    "(peak-to-trough price fall ~50%, far exceeding any other UK region) "
                    "driven by a concentrated construction boom and subsequent credit "
                    "withdrawal specific to the Belfast market. This structural break "
                    "inverts the normal LTI-to-price-growth relationship over the "
                    "estimation window and makes the affordability coefficient "
                    "unreliable for forward projections. Leave estimate as-is; "
                    "use beta_affordability_low_confidence flag to suppress in scoring."
                ) if beta_affordability_low_confidence else "",
                "rho_raw": float(fit.params.get("price_growth_lag1", np.nan)),
                "rho_final": rho,
                "beta_rate_guardrail": "non_positive_sign_prior",
                "beta_financial_stress_guardrail": "non_positive_episodic_overlay_prior",
                "beta_income_guardrail": "non_negative_income_support_prior",
                "beta_transaction_guardrail": "non_negative_activity_prior",
                "beta_supply_guardrail": "non_positive_supply_prior" if supply_available else "fixed_nan_no_data",
                "beta_affordability_guardrail": "none_data_driven" if affordability_available else "fixed_nan_no_data",
                "rho_guardrail": "mean_reversion_stability_cap",
                "gamma_guardrail": "annual_drift_bound",
                "sigma_guardrail": "winsorised_blend_with_floor_cap",
            }
        )

    # Print beta_supply distribution before and after the non-positive guardrail.
    print(f"\n{'─'*70}")
    print(f"  SUPPLY CHANNEL ({SUPPLY_FEATURE})  —  beta_supply distribution")
    print(f"  supply_available={supply_available}  |  pooled estimate: "
          f"{pooled_lookup.get(SUPPLY_FEATURE, float('nan')):.6f}")
    print(
        f"  {'Region':<30}  {'Raw (shrunk)':>14}  {'After guardrail':>15}  {'Clipped?':>8}")
    print(f"  {'─'*30}  {'─'*14}  {'─'*15}  {'─'*8}")
    for reg, raw, final in sorted(supply_raw_vals):
        clipped = "YES" if np.isfinite(raw) and raw > 0.0 else ""
        raw_s = f"{raw:+.6f}" if np.isfinite(raw) else "    n/a"
        fin_s = f"{final:+.6f}" if np.isfinite(final) else "    n/a"
        print(f"  {reg:<30}  {raw_s:>14}  {fin_s:>15}  {clipped:>8}")
    raw_arr = np.array([r for _, r, _ in supply_raw_vals if np.isfinite(r)])
    fin_arr = np.array([f for _, _, f in supply_raw_vals if np.isfinite(f)])
    if len(raw_arr):
        print(f"  {'─'*70}")
        print(f"  raw  — min: {raw_arr.min():+.6f}  median: {np.median(raw_arr):+.6f}  max: {raw_arr.max():+.6f}  n_clipped: {int((raw_arr > 0).sum())}/{len(raw_arr)}")
        print(
            f"  final— min: {fin_arr.min():+.6f}  median: {np.median(fin_arr):+.6f}  max: {fin_arr.max():+.6f}")
    print(f"{'─'*70}\n")

    # Print beta_affordability distribution (no guardrail — data-driven).
    print(f"\n{'─'*70}")
    print(
        f"  AFFORDABILITY CHANNEL ({AFFORDABILITY_FEATURE})  —  beta_affordability distribution")
    print(f"  affordability_available={affordability_available}  |  pooled estimate: "
          f"{pooled_lookup.get(AFFORDABILITY_FEATURE, float('nan')):.6f}")
    print(f"  {'Region':<30}  {'Estimate (shrunk)':>18}")
    print(f"  {'─'*30}  {'─'*18}")
    for reg, est in sorted(affordability_raw_vals):
        est_s = f"{est:+.6f}" if np.isfinite(est) else "    n/a"
        print(f"  {reg:<30}  {est_s:>18}")
    aff_arr = np.array(
        [e for _, e in affordability_raw_vals if np.isfinite(e)])
    if len(aff_arr):
        print(f"  {'─'*70}")
        print(f"  min: {aff_arr.min():+.6f}  median: {np.median(aff_arr):+.6f}  max: {aff_arr.max():+.6f}"
              f"  n_negative: {int((aff_arr < 0).sum())}/{len(aff_arr)}")
    print(f"{'─'*70}\n")

    return pd.DataFrame(region_rows), pd.DataFrame(diagnostics_rows), pooled_coeffs


def _estimate_kappa_by_region(fair_panel: pd.DataFrame) -> pd.DataFrame:
    config = load_config()
    rows = []
    stable_phis = []
    for region, group in fair_panel.groupby("region"):
        grp = group.sort_values("date").copy()
        grp["gap_lag1"] = grp["fair_value_gap_log"].shift(1)
        sample = grp.dropna(subset=["fair_value_gap_log", "gap_lag1"])
        fit = sm.OLS(sample["fair_value_gap_log"], sm.add_constant(
            sample[["gap_lag1"]], has_constant="add")).fit()
        raw_phi = float(fit.params["gap_lag1"])
        if 0.0 < raw_phi < config.fair_value.kappa_phi_upper_bound:
            stable_phis.append(raw_phi)
        rows.append({"region": region, "raw_phi": raw_phi})

    fallback_phi = float(np.median(
        stable_phis)) if stable_phis else config.fair_value.kappa_fallback_phi
    result_rows = []
    for row in rows:
        raw_phi = row["raw_phi"]
        if 0.0 < raw_phi < config.fair_value.kappa_phi_upper_bound:
            phi = raw_phi
            mode = "estimated"
        else:
            phi = fallback_phi
            mode = config.guardrails.kappa_fallback_mode
        result_rows.append(
            {
                "region": row["region"],
                "phi_gap_ar1_raw": raw_phi,
                "phi_gap_ar1": phi,
                "kappa": -log(phi) * 12.0,
                "kappa_estimation_mode": mode,
            }
        )
    return pd.DataFrame(result_rows)


def _fit_baseline_return_model(fair_panel: pd.DataFrame) -> tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    rows = []
    for region, group in fair_panel.groupby("region"):
        grp = group.sort_values("date").reset_index(drop=True)
        for index in range(len(grp) - 60):
            rows.append(
                {
                    "region": region,
                    "date": grp.loc[index, "date"],
                    "forward_5y_log_return": float(np.log(grp.loc[index + 60, "nominal_house_price"] / grp.loc[index, "nominal_house_price"])),
                    "fair_value_gap_log": float(grp.loc[index, "fair_value_gap_log"]),
                    RATE_FEATURE: float(grp.loc[index, RATE_FEATURE]) if pd.notna(grp.loc[index, RATE_FEATURE]) else np.nan,
                    STRESS_FEATURE: float(grp.loc[index, STRESS_FEATURE]) if pd.notna(grp.loc[index, STRESS_FEATURE]) else np.nan,
                    "transaction_growth": float(grp.loc[index, "transaction_growth"]) if pd.notna(grp.loc[index, "transaction_growth"]) else np.nan,
                    EARNINGS_FEATURE: float(grp.loc[index, EARNINGS_FEATURE]) if pd.notna(grp.loc[index, EARNINGS_FEATURE]) else np.nan,
                }
            )
    sample = pd.DataFrame(rows).dropna().reset_index(drop=True)
    region_dummies = pd.get_dummies(
        sample["region"], prefix="region", drop_first=True, dtype=float)
    regressors = ["fair_value_gap_log", RATE_FEATURE,
                  STRESS_FEATURE, "transaction_growth", EARNINGS_FEATURE]
    X = pd.concat([sample[regressors], region_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(sample["forward_5y_log_return"], X).fit(cov_type="HC3")
    coefficients = pd.DataFrame(
        {
            "feature": fit.params.index,
            "coefficient": fit.params.values,
            "std_error": fit.bse.values,
            "p_value": fit.pvalues.values,
        }
    )
    coefficients.to_csv(BASELINE_FIT_PATH, index=False)
    return coefficients, fit


def _predict_baseline_return(
    fit: sm.regression.linear_model.RegressionResultsWrapper,
    row: pd.Series,
) -> float:
    regressors = ["fair_value_gap_log", RATE_FEATURE,
                  STRESS_FEATURE, "transaction_growth", EARNINGS_FEATURE]
    features = {feature: float(row.get(feature, 0.0))
                for feature in regressors}
    for name in fit.params.index:
        if name.startswith("region_"):
            features[name] = 1.0 if name == f"region_{row['region']}" else 0.0
    X = pd.DataFrame([features])
    X = sm.add_constant(X, has_constant="add")
    X = X.reindex(columns=fit.params.index, fill_value=0.0)
    return float(fit.predict(X).iloc[0])


def _simulation_plausibility_from_summary(master: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, group in master.groupby("region"):
        grp = group.sort_values("date").reset_index(drop=True)
        forward_returns = np.asarray(
            [
                (float(grp.loc[index + 60, "nominal_house_price"]) /
                 float(grp.loc[index, "nominal_house_price"]) - 1.0) * 100.0
                for index in range(len(grp) - 60)
            ],
            dtype=float,
        )
        baseline = summary[(summary["region"] == region) & (
            summary["scenario"] == "Baseline")].iloc[0]
        hist_loss = float(np.mean(forward_returns <= -10.0))
        rows.append(
            {
                "region": region,
                "historical_5y_p10": float(np.percentile(forward_returns, 10)),
                "historical_5y_p50": float(np.percentile(forward_returns, 50)),
                "historical_5y_p90": float(np.percentile(forward_returns, 90)),
                "historical_prob_loss_10pct": hist_loss,
                "baseline_sim_median_5y": float(baseline["median_5yr_growth"]),
                "baseline_sim_prob_loss_10pct": float(baseline["prob_terminal_loss_10pct"]),
                "median_minus_hist_p50": float(baseline["median_5yr_growth"] - np.percentile(forward_returns, 50)),
                "tail_risk_gap": float(baseline["prob_terminal_loss_10pct"] - hist_loss),
                "within_hist_10_90_band": bool(
                    np.percentile(forward_returns, 10) <=
                    float(baseline["median_5yr_growth"]) <=
                    np.percentile(forward_returns, 90)
                ),
            }
        )
    plausibility = pd.DataFrame(rows).sort_values(
        "region").reset_index(drop=True)
    plausibility.to_csv(SIMULATION_PLAUSIBILITY_REVISED_PATH, index=False)
    plausibility.to_csv(REGION_PLAUSIBILITY_PATH, index=False)
    return plausibility


def _within_demean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Subtract region mean from each column to partial out entity fixed effects."""
    out = df.copy()
    for col in cols:
        out[col] = out[col] - out.groupby("region")[col].transform("mean")
    return out


def _chow_f_test(
    clean: pd.DataFrame,
    break_ts: pd.Timestamp,
    active_features: list[str],
) -> dict:
    """Run a Chow F-test for a structural break at break_ts using within-demeaned OLS."""
    y_col = "price_growth"
    pre = clean[clean["date"] < break_ts].copy()
    post = clean[clean["date"] >= break_ts].copy()

    if len(pre) < len(active_features) * 2 or len(post) < len(active_features) * 2:
        return {"break_date": str(break_ts.date()), "f_stat": np.nan, "p_value": np.nan, "significant_5pct": False}

    pre_d = _within_demean(pre, [y_col] + active_features)
    post_d = _within_demean(post, [y_col] + active_features)
    full_d = _within_demean(clean, [y_col] + active_features)

    def _ssr(df_d: pd.DataFrame) -> float:
        X = sm.add_constant(df_d[active_features], has_constant="add")
        res = sm.OLS(df_d[y_col], X).fit()
        return float(res.ssr)

    ssr_full = _ssr(full_d)
    ssr_u = _ssr(pre_d) + _ssr(post_d)
    k = len(active_features) + 1  # +1 for intercept
    n_u = len(pre_d) + len(post_d)
    denom = ssr_u / (n_u - 2 * k)
    if denom <= 0:
        return {"break_date": str(break_ts.date()), "f_stat": np.nan, "p_value": np.nan, "significant_5pct": False}

    f_stat = ((ssr_full - ssr_u) / k) / denom
    p_value = float(sp_stats.f.sf(f_stat, dfn=k, dfd=n_u - 2 * k))
    return {
        "break_date": str(break_ts.date()),
        "f_stat": float(f_stat),
        "p_value": p_value,
        "significant_5pct": p_value < 0.05,
    }


def _region_dw_stats(clean: pd.DataFrame, active_features: list[str]) -> pd.Series:
    """Return per-region Durbin-Watson statistics from HC3 OLS residuals."""
    y_col = "price_growth"
    dw_vals = {}
    for region, grp in clean.groupby("region"):
        grp = grp.sort_values("date").dropna(subset=[y_col] + active_features)
        if len(grp) < len(active_features) + 2:
            continue
        X = sm.add_constant(grp[active_features], has_constant="add")
        res = sm.OLS(grp[y_col], X).fit(cov_type="HC3")
        dw_vals[region] = float(sm.stats.stattools.durbin_watson(res.resid))
    return pd.Series(dw_vals)


def run_stability_diagnostics(
    panel: pd.DataFrame,
    full_sample_diagnostics: pd.DataFrame,
) -> dict:
    """Run Chow structural-break tests and optionally produce post-2015 parameters.

    Returns a dict with keys ``any_significant`` and ``post2015_preferred``.
    """
    y_col = "price_growth"

    # Replicate active_features logic from _estimate_growth_coefficients
    indexed = panel.copy()
    if AFFORDABILITY_FEATURE not in indexed.columns:
        if "nominal_house_price" in indexed.columns and "real_annual_earnings" in indexed.columns:
            lti = indexed["nominal_house_price"] / (
                4.0 * indexed["real_annual_earnings"].replace(0.0, np.nan)
            )
            lti_mean = lti.groupby(indexed["region"]).transform(
                lambda s: s.rolling(120, min_periods=60).mean()
            )
            gap = lti - lti_mean
            indexed[AFFORDABILITY_FEATURE] = gap.groupby(
                indexed["region"]).shift(1)

    supply_available = (
        SUPPLY_FEATURE in indexed.columns and
        float(indexed[SUPPLY_FEATURE].notna().mean()) >= 0.50
    )
    affordability_available = (
        AFFORDABILITY_FEATURE in indexed.columns and
        float(indexed[AFFORDABILITY_FEATURE].notna().mean()) >= 0.50
    )
    inactive = (
        (set() if supply_available else {SUPPLY_FEATURE}) |
        (set() if affordability_available else {AFFORDABILITY_FEATURE})
    )
    active_features = [f for f in FEATURE_COLUMNS if f not in inactive]

    required_cols = [y_col, "date", "region"] + active_features
    clean = indexed.dropna(subset=required_cols).copy()

    # --- Chow tests ---
    chow_rows = []
    any_significant = False
    for bd_str in CHOW_BREAK_DATES:
        result = _chow_f_test(clean, pd.Timestamp(bd_str), active_features)
        chow_rows.append(result)
        if result["significant_5pct"]:
            any_significant = True

    # --- Post-2015 re-estimation if any break is significant ---
    post2015_preferred = False
    post2015_params_rows: list[dict] = []

    if any_significant:
        post_clean = clean[clean["date"] >= POST2015_START].copy()

        # Full-sample DW deviation
        dw_full = _region_dw_stats(clean, active_features)
        dw_post = _region_dw_stats(post_clean, active_features)
        median_dw_dev_full = float((dw_full - 2.0).abs().median())
        median_dw_dev_post = float((dw_post - 2.0).abs().median())
        post2015_preferred = median_dw_dev_post < median_dw_dev_full

        # PanelOLS on post-2015 subsample
        post_indexed = post_clean.set_index(["region", "date"])
        for col in [y_col] + active_features:
            post_indexed[col] = post_indexed[col].astype(float)

        try:
            pooled_post = PanelOLS(
                post_indexed[y_col],
                sm.add_constant(
                    post_indexed[active_features], has_constant="add"),
                entity_effects=True,
            ).fit(cov_type="clustered", cluster_entity=True)

            for region, grp in post_clean.groupby("region"):
                grp = grp.sort_values("date").dropna(
                    subset=[y_col] + active_features)
                if len(grp) < len(active_features) + 2:
                    continue
                n_region = len(grp)
                shrink = n_region / (n_region + SHRINKAGE_MONTHS)
                X = sm.add_constant(grp[active_features], has_constant="add")
                fit = sm.OLS(grp[y_col], X).fit(cov_type="HC3")

                def shrunk_post(feat: str) -> float:
                    regional = float(fit.params.get(feat, np.nan))
                    pool = float(pooled_post.params.get(feat, np.nan))
                    if not np.isfinite(regional) or not np.isfinite(pool):
                        return pool if np.isfinite(pool) else 0.0
                    return shrink * regional + (1.0 - shrink) * pool

                row: dict = {"region": region}
                for feat in active_features:
                    row[feat] = shrunk_post(feat)
                post2015_params_rows.append(row)

            pd.DataFrame(post2015_params_rows).to_csv(
                POST2015_PARAMS_PATH, index=False)
        except Exception:
            pass  # post-2015 estimation failed; skip file write

    # --- Write diagnostics ---
    diag_rows = []
    for cr in chow_rows:
        diag_rows.append(
            {
                "break_date": cr["break_date"],
                "f_stat": cr["f_stat"],
                "p_value": cr["p_value"],
                "significant_5pct": cr["significant_5pct"],
                "any_significant": any_significant,
                "post2015_preferred": post2015_preferred,
                "active_features": "|".join(active_features),
            }
        )
    pd.DataFrame(diag_rows).to_csv(
        STRUCTURAL_BREAK_DIAGNOSTICS_PATH, index=False)

    print(
        "\n[Structural Break Diagnostics]\n" +
        "\n".join(
            f"  {r['break_date']}: F={r['f_stat']:.3f}  p={r['p_value']:.4f}  "
            f"{'** SIGNIFICANT **' if r['significant_5pct'] else 'not significant'}"
            for r in chow_rows
        ) +
        f"\n  any_significant={any_significant}  post2015_preferred={post2015_preferred}"
    )
    return {"any_significant": any_significant, "post2015_preferred": post2015_preferred}


def _calibrate_sigma_by_region(
    panel: pd.DataFrame,
    params: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """Conditionally calibrate sigma per region to match the IQR of gap-conditional 5-year returns.

    For each region:
      1. Filter historical months where fair_value_gap_log is within ±0.15 of the
         current valuation_gap_current.  If fewer than 24 such months exist, fall back
         to the full unconditional sample (flagged via sigma_calibration_conditional=False).
      2. Compute the empirical p25 and p75 of 60-month forward log returns from those
         starting months only.  Using IQR rather than p10/p90 is deliberate: the OU
         process is designed to fit the central tendency of returns, not crisis tails —
         those are handled by the scenario layer.
      3. Simulate 1000 paths starting at the region's current valuation gap (not zero),
         evaluating all candidate sigma values (0.02–0.20 in steps of 0.005) via
         vectorised NumPy broadcasting.  Paths start at the current gap so mean-reversion
         force is correctly directed from the same starting state as the empirical filter.
      4. Select sigma minimising |sim_p25 − emp_p25| + |sim_p75 − emp_p75|.
      5. Apply floor/cap as hard limits only (not blend targets).

    Uses a fixed RNG seed (42) so results are reproducible across runs.
    Writes diagnostics to SIGMA_CALIBRATION_PATH and returns the same DataFrame.
    """
    N_PATHS = 1000
    N_MONTHS = 60
    SIGMA_GRID = np.arange(0.02, 0.205, 0.005)  # 37 candidate values
    GAP_BANDWIDTH = 0.15
    MIN_CONDITIONAL_MONTHS = 24

    rng = np.random.default_rng(42)
    # shape (1000, 60) — fixed across regions/sigmas
    eps = rng.standard_normal((N_PATHS, N_MONTHS))

    sigma_ms = SIGMA_GRID / np.sqrt(12.0)  # annualised → monthly, shape (37,)

    rows = []
    for _, pr in params.iterrows():
        region = pr["region"]
        kappa_annual = float(pr["kappa"])
        gamma_annual = float(pr["gamma_annual_pp"]) / 100.0  # pp → fraction
        sigma_old = float(pr["sigma"])
        gap_current = float(pr["valuation_gap_current"])

        kappa_m = kappa_annual / 12.0
        gamma_m = gamma_annual / 12.0
        a = 1.0 - kappa_m  # monthly AR(1) coefficient in gap space
        # If gap_current is missing (fair value model didn't converge for this region),
        # start simulation at equilibrium (gap=0) and treat as unconditional.
        gap_sim_start = gap_current if np.isfinite(gap_current) else 0.0

        grp = (
            panel[panel["region"] == region]
            .sort_values("date")
            .reset_index(drop=True)
        )

        # Require fair_value_gap_log for conditional filtering
        has_gap = "fair_value_gap_log" in grp.columns and grp["fair_value_gap_log"].notna(
        ).any()

        if len(grp) < N_MONTHS + 1:
            rows.append({
                "region": region, "sigma_old": sigma_old, "sigma_calibrated": sigma_old,
                "simulated_p25": np.nan, "empirical_p25": np.nan,
                "simulated_p75": np.nan, "empirical_p75": np.nan,
                "calibration_error": np.nan, "n_conditional_months": 0,
                "sigma_calibration_conditional": False,
            })
            continue

        prices = grp["nominal_house_price"].values
        gaps = grp["fair_value_gap_log"].values if has_gap else np.full(
            len(grp), np.nan)

        # --- Select conditional or unconditional starting indices ---
        max_start = len(grp) - N_MONTHS
        conditional = False
        if has_gap:
            cond_idx = [
                i for i in range(max_start)
                if np.isfinite(gaps[i]) and abs(gaps[i] - gap_current) <= GAP_BANDWIDTH
            ]
            if len(cond_idx) >= MIN_CONDITIONAL_MONTHS:
                start_indices = cond_idx
                conditional = True

        if not conditional:
            start_indices = list(range(max_start))

        emp_returns_pct = np.array([
            np.log(prices[i + N_MONTHS] / prices[i]) * 100.0
            for i in start_indices
        ])
        emp_p25 = float(np.percentile(emp_returns_pct, 25))
        emp_p75 = float(np.percentile(emp_returns_pct, 75))

        # --- Vectorised grid search — paths start at current gap, not zero ---
        # log_gap shape: (n_sigma, N_PATHS), initialised to gap_sim_start for all paths
        log_gap = np.full((len(SIGMA_GRID), N_PATHS), gap_sim_start)
        for t in range(N_MONTHS):
            log_gap = (
                log_gap * a +
                gamma_m +
                sigma_ms[:, np.newaxis] * eps[np.newaxis, :, t]
            )
        # 60-month log return = final gap − initial gap (since log P* cancels)
        log_returns_pct = (log_gap - gap_sim_start) * \
            100.0  # shape (n_sigma, N_PATHS)

        sim_p25s = np.percentile(
            log_returns_pct, 25, axis=1)  # shape (n_sigma,)
        sim_p75s = np.percentile(log_returns_pct, 75, axis=1)

        losses = np.abs(sim_p25s - emp_p25) + np.abs(sim_p75s - emp_p75)
        best_idx = int(np.argmin(losses))
        best_sigma = float(SIGMA_GRID[best_idx])

        # Per-region sigma cap override (falls back to global cap if not present).
        region_sigma_cap = config.fair_value.sigma_cap_overrides.get(
            region, config.fair_value.sigma_cap_annual
        )
        if region_sigma_cap != config.fair_value.sigma_cap_annual:
            print(f"  _calibrate_sigma: {region} using sigma_cap_override={region_sigma_cap:.3f} "
                  f"(global cap={config.fair_value.sigma_cap_annual:.3f})")
        sigma_calibrated = float(np.clip(
            best_sigma,
            config.fair_value.sigma_floor_annual,
            region_sigma_cap,
        ))

        rows.append({
            "region": region,
            "sigma_old": round(sigma_old, 6),
            "sigma_calibrated": round(sigma_calibrated, 6),
            "simulated_p25": round(float(sim_p25s[best_idx]), 4),
            "empirical_p25": round(emp_p25, 4),
            "simulated_p75": round(float(sim_p75s[best_idx]), 4),
            "empirical_p75": round(emp_p75, 4),
            "calibration_error": round(float(losses[best_idx]), 4),
            "n_conditional_months": len(start_indices),
            "sigma_calibration_conditional": conditional,
        })

    cal = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    ensure_directory(STAGE5_OUTPUT_DIR)
    cal.to_csv(SIGMA_CALIBRATION_PATH, index=False)
    return cal


def estimate_pti_ptr_blend(panel_df, min_w=0.25, max_w=0.75):
    """
    Grid-search for PTI weight w in [min_w, max_w] that minimises pooled
    squared log-price deviation from the blended fair-value anchor:

        log_fv_blend = w * log_income_proxy + (1-w) * log_rent_proxy

    Falls back to expert prior w=0.60 if required columns are absent or
    panel has fewer than 100 rows.
    """
    import warnings
    import numpy as np

    price_col = next(
        (c for c in ["log_real_price"] if c in panel_df.columns), None)
    income_col = next((c for c in ["log_income_asof", "log_real_earnings",
                                   "real_annual_earnings"] if c in panel_df.columns), None)
    rent_col = next((c for c in ["log_real_monthly_rent", "real_monthly_rent",
                                 "log_rent"] if c in panel_df.columns), None)

    if not all([price_col, income_col, rent_col]):
        warnings.warn(
            f"estimate_pti_ptr_blend: columns not found "
            f"(price={price_col}, income={income_col}, rent={rent_col}). "
            "Using expert prior w=0.60.", UserWarning)
        return 0.60

    df = panel_df[[price_col, income_col, rent_col]].dropna()
    if len(df) < 100:
        warnings.warn(
            "estimate_pti_ptr_blend: <100 rows. Using expert prior w=0.60.", UserWarning)
        return 0.60

    lp = df[price_col].values.copy()
    li = np.log(df[income_col].values) if df[income_col].min(
    ) > 0 else df[income_col].values.copy()
    lr = np.log(df[rent_col].values) if df[rent_col].min(
    ) > 0 else df[rent_col].values.copy()

    # de-mean so scale does not dominate
    lp -= lp.mean()
    li -= li.mean()
    lr -= lr.mean()

    best_w, best_sse = 0.60, np.inf
    for w in np.arange(min_w, max_w + 0.01, 0.05):
        lf = w * li + (1.0 - w) * lr
        sse = float(((lp - lf) ** 2).sum())
        if sse < best_sse:
            best_sse, best_w = sse, float(w)

    best_w = float(np.clip(best_w, min_w, max_w))
    print(f"[PTI/PTR blend] estimated w_PTI={best_w:.2f}  SSE={best_sse:.4f}")
    return best_w


def estimate_fair_value_blend_weights(
    panel_df: pd.DataFrame,
    min_weight: float = 0.20,
    max_weight: float = 0.80,
) -> float:
    """Estimate the optimal PTI/PTR blend weight from data.

    Fits two single-regressor fair-value models (PTI only, PTR only) on anchor
    months then grid-searches w in [min_weight, max_weight] at step 0.05 to
    minimise the pooled sum of squared log-deviations from the blended fair value:
        log_fv = w * log(pti_fv) + (1-w) * log(ptr_fv)

    Returns the pooled optimal weight, clipped to [0.30, 0.70] with a warning
    if the unconstrained estimate falls outside that range.
    """
    import warnings as _warnings

    config = load_config()
    anchor_months = config.fair_value.anchor_months
    sample = panel_df[panel_df["date"].dt.month.isin(anchor_months)].copy()
    sample = sample.dropna(
        subset=["log_real_price", "log_income_asof", "log_rent"]).reset_index(drop=True)

    if len(sample) < 30:
        _warnings.warn(
            "estimate_fair_value_blend_weights: fewer than 30 anchor observations — "
            "falling back to expert prior 0.60/0.40.",
            UserWarning,
            stacklevel=2,
        )
        return 0.60

    region_dummies = pd.get_dummies(
        sample["region"], prefix="region", drop_first=True, dtype=float)

    # PTI-only model (income-based fair value)
    X_pti = sm.add_constant(pd.concat(
        [sample[["log_income_asof"]], region_dummies], axis=1), has_constant="add")
    fit_pti = sm.OLS(sample["log_real_price"], X_pti).fit()
    pti_log_fv = fit_pti.predict(X_pti).values

    # PTR-only model (rent-based fair value)
    X_ptr = sm.add_constant(
        pd.concat([sample[["log_rent"]], region_dummies], axis=1), has_constant="add")
    fit_ptr = sm.OLS(sample["log_real_price"], X_ptr).fit()
    ptr_log_fv = fit_ptr.predict(X_ptr).values

    actual_log_price = sample["log_real_price"].values

    # Grid search over w
    best_w = 0.60
    best_sse = np.inf
    for w in np.arange(min_weight, max_weight + 1e-9, 0.05):
        blended = w * pti_log_fv + (1.0 - w) * ptr_log_fv
        sse = float(np.sum((actual_log_price - blended) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_w = float(w)

    if best_w < 0.30 or best_w > 0.70:
        _warnings.warn(
            f"estimate_fair_value_blend_weights: optimal w={best_w:.3f} is outside "
            f"[0.30, 0.70] — clipping to bounds.",
            UserWarning,
            stacklevel=2,
        )
        best_w = float(np.clip(best_w, 0.30, 0.70))

    return best_w


def run_subsample_ols_validation(
    panel_df: pd.DataFrame | None = None,
    split_date: str = "2018-01-01",
) -> dict:
    """Subsample OLS validation: refit on train, evaluate on holdout.

    Splits the canonical panel at split_date.  Refits the fair-value OLS and
    pooled growth OLS on the train window.  Evaluates price-growth predictions
    on the holdout against the zero-growth benchmark.

    Returns a summary dict and writes to data/outputs/validation/subsample_ols_validation.json.
    """
    from linearmodels.panel import PanelOLS as _PanelOLS

    config = load_config()
    master = panel_df if panel_df is not None else _load_master_dataset()

    split_ts = pd.Timestamp(split_date)
    train = master[master["date"] < split_ts].copy()
    # Re-apply train OLS coefficients to holdout (using the coefficients table)
    # Build region dummies for holdout consistent with train
    anchor_train = train[train["date"].dt.month.isin(config.fair_value.anchor_months)].dropna(
        subset=["log_real_price", *config.fair_value.regressors])
    region_dummies_train = pd.get_dummies(
        anchor_train["region"], prefix="region", drop_first=True, dtype=float)
    full_dummies = pd.get_dummies(
        master["region"], prefix="region", drop_first=True, dtype=float)
    for col in region_dummies_train.columns:
        if col not in full_dummies.columns:
            full_dummies[col] = 0.0
    full_dummies = full_dummies.reindex(
        columns=region_dummies_train.columns, fill_value=0.0)

    X_full = sm.add_constant(pd.concat([master[config.fair_value.regressors].astype(
        float), full_dummies], axis=1), has_constant="add")
    X_train_anchor = sm.add_constant(pd.concat([anchor_train[config.fair_value.regressors].astype(
        float), region_dummies_train], axis=1), has_constant="add")
    fit_train_ols = sm.OLS(
        anchor_train["log_real_price"], X_train_anchor).fit(cov_type="HC3")

    master_with_gap = master.copy()
    master_with_gap["fair_value_log_real"] = fit_train_ols.predict(X_full)
    master_with_gap["fair_value_gap_log"] = master_with_gap["log_real_price"] - \
        master_with_gap["fair_value_log_real"]

    holdout_with_gap = master_with_gap[master_with_gap["date"] >= split_ts].copy(
    )
    train_with_gap = master_with_gap[master_with_gap["date"] < split_ts].copy()

    # --- Refit growth OLS on train ---
    train_indexed = train_with_gap.set_index(["region", "date"]).sort_index()
    active_features = [f for f in FEATURE_COLUMNS if f in train_indexed.columns and float(
        train_indexed[f].notna().mean()) >= 0.50]
    growth_features = ["fair_value_gap_log"] + active_features + MONTH_COLUMNS
    clean_train = train_indexed.dropna(
        subset=["price_growth"] + growth_features)

    if len(clean_train) < 50:
        result: dict = {
            "train_end": str(split_ts.date()),
            "holdout_start": str(split_ts.date()),
            "holdout_rmse": None,
            "zero_growth_rmse": None,
            "beats_zero_growth": False,
            "directional_1m": None,
            "directional_6m": None,
            "directional_12m": None,
            "error": "Insufficient train observations",
        }
        ensure_directory(VALIDATION_DIR)
        (VALIDATION_DIR / "subsample_ols_validation.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        return result

    growth_model = _PanelOLS(
        clean_train["price_growth"],
        sm.add_constant(clean_train[growth_features], has_constant="add"),
        entity_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    # --- Predict on holdout ---
    holdout_indexed = holdout_with_gap.set_index(
        ["region", "date"]).sort_index()
    clean_holdout = holdout_indexed.dropna(
        subset=["price_growth"] + growth_features)

    if len(clean_holdout) < 10:
        result = {
            "train_end": str(split_ts.date()),
            "holdout_start": str(split_ts.date()),
            "holdout_rmse": None,
            "zero_growth_rmse": None,
            "beats_zero_growth": False,
            "directional_1m": None,
            "directional_6m": None,
            "directional_12m": None,
            "error": "Insufficient holdout observations",
        }
        ensure_directory(VALIDATION_DIR)
        (VALIDATION_DIR / "subsample_ols_validation.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        return result

    X_holdout = sm.add_constant(
        clean_holdout[growth_features], has_constant="add")
    # linearmodels PanelOLS params are a Series; use .values for dotting
    coef_names = growth_model.params.index.tolist()
    X_holdout_aligned = X_holdout.reindex(columns=coef_names, fill_value=0.0)
    predicted = X_holdout_aligned.values @ growth_model.params.values
    actual = clean_holdout["price_growth"].values

    holdout_rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    zero_growth_rmse = float(np.sqrt(np.mean(actual ** 2)))
    beats_zero_growth = bool(holdout_rmse < zero_growth_rmse)

    # Directional accuracy at 1m/6m/12m
    def _directional_accuracy(actual_arr: np.ndarray, pred_arr: np.ndarray, horizon: int) -> float:
        """% of windows where sign(cumulative_actual) == sign(cumulative_predicted)."""
        n = len(actual_arr)
        correct = 0
        total = 0
        for i in range(n - horizon + 1):
            cum_actual = float(np.sum(actual_arr[i: i + horizon]))
            cum_pred = float(np.sum(pred_arr[i: i + horizon]))
            if cum_actual == 0 or cum_pred == 0:
                continue
            correct += int(np.sign(cum_actual) == np.sign(cum_pred))
            total += 1
        return float(correct / total) if total > 0 else float("nan")

    dir_1m = _directional_accuracy(actual, predicted, 1)
    dir_6m = _directional_accuracy(actual, predicted, 6)
    dir_12m = _directional_accuracy(actual, predicted, 12)

    result = {
        "train_end": str((split_ts - pd.DateOffset(months=1)).date()),
        "holdout_start": str(split_ts.date()),
        "holdout_n_obs": int(len(actual)),
        "holdout_rmse": round(holdout_rmse, 6),
        "zero_growth_rmse": round(zero_growth_rmse, 6),
        "beats_zero_growth": beats_zero_growth,
        "directional_1m": round(dir_1m, 4),
        "directional_6m": round(dir_6m, 4),
        "directional_12m": round(dir_12m, 4),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    ensure_directory(VALIDATION_DIR)
    (VALIDATION_DIR / "subsample_ols_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def _apply_iqr_prior(
    sigma: float,
    kappa: float,
    hist_iqr: float,
    target_iqr_ratio: float = 0.35,
    dt: float = 1.0 / 12,
    n_sim: int = 5000,
    horizon_months: int = 60,
    seed: int = 42,
    sigma_iqr_cap: float = 0.115,
) -> tuple[float, bool]:
    """Apply a minimum historical IQR prior to sigma.

    # IQR prior: sigma is raised iteratively until simulated 5-yr IQR is at
    # least 35% of historical IQR. This replaces the rejected hard sigma_floor_override.
    # Cap at 0.115 to prevent implausible volatility. Approved 2026-03-31 governance review.

    Given sigma and kappa, simulates a mean-zero OU process (dX = -kappa*X*dt +
    sigma*sqrt(dt)*N(0,1), X0=0) for horizon_months steps across n_sim paths.
    If sim_IQR / hist_IQR < target_iqr_ratio, sigma is increased by 10% and
    the simulation is re-run, repeating up to 20 iterations.  The adjustment is
    capped at sigma_iqr_cap (default 0.115) — reached via the IQR criterion,
    not by fiat (distinct from the rejected hard sigma_floor_override).

    Returns (adjusted_sigma, was_adjusted).
    """
    import warnings as _warnings

    if hist_iqr <= 0.0:
        return sigma, False

    rng = np.random.default_rng(seed)
    # Fix random draws so all iterations compare the same paths (only sigma varies)
    eps = rng.standard_normal((n_sim, horizon_months))

    sigma_adjusted = sigma
    was_adjusted = False

    for _ in range(20):
        kappa_m = kappa * dt
        sigma_m = sigma_adjusted * np.sqrt(dt)
        X = np.zeros(n_sim)
        for t in range(horizon_months):
            X = X * (1.0 - kappa_m) + sigma_m * eps[:, t]
        sim_iqr = float(np.percentile(X * 100.0, 75) -
                        np.percentile(X * 100.0, 25))

        if sim_iqr / hist_iqr >= target_iqr_ratio:
            break

        new_sigma = sigma_adjusted * 1.10
        if new_sigma >= sigma_iqr_cap:
            # Only raise to cap if it is actually higher than current sigma
            if sigma_iqr_cap > sigma_adjusted:
                sigma_adjusted = sigma_iqr_cap
                was_adjusted = True
            # Either way, ceiling reached — cannot improve further
            break
        sigma_adjusted = new_sigma
        was_adjusted = True

    if was_adjusted:
        _warnings.warn(
            f"IQR prior raised sigma {sigma:.5f} → {sigma_adjusted:.5f} "
            f"(hist_IQR={hist_iqr:.2f}pp, target_ratio={target_iqr_ratio:.2f}).",
            UserWarning,
            stacklevel=2,
        )

    return sigma_adjusted, was_adjusted


def build_stage4_parameters() -> pd.DataFrame:
    """Rebuild Stage 4 parameters using structural fair value and stable repricing features."""
    import warnings as _warnings

    config = load_config()
    build_canonical_panel()
    export_stage4_ready_panel()
    master = _load_master_dataset()

    # --- Estimate PTI/PTR blend weight (grid-search pooled SSE) ---
    _pti_w = estimate_pti_ptr_blend(master)
    1.0 - _pti_w

    # --- Estimate PTI/PTR blend weight (governance logging) ---
    _pti_weight_estimated: float | None = None
    _pti_weight_method = "expert_prior_0.60_0.40"
    try:
        _estimated_w = estimate_fair_value_blend_weights(master)
        if 0.30 <= _estimated_w <= 0.70:
            _pti_weight_estimated = round(_estimated_w, 4)
            _pti_weight_method = "data_driven"
            print(
                f"[build_stage4_parameters] PTI blend weight estimated from data: {_pti_weight_estimated:.4f}")
        else:
            _warnings.warn(
                f"[build_stage4_parameters] Estimated PTI blend weight {_estimated_w:.4f} "
                "is outside [0.30, 0.70] after clipping — falling back to expert prior 0.60/0.40.",
                UserWarning, stacklevel=2,
            )
    except Exception as _exc:
        _warnings.warn(
            f"[build_stage4_parameters] estimate_fair_value_blend_weights failed ({_exc}) — "
            "falling back to expert prior 0.60/0.40.",
            UserWarning, stacklevel=2,
        )

    # Persist estimated weight to parameters.json (governance field only; does not
    # change the OLS which already estimates data-driven coefficients for all regressors)
    try:
        from src.core.config import DEFAULT_CONFIG_PATH as _CONFIG_PATH
        _params_json = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        # Update inside the fair_value block; also remove any stale top-level keys
        _params_json.setdefault("fair_value", {})[
                                "fair_value_pti_weight_estimated"] = _pti_weight_estimated
        _params_json["fair_value"]["fair_value_pti_weight_method"] = _pti_weight_method
        _params_json.pop("fair_value_pti_weight_estimated", None)
        _params_json.pop("fair_value_pti_weight_method", None)
        _CONFIG_PATH.write_text(json.dumps(
            _params_json, indent=2), encoding="utf-8")
    except Exception as _exc:
        _warnings.warn(
            f"[build_stage4_parameters] Could not persist PTI weight to parameters.json: {_exc}", UserWarning, stacklevel=2)

    fair_value = fit_structural_fair_value(
        master,
        regressors=config.fair_value.regressors,
        anchor_months=config.fair_value.anchor_months,
    )
    fair_panel = fair_value.fitted_panel.copy()
    cpi_base = float(
        fair_panel.loc[fair_panel["date"] == pd.Timestamp("2015-01-01"), "cpi"].iloc[0])
    fair_panel["fair_value_nominal"] = fair_panel["fair_value_real"] * \
        (fair_panel["cpi"] / cpi_base)

    # ------------------------------------------------------------------
    # 90% prediction-interval bands on fair_value_gap_log
    # Approximation: SE_pred ≈ σ_resid (hat-matrix leverage term omitted;
    # valid for n >> p, which holds here with ~200+ anchor observations).
    # Band in log-gap space: gap_lower = gap - z*σ, gap_upper = gap + z*σ
    # where z=1.645 is the 90% two-sided normal critical value.
    # Semantics (consistent with task spec):
    #   fair_value_gap_lower_90 — most favourable (P* high → gap small)
    #   fair_value_gap_upper_90 — least favourable (P* low  → gap large)
    # ------------------------------------------------------------------
    _resid_std = float(fair_value.diagnostics["resid_std"].iloc[0])
    _Z_90 = 1.645
    fair_panel["fair_value_gap_lower_90"] = fair_panel["fair_value_gap_log"] - \
        _Z_90 * _resid_std
    fair_panel["fair_value_gap_upper_90"] = fair_panel["fair_value_gap_log"] + \
        _Z_90 * _resid_std

    _rate_channel_diagnostics(master)
    coefficient_rows, diagnostics, pooled_coeffs = _estimate_growth_coefficients(
        fair_panel)
    stability = run_stability_diagnostics(fair_panel, diagnostics)
    kappa_rows = _estimate_kappa_by_region(fair_panel)

    valuation_rows = []
    for region, group in fair_panel.groupby("region"):
        latest = group.sort_values("date").iloc[-1]
        valuation_rows.append(
            {
                "region": region,
                "mu_equilibrium": float(latest["fair_value_nominal"]),
                "valuation_gap_current": float(latest["fair_value_gap_log"]),
                "valuation_gap_pct": float((np.exp(latest["fair_value_gap_log"]) - 1.0) * 100.0),
                "latest_price": float(latest["nominal_house_price"]),
                "current_rate": float(latest["mortgage_rate"]),
                "latest_gross_yield_pct": float(latest["gross_yield_pct"]),
                "latest_transaction_growth": float(latest.get("transaction_growth", 0.0)),
                "latest_unemployment_rate": float(latest.get("unemployment_rate", np.nan)),
                "latest_approvals_growth": float(latest.get("approvals_growth", np.nan)),
                "latest_financial_stress": float(latest.get("financial_stress_lag3", 0.0)),
                "latest_financial_stress_excess": float(latest.get(STRESS_FEATURE, 0.0)),
                "earnings_obs_date": str(pd.to_datetime(latest["earnings_obs_date"]).date()),
                "earnings_staleness_months": int(latest.get("earnings_staleness_months", 0)),
            }
        )

    params = coefficient_rows.merge(pd.DataFrame(
        valuation_rows), on="region", how="left").merge(kappa_rows, on="region", how="left")

    sigma_cal = _calibrate_sigma_by_region(fair_panel, params, config)
    params = params.merge(
        sigma_cal[["region", "sigma_calibrated"]], on="region", how="left")
    params["sigma_heuristic"] = params["sigma"]
    params["sigma"] = params["sigma_calibrated"].combine_first(params["sigma"])
    params = params.drop(columns=["sigma_calibrated"])

    # --- Apply minimum historical IQR prior to all regions ---
    # Raises sigma iteratively until sim 5-yr IQR >= 35% of historical IQR.
    _iqr_prior_rows: list[dict] = []
    _sigma_updates: dict[str, float] = {}
    # Load per-region sigma floor overrides once before loop
    try:
        from src.core.config import DEFAULT_CONFIG_PATH as _CFG_PATH_SIGMA
        _sigma_floor_overrides: dict = json.loads(_CFG_PATH_SIGMA.read_text(
            encoding="utf-8")).get("sigma_floor_overrides", {})
    except Exception:
        _sigma_floor_overrides = {}
    for _, pr in params.iterrows():
        region = str(pr["region"])
        sigma_base = float(pr["sigma"])
        kappa_val = float(pr["kappa"])

        grp = (
            fair_panel[fair_panel["region"] == region]
            .sort_values("date")
            .reset_index(drop=True)
        )
        if "nominal_house_price" in grp.columns and len(grp) > 60:
            prices = grp["nominal_house_price"].values
            hist_returns = [
                np.log(prices[i + 60] / prices[i]) * 100.0
                for i in range(len(grp) - 60)
            ]
            hist_iqr = float(np.percentile(hist_returns, 75) -
                             np.percentile(hist_returns, 25))
        else:
            hist_iqr = 0.0

        sigma_new, adjusted = _apply_iqr_prior(sigma_base, kappa_val, hist_iqr)

        # Apply per-region sigma floor overrides (e.g. South West IQR governance)
        sigma_floors = _sigma_floor_overrides
        if region in sigma_floors:
            floor_val = sigma_floors[region]
            if sigma_new < floor_val:
                print(
                    f"[sigma floor] {region}: sigma {sigma_new:.4f} -> {floor_val:.4f} (floor override)")
                sigma_new = floor_val

        _iqr_prior_rows.append({
            "region": region,
            "sigma_before_iqr_prior": round(sigma_base, 6),
            "sigma_after_iqr_prior": round(sigma_new, 6),
            "hist_iqr_pp": round(hist_iqr, 4),
            "iqr_prior_adjusted": adjusted,
        })
        _sigma_updates[region] = sigma_new

    params["sigma"] = params["region"].map(
        _sigma_updates).combine_first(params["sigma"])
    _iqr_prior_df = pd.DataFrame(_iqr_prior_rows)
    print("\n--- IQR Prior: sigma before/after ---")
    print(_iqr_prior_df.to_string(index=False))

    params["post2015_preferred"] = stability["post2015_preferred"]
    params["last_updated"] = datetime.now(
        timezone.utc).isoformat(timespec="seconds")
    params["final_spec_note"] = (
        "Canonical rebuild on the historical research panel. "
        "Fair value is estimated from a semi-structural real-price regression on as-of earnings, rent, and mortgage rates. "
        "Monthly rate signal uses the lagged mortgage-rate level; financial stress enters only as an episodic excess-above-threshold term; "
        "annual earnings enter as lagged 12m growth to avoid fake monthly precision. "
        "Directional coefficient restrictions, volatility clipping, and kappa fallback are treated as explicit modelling guardrails rather than hidden identification claims."
    )
    params["rho_source"] = "region_hc3_shrunk"
    params["rate_channel_mode"] = "rate_level_plus_aux_stress"
    params["earnings_growth_12m_lag1_is_interpolated"] = params["earnings_staleness_months"].gt(
        0)
    params["net_additions_monthly_is_interpolated"] = pd.to_datetime(
        params["last_observation_date"]).dt.month.ne(4)

    ensure_directory(STAGE4_OUTPUT_DIR)
    params.to_csv(STAGE4_OUTPUT_PATH, index=False)
    fair_panel[
        [
            "region",
            "date",
            "nominal_house_price",
            "real_house_price",
            "fair_value_real",
            "fair_value_nominal",
            "fair_value_gap_log",
            "fair_value_gap_lower_90",
            "fair_value_gap_upper_90",
            RATE_FEATURE,
            STRESS_FEATURE,
            "transaction_growth",
            EARNINGS_FEATURE,
            "price_growth",
        ]
    ].to_csv(STAGE4_FAIR_VALUE_PANEL_PATH, index=False)
    diagnostics.to_csv(STAGE4_DIAGNOSTICS_PATH, index=False)
    pd.concat(
        [
            fair_value.coefficients.assign(model_block="fair_value"),
            pooled_coeffs.assign(model_block="growth"),
        ],
        ignore_index=True,
    ).to_csv(STAGE4_POOLED_COEFFS_PATH, index=False)
    params[
        [
            "region",
            "sigma_raw_annual",
            "sigma_winsor_annual",
            "sigma_heuristic",
            "sigma",
            "sigma_winsor_lo",
            "sigma_winsor_hi",
        ]
    ].rename(columns={"sigma_heuristic": "sigma_heuristic_annual", "sigma": "sigma_final_annual"}).to_csv(SIGMA_MAPPING_DIAGNOSTICS_PATH, index=False)
    return params


def build_stage5_summary(stage4_params: pd.DataFrame) -> pd.DataFrame:
    """Build canonical Stage 5 scenario summaries from the structural fair-value rebuild."""
    config = load_config()
    master = _load_master_dataset()
    fair_value = fit_structural_fair_value(
        master,
        regressors=config.fair_value.regressors,
        anchor_months=config.fair_value.anchor_months,
    )
    fair_panel = fair_value.fitted_panel.copy()
    cpi_base = float(
        fair_panel.loc[fair_panel["date"] == pd.Timestamp("2015-01-01"), "cpi"].iloc[0])
    fair_panel["fair_value_nominal"] = fair_panel["fair_value_real"] * \
        (fair_panel["cpi"] / cpi_base)
    _, baseline_fit = _fit_baseline_return_model(fair_panel)

    latest = (
        fair_panel.sort_values(["region", "date"])
        .groupby("region", as_index=False)
        .tail(1)
        .sort_values("region")
        .reset_index(drop=True)
    )
    params = stage4_params.set_index("region")

    rows: list[dict] = []
    for _, latest_row in latest.iterrows():
        region = str(latest_row["region"])
        start_price = float(latest_row["nominal_house_price"])
        current_rate = float(latest_row["mortgage_rate"])
        current_stress = float(latest_row.get("financial_stress_lag3", 0.0))
        baseline_log_return = _predict_baseline_return(
            baseline_fit, latest_row)
        base_drift = float(params.loc[region, "gamma_annual_pp"]) / 100.0
        base_sigma = float(params.loc[region, "sigma"])
        kappa = float(params.loc[region, "kappa"])
        beta_rate = float(params.loc[region, "beta_rate"])
        beta_financial_stress = float(
            params.loc[region, "beta_financial_stress"])
        beta_transaction = float(params.loc[region, "beta_transaction_growth"])
        current_fair_value = float(latest_row["fair_value_nominal"])

        for scenario_name in config.ui.scenario_labels:
            scenario = config.scenarios[scenario_name]
            rate_shift_pp = scenario.rate_shift_bps / 100.0
            annual_overlay = (
                3.0 * beta_rate * rate_shift_pp +
                12.0 * beta_financial_stress * scenario.financial_stress_shift +
                12.0 * beta_transaction * scenario.transaction_shift +
                scenario.drift_shift_pct / 100.0
            )
            target_log_return = baseline_log_return + 5.0 * annual_overlay
            target_terminal_price = start_price * np.exp(target_log_return)
            target_terminal_price = (1.0 - scenario.valuation_gap_closure) * \
                target_terminal_price + scenario.valuation_gap_closure * current_fair_value
            target_terminal_price *= 1.0 + scenario.income_growth_shift_pct / \
                150.0 - scenario.supply_shift_pct / 400.0
            target_terminal_price = max(
                start_price * scenario.min_target_price_ratio, target_terminal_price)

            mean_path = np.empty(61)
            mean_path[0] = start_price
            for t in range(1, 61):
                mean_path[t] = mean_path[t - 1] + kappa * \
                    (target_terminal_price - mean_path[t - 1]) * (1 / 12)
            paths = run_monte_carlo(
                start_price=start_price,
                n_months=60,
                n_paths=config.model.n_simulations,
                params={
                    "kappa": kappa,
                    "sigma": min(base_sigma * scenario.volatility_multiplier, base_sigma * config.simulation.stress_sigma_multiplier_cap),
                    "drift_annual": base_drift + annual_overlay,
                    "long_run_mean": mean_path,
                },
                seed=config.controls.random_seed,
            )
            terminal_returns = (
                paths.iloc[-1].to_numpy() / start_price - 1.0) * 100.0
            rows.append(
                {
                    "scenario": scenario_name,
                    "region": region,
                    "median_5yr_growth": round(float(np.median(terminal_returns)), 2),
                    "p10_5yr_growth": round(float(np.percentile(terminal_returns, 10)), 2),
                    "p25_5yr_growth": round(float(np.percentile(terminal_returns, 25)), 2),
                    "p75_5yr_growth": round(float(np.percentile(terminal_returns, 75)), 2),
                    "p90_5yr_growth": round(float(np.percentile(terminal_returns, 90)), 2),
                    "prob_terminal_loss_10pct": round(float(np.mean(terminal_returns <= -10.0)), 4),
                    "prob_terminal_loss_5pct": round(float(np.mean(terminal_returns <= -5.0)), 4),
                    "p_normal_T60": scenario.regime_prob_normal,
                    "p_stress_T60": scenario.regime_prob_stress,
                    "p_boom_T60": scenario.regime_prob_boom,
                    "mean_fs_T60": round(current_stress + scenario.financial_stress_shift, 4),
                    "mean_rate_T60": round(current_rate + rate_shift_pp, 3),
                    "mean_pstar_T60": round(target_terminal_price, 2),
                    "p_star_base": round(current_fair_value, 2),
                    "scenario_spread_pp": np.nan,
                    "alpha_used": 0.0,
                    "r_bar": round(current_rate, 3),
                    "baseline_log_return_model": round(baseline_log_return, 6),
                }
            )

    summary = pd.DataFrame(rows)
    spread = summary.groupby("region", as_index=False)[
                             "median_5yr_growth"].agg(["max", "min"]).reset_index()
    spread["scenario_spread_pp"] = spread["max"] - spread["min"]
    summary = summary.drop(columns=["scenario_spread_pp"]).merge(
        spread[["region", "scenario_spread_pp"]], on="region", how="left")
    summary = summary.sort_values(
        ["region", "scenario"]).reset_index(drop=True)

    ensure_directory(STAGE5_OUTPUT_DIR)
    summary.to_csv(STAGE5_OUTPUT_PATH, index=False)
    _simulation_plausibility_from_summary(master, summary)
    return summary


def build_stage6_handoff(stage4_params: pd.DataFrame, stage5_summary: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical Stage 6 handoff from the rebuilt Stage 4/5 artifacts."""
    latest = (
        _load_master_dataset()
        .sort_values(["region", "date"])
        .groupby("region", as_index=False)
        .tail(1)
        .sort_values("region")
        .reset_index(drop=True)
    )
    params = stage4_params.set_index("region")

    # WEIGHT: see config/parameters.json → scoring.scenario_consumer_weights
    # JUSTIFICATION: expert-prior — scenario probability weights assigned judgementally;
    #   not yet calibrated to historical scenario frequencies. Downside-skewed:
    #   adverse scenarios (Rate_Shock + Stagflation) weighted 0.45 combined.
    # LAST_REVIEWED: 2026-03-21
    consumer_weights = load_config().scoring.scenario_consumer_weights
    # WEIGHT: see config/parameters.json → scoring.scenario_reit_weights
    # JUSTIFICATION: expert-prior — equal weight (1/N) across all 5 scenarios.
    #   Previously computed dynamically at runtime; now explicit in config so that
    #   adding or renaming a scenario requires a deliberate config edit rather than
    #   a silent automatic rebalancing.
    # LAST_REVIEWED: 2026-03-21
    reit_weights = load_config().scoring.scenario_reit_weights

    rows: list[dict] = []
    for _, latest_row in latest.iterrows():
        region = str(latest_row["region"])
        region_sim = stage5_summary[stage5_summary["region"] == region].set_index(
            "scenario")

        weighted_consumer_return = sum(consumer_weights[name] * float(
            region_sim.loc[name, "median_5yr_growth"]) for name in consumer_weights)
        weighted_reit_return = sum(
            reit_weights[name] * float(region_sim.loc[name, "median_5yr_growth"]) for name in reit_weights)
        tail_risk = float(region_sim["prob_terminal_loss_10pct"].mean())
        scenario_spread = float(
            region_sim["median_5yr_growth"].max() - region_sim["median_5yr_growth"].min())
        p_star_now = float(params.loc[region, "mu_equilibrium"])
        p_last = float(latest_row["nominal_house_price"])
        pct_above_pstar = (p_last / p_star_now - 1.0) * 100.0

        c1_mispricing = _clip(50.0 - 0.95 * pct_above_pstar, 0.0, 100.0)
        c2_consumer = _clip(50.0 + 1.8 * weighted_consumer_return, 0.0, 100.0)
        c2_reit = _clip(50.0 + 1.8 * weighted_reit_return, 0.0, 100.0)
        yield_score = _clip(
            50.0 + (float(latest_row["gross_yield_pct"]) - 5.5) * 16.0, 0.0, 100.0)
        tail_score = _clip(100.0 * (1.0 - tail_risk), 0.0, 100.0)
        c3_regime = _clip(
            60.0 +
            65.0 * float(latest_row.get("transaction_growth", 0.0)) -
            1.0 * scenario_spread -
            5.0 * max(float(latest_row.get("unemployment_rate", 0.0)) - 5.0, 0.0),
            0.0,
            100.0,
        )

        consumer_score = _clip(
            0.40 * c1_mispricing + 0.35 * c2_consumer + 0.25 * tail_score, 0.0, 100.0)
        reit_score = _clip(0.20 * c1_mispricing + 0.35 * c2_reit +
                           0.25 * yield_score + 0.20 * tail_score, 0.0, 100.0)

        rows.append(
            {
                "region": region,
                "C1_mispricing": round(c1_mispricing, 1),
                "C2_consumer": round(c2_consumer, 1),
                "C2_reit": round(c2_reit, 1),
                "C3_regime": round(c3_regime, 1),
                "tail_penalty": round(100.0 - tail_score, 1),
                "consumer_score": round(consumer_score, 1),
                "consumer_band": label_score(consumer_score),
                "reit_score": round(reit_score, 1),
                "reit_band": label_score(reit_score),
                "score_label_mode": "descriptive_signal",
                "scenario_spread": round(scenario_spread, 1),
                "pct_above_pstar": round(pct_above_pstar, 1),
                "wtd_return_consumer": round(weighted_consumer_return, 2),
                "p_terminal_loss_10_avg": round(tail_risk, 4),
                "median_Baseline": round(float(region_sim.loc["Baseline", "median_5yr_growth"]), 2),
                "median_Rate_Shock": round(float(region_sim.loc["Rate_Shock", "median_5yr_growth"]), 2),
                "median_Recovery_Boom": round(float(region_sim.loc["Recovery_Boom", "median_5yr_growth"]), 2),
                "median_Soft_Landing": round(float(region_sim.loc["Soft_Landing", "median_5yr_growth"]), 2),
                "median_Stagflation": round(float(region_sim.loc["Stagflation", "median_5yr_growth"]), 2),
                "P_star_base": round(p_star_now, 2),
                "P_star_now": round(p_star_now, 2),
                "P_last_obs": round(p_last, 2),
                "kappa": round(float(params.loc[region, "kappa"]), 4),
                "gamma_annual": round(float(params.loc[region, "gamma_annual_pp"]) / 100.0, 4),
                "scoring_date": str(pd.to_datetime(latest_row["date"]).date()),
                "alpha_used": 0.0,
                "r_bar": round(float(latest_row["mortgage_rate"]), 3),
                "r_current": round(float(latest_row["mortgage_rate"]), 3),
                "regime_prob_normal": 0.35,
                "regime_prob_stress": 0.40,
                "regime_prob_boom": 0.25,
                "rate_channel_mode": "rate_level_plus_aux_stress",
                "valuation_gap_current": round(float(params.loc[region, "valuation_gap_current"]), 4),
                "beta_rate": round(float(params.loc[region, "beta_rate"]), 6),
                "beta_financial_stress": round(float(params.loc[region, "beta_financial_stress"]), 6),
                "earnings_staleness_months": int(params.loc[region, "earnings_staleness_months"]),
            }
        )

    handoff = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    ensure_directory(STAGE6_OUTPUT_DIR)
    handoff.to_csv(STAGE6_OUTPUT_PATH, index=False)
    return handoff


def run_canonical_rebuild() -> CanonicalBuildResult:
    """Run the canonical Stage 4-6 rebuild and write app-ready artifacts."""
    stage4 = build_stage4_parameters()
    stage5 = build_stage5_summary(stage4)
    build_stage6_handoff(stage4, stage5)

    # Pull publication-lag audit fields from the panel metadata written by
    # build_canonical_panel (Stage 2/3).  Absent if panel was not rebuilt in
    # this run — fields are null rather than raising so Stage 4-6 can still run.
    _max_lag: int | None = None
    _effective_as_of: str | None = None
    try:
        _panel_meta = json.loads(
            CANONICAL_PANEL_METADATA_PATH.read_text(encoding="utf-8"))
        _max_lag = _panel_meta.get("max_publication_lag_months")
        _effective_as_of = _panel_meta.get("effective_data_as_of")
    except Exception:
        pass

    metadata = CanonicalBuildResult(
        stage4_path=str(STAGE4_OUTPUT_PATH),
        stage5_path=str(STAGE5_OUTPUT_PATH),
        stage6_path=str(STAGE6_OUTPUT_PATH),
        generated_at_utc=datetime.now(
            timezone.utc).isoformat(timespec="seconds"),
        regions=int(stage4["region"].nunique()),
        scenarios=int(stage5["scenario"].nunique()),
        max_publication_lag_months=_max_lag,
        effective_data_as_of=_effective_as_of,
    )
    PIPELINE_METADATA_PATH.write_text(json.dumps(
        asdict(metadata), indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    metadata = run_canonical_rebuild()
    print(json.dumps(asdict(metadata), indent=2))


if __name__ == "__main__":
    main()
