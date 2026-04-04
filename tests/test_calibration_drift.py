"""
Calibration drift acceptance tests for the canonical Stage 4 pipeline.

Tests compare current SDE/OLS parameters against stored baselines.
On first run, baselines are written to data/outputs/stage4/ and the
comparison step is skipped (pytest.skip). On subsequent runs, each test
asserts that no region has drifted beyond its specified threshold.

Baseline files:
  data/outputs/stage4/sigma_baseline.csv
  data/outputs/stage4/gamma_baseline.csv
  data/outputs/stage4/kappa_baseline.csv
  data/outputs/stage4/rsquared_baseline.csv

Sources:
  canonical_stage4_diagnostics.csv  — sigma, r_squared, raw/final coefficients
  sde_parameters_bankgrade.csv      — gamma_annual_pp, kappa

Out-of-cycle triggers (from SIGN_OFF_CHECKLIST.md §4.2):
  - R-squared drops > 15pp for any region         → test_r_squared_drift
  - Kappa moves > 50% from prior calibration       → test_kappa_drift
  - Guardrail override fraction > 40%              → test_guardrail_override_frequency
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT           = Path(__file__).resolve().parents[1]
STAGE4_DIR     = ROOT / "data" / "outputs" / "stage4_final"
DIAGNOSTICS    = STAGE4_DIR / "canonical_stage4_diagnostics.csv"
SDE_PARAMS     = STAGE4_DIR / "sde_parameters_bankgrade.csv"
BASELINE_DIR   = ROOT / "data" / "outputs" / "stage4"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_diagnostics() -> pd.DataFrame:
    assert DIAGNOSTICS.exists(), (
        f"canonical_stage4_diagnostics.csv not found at {DIAGNOSTICS}. "
        "Run python -m src.model.canonical_rebuild to generate it."
    )
    return pd.read_csv(DIAGNOSTICS)


def _load_sde_params() -> pd.DataFrame:
    assert SDE_PARAMS.exists(), (
        f"sde_parameters_bankgrade.csv not found at {SDE_PARAMS}. "
        "Run python -m src.model.canonical_rebuild to generate it."
    )
    return pd.read_csv(SDE_PARAMS)


def _write_baseline(path: Path, series: pd.Series) -> None:
    """Write a baseline Series (indexed by region) to CSV."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    series.to_frame(name="baseline").to_csv(path)


# ---------------------------------------------------------------------------
# 1. Sigma drift
# ---------------------------------------------------------------------------

def test_sigma_drift():
    """
    Assert that no region's blended annual sigma has drifted more than 30%
    from the stored baseline.

    Reads sigma_final_annual from canonical_stage4_diagnostics.csv (the
    winsorised-blended annual volatility, after floor/cap guardrails).

    On first run, writes the baseline and skips comparison.
    Threshold: ratio current/baseline must lie in [0.70, 1.30] for every region.

    Corresponds to the sigma calibration out-of-cycle trigger in
    SIGN_OFF_CHECKLIST.md §4.3 ("realised σ diverges > 50% from deployed σ").
    The 30% drift guard is tighter than the annual review threshold.
    """
    diag    = _load_diagnostics()
    current = diag.set_index("region")["sigma_final_annual"]

    baseline_path = BASELINE_DIR / "sigma_baseline.csv"
    if not baseline_path.exists():
        _write_baseline(baseline_path, current)
        print("Baseline created — rerun to compare")
        pytest.skip("Baseline created — rerun to compare")

    baseline = pd.read_csv(baseline_path, index_col=0)["baseline"]

    failures = []
    for region in current.index:
        if region not in baseline.index:
            continue
        b = float(baseline[region])
        c = float(current[region])
        if b == 0:
            continue
        ratio = c / b
        if abs(ratio - 1.0) > 0.30:
            failures.append(
                dict(region=region, baseline=b, current=c, ratio=ratio)
            )

    if failures:
        print(f"\nSIGMA DRIFT — {len(failures)} region(s) exceed 30% threshold:")
        for f in failures:
            print(
                f"  {f['region']}: baseline={f['baseline']:.6f}  "
                f"current={f['current']:.6f}  ratio={f['ratio']:.3f}"
            )

    assert not failures, (
        f"sigma_final_annual has drifted > 30% from baseline for "
        f"{len(failures)} region(s). Review sigma calibration or update "
        f"baseline after deliberate recalibration."
    )


# ---------------------------------------------------------------------------
# 2. Gamma drift
# ---------------------------------------------------------------------------

def test_gamma_drift():
    """
    Assert that no region's annual drift (gamma) has moved more than 0.8
    percentage points (0.008 in decimal) from the stored baseline.

    Reads gamma_annual_pp from sde_parameters_bankgrade.csv (the blended
    40% current-real / 60% long-run-nominal drift, clipped to [-1%, +2.5%]).
    Values are converted to decimal for comparison against the 0.008 threshold.

    On first run, writes the baseline (stored in decimal) and skips comparison.

    Corresponds to the drift bound guardrail in SIGN_OFF_CHECKLIST.md §2.3.
    A shift of 0.8pp in annual drift materially changes long-run simulation
    paths and warrants review before the updated calibration is accepted.
    """
    params  = _load_sde_params()
    # gamma_annual_pp is in percentage points; convert to decimal for comparison
    current = (params.set_index("region")["gamma_annual_pp"] / 100.0)

    baseline_path = BASELINE_DIR / "gamma_baseline.csv"
    if not baseline_path.exists():
        _write_baseline(baseline_path, current)
        print("Baseline created — rerun to compare")
        pytest.skip("Baseline created — rerun to compare")

    baseline = pd.read_csv(baseline_path, index_col=0)["baseline"]

    THRESHOLD = 0.008  # 0.8 percentage points in decimal

    failures = []
    for region in current.index:
        if region not in baseline.index:
            continue
        b = float(baseline[region])
        c = float(current[region])
        if abs(c - b) >= THRESHOLD:
            failures.append(
                dict(region=region, baseline=b, current=c, abs_diff=abs(c - b))
            )

    if failures:
        print(f"\nGAMMA DRIFT — {len(failures)} region(s) exceed 0.8pp threshold:")
        for f in failures:
            print(
                f"  {f['region']}: baseline={f['baseline']:.5f}  "
                f"current={f['current']:.5f}  "
                f"abs_diff={f['abs_diff']:.5f} ({f['abs_diff']*100:.2f}pp)"
            )

    assert not failures, (
        f"gamma (annual drift) has moved > 0.8pp from baseline for "
        f"{len(failures)} region(s). Review drift calibration or update "
        f"baseline after deliberate recalibration."
    )


# ---------------------------------------------------------------------------
# 3. Kappa drift
# ---------------------------------------------------------------------------

def test_kappa_drift():
    """
    Assert that no region's mean-reversion speed (kappa) has changed by more
    than 50% from the stored baseline.

    Reads kappa from sde_parameters_bankgrade.csv (the region-specific
    mean-reversion speed, derived from an AR(1) on the log fair-value gap,
    with shrinkage toward the cross-regional median and fallback to
    kappa_fallback_phi where the AR fit is unstable).

    On first run, writes the baseline and skips comparison.
    Threshold: ratio current/baseline must lie in [0.50, 1.50] for every region.

    Matches the out-of-cycle trigger in SIGN_OFF_CHECKLIST.md §4.2:
    "Regional kappa estimate moves by more than 50% from one calibration
    to the next."
    """
    params  = _load_sde_params()
    current = params.set_index("region")["kappa"]

    baseline_path = BASELINE_DIR / "kappa_baseline.csv"
    if not baseline_path.exists():
        _write_baseline(baseline_path, current)
        print("Baseline created — rerun to compare")
        pytest.skip("Baseline created — rerun to compare")

    baseline = pd.read_csv(baseline_path, index_col=0)["baseline"]

    failures = []
    for region in current.index:
        if region not in baseline.index:
            continue
        b = float(baseline[region])
        c = float(current[region])
        if b == 0:
            continue
        ratio = c / b
        if abs(ratio - 1.0) > 0.50:
            failures.append(
                dict(region=region, baseline=b, current=c, ratio=ratio)
            )

    if failures:
        print(f"\nKAPPA DRIFT — {len(failures)} region(s) exceed 50% threshold:")
        for f in failures:
            print(
                f"  {f['region']}: baseline={f['baseline']:.6f}  "
                f"current={f['current']:.6f}  ratio={f['ratio']:.3f}"
            )

    assert not failures, (
        f"kappa (mean-reversion speed) has drifted > 50% from baseline for "
        f"{len(failures)} region(s). This matches the out-of-cycle trigger "
        f"in SIGN_OFF_CHECKLIST.md §4.2. Investigate before accepting the "
        f"updated calibration."
    )


# ---------------------------------------------------------------------------
# 4. R-squared drift
# ---------------------------------------------------------------------------

def test_r_squared_drift():
    """
    Assert that no region's within-R-squared has dropped more than 15
    percentage points from the stored baseline.

    Reads r_squared from canonical_stage4_diagnostics.csv (the within-R²
    from the regional OLS growth equation).

    On first run, writes the baseline and skips comparison.
    Threshold: current R² must not fall more than 0.15 below the baseline.

    Matches the out-of-cycle trigger in SIGN_OFF_CHECKLIST.md §4.2:
    "Adjusted R² within drops more than 15 percentage points for three or
    more regions simultaneously."
    This test applies the threshold per-region (tighter than the 3-region
    simultaneous rule) to provide early warning of individual region degradation.
    """
    diag    = _load_diagnostics()
    current = diag.set_index("region")["r_squared"]

    baseline_path = BASELINE_DIR / "rsquared_baseline.csv"
    if not baseline_path.exists():
        _write_baseline(baseline_path, current)
        print("Baseline created — rerun to compare")
        pytest.skip("Baseline created — rerun to compare")

    baseline = pd.read_csv(baseline_path, index_col=0)["baseline"]

    THRESHOLD = 0.15  # 15 percentage points

    failures = []
    for region in current.index:
        if region not in baseline.index:
            continue
        b = float(baseline[region])
        c = float(current[region])
        drop = b - c  # positive means R² fell
        if drop > THRESHOLD:
            failures.append(
                dict(region=region, baseline=b, current=c, drop=drop)
            )

    if failures:
        print(f"\nR² DRIFT — {len(failures)} region(s) dropped > 15pp from baseline:")
        for f in failures:
            print(
                f"  {f['region']}: baseline={f['baseline']:.4f}  "
                f"current={f['current']:.4f}  drop={f['drop']:.4f} ({f['drop']*100:.1f}pp)"
            )

    assert not failures, (
        f"within-R² has dropped > 15pp from baseline for {len(failures)} "
        f"region(s). This matches the out-of-cycle trigger in "
        f"SIGN_OFF_CHECKLIST.md §4.2. Investigate model fit deterioration "
        f"before accepting the updated calibration."
    )


# ---------------------------------------------------------------------------
# 5. Guardrail override frequency
# ---------------------------------------------------------------------------

# Coefficient raw/final pairs to evaluate for guardrail overrides.
# Each tuple: (raw_column, final_column, display_name)
_COEF_PAIRS = [
    ("beta_rate_raw",              "beta_rate_final",              "beta_rate"),
    ("beta_financial_stress_raw",  "beta_financial_stress_final",  "beta_financial_stress"),
    ("beta_transaction_raw",       "beta_transaction_final",       "beta_transaction"),
    ("beta_supply_raw",            "beta_supply_final",            "beta_supply"),
    ("beta_affordability_raw",     "beta_affordability_final",     "beta_affordability"),
    ("rho_raw",                    "rho_final",                    "rho"),
]

# A coefficient is counted as guardrail-overridden when the final value
# differs from the raw OLS estimate by more than 10% in relative terms.
# This threshold excludes routine shrinkage adjustments (typically < 3%)
# while capturing material guardrail interventions (sign reversals,
# bound violations) which typically produce > 50% changes.
# The denominator uses max(|raw|, 1e-8) to handle near-zero raw estimates
# (e.g. financial_stress in Scotland/NI during the ZLB era).
_OVERRIDE_RTOL = 0.10


def test_guardrail_override_frequency():
    """
    Assert that guardrails are not overriding an excessive fraction of
    coefficient estimates across all regions.

    Reads raw and final values for six coefficient types from
    canonical_stage4_diagnostics.csv. A coefficient-region pair is counted
    as overridden when the final value differs from the raw estimate by more
    than 10% in relative terms (see _OVERRIDE_RTOL above).

    Total observations: 6 coefficient types × 12 regions = 72 pairs.
    Threshold: fraction overridden < 0.40 (40%).

    A high override fraction indicates that the guardrail system is
    substituting for structural identification rather than correcting
    occasional implausible estimates — an MRM concern documented in
    SIGN_OFF_CHECKLIST.md §2 (Guardrails Register).

    On failure, prints every overridden coefficient-region pair with the
    raw value, final value, and percentage change.
    """
    diag = _load_diagnostics()

    # Verify all required columns are present
    required = {col for pair in _COEF_PAIRS for col in pair[:2]}
    missing  = required - set(diag.columns)
    assert not missing, (
        f"canonical_stage4_diagnostics.csv is missing columns: {missing}"
    )

    total            = 0
    overridden_count = 0
    overridden_pairs = []

    for raw_col, final_col, coef_name in _COEF_PAIRS:
        for _, row in diag.iterrows():
            region    = row["region"]
            raw_val   = float(row[raw_col])
            final_val = float(row[final_col])
            total    += 1

            # Relative change, guarded against near-zero denominators
            raw_abs  = max(abs(raw_val), 1e-8)
            rel_diff = abs(final_val - raw_val) / raw_abs

            if rel_diff > _OVERRIDE_RTOL:
                overridden_count += 1
                overridden_pairs.append(
                    dict(
                        region      = region,
                        coefficient = coef_name,
                        raw         = raw_val,
                        final       = final_val,
                        rel_diff_pct= rel_diff * 100.0,
                    )
                )

    fraction = overridden_count / total if total > 0 else 0.0

    if overridden_pairs:
        print(
            f"\nGUARDRAIL OVERRIDES — "
            f"{overridden_count}/{total} pairs = {fraction:.1%}:"
        )
        for p in sorted(overridden_pairs, key=lambda x: -x["rel_diff_pct"]):
            print(
                f"  {p['region']} | {p['coefficient']}: "
                f"raw={p['raw']:.6g}  final={p['final']:.6g}  "
                f"({p['rel_diff_pct']:.1f}% change)"
            )

    assert fraction < 0.40, (
        f"Guardrail override fraction {fraction:.1%} exceeds 40% threshold "
        f"({overridden_count}/{total} coefficient-region pairs overridden). "
        f"Review canonical_stage4_diagnostics.csv guardrail columns. "
        f"A high override rate suggests the guardrail system is compensating "
        f"for structural misspecification rather than correcting outliers."
    )
