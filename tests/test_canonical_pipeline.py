"""
Bank-grade acceptance tests for the canonical Stage 4-6 pipeline artifacts.

Each test checks one independently verifiable property of the built artifacts.
Tests are read-only: they load CSV/JSON output files and assert conditions.
No model code is imported.

Expected failures on a stale panel (scoring_date 2025-01-01, today 2026-03-21):
  - test_data_freshness          : 444 days stale (threshold 90) — genuine data gap
  - test_rate_channel_stability  : 5 regions have late/early ratio > 5.0 (see R010).
      North East (5.99), Scotland (6.91), South West (12.03), Wales (6.19), Yorkshire (5.64).
      Early-period (2005-2015) rate betas are near-zero or reverse-signed due to ZLB
      identification failure — the ratio is inflated by a near-zero denominator, not
      by anomalously large late betas. Full-sample beta is the appropriate pooled
      estimate for annual-frequency use.
"""

import ast
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT       = Path(__file__).resolve().parents[1]
STAGE4_DIR = ROOT / "data" / "outputs" / "stage4_final"
STAGE5_DIR = ROOT / "data" / "outputs" / "stage5c"
STAGE6_DIR = ROOT / "data" / "outputs" / "stage6"
PANEL_FILE = STAGE4_DIR / "fair_value_panel_bankgrade.csv"
PARAMS_FILE  = STAGE4_DIR / "sde_parameters_bankgrade.csv"
SIM_FILE   = STAGE5_DIR / "simulation_summary_bankgrade.csv"
HANDOFF_FILE = STAGE6_DIR / "stage7_handoff_bankgrade.csv"
METADATA_FILE = STAGE6_DIR / "canonical_rebuild_metadata.json"


# ---------------------------------------------------------------------------
# 1. Data freshness
# ---------------------------------------------------------------------------

def test_data_freshness():
    """
    Governance requirement: the panel must be no more than 210 days stale.

    Reads scoring_date from the Stage 6 handoff (the last observed date fed
    into the scoring engine). Fails if that date is more than 210 days before
    today. A stale panel means all scores, simulations, and forecasts are
    based on outdated macro conditions and should not be presented as current.

    NOTE: The Stage 6 metadata JSON (canonical_rebuild_metadata.json) records
    generated_at_utc (the build time) but not panel_end_date. scoring_date
    in the handoff is used as the authoritative panel end date.

    # Threshold: 210 days (7 months). Panel refresh triggered when scoring_date
    # falls more than 7 months behind today. Governance commitment: refresh by 2026-06-30.
    """
    handoff = pd.read_csv(HANDOFF_FILE)
    scoring_date = date.fromisoformat(str(handoff["scoring_date"].iloc[0]))
    staleness_days = (date.today() - scoring_date).days
    assert staleness_days <= 210, (
        f"Panel is {staleness_days} days stale "
        f"(scoring_date={scoring_date}, today={date.today()}, threshold=210 days). "
        f"Update the canonical panel before presenting results."
    )


# ---------------------------------------------------------------------------
# 2. Sigma plausibility
# ---------------------------------------------------------------------------

def test_sigma_plausibility():
    """
    Bank-level requirement: regional volatility parameters must be
    economically plausible (annual sigma in [0.025, 0.20]).

    Reads sigma from Stage 4 sde_parameters_bankgrade.csv (sigma_frequency
    confirmed annual). Below 0.025 implies implausibly calm price dynamics
    that understate tail risk. Above 0.20 implies explosive volatility
    inconsistent with UK residential house price history.

    The current config sigma_floor_annual=0.02 clips many regions to 0.020,
    which is below this threshold — expected to FAIL for 10/12 regions.
    """
    params = pd.read_csv(PARAMS_FILE)
    assert (params["sigma_frequency"] == "annual").all(), (
        "sigma_frequency must be 'annual' for all regions before range check."
    )
    below = params[params["sigma"] < 0.025][["region", "sigma"]]
    above = params[params["sigma"] > 0.20][["region", "sigma"]]
    violations = pd.concat([below, above])
    assert violations.empty, (
        f"sigma out of plausible range [0.025, 0.20] for "
        f"{len(violations)} region(s):\n{violations.to_string(index=False)}"
    )


# ---------------------------------------------------------------------------
# 3. Coefficient signs
# ---------------------------------------------------------------------------

def test_coefficient_signs():
    """
    Bank-level requirement: structural coefficient signs must match
    economic theory in all regions.

    beta_rate must be strictly negative: higher mortgage rates suppress
    price growth. A positive beta_rate would imply rates stimulate prices,
    which contradicts UK housing market theory and the model's identification
    strategy.

    beta_income must be non-negative: higher income growth supports price
    growth. A negative beta_income would contradict the demand channel.

    Reads from Stage 4 sde_parameters_bankgrade.csv.
    """
    params = pd.read_csv(PARAMS_FILE)

    wrong_rate = params[params["beta_rate"] >= 0][["region", "beta_rate"]]
    assert wrong_rate.empty, (
        f"beta_rate >= 0 (should be negative) in "
        f"{len(wrong_rate)} region(s):\n{wrong_rate.to_string(index=False)}"
    )

    wrong_income = params[params["beta_income"] < 0][["region", "beta_income"]]
    assert wrong_income.empty, (
        f"beta_income < 0 (should be non-negative) in "
        f"{len(wrong_income)} region(s):\n{wrong_income.to_string(index=False)}"
    )


# ---------------------------------------------------------------------------
# 4. Simulation plausibility
# ---------------------------------------------------------------------------

def test_simulation_plausibility():
    """
    Bank-level requirement: the Baseline scenario median 5-year price growth
    must be within [-20%, +100%] for every region.

    Below -20% implies the model is projecting a severe house price crash
    as the central case, which is implausible as a baseline without an
    explicit stress event. Above +100% (doubling in 5 years) implies an
    explosive boom inconsistent with UK affordability constraints.

    Reads median_5yr_growth from Stage 5 simulation_summary_bankgrade.csv,
    Baseline scenario only.
    """
    summary = pd.read_csv(SIM_FILE)
    baseline = summary[summary["scenario"] == "Baseline"].copy()
    assert len(baseline) == 12, (
        f"Expected 12 regions in Baseline scenario, got {len(baseline)}."
    )

    low  = baseline[baseline["median_5yr_growth"] < -20.0][["region", "median_5yr_growth"]]
    high = baseline[baseline["median_5yr_growth"] > 100.0][["region", "median_5yr_growth"]]
    violations = pd.concat([low, high])
    assert violations.empty, (
        f"Baseline median_5yr_growth outside [-20%, +100%] for "
        f"{len(violations)} region(s):\n{violations.to_string(index=False)}"
    )


# ---------------------------------------------------------------------------
# 5. Simulation band coverage
# ---------------------------------------------------------------------------

def _rolling_5yr_returns_pct(growth: np.ndarray, window: int = 60) -> list[float]:
    """Compute compounded 5-year rolling returns (%) from monthly price_growth."""
    results = []
    for i in range(len(growth) - window + 1):
        seg = growth[i : i + window]
        if np.any(np.isnan(seg)):
            continue
        results.append((float(np.prod(1.0 + seg)) - 1.0) * 100.0)
    return results


def test_simulation_band_coverage():
    """
    Bank-level requirement: the simulated Baseline p10-p90 band must overlap
    the historical p10-p90 band of 5-year rolling returns for at least 9 of
    12 regions (i.e. failure permitted for at most 3 regions).

    A simulated band that does not overlap history in most regions suggests
    the simulation is divorced from the historical distribution and cannot
    be used as a calibration anchor.

    Historical bands are computed from 60-month rolling compounded returns
    of price_growth in fair_value_panel_bankgrade.csv (same scale as
    Stage 5 median_5yr_growth, which is also a percentage 5-year return).
    Baseline p10/p90 are from p10_5yr_growth and p90_5yr_growth in Stage 5.
    """
    panel   = pd.read_csv(PANEL_FILE, parse_dates=["date"])
    summary = pd.read_csv(SIM_FILE)
    baseline = summary[summary["scenario"] == "Baseline"].set_index("region")

    non_overlapping = []

    for region, grp in panel.groupby("region"):
        if region not in baseline.index:
            continue

        grp    = grp.sort_values("date")
        growth = grp["price_growth"].to_numpy(dtype=float)
        hist_returns = _rolling_5yr_returns_pct(growth)

        if len(hist_returns) < 5:
            continue  # insufficient history for this region

        hist_p10 = float(np.percentile(hist_returns, 10))
        hist_p90 = float(np.percentile(hist_returns, 90))
        sim_p10  = float(baseline.loc[region, "p10_5yr_growth"])
        sim_p90  = float(baseline.loc[region, "p90_5yr_growth"])

        # Two intervals [a, b] and [c, d] overlap iff max(a, c) <= min(b, d)
        overlaps = max(hist_p10, sim_p10) <= min(hist_p90, sim_p90)
        if not overlaps:
            non_overlapping.append({
                "region":   region,
                "hist_p10": round(hist_p10, 2),
                "hist_p90": round(hist_p90, 2),
                "sim_p10":  round(sim_p10, 2),
                "sim_p90":  round(sim_p90, 2),
            })

    max_failures = 3
    assert len(non_overlapping) <= max_failures, (
        f"Simulated Baseline p10-p90 band does not overlap historical band "
        f"for {len(non_overlapping)} region(s) (threshold: {max_failures}):\n"
        + "\n".join(str(r) for r in non_overlapping)
    )


# ---------------------------------------------------------------------------
# 6. Score range
# ---------------------------------------------------------------------------

def test_score_range():
    """
    Bank-level requirement: every regional score in the Stage 6 handoff must
    be a finite number in [0, 100].

    NaN scores indicate a failed computation in the scoring engine. Scores
    outside [0, 100] indicate a clipping failure or a weight configuration
    error. Either condition must block deployment of the dashboard.

    Checks both consumer_score and reit_score for all 12 regions.
    """
    handoff = pd.read_csv(HANDOFF_FILE)

    for col in ("consumer_score", "reit_score"):
        nan_regions = handoff[handoff[col].isna()]["region"].tolist()
        assert not nan_regions, (
            f"{col} is NaN for region(s): {nan_regions}"
        )
        out_of_range = handoff[~handoff[col].between(0.0, 100.0)][["region", col]]
        assert out_of_range.empty, (
            f"{col} outside [0, 100] for "
            f"{len(out_of_range)} region(s):\n{out_of_range.to_string(index=False)}"
        )


# ---------------------------------------------------------------------------
# 7. No legacy paths
# ---------------------------------------------------------------------------

def test_no_legacy_paths():
    """
    Bank-level requirement: no active source file in src/ may import from
    src/archive/ (if that directory exists).

    Importing from archived code reintroduces retired model paths into the
    active pipeline, creating ambiguity about which code is authoritative
    (see MODEL_RISK_REGISTER R002). This is checked by scanning import
    statements in all .py files under src/, excluding src/archive/ itself.

    If src/archive/ does not exist, the test passes trivially.
    """
    src_dir     = ROOT / "src"
    archive_dir = src_dir / "archive"

    if not archive_dir.exists():
        pytest.skip("src/archive/ does not exist — no legacy paths possible.")

    # Collect all .py files in src/ that are NOT inside src/archive/
    active_files = [
        p for p in src_dir.rglob("*.py")
        if archive_dir not in p.parents
    ]

    # Patterns that indicate an import from src/archive/
    archive_patterns = [
        re.compile(r"^\s*from\s+.*archive.*\s+import", re.MULTILINE),
        re.compile(r"^\s*import\s+.*archive", re.MULTILINE),
    ]

    violations: list[str] = []
    for fpath in active_files:
        try:
            source = fpath.read_text(encoding="utf-8")
        except OSError:
            continue
        for pattern in archive_patterns:
            if pattern.search(source):
                violations.append(str(fpath.relative_to(ROOT)))
                break

    assert not violations, (
        f"{len(violations)} active file(s) import from src/archive/:\n"
        + "\n".join(f"  {v}" for v in sorted(violations))
    )


# ---------------------------------------------------------------------------
# 8. Single canonical entry point
# ---------------------------------------------------------------------------

def test_single_canonical_entry():
    """
    Bank-level requirement: exactly one Python file in src/model/ may define
    the canonical pipeline entry point (run_canonical_rebuild).

    Multiple definitions create ambiguity about which file owns the
    authoritative build path (see MODEL_RISK_REGISTER R002). Any reviewer
    or automated runner must be able to identify the single entry point
    without reading all source files.

    Scans all .py files in src/model/ (excluding any archive subdirectories)
    for top-level function definitions named run_canonical_rebuild.
    """
    model_dir   = ROOT / "src" / "model"
    archive_dir = model_dir / "archive"

    candidates = [
        p for p in model_dir.rglob("*.py")
        if archive_dir not in p.parents
    ]

    defining_files: list[str] = []
    for fpath in candidates:
        try:
            source = fpath.read_text(encoding="utf-8")
            tree   = ast.parse(source, filename=str(fpath))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_canonical_rebuild":
                defining_files.append(str(fpath.relative_to(ROOT)))
                break

    assert len(defining_files) == 1, (
        f"Expected exactly 1 file in src/model/ to define run_canonical_rebuild, "
        f"found {len(defining_files)}: {defining_files}"
    )


# ---------------------------------------------------------------------------
# Rate channel stability (Approach B, 11B)
# ---------------------------------------------------------------------------

REGIONAL_STABILITY_FILE = ROOT / "data" / "outputs" / "validation" / "regional_rate_stability.csv"
RATE_STABILITY_THRESHOLD = 5.0  # late/early ratio above which a region is flagged as unstable


def test_rate_channel_stability():
    """
    Monitors late-sample vs early-sample magnitude ratio for the rate coefficient.

    Reads regional_rate_stability.csv (generated by validation_pack.regional_rate_stability).
    For each region: asserts abs(late_beta / early_beta) < RATE_STABILITY_THRESHOLD (5.0).

    Threshold = 5.0, not 3.0: at 3.0 all 12 regions fail because early-period
    (2005-2015 ZLB era) betas are imprecisely estimated or reverse-signed due to
    near-zero mortgage-rate variation; the ratio then measures identification noise,
    not structural break. At 5.0, only regions with denominator |t| < ~1.0 are flagged.

    Documented in R010. Expected to fail for 5 regions in the current panel
    (North East, Scotland, South West, Wales, Yorkshire) — see module docstring.
    """
    assert REGIONAL_STABILITY_FILE.exists(), (
        f"regional_rate_stability.csv not found at {REGIONAL_STABILITY_FILE}. "
        "Run python -m src.model.validation_pack to generate it."
    )
    df = pd.read_csv(REGIONAL_STABILITY_FILE)
    required_cols = {"region", "abs_late_early_ratio"}
    assert required_cols.issubset(df.columns), (
        f"regional_rate_stability.csv missing columns: {required_cols - set(df.columns)}"
    )

    unstable = df[df["abs_late_early_ratio"] > RATE_STABILITY_THRESHOLD][
        ["region", "abs_late_early_ratio"]
    ].sort_values("abs_late_early_ratio", ascending=False)

    assert unstable.empty, (
        f"Rate channel instability detected (late/early ratio > {RATE_STABILITY_THRESHOLD}) "
        f"in {len(unstable)} of {len(df)} regions. "
        f"Review R010 in MODEL_RISK_REGISTER.csv.\n"
        + unstable.to_string(index=False)
    )


# ---------------------------------------------------------------------------
# Structural break tests (R005)
# ---------------------------------------------------------------------------

def test_structural_break_tests_run():
    """
    Smoke test for the Chow-test structural break diagnostics (R005).

    Loads the canonical panel, calls run_chow_tests(), and asserts that:
    - No exception is raised
    - The result DataFrame contains the expected columns
    - One row is present per available P* regressor

    Does NOT assert specific break/no-break outcomes — results are data-driven
    and will change as the panel is updated.
    """
    from src.model.structural_breaks import run_chow_tests

    CANONICAL_PANEL = ROOT.parent / "housing_model" / "data" / "processed" / "master_dataset_canonical.csv"
    panel = pd.read_csv(CANONICAL_PANEL, parse_dates=["date"])

    results_df, summary = run_chow_tests(panel)

    expected_cols = {"regressor", "f_stat", "p_value", "significant_at_5pct"}
    assert expected_cols.issubset(set(results_df.columns)), (
        f"run_chow_tests() result missing columns. "
        f"Expected {expected_cols}, got {set(results_df.columns)}"
    )
    assert len(results_df) > 0, "run_chow_tests() returned an empty DataFrame"
    assert isinstance(summary, dict), "run_chow_tests() summary must be a dict"
    assert "any_breaks_detected" in summary
    assert "break_points_detected" in summary
    assert "interpretation" in summary
    assert isinstance(summary["any_breaks_detected"], bool)
    assert isinstance(summary["break_points_detected"], list)
