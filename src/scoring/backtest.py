"""Stage 6 scoring backtest: precision/recall/IR sweep across threshold values.

Loads the Stage 4 fair-value panel, reconstructs the c1_mispricing valuation
signal (identical formula to canonical_rebuild.py), computes compounded forward
returns at 12/24/36-month horizons, and evaluates signal quality for threshold
values from 30 to 70 in steps of 5.

Upside metrics (signal >= threshold → Supportive): thresholds 30–70.
Downside metrics (signal < threshold → Cautious): thresholds 30–50 only.

Outputs
-------
data/outputs/validation/scoring_backtest_results.csv
data/outputs/validation/scoring_backtest_curve.png
"""

from __future__ import annotations
from src.core.paths import STAGE4_OUTPUT_DIR, OUTPUT_DATA_DIR, ensure_directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Path bootstrap (run as script or imported from project root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAIR_VALUE_PANEL = STAGE4_OUTPUT_DIR / "fair_value_panel_bankgrade.csv"
VALIDATION_DIR = OUTPUT_DATA_DIR / "validation"
RESULTS_CSV = VALIDATION_DIR / "scoring_backtest_results.csv"
CURVE_PNG = VALIDATION_DIR / "scoring_backtest_curve.png"

HORIZONS = [12, 24, 36]
PRIMARY_HORIZON = 36          # used for precision/recall/IR
THRESHOLDS = list(range(30, 75, 5))          # 30, 35, ..., 70  (upside sweep)
CAUTIOUS_THRESHOLDS = [30, 35, 40, 45, 50]  # downside sweep

MIN_SUBSEQUENT_MONTHS = 36    # rows must have this many months of forward data


# ---------------------------------------------------------------------------
# Signal formula (verbatim from canonical_rebuild.py lines 1221-1223)
# ---------------------------------------------------------------------------

def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_valuation_signal(fair_value_gap_log: pd.Series) -> pd.Series:
    """Replicate c1_mispricing: clip(50 - 0.95 * pct_above_pstar, 0, 100)."""
    pct_above_pstar = (np.exp(fair_value_gap_log) - 1.0) * 100.0
    return (50.0 - 0.95 * pct_above_pstar).clip(0.0, 100.0)


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------

def _compound_forward_return(price_growth: np.ndarray, start: int, n_months: int) -> float:
    """Compound n_months of monthly returns starting one period after `start`."""
    window = price_growth[start + 1: start + 1 + n_months]
    if len(window) < n_months or np.any(np.isnan(window)):
        return np.nan
    return float(np.prod(1.0 + window) - 1.0)


def compute_forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add forward_12m_return, forward_24m_return, forward_36m_return columns."""
    records = []
    for region, grp in panel.groupby("region", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        growth = grp["price_growth"].to_numpy(dtype=float)
        for i in range(len(grp)):
            row = grp.iloc[i].to_dict()
            for h in HORIZONS:
                row[f"forward_{h}m_return"] = _compound_forward_return(
                    growth, i, h)
            records.append(row)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Threshold metrics
# ---------------------------------------------------------------------------

def _precision_recall(signal: pd.Series, fwd: pd.Series, threshold: float):
    """Precision and recall for a single threshold / horizon pair (upside)."""
    above = signal >= threshold
    positive = fwd > 0.0

    tp = (above & positive).sum()
    fp = (above & ~positive).sum()
    fn = (~above & positive).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return float(precision), float(recall)


def _precision_recall_cautious(signal: pd.Series, fwd: pd.Series, threshold: float):
    """Precision and recall for the Cautious bucket (signal < threshold, return < 0)."""
    below = signal < threshold
    negative = fwd < 0.0

    tp = (below & negative).sum()
    fp = (below & ~negative).sum()
    fn = (~below & negative).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return float(precision), float(recall)


def compute_threshold_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per threshold with precision, recall, IR, and counts."""
    signal = df["valuation_signal"]
    fwd_primary = df[f"forward_{PRIMARY_HORIZON}m_return"]
    std_all = float(fwd_primary.std())

    rows = []
    for thr in THRESHOLDS:
        above = signal >= thr
        below = ~above

        prec_36, rec_36 = _precision_recall(signal, fwd_primary, thr)
        prec_12, rec_12 = _precision_recall(
            signal, df["forward_12m_return"], thr)
        prec_24, rec_24 = _precision_recall(
            signal, df["forward_24m_return"], thr)

        n_supportive = int(above.sum())
        n_cautious = int(below.sum())

        mean_sup = float(fwd_primary[above].mean()
                         ) if n_supportive > 0 else np.nan
        mean_cau = float(fwd_primary[below].mean()
                         ) if n_cautious > 0 else np.nan

        if np.isnan(mean_sup) or np.isnan(mean_cau) or std_all == 0:
            ir = np.nan
        else:
            ir = (mean_sup - mean_cau) / std_all

        # Downside (Cautious) metrics — only for CAUTIOUS_THRESHOLDS
        if thr in CAUTIOUS_THRESHOLDS:
            prec_cau, rec_cau = _precision_recall_cautious(
                signal, fwd_primary, thr)
            ir_cau = (mean_cau - mean_sup) / std_all if not (np.isnan(mean_cau) or
                                                             np.isnan(mean_sup) or std_all == 0) else np.nan
        else:
            prec_cau = rec_cau = ir_cau = np.nan

        rows.append({
            "threshold": thr,
            "n_supportive": n_supportive,
            "n_cautious": n_cautious,
            "mean_return_supportive": round(mean_sup, 6) if not np.isnan(mean_sup) else np.nan,
            "mean_return_cautious": round(mean_cau, 6) if not np.isnan(mean_cau) else np.nan,
            "precision_36m": round(prec_36, 4) if not np.isnan(prec_36) else np.nan,
            "recall_36m": round(rec_36, 4) if not np.isnan(rec_36) else np.nan,
            "ir_36m": round(ir, 4) if not np.isnan(ir) else np.nan,
            "precision_12m": round(prec_12, 4) if not np.isnan(prec_12) else np.nan,
            "recall_12m": round(rec_12, 4) if not np.isnan(rec_12) else np.nan,
            "precision_24m": round(prec_24, 4) if not np.isnan(prec_24) else np.nan,
            "recall_24m": round(rec_24, 4) if not np.isnan(rec_24) else np.nan,
            "precision_cautious_36m": round(prec_cau, 4) if not np.isnan(prec_cau) else np.nan,
            "recall_cautious_36m": round(rec_cau, 4) if not np.isnan(rec_cau) else np.nan,
            "ir_cautious_36m": round(ir_cau, 4) if not np.isnan(ir_cau) else np.nan,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_precision_recall(metrics: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        metrics["recall_36m"],
        metrics["precision_36m"],
        marker="o",
        color="#1f4e79",
        linewidth=1.5,
        markersize=6,
    )
    for _, row in metrics.iterrows():
        ax.annotate(
            f"t={int(row['threshold'])}",
            xy=(row["recall_36m"], row["precision_36m"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
        )
    ax.set_xlabel(
        "Recall (fraction of positive-return periods caught)", fontsize=10)
    ax.set_ylabel(
        "Precision (P[36m return > 0 | signal ≥ threshold])", fontsize=10)
    ax.set_title(
        "Valuation Signal — Precision / Recall at 36-month horizon\n"
        "Threshold sweep: 30 → 70 in steps of 5",
        fontsize=10,
    )
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    ensure_directory(VALIDATION_DIR)

    # 1. Load panel
    print(f"Loading panel: {FAIR_VALUE_PANEL}")
    panel = pd.read_csv(FAIR_VALUE_PANEL, parse_dates=["date"])
    print(f"  Raw rows: {len(panel)}")

    # 2. Compute valuation signal
    panel["valuation_signal"] = compute_valuation_signal(
        panel["fair_value_gap_log"])

    # 3. Compute forward returns
    print("Computing forward returns (12m / 24m / 36m) …")
    panel = compute_forward_returns(panel)

    # 4. Filter to rows with complete 36m forward window
    keep_cols = ["forward_12m_return",
                 "forward_24m_return", "forward_36m_return"]
    panel_filtered = panel.dropna(
        subset=keep_cols + ["valuation_signal", "fair_value_gap_log"])
    print(
        f"  Rows with ≥36m subsequent data and valid signal: {len(panel_filtered)}")

    # 5. Threshold sweep
    print("Running threshold sweep …")
    metrics = compute_threshold_metrics(panel_filtered)

    # 6. Save CSV
    metrics.to_csv(RESULTS_CSV, index=False)
    print(f"  Saved: {RESULTS_CSV}")

    # 7. Plot
    plot_precision_recall(metrics, CURVE_PNG)

    # 8. Best upside IR
    best_row = metrics.loc[metrics["ir_36m"].idxmax()]
    print(
        f"\nBest upside IR threshold: {int(best_row['threshold'])} "
        f"(IR={best_row['ir_36m']:.4f}, "
        f"precision={best_row['precision_36m']:.3f}, "
        f"recall={best_row['recall_36m']:.3f})"
    )

    # 9. Best downside IR (most negative = strongest cautious signal)
    cau_subset = metrics.dropna(subset=["ir_cautious_36m"])
    if not cau_subset.empty:
        best_cau = cau_subset.loc[cau_subset["ir_cautious_36m"].idxmin()]
        print(
            f"Best downside IR threshold: {int(best_cau['threshold'])} "
            f"(IR_cautious={best_cau['ir_cautious_36m']:.4f}, "
            f"precision_cautious={best_cau['precision_cautious_36m']:.3f}, "
            f"recall_cautious={best_cau['recall_cautious_36m']:.3f})"
        )

    return metrics


if __name__ == "__main__":
    _run_result = run()


# ---------------------------------------------------------------------------
# C2 — Downside probability directional backtest
# ---------------------------------------------------------------------------

def backtest_c2_downside(
    panel_df: pd.DataFrame,
    horizon_months: int = 24,
) -> tuple[pd.DataFrame, dict]:
    """
    Backtest the C2 downside probability signal.

    For each region at each point in time, checks whether downside_probability
    exceeded 0.5 (model predicted a >50% chance of price decline) and compares
    against the realised nominal price change over the next *horizon_months*.

    Required column
    ---------------
    downside_probability : float in [0, 1]
        This column is NOT present in master_dataset_canonical.csv by default.
        It must be added by merging the simulation summary
        (prob_terminal_loss_10pct, Baseline scenario from
        data/outputs/stage5c/simulation_summary_bankgrade.csv) into the panel
        by region, or by computing a time-varying equivalent in the canonical
        rebuild.  Until that column is available the function returns an empty
        DataFrame with the correct schema and a UserWarning.

    Returns
    -------
    result_df : pd.DataFrame
        Columns: region, date, predicted_downside_flag, realised_decline,
        correct_direction
    summary : dict
        Keys: accuracy, precision, recall
    """
    import warnings

    SIGNAL_COL = "downside_probability"
    EXPECTED_COLS = [
        "region", "date", "predicted_downside_flag",
        "realised_decline", "correct_direction",
    ]
    EMPTY_SUMMARY: dict = {"accuracy": np.nan,
                           "precision": np.nan, "recall": np.nan}

    if SIGNAL_COL not in panel_df.columns:
        warnings.warn(
            f"backtest_c2_downside: '{SIGNAL_COL}' column not found in panel_df. "
            "Merge simulation_summary_bankgrade.csv (prob_terminal_loss_10pct, "
            "Baseline scenario) by region, or add a time-varying equivalent to "
            "the canonical build, then re-run this backtest. "
            "Returning empty DataFrame.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=EXPECTED_COLS), EMPTY_SUMMARY

    df = panel_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    records: list[dict] = []
    for region, grp in df.groupby("region", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        grp = grp.dropna(
            subset=[SIGNAL_COL, "nominal_house_price"]).reset_index(drop=True)
        prices = grp["nominal_house_price"].to_numpy(dtype=float)

        for i in range(len(grp)):
            j = i + horizon_months
            if j >= len(grp):
                continue
            predicted_flag = bool(float(grp.at[i, SIGNAL_COL]) > 0.5)
            realised_decline = bool(prices[j] < prices[i])
            records.append({
                "region":                 region,
                "date":                   grp.at[i, "date"],
                "predicted_downside_flag": predicted_flag,
                "realised_decline":        realised_decline,
                "correct_direction":       predicted_flag == realised_decline,
            })

    result_df = pd.DataFrame(records)

    if result_df.empty:
        return result_df, EMPTY_SUMMARY

    accuracy = float(result_df["correct_direction"].mean())

    downside_calls = result_df[result_df["predicted_downside_flag"]]
    precision = (
        float(downside_calls["realised_decline"].mean())
        if len(downside_calls) > 0 else np.nan
    )

    actual_declines = result_df[result_df["realised_decline"]]
    recall = (
        float(actual_declines["predicted_downside_flag"].mean())
        if len(actual_declines) > 0 else np.nan
    )

    summary = {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4) if not np.isnan(precision) else np.nan,
        "recall":    round(recall, 4) if not np.isnan(recall) else np.nan,
    }
    return result_df, summary


# ---------------------------------------------------------------------------
# C3 — Affordability signal directional backtest
# ---------------------------------------------------------------------------

def backtest_c3_affordability(
    panel_df: pd.DataFrame,
    horizon_months: int = 24,
) -> tuple[pd.DataFrame, dict]:
    """
    Backtest the C3 affordability signal.

    A positive affordability_gap (housing more expensive than the income-implied
    historical norm) is treated as a signal of correction pressure.  The backtest
    checks whether that directional call (positive gap → predicts correction)
    aligned with the realised real house price change over *horizon_months*.

    Signal column priority
    ----------------------
    1. affordability_gap       — primary; available in master_dataset_canonical.csv
    2. payment_burden_gap      — fallback; also in master_dataset_canonical.csv

    If neither column is present the function returns an empty DataFrame with
    the correct schema and a UserWarning.

    Returns
    -------
    result_df : pd.DataFrame
        Columns: region, date, affordability_signal, realised_correction,
        correct_direction
    summary : dict
        Keys: accuracy, directional_information_ratio
    """
    import warnings

    EXPECTED_COLS = [
        "region", "date", "affordability_signal",
        "realised_correction", "correct_direction",
    ]
    EMPTY_SUMMARY: dict = {
        "accuracy": np.nan,
        "directional_information_ratio": np.nan,
    }

    signal_col: str | None = None
    for candidate in ("affordability_gap", "payment_burden_gap"):
        if candidate in panel_df.columns:
            signal_col = candidate
            break

    if signal_col is None:
        warnings.warn(
            "backtest_c3_affordability: neither 'affordability_gap' nor "
            "'payment_burden_gap' found in panel_df. "
            "Returning empty DataFrame.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=EXPECTED_COLS), EMPTY_SUMMARY

    price_col = "real_house_price" if "real_house_price" in panel_df.columns else "nominal_house_price"

    df = panel_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    records: list[dict] = []
    for region, grp in df.groupby("region", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        grp = grp.dropna(subset=[signal_col, price_col]).reset_index(drop=True)
        prices = grp[price_col].to_numpy(dtype=float)

        for i in range(len(grp)):
            j = i + horizon_months
            if j >= len(grp):
                continue
            signal_val = float(grp.at[i, signal_col])
            # Positive gap → unaffordable → predicts correction
            predicts_correction = signal_val > 0.0
            realised_correction = bool(prices[j] < prices[i])
            records.append({
                "region":               region,
                "date":                 grp.at[i, "date"],
                "affordability_signal": round(signal_val, 6),
                "realised_correction":  realised_correction,
                "correct_direction":    predicts_correction == realised_correction,
            })

    result_df = pd.DataFrame(records)

    if result_df.empty:
        return result_df, EMPTY_SUMMARY

    correct = result_df["correct_direction"].astype(float).to_numpy()
    accuracy = float(correct.mean())
    std_c = float(correct.std())
    dir_ir = float(correct.mean() / std_c) if std_c > 0 else np.nan

    summary = {
        "accuracy": round(accuracy, 4),
        "directional_information_ratio": (
            round(dir_ir, 4) if not np.isnan(dir_ir) else np.nan
        ),
    }
    return result_df, summary


# ---------------------------------------------------------------------------
# C4 — Rental yield directional backtest
# ---------------------------------------------------------------------------

def backtest_c4_rental_yield(
    panel_df: pd.DataFrame,
    horizon_months: int = 24,
) -> tuple[pd.DataFrame, dict]:
    """
    Backtest the C4 rental yield signal.

    The signal in engine.py is ``yield_signal = clip(50 + (gross_yield - target_yield) * 18)``,
    which is monotonically increasing in gross_yield.  The directional equivalent
    used here: if the current regional yield is *above* its own expanding-window
    historical average, the region is cheap on yield → predicts outperformance
    over the benchmark (national mean return) in the next *horizon_months*.

    Signal column priority
    ----------------------
    1. gross_yield_pct  — primary; available in master_dataset_canonical.csv
    2. yield_lag1       — fallback; one-month lagged yield

    Realised outcome
    ----------------
    Region compound return over [t, t + horizon_months] vs. cross-sectional
    mean return across all regions over the same horizon.  Outperformance
    = region return > national mean return for that date.

    Returns
    -------
    result_df : pd.DataFrame
        Columns: region, date, yield_signal_direction, realised_outperformance,
        correct_direction
    summary : dict
        Keys: accuracy, information_ratio_proxy
            accuracy               — fraction of directional calls correct
            information_ratio_proxy — (accuracy − 0.5) * sqrt(12 / horizon_months)
    """
    import warnings

    EXPECTED_COLS = [
        "region", "date", "yield_signal_direction",
        "realised_outperformance", "correct_direction",
    ]
    EMPTY_SUMMARY: dict = {"accuracy": np.nan,
                           "information_ratio_proxy": np.nan}

    signal_col: str | None = None
    for candidate in ("gross_yield_pct", "yield_lag1"):
        if candidate in panel_df.columns:
            signal_col = candidate
            break

    if signal_col is None:
        warnings.warn(
            "backtest_c4_rental_yield: neither 'gross_yield_pct' nor 'yield_lag1' "
            "found in panel_df. Returning empty DataFrame.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=EXPECTED_COLS), EMPTY_SUMMARY

    if "nominal_house_price" not in panel_df.columns:
        warnings.warn(
            "backtest_c4_rental_yield: 'nominal_house_price' not found. "
            "Returning empty DataFrame.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=EXPECTED_COLS), EMPTY_SUMMARY

    df = panel_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Step 1 — per-region records with raw forward returns
    raw_records: list[dict] = []
    for region, grp in df.groupby("region", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        grp = grp.dropna(
            subset=[signal_col, "nominal_house_price"]).reset_index(drop=True)
        yields = grp[signal_col].to_numpy(dtype=float)
        prices = grp["nominal_house_price"].to_numpy(dtype=float)

        for i in range(len(grp)):
            j = i + horizon_months
            if j >= len(grp):
                continue
            # Expanding-window regional average yield up to and including t=i.
            # Avoids look-ahead: only uses information available at time i.
            regional_avg = float(np.mean(yields[: i + 1]))
            yield_above_avg = bool(yields[i] > regional_avg)
            fwd_return = float(prices[j] / prices[i] - 1.0)
            raw_records.append({
                "region":               region,
                "date":                 grp.at[i, "date"],
                "yield_signal_direction": yield_above_avg,
                "fwd_return":           fwd_return,
            })

    if not raw_records:
        return pd.DataFrame(columns=EXPECTED_COLS), EMPTY_SUMMARY

    result_df = pd.DataFrame(raw_records)

    # Step 2 — cross-sectional benchmark: national mean return per date
    nat_mean = (
        result_df.groupby("date")["fwd_return"]
        .mean()
        .rename("nat_mean_return")
    )
    result_df = result_df.merge(nat_mean, on="date", how="left")
    result_df["realised_outperformance"] = (
        result_df["fwd_return"] > result_df["nat_mean_return"]
    )
    # Buy (yield_above_avg=True) predicts outperformance; Sell predicts underperformance
    result_df["correct_direction"] = (
        result_df["yield_signal_direction"] == result_df["realised_outperformance"]
    )
    result_df = result_df.drop(columns=["fwd_return", "nat_mean_return"])

    accuracy = float(result_df["correct_direction"].mean())
    # Annualise the edge above random: scale by sqrt(periods per year)
    ir_proxy = float((accuracy - 0.5) * np.sqrt(12.0 / max(horizon_months, 1)))

    summary = {
        "accuracy":                round(accuracy, 4),
        "information_ratio_proxy": round(ir_proxy, 4),
    }
    return result_df, summary


# ---------------------------------------------------------------------------
# C5 — Cycle / momentum directional backtest
# ---------------------------------------------------------------------------

def backtest_c5_cycle(
    panel_df: pd.DataFrame,
    horizon_months: int = 12,
) -> tuple[pd.DataFrame, dict]:
    """
    Backtest the C5 cycle / momentum signal.

    In engine.py the cycle component feeds into the consumer and REIT composites
    as ``cycle_outlook = clip(0.65 * cycle_signal + 0.35 * scenario_signal)``.
    The directional equivalent used here: positive trailing momentum
    (price_growth_12m > 0) predicts continued positive price growth over the
    next *horizon_months*.  A 12-month horizon is appropriate for momentum — it
    is short enough to capture the autocorrelation regime but long enough to
    avoid pure noise.

    Signal column priority
    ----------------------
    1. price_growth_12m  — trailing 12m log-price growth
    2. price_growth_3m   — trailing 3m price growth
    3. approvals_growth  — mortgage approvals growth (lead indicator)
    4. price_growth_lag1 — lagged monthly price growth

    Realised outcome
    ----------------
    Did the compound nominal price growth over the next *horizon_months* remain
    positive?  (price[t + h] > price[t])

    If no signal column is present the function returns an empty DataFrame and
    emits a UserWarning — consistent with the C2 pattern.

    Returns
    -------
    result_df : pd.DataFrame
        Columns: region, date, momentum_signal_direction, realised_continuation,
        correct_direction
    summary : dict
        Keys: accuracy, information_ratio_proxy
    """
    import warnings

    EXPECTED_COLS = [
        "region", "date", "momentum_signal_direction",
        "realised_continuation", "correct_direction",
    ]
    EMPTY_SUMMARY: dict = {"accuracy": np.nan,
                           "information_ratio_proxy": np.nan}

    signal_col: str | None = None
    for candidate in (
        "price_growth_12m", "price_growth_3m",
        "approvals_growth", "price_growth_lag1",
    ):
        if candidate in panel_df.columns:
            signal_col = candidate
            break

    if signal_col is None:
        warnings.warn(
            "backtest_c5_cycle: no momentum/cycle column found in panel_df. "
            "Expected one of: 'price_growth_12m', 'price_growth_3m', "
            "'approvals_growth', 'price_growth_lag1'. "
            "Returning empty DataFrame.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=EXPECTED_COLS), EMPTY_SUMMARY

    if "nominal_house_price" not in panel_df.columns:
        warnings.warn(
            "backtest_c5_cycle: 'nominal_house_price' not found. "
            "Returning empty DataFrame.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=EXPECTED_COLS), EMPTY_SUMMARY

    df = panel_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    records: list[dict] = []
    for region, grp in df.groupby("region", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        grp = grp.dropna(
            subset=[signal_col, "nominal_house_price"]).reset_index(drop=True)
        signals = grp[signal_col].to_numpy(dtype=float)
        prices = grp["nominal_house_price"].to_numpy(dtype=float)

        for i in range(len(grp)):
            j = i + horizon_months
            if j >= len(grp):
                continue
            # Positive trailing momentum → predicts positive forward return
            positive_momentum = bool(signals[i] > 0.0)
            fwd_return = float(prices[j] / prices[i] - 1.0)
            realised_continuation = fwd_return > 0.0
            records.append({
                "region":                   region,
                "date":                     grp.at[i, "date"],
                "momentum_signal_direction": positive_momentum,
                "realised_continuation":     realised_continuation,
                "correct_direction":         positive_momentum == realised_continuation,
            })

    result_df = pd.DataFrame(records)

    if result_df.empty:
        return result_df, EMPTY_SUMMARY

    accuracy = float(result_df["correct_direction"].mean())
    ir_proxy = float((accuracy - 0.5) * np.sqrt(12.0 / max(horizon_months, 1)))

    summary = {
        "accuracy":                round(accuracy, 4),
        "information_ratio_proxy": round(ir_proxy, 4),
    }
    return result_df, summary


# ---------------------------------------------------------------------------
# CLI entry point — C2 and C3 backtests against canonical panel
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import pathlib
    import warnings as _w
    # Load the canonical panel -- find the correct path by checking src/core/paths.py or config/settings.py
    # The panel is likely at data/outputs/stage4/ or data/processed/
    panel_path = next(pathlib.Path("data").rglob(
        "master_dataset_canonical.csv"), None)
    if panel_path is None:
        panel_path = next(pathlib.Path("data").rglob("*canonical*.csv"), None)
    print(f"Loading panel from: {panel_path}")
    panel_df = pd.read_csv(panel_path, parse_dates=["date"])

    print("\n=== C2 Downside Backtest (24m horizon) ===")
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)
        result_c2, stats_c2 = backtest_c2_downside(panel_df, horizon_months=24)
    print(json.dumps(stats_c2, indent=2, default=str))

    print("\n=== C3 Affordability Backtest (24m horizon) ===")
    result_c3, stats_c3 = backtest_c3_affordability(
        panel_df, horizon_months=24)
    print(json.dumps(stats_c3, indent=2, default=str))

    print("\n=== C4 Rental Yield Backtest (24m horizon) ===")
    result_c4, stats_c4 = backtest_c4_rental_yield(panel_df, horizon_months=24)
    print(json.dumps(stats_c4, indent=2, default=str))

    print("\n=== C5 Cycle / Momentum Backtest (12m horizon) ===")
    result_c5, stats_c5 = backtest_c5_cycle(panel_df, horizon_months=12)
    print(json.dumps(stats_c5, indent=2, default=str))

    print("\n=== Summary ===")
    for label, stats in [("C2 Downside", stats_c2), ("C3 Affordability", stats_c3),
                         ("C4 Rental Yield", stats_c4), ("C5 Cycle", stats_c5)]:
        acc = stats.get("accuracy")
        acc_str = f"{acc:.1%}" if acc == acc else "N/A"  # NaN check
        print(f"  {label}: accuracy={acc_str}")
