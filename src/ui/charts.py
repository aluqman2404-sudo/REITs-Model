"""Matplotlib charts for the Stage 7 research dashboard."""

from __future__ import annotations
from src.ui.formatters import band_colour
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
import tempfile

_CACHE_ROOT = os.path.join(tempfile.gettempdir(), "uk_housing_stage7")
os.makedirs(_CACHE_ROOT, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _CACHE_ROOT)


matplotlib.use("Agg")


INK = "#12355b"
TEAL = "#0f766e"
SKY = "#90cdf4"
SLATE = "#4f5b66"
GRID = "#d9e2ec"


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=SLATE)
    ax.xaxis.label.set_color(SLATE)
    ax.yaxis.label.set_color(SLATE)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.6)


def fan_chart(history: pd.DataFrame, forecast: pd.DataFrame, region: str, scenario_name: str):
    future_months = pd.date_range(
        history["date"].max(), periods=len(forecast), freq="MS")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(history["date"], history["nominal_house_price"],
            color=INK, lw=2, label="Historical price")
    ax.fill_between(
        future_months, forecast["p10"], forecast["p90"], color=SKY, alpha=0.25, label="10-90 range")
    ax.fill_between(future_months, forecast["p25"], forecast["p75"],
                    color="#3182ce", alpha=0.25, label="25-75 range")
    ax.plot(future_months, forecast["p50"],
            color=TEAL, lw=2.5, label="Median path")
    ax.set_title(f"{region}: historical and simulated price path",
                 fontsize=12, loc="left")
    ax.set_xlabel("Date")
    ax.set_ylabel("Nominal average price (GBP)")
    ax.legend(frameon=False, ncol=3)
    _style_axes(ax)
    ax.text(
        0,
        -0.18,
        f"Scenario lab path under {scenario_name}. Fan widths show simulated uncertainty, not forecast certainty.",
        transform=ax.transAxes,
        fontsize=10,
        color=SLATE,
    )
    fig.tight_layout()
    return fig


def score_breakdown_chart(component_scores: dict[str, float], title: str):
    labels = [name.replace("_", " ").title() for name in component_scores]
    values = list(component_scores.values())
    colours = ["#1f7a45" if v >= 67 else "#b26a00" if v >=
               40 else "#a63f3f" for v in values]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(labels, values, color=colours)
    for idx, value in enumerate(values):
        ax.text(value + 1.5, idx, f"{value:.0f}", va="center", fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Score (0-100)")
    ax.set_title(title, fontsize=12, loc="left")
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    return fig


def affordability_gauge(score: float, title: str):
    fig, ax = plt.subplots(figsize=(6.5, 1.8))
    ax.barh([0], [100], color="#e5e7eb", height=0.45)
    colour = "#1f7a45" if score >= 67 else "#b26a00" if score >= 40 else "#a63f3f"
    ax.barh([0], [score], color=colour, height=0.45)
    ax.text(score + 1.5, 0, f"{score:.0f}/100",
            va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.set_yticks([])
    ax.set_xlabel("Affordability")
    ax.set_title(title, fontsize=12, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=SLATE)
    fig.tight_layout()
    return fig


def scenario_comparison_chart(sim_region: pd.DataFrame):
    table = sim_region.sort_values("median_5yr_growth").copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(table))
    err_low = table["median_5yr_growth"] - table["p10_5yr_growth"]
    err_high = table["p90_5yr_growth"] - table["median_5yr_growth"]
    colours = plt.cm.RdYlGn_r(table["prob_terminal_loss_10pct"].to_numpy())
    ax.barh(y, table["median_5yr_growth"], color=colours)
    ax.errorbar(
        table["median_5yr_growth"],
        y,
        xerr=[err_low, err_high],
        fmt="none",
        ecolor=INK,
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(table["scenario"])
    ax.set_xlabel("Median 5-year price change (%)")
    ax.set_title("Scenario comparison", fontsize=12, loc="left")
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    return fig


def regional_rank_chart(region_table: pd.DataFrame, score_col: str, selected_region: str, title: str):
    table = region_table.sort_values(score_col).copy()
    band_col = f"{score_col.split('_')[0]}_band"
    colours = [
        "#12355b" if region == selected_region else band_colour(label)
        for region, label in zip(table["region"], table[band_col])
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.barh(table["region"], table[score_col], color=colours)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score")
    ax.set_title(title, fontsize=12, loc="left")
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    return fig


def yield_risk_chart(region_table: pd.DataFrame, selected_region: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colours = [band_colour(label) for label in region_table["reit_band"]]
    sizes = [180 if region ==
             selected_region else 80 for region in region_table["region"]]
    scatter = ax.scatter(region_table["gross_yield_pct"],
                         region_table["p_terminal_loss_10_avg"], c=colours, s=sizes, alpha=0.8, edgecolors=INK)
    del scatter
    for _, row in region_table.iterrows():
        if row["region"] == selected_region:
            ax.annotate(row["region"], (row["gross_yield_pct"], row["p_terminal_loss_10_avg"]), xytext=(
                6, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Current gross yield (%)")
    ax.set_ylabel("P(terminal loss >10%)")
    ax.set_title("Yield versus downside risk", fontsize=12, loc="left")
    _style_axes(ax)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    return fig


def terminal_distribution_chart(paths: pd.DataFrame, start_price: float, title: str):
    terminal_returns = (paths.iloc[-1] / start_price - 1.0) * 100.0
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(terminal_returns, bins=35, color=SKY, edgecolor="white")
    ax.axvline(np.median(terminal_returns), color=INK, linestyle="--", lw=2)
    ax.set_xlabel("5-year total return (%)")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=12, loc="left")
    _style_axes(ax)
    fig.tight_layout()
    return fig


def tail_risk_heatmap(simulation: pd.DataFrame):
    pivot = simulation.pivot(
        index="region", columns="scenario", values="prob_terminal_loss_10pct")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r",
                   aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Regional downside heatmap", fontsize=12, loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(loss >10%)")
    fig.tight_layout()
    return fig


_COMPARISON_COLOURS = ["#12355b", "#0f766e", "#b45309", "#6d28d9"]

_METRIC_COLUMN: dict[str, str | None] = {
    "Nominal House Price Index (2005=100)": "nominal_house_price",
    "Real House Price Index (2005=100)": "real_house_price",
    "Fair Value Gap (%)": "fair_value_gap_log",
    "Downside Probability (%)": None,
    "Rental Yield (%)": "gross_yield_pct",
}

_PERIOD_START: dict[str, str] = {
    "Full history (2005\u2013present)": "2005-01-01",
    "Post-GFC (2010\u2013present)": "2010-01-01",
    "Post-pandemic (2020\u2013present)": "2020-01-01",
}


def plot_region_comparison(
    regions: list[str],
    metric: str,
    period: str,
    panel_df: pd.DataFrame,
):
    """Plot selected regions on a single figure for the chosen metric and period.

    Returns a Matplotlib figure.
    """
    col = _METRIC_COLUMN.get(metric)
    fig, ax = plt.subplots(figsize=(10, 5.2))

    if col is None:
        ax.text(
            0.5,
            0.5,
            f'"{metric}" is not available as a historical time series.\n'
            "Use the Scenario Lab for simulation-based downside estimates.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color=SLATE,
        )
        ax.set_title(metric, fontsize=12, loc="left")
        _style_axes(ax)
        fig.tight_layout()
        return fig

    if col not in panel_df.columns:
        ax.text(
            0.5,
            0.5,
            f'Column "{col}" not found in panel data.',
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color=SLATE,
        )
        ax.set_title(metric, fontsize=12, loc="left")
        _style_axes(ax)
        fig.tight_layout()
        return fig

    start_date = pd.Timestamp(_PERIOD_START.get(period, "2005-01-01"))
    is_index_metric = col in ("nominal_house_price", "real_house_price")
    is_log_gap = col == "fair_value_gap_log"

    _ci_lower_col = "fair_value_gap_lower_90"
    _ci_upper_col = "fair_value_gap_upper_90"
    _has_ci_band = is_log_gap and _ci_lower_col in panel_df.columns and _ci_upper_col in panel_df.columns

    for i, region in enumerate(regions[:4]):
        colour = _COMPARISON_COLOURS[i % len(_COMPARISON_COLOURS)]
        subset = (
            panel_df[(panel_df["region"] == region) &
                     (panel_df["date"] >= start_date)]
            .sort_values("date")
            .dropna(subset=[col])
        )
        if subset.empty:
            continue

        values = subset[col].copy()
        if is_index_metric:
            base = values.iloc[0]
            if base != 0:
                values = values / base * 100.0
        elif is_log_gap:
            values = values * 100.0

        ax.plot(subset["date"], values, color=colour, lw=2, label=region)

        # Optional 90% CI shaded band — only rendered for fair value gap when CI
        # columns exist in the data.  Alpha=0.15 keeps the band unobtrusive.
        if _has_ci_band:
            _ci_subset = subset.dropna(subset=[_ci_lower_col, _ci_upper_col])
            if not _ci_subset.empty:
                ax.fill_between(
                    _ci_subset["date"],
                    _ci_subset[_ci_lower_col] * 100.0,
                    _ci_subset[_ci_upper_col] * 100.0,
                    color=colour,
                    alpha=0.15,
                    linewidth=0,
                )

    if is_index_metric:
        ax.axhline(100, color=GRID, lw=1, linestyle="--")
        ax.set_ylabel("Index (100 = start of period)")
    elif is_log_gap:
        ax.axhline(0, color=GRID, lw=1, linestyle="--")
        ax.set_ylabel("Fair value gap (approx. %)")
    else:
        ax.set_ylabel(metric)

    ax.set_xlabel("Date")
    ax.set_title(f"{metric} \u2014 regional comparison",
                 fontsize=12, loc="left")
    ax.legend(frameon=False, ncol=min(len(regions), 4))
    _style_axes(ax)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.3)
    fig.tight_layout()
    return fig


def historical_vs_simulated_distribution_chart(distribution: pd.DataFrame, region: str):
    subset = distribution[distribution["region"] == region].copy()
    ordered = ["p10", "p25", "p50", "p75", "p90"]
    subset["percentile"] = pd.Categorical(
        subset["percentile"], categories=ordered, ordered=True)
    subset = subset.sort_values("percentile")

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(subset["percentile"].astype(str), subset["historical_return_pct"],
            marker="o", lw=2, color=INK, label="Historical realised 5y")
    ax.plot(subset["percentile"].astype(str), subset["simulated_return_pct"],
            marker="o", lw=2, color=TEAL, label="Simulated baseline 5y")
    ax.axhline(0.0, color="#94a3b8", lw=1, linestyle="--")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("5-year return (%)")
    ax.set_title(
        f"{region}: historical vs simulated 5-year distribution", fontsize=12, loc="left")
    ax.legend(frameon=False)
    _style_axes(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    return fig
