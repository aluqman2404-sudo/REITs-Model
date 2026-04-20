"""Interactive Plotly charts for the Stage 7 research dashboard."""

from __future__ import annotations
from src.ui.formatters import band_colour
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

INK = "#12355b"
TEAL = "#0f766e"
SKY = "#90cdf4"
SLATE = "#4f5b66"
GRID = "#d9e2ec"


def _hex_to_rgba(hex_colour: str, alpha: float) -> str:
    """Convert a #rrggbb hex string to an rgba(...) CSS string."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


BG = "rgba(255,255,255,0)"

_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="IBM Plex Sans, sans-serif", color=SLATE, size=13),
    margin=dict(l=60, r=30, t=50, b=110),
    hoverlabel=dict(
        bgcolor="white",
        bordercolor=GRID,
        font_size=13,
        font_family="IBM Plex Sans, sans-serif",
    ),
    height=420,
)

_AXIS = dict(
    showgrid=True,
    gridcolor=GRID,
    gridwidth=0.6,
    zeroline=False,
    linecolor=GRID,
    tickfont=dict(color=SLATE),
    title_font=dict(color=SLATE),
)


def _apply_layout(fig: go.Figure, **extra) -> go.Figure:
    fig.update_layout(**{**_LAYOUT, **extra})
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    return fig


# ---------------------------------------------------------------------------
# Fan chart
# ---------------------------------------------------------------------------

def fan_chart(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    region: str,
    scenario_name: str,
    subtitle: str | None = None,
) -> go.Figure:
    future_months = pd.date_range(
        history["date"].max(), periods=len(forecast), freq="MS"
    )

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=history["date"],
        y=history["nominal_house_price"],
        name="Historical price",
        line=dict(color=INK, width=2),
        hovertemplate="<b>Historical</b><br>%{x|%b %Y}: £%{y:,.0f}<extra></extra>",
    ))

    # P10–P90 band
    fig.add_trace(go.Scatter(
        x=pd.concat([pd.Series(future_months), pd.Series(future_months[::-1])]),
        y=pd.concat([forecast["p90"], forecast["p10"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(144,205,244,0.20)",
        line=dict(width=0),
        name="10–90 range",
        hoverinfo="skip",
    ))

    # P25–P75 band
    fig.add_trace(go.Scatter(
        x=pd.concat([pd.Series(future_months), pd.Series(future_months[::-1])]),
        y=pd.concat([forecast["p75"], forecast["p25"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(49,130,206,0.18)",
        line=dict(width=0),
        name="25–75 range",
        hoverinfo="skip",
    ))

    # Median path
    fig.add_trace(go.Scatter(
        x=future_months,
        y=forecast["p50"],
        name="Median path",
        line=dict(color=TEAL, width=2.5),
        hovertemplate="<b>Median</b><br>%{x|%b %Y}: £%{y:,.0f}<extra></extra>",
    ))

    # P10 / P90 boundary lines for hover
    fig.add_trace(go.Scatter(
        x=future_months, y=forecast["p10"],
        name="P10", line=dict(color=SKY, width=1, dash="dot"),
        hovertemplate="<b>P10</b><br>%{x|%b %Y}: £%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=future_months, y=forecast["p90"],
        name="P90", line=dict(color=SKY, width=1, dash="dot"),
        hovertemplate="<b>P90</b><br>%{x|%b %Y}: £%{y:,.0f}<extra></extra>",
    ))

    _subtitle = subtitle if subtitle is not None else (
        f"Scenario lab path under {scenario_name}. Fan widths show simulated uncertainty, not forecast certainty."
    )
    _apply_layout(
        fig,
        title=dict(text=f"{region}: historical and simulated price path", font=dict(size=13, color=INK), x=0),
        xaxis_title="Date",
        yaxis_title="Nominal average price (GBP)",
        legend=dict(orientation="h", y=-0.22, x=0),
        annotations=[dict(
            text=_subtitle,
            xref="paper", yref="paper",
            x=0, y=-0.36,
            showarrow=False,
            font=dict(size=11, color=SLATE),
            align="left",
        )],
        hovermode="x unified",
        height=480,
        margin=dict(l=60, r=30, t=50, b=140),
    )
    return fig


# ---------------------------------------------------------------------------
# Score breakdown (horizontal bar)
# ---------------------------------------------------------------------------

def score_breakdown_chart(component_scores: dict[str, float], title: str) -> go.Figure:
    labels = [name.replace("_", " ").title() for name in component_scores]
    values = list(component_scores.values())
    colours = ["#1f7a45" if v >= 67 else "#b26a00" if v >= 40 else "#a63f3f" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colours,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.0f}/100<extra></extra>",
    ))
    _apply_layout(
        fig,
        title=dict(text=title, font=dict(size=13, color=INK), x=0),
        xaxis=dict(range=[0, 115], title="Score (0–100)", **_AXIS),
        bargap=0.35,
    )
    return fig


# ---------------------------------------------------------------------------
# Affordability gauge (single horizontal bar)
# ---------------------------------------------------------------------------

def affordability_gauge(score: float, title: str) -> go.Figure:
    colour = "#1f7a45" if score >= 67 else "#b26a00" if score >= 40 else "#a63f3f"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[100], y=[""], orientation="h",
        marker_color="#e5e7eb", showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Bar(
        x=[score], y=[""], orientation="h",
        marker_color=colour, showlegend=False,
        text=[f"{score:.0f}/100"], textposition="outside",
        hovertemplate=f"Affordability score: {score:.0f}/100<extra></extra>",
    ))
    _apply_layout(
        fig,
        title=dict(text=title, font=dict(size=13, color=INK), x=0),
        xaxis=dict(range=[0, 120], title="Affordability", **_AXIS),
        yaxis=dict(visible=False),
        barmode="overlay",
        height=180,
        margin=dict(l=60, r=30, t=50, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Scenario comparison (horizontal bars + error bars + colour = downside risk)
# ---------------------------------------------------------------------------

def scenario_comparison_chart(sim_region: pd.DataFrame) -> go.Figure:
    table = sim_region.sort_values("median_5yr_growth").copy()
    loss_vals = table["prob_terminal_loss_10pct"].to_numpy()
    err_low = (table["median_5yr_growth"] - table["p10_5yr_growth"]).to_numpy()
    err_high = (table["p90_5yr_growth"] - table["median_5yr_growth"]).to_numpy()

    # Map loss probability to RdYlGn_r colour scale
    norm = (loss_vals - loss_vals.min()) / max(loss_vals.max() - loss_vals.min(), 1e-9)
    cmap = px.colors.diverging.RdYlGn
    n = len(cmap)
    colours = [cmap[min(int(v * (n - 1)), n - 1)] for v in norm]

    fig = go.Figure()
    for i, (scenario, med, elo, ehi, colour, loss) in enumerate(zip(
        table["scenario"], table["median_5yr_growth"],
        err_low, err_high, colours, loss_vals
    )):
        fig.add_trace(go.Bar(
            x=[med], y=[scenario],
            orientation="h",
            marker_color=colour,
            error_x=dict(type="data", symmetric=False, array=[ehi], arrayminus=[elo], color=INK, thickness=1.5, width=4),
            name=scenario,
            showlegend=False,
            hovertemplate=(
                f"<b>{scenario}</b><br>"
                f"Median 5y return: {med:+.1f}%<br>"
                f"P10–P90: {med - elo:+.1f}% to {med + ehi:+.1f}%<br>"
                f"P(loss >10%): {loss:.0%}<extra></extra>"
            ),
        ))

    # Invisible colourscale trace for the colorbar
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            colorscale="RdYlGn",
            reversescale=True,
            cmin=float(loss_vals.min()),
            cmax=float(loss_vals.max()),
            colorbar=dict(
                title=dict(text="P(loss >10%)", side="right"),
                tickformat=".0%",
                thickness=12,
                len=0.8,
            ),
            showscale=True,
            color=[],
        ),
        showlegend=False,
        hoverinfo="skip",
    ))

    _apply_layout(
        fig,
        title=dict(text="Scenario comparison — bar colour = downside risk (red = higher)", font=dict(size=12, color=INK), x=0),
        xaxis_title="Median 5-year price change (%)",
        bargap=0.3,
    )
    return fig


# ---------------------------------------------------------------------------
# Regional rank chart (horizontal bar)
# ---------------------------------------------------------------------------

def regional_rank_chart(
    region_table: pd.DataFrame,
    score_col: str,
    selected_region: str,
    title: str,
) -> go.Figure:
    table = region_table.sort_values(score_col).copy()
    band_col = f"{score_col.split('_')[0]}_band"
    colours = [
        INK if region == selected_region else band_colour(label)
        for region, label in zip(table["region"], table[band_col])
    ]
    scores = table[score_col].tolist()

    fig = go.Figure(go.Bar(
        x=scores,
        y=table["region"].tolist(),
        orientation="h",
        marker_color=colours,
        hovertemplate="<b>%{y}</b><br>Score: %{x:.0f}/100<extra></extra>",
    ))
    _apply_layout(
        fig,
        title=dict(text=title, font=dict(size=13, color=INK), x=0),
        xaxis=dict(range=[0, 100], title="Score", **_AXIS),
        bargap=0.3,
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Yield vs risk scatter
# ---------------------------------------------------------------------------

def yield_risk_chart(region_table: pd.DataFrame, selected_region: str) -> go.Figure:
    sizes = [18 if region == selected_region else 10 for region in region_table["region"]]

    fig = go.Figure()
    for i, row in region_table.iterrows():
        is_selected = row["region"] == selected_region
        fig.add_trace(go.Scatter(
            x=[row["gross_yield_pct"]],
            y=[row["p_terminal_loss_10_avg"]],
            mode="markers+text" if is_selected else "markers",
            marker=dict(
                color=band_colour(row["reit_band"]),
                size=sizes[list(region_table.index).index(i)],
                line=dict(color=INK, width=1),
            ),
            text=[row["region"]] if is_selected else [],
            textposition="top right",
            name=row["region"],
            hovertemplate=(
                f"<b>{row['region']}</b><br>"
                f"Gross yield: {row['gross_yield_pct']:.1f}%<br>"
                f"P(loss >10%): {row['p_terminal_loss_10_avg']:.1%}<extra></extra>"
            ),
            showlegend=False,
        ))

    _apply_layout(
        fig,
        title=dict(text="Yield versus downside risk", font=dict(size=13, color=INK), x=0),
        xaxis_title="Current gross yield (%)",
        yaxis_title="P(terminal loss >10%)",
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# Terminal return distribution (histogram)
# ---------------------------------------------------------------------------

def terminal_distribution_chart(paths: pd.DataFrame, start_price: float, title: str) -> go.Figure:
    terminal_returns = (paths.iloc[-1] / start_price - 1.0) * 100.0
    median_val = float(np.median(terminal_returns))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=terminal_returns,
        nbinsx=35,
        marker_color=SKY,
        marker_line_color="white",
        marker_line_width=1,
        name="Simulated returns",
        hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(
        x=median_val,
        line=dict(color=INK, width=2, dash="dash"),
        annotation_text=f"Median: {median_val:.1f}%",
        annotation_position="top right",
        annotation_font=dict(color=INK, size=11),
    )
    _apply_layout(
        fig,
        title=dict(text=title, font=dict(size=13, color=INK), x=0),
        xaxis_title="5-year total return (%)",
        yaxis_title="Count",
    )
    return fig


# ---------------------------------------------------------------------------
# Tail risk heatmap
# ---------------------------------------------------------------------------

def tail_risk_heatmap(simulation: pd.DataFrame) -> go.Figure:
    pivot = simulation.pivot(
        index="region", columns="scenario", values="prob_terminal_loss_10pct"
    )

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        reversescale=True,
        zmin=0,
        zmax=1,
        colorbar=dict(title="P(loss >10%)", tickformat=".0%"),
        hovertemplate="<b>%{y}</b><br>Scenario: %{x}<br>P(loss >10%): %{z:.1%}<extra></extra>",
    ))
    _apply_layout(
        fig,
        title=dict(text="Regional downside heatmap", font=dict(size=13, color=INK), x=0),
        xaxis=dict(**_AXIS, tickangle=-25),
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Region comparison line chart
# ---------------------------------------------------------------------------

_COMPARISON_COLOURS = ["#12355b", "#0f766e", "#b45309", "#6d28d9"]

_METRIC_COLUMN: dict[str, str | None] = {
    "Nominal House Price Index (2005=100)": "nominal_house_price",
    "Real House Price Index (2005=100)": "real_house_price",
    "Fair Value Gap (%)": "fair_value_gap_log",
    "Downside Probability (%) — simulation only, no historical series": None,
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
) -> go.Figure:
    col = _METRIC_COLUMN.get(metric)
    fig = go.Figure()

    if col is None:
        fig.add_annotation(
            text=(
                f'"{metric}" is not available as a historical time series.<br>'
                "Use the Scenario Lab for simulation-based downside estimates."
            ),
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12, color=SLATE),
            align="center",
        )
        _apply_layout(fig, title=dict(text=metric, font=dict(size=13, color=INK), x=0))
        return fig

    if col not in panel_df.columns:
        fig.add_annotation(
            text=f'Column "{col}" not found in panel data.',
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12, color=SLATE),
        )
        _apply_layout(fig, title=dict(text=metric, font=dict(size=13, color=INK), x=0))
        return fig

    start_date = pd.Timestamp(_PERIOD_START.get(period, "2005-01-01"))
    is_index_metric = col in ("nominal_house_price", "real_house_price")
    is_log_gap = col == "fair_value_gap_log"

    # Default y_label in case all subsets are empty
    y_label = "Index (100 = start of period)" if is_index_metric else "Fair value gap (approx. %)" if is_log_gap else metric

    _ci_lower_col = "fair_value_gap_lower_90"
    _ci_upper_col = "fair_value_gap_upper_90"
    _has_ci_band = is_log_gap and _ci_lower_col in panel_df.columns and _ci_upper_col in panel_df.columns

    for i, region in enumerate(regions[:4]):
        colour = _COMPARISON_COLOURS[i % len(_COMPARISON_COLOURS)]
        subset = (
            panel_df[(panel_df["region"] == region) & (panel_df["date"] >= start_date)]
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

        hover_fmt = "%{y:.1f}" + ("%" if (is_log_gap or "%" in metric) else "")

        fig.add_trace(go.Scatter(
            x=subset["date"],
            y=values,
            name=region,
            line=dict(color=colour, width=2),
            hovertemplate=f"<b>{region}</b><br>%{{x|%b %Y}}: {hover_fmt}<extra></extra>",
        ))

        if _has_ci_band:
            _ci_subset = subset.dropna(subset=[_ci_lower_col, _ci_upper_col])
            if not _ci_subset.empty:
                fig.add_trace(go.Scatter(
                    x=pd.concat([_ci_subset["date"], _ci_subset["date"].iloc[::-1]]),
                    y=pd.concat([_ci_subset[_ci_upper_col] * 100.0, _ci_subset[_ci_lower_col].iloc[::-1] * 100.0]),
                    fill="toself",
                    fillcolor=_hex_to_rgba(colour, 0.12),
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    name=f"{region} 90% CI",
                ))

    if is_index_metric:
        fig.add_hline(y=100, line=dict(color=GRID, width=1, dash="dash"))
    elif is_log_gap:
        fig.add_hline(y=0, line=dict(color=GRID, width=1, dash="dash"))

    _period_short = period.split("(")[0].strip()
    _apply_layout(
        fig,
        title=dict(text=f"{metric} — {_period_short} — regional comparison", font=dict(size=13, color=INK), x=0),
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.22, x=0),
        height=460,
    )
    return fig


# ---------------------------------------------------------------------------
# Historical vs simulated distribution
# ---------------------------------------------------------------------------

def historical_vs_simulated_distribution_chart(distribution: pd.DataFrame, region: str) -> go.Figure:
    subset = distribution[distribution["region"] == region].copy()
    ordered = ["p10", "p25", "p50", "p75", "p90"]
    subset["percentile"] = pd.Categorical(subset["percentile"], categories=ordered, ordered=True)
    subset = subset.sort_values("percentile")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subset["percentile"].astype(str),
        y=subset["historical_return_pct"],
        name="Historical realised 5y",
        line=dict(color=INK, width=2),
        mode="lines+markers",
        marker=dict(size=7),
        hovertemplate="<b>Historical</b><br>%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=subset["percentile"].astype(str),
        y=subset["simulated_return_pct"],
        name="Simulated baseline 5y",
        line=dict(color=TEAL, width=2),
        mode="lines+markers",
        marker=dict(size=7),
        hovertemplate="<b>Simulated</b><br>%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dash"))

    _apply_layout(
        fig,
        title=dict(text=f"{region}: historical vs simulated 5-year distribution", font=dict(size=13, color=INK), x=0),
        xaxis_title="Percentile",
        yaxis_title="5-year return (%)",
        legend=dict(orientation="h", y=-0.22, x=0),
        hovermode="x unified",
        height=400,
    )
    return fig
