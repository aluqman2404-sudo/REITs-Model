"""Render functions for the Stage 7 Streamlit dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import datetime
import json

from src.core.config import DEFAULT_CONFIG_PATH, load_config
from src.core.logging_utils import get_logger, log_model_event
from src.scoring.engine import consumer_signal_dashboard, regional_market_signal_dashboard
from src.simulation.sde import run_monte_carlo, summarize_paths
from src.ui.charts import (
    fan_chart,
    historical_vs_simulated_distribution_chart,
    plot_region_comparison,
    regional_rank_chart,
    scenario_comparison_chart,
    score_breakdown_chart,
    tail_risk_heatmap,
    yield_risk_chart,
)
from src.ui.explainers import (
    consumer_commentary,
    consumer_risk_flags,
    override_warning,
    plain_signal_takeaways,
    reit_risk_flags,
    top_signal_contributors,
)
from src.ui.formatters import apply_app_style, format_currency, format_pct, render_badge, render_evidence_badge
from src.ui.loaders import (
    get_region_distribution_comparison,
    get_region_history,
    get_region_moment_comparison,
    get_region_record,
    get_region_simulation_summary,
    load_dashboard_data,
)
from src.ui.glossary import tooltip
from src.ui.state import ensure_state_defaults, reset_override_state

_logger = get_logger(__name__)

VALUATION_GAP_CLOSURE_UI = {
    "Baseline": 0.15,
    "Soft_Landing": 0.08,
    "Recovery_Boom": 0.05,
    "Rate_Shock": 0.45,
    "Stagflation": 0.40,
    "Base": 0.15,
    "Bull": 0.05,
    "Bear": 0.45,
}


def _live_terminal_percentiles(paths: pd.DataFrame, start_price: float) -> tuple[dict, float]:
    terminal_returns = (paths.iloc[-1].to_numpy() / start_price - 1.0) * 100.0
    percentiles = {
        "p10": float(np.percentile(terminal_returns, 10)),
        "p50": float(np.percentile(terminal_returns, 50)),
        "p90": float(np.percentile(terminal_returns, 90)),
        "start_price": float(start_price),
    }
    downside_prob = float(np.mean(terminal_returns <= -10.0))
    return percentiles, downside_prob


def _scenario_summary_row(region_summary: pd.DataFrame, scenario_name: str) -> dict | None:
    matched = region_summary[region_summary["scenario"] == scenario_name]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def _implied_annual_sigma_from_stage5(stage5_row: dict | pd.Series | None, horizon_years: int, fallback: float) -> float:
    if stage5_row is None:
        return fallback
    try:
        p10 = float(stage5_row["p10_5yr_growth"]) / 100.0
        p90 = float(stage5_row["p90_5yr_growth"]) / 100.0
        if p10 <= -0.99 or p90 <= -0.99:
            return fallback
        z_90 = 1.2815515655446004
        sigma_terminal = (np.log1p(p90) - np.log1p(p10)) / (2.0 * z_90)
        if not np.isfinite(sigma_terminal) or sigma_terminal <= 0:
            return fallback
        return float(max(0.0001, sigma_terminal / np.sqrt(max(float(horizon_years), 1.0 / 12.0))))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return fallback


def _scenario_parameters(
    region_row: dict,
    scenario_name: str,
    *,
    stage5_row: dict | pd.Series | None = None,
    rate_shift_bps: float = 0.0,
    sigma_multiplier: float = 1.0,
    drift_shift_pct: float = 0.0,
    horizon_years: int = 5,
) -> tuple[dict, dict]:
    config = load_config()
    scenario = config.scenarios[scenario_name]

    start_price = float(region_row["nominal_house_price"])
    current_rate = float(region_row.get(
        "r_current", region_row.get("mortgage_rate", 5.25)))
    target_rate = max(0.5, current_rate +
                      (scenario.rate_shift_bps + rate_shift_bps) / 100.0)
    rate_channel_mode = str(region_row.get(
        "rate_channel_mode", "rate_level_anchor"))
    alpha = float(region_row.get("alpha_used", 0.5))
    r_bar = float(region_row.get("r_bar", 3.54))
    p_star_base = float(region_row.get(
        "P_star_base", region_row.get("mu_equilibrium", start_price)))
    current_fair_value = float(region_row.get("P_star_now", p_star_base))
    if rate_channel_mode in {"rate_change_plus_stress", "rate_change_plus_aux_stress", "rate_gap_plus_aux_stress"}:
        gap_current = float(region_row.get("valuation_gap_current", 0.0))
        gap_closure = VALUATION_GAP_CLOSURE_UI.get(scenario_name, 0.15)
        target_fair_value = start_price * np.exp(-gap_closure * gap_current)
        target_fair_value *= 1.0 + scenario.income_growth_shift_pct / \
            150.0 - scenario.supply_shift_pct / 400.0
    else:
        target_fair_value = p_star_base * (r_bar / target_rate) ** alpha
        target_fair_value *= 1.0 + scenario.income_growth_shift_pct / \
            100.0 - scenario.supply_shift_pct / 200.0
    if stage5_row is not None and "median_5yr_growth" in stage5_row:
        stage5_terminal = start_price * \
            (1.0 + float(stage5_row["median_5yr_growth"]) / 100.0)
        if np.isfinite(stage5_terminal) and stage5_terminal > 0:
            blend = 0.65 if rate_channel_mode in {
                "rate_change_plus_stress", "rate_change_plus_aux_stress", "rate_gap_plus_aux_stress"} else 0.50
            target_fair_value = (1.0 - blend) * \
                target_fair_value + blend * stage5_terminal
    target_fair_value = max(
        start_price * config.controls.minimum_price_floor_ratio, target_fair_value)

    drift_annual = float(region_row.get(
        "gamma_annual", region_row.get("gamma_annual_pp", 0.0)))
    if abs(drift_annual) > 1.0:
        drift_annual /= 100.0
    drift_annual += (scenario.drift_shift_pct + drift_shift_pct) / 100.0

    stage4_sigma_raw = float(region_row.get("sigma", 0.003))
    sigma_frequency = str(region_row.get("sigma_frequency", "monthly")).lower()
    if sigma_frequency == "annual":
        stage4_sigma_annual = stage4_sigma_raw
        stage4_sigma_monthly = stage4_sigma_raw / np.sqrt(12.0)
    else:
        stage4_sigma_monthly = stage4_sigma_raw
        stage4_sigma_annual = stage4_sigma_monthly * np.sqrt(12.0)
    hist_vol_annual = float(region_row.get(
        "hist_price_growth_vol_annual", stage4_sigma_annual))
    if not np.isfinite(hist_vol_annual) or hist_vol_annual <= 0:
        hist_vol_annual = stage4_sigma_annual
    sigma_implied_annual = _implied_annual_sigma_from_stage5(
        stage5_row, horizon_years, hist_vol_annual)
    sigma_annual = max(stage4_sigma_annual, 0.5 *
                       (hist_vol_annual + sigma_implied_annual))

    horizon_months = int(horizon_years * 12)
    mean_path = np.linspace(
        current_fair_value, target_fair_value, horizon_months + 1)
    params = {
        "kappa": float(region_row.get("kappa", 0.12)),
        "sigma": max(0.0001, sigma_annual * scenario.volatility_multiplier * sigma_multiplier),
        "drift_annual": drift_annual,
        "long_run_mean": mean_path,
    }
    meta = {
        "scenario_name": scenario.display_name,
        "target_rate": target_rate,
        "target_fair_value": target_fair_value,
        "stage4_sigma_monthly": stage4_sigma_monthly,
        "stage4_sigma_frequency": sigma_frequency,
        "sigma_annual": sigma_annual,
        "rate_channel_mode": rate_channel_mode,
        "note": scenario.note,
    }
    return params, meta


def _render_limitations_notice() -> None:
    st.caption(
        "Historical research panel only. The model combines historical calibration, scenario assumptions, "
        "and simulation outputs. It does not provide current real-time UK housing advice and does not observe idiosyncratic property quality, financing structure, or policy shocks in real time."
    )


def _direction_label(bucket: str) -> str:
    return {
        "Supportive": "Favourable",
        "Mixed": "Neutral",
        "Cautious": "Stretched",
    }.get(bucket, "Neutral")


def _evidence_label(raw: str) -> str:
    return {
        "empirically_supported": "Supported",
        "partially_supported": "Partial",
        "descriptive": "Descriptive",
    }.get(raw, raw.replace("_", " ").title())


def _confidence_lead(raw: str) -> str:
    return {
        "empirically_supported": "Validated",
        "partially_supported": "Partial",
        "descriptive": "Context only",
    }.get(raw, raw.replace("_", " ").title())


def _signal_confidence_summary(signals: list[dict]) -> str:
    counts = {
        "empirically_supported": 0,
        "partially_supported": 0,
        "descriptive": 0,
    }
    for signal in signals:
        evidence_level = str(signal.get("evidence_level", "descriptive"))
        counts[evidence_level] = counts.get(evidence_level, 0) + 1
    return (
        f"Confidence mix: {counts.get('empirically_supported', 0)} validated, "
        f"{counts.get('partially_supported', 0)} partial, "
        f"{counts.get('descriptive', 0)} contextual. "
        "Read validated signals first and use contextual signals as framing rather than proof."
    )


def _render_research_header(data, *, compact: bool = False, show_title: bool = True) -> None:
    model_start = pd.to_datetime(data.metadata["data_coverage_start"]).year
    model_end = pd.to_datetime(data.metadata["model_panel_coverage_end"]).year
    descriptive_end = pd.to_datetime(
        data.metadata["latest_market_data_end"]).year
    if show_title:
        st.markdown("### Housing Market Research Dashboard")
        st.caption(
            "Model estimated on the harmonised panel. Newer descriptive market data are shown separately and are not used for Stage 4 estimation."
        )
    cols = st.columns(3)
    for _col, (_lbl, _val) in zip(cols, [
        ("Model coverage window", f"{model_start}–{model_end}"),
        ("Latest market data", str(descriptive_end)),
        ("Last pipeline rebuild", data.metadata["latest_model_run_utc"][:10]),
    ]):
        _col.markdown(
            f"""<div style="background:rgba(255,255,255,0.82);border:1px solid #d3dce5;
                border-radius:14px;padding:12px 16px;">
              <div style="font-size:0.76rem;color:#51606f;margin-bottom:3px;">{_lbl}</div>
              <div style="font-size:1.0rem;font-weight:600;color:#132238;">{_val}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    if not compact:
        st.caption(
            f"Latest descriptive housing context: house prices through {data.metadata.get('latest_house_price_end', 'n/a')} "
            f"and rent context through {data.metadata.get('latest_rental_level_end', 'n/a')}."
        )


def _signal_display_value(signal: dict, region_row: dict | None = None) -> str:
    if region_row is None:
        return signal["simple_status"]
    if signal["id"] == "valuation":
        pct = float(region_row.get("valuation_stretch_percentile", np.nan))
        if np.isfinite(pct):
            if pct >= 75:
                return "Historically stretched"
            if pct <= 25:
                return "Supportive vs history"
            return "Around long-run neutral"
    if signal["id"] in {"cycle", "cycle_state"}:
        pct = float(region_row.get("cycle_strength_percentile", np.nan))
        if np.isfinite(pct):
            if pct >= 75:
                return "Firming vs history"
            if pct <= 25:
                return "Cooling vs history"
            return "Mixed cycle backdrop"
    if signal["id"] in {"downside", "downside_vulnerability"}:
        pct = float(region_row.get(
            "downside_vulnerability_percentile", np.nan))
        if np.isfinite(pct):
            if pct >= 75:
                return "Top quartile vulnerability"
            if pct <= 25:
                return "Bottom quartile vulnerability"
            return "Middle-range vulnerability"
    if signal["id"] in {"yield", "rental_support"}:
        pct = float(region_row.get("rent_support_percentile", np.nan))
        if np.isfinite(pct):
            if pct >= 75:
                return "Top quartile rent support"
            if pct <= 25:
                return "Bottom quartile rent support"
            return "Middle-range rent support"
    return signal["simple_status"]


def _colour_signal_direction(direction: str) -> str:
    return {
        "Favourable": "background-color:#e8f5e9; color:#2e7d32",
        "Neutral": "background-color:#fff9c4; color:#795548",
        "Stretched": "background-color:#fce4ec; color:#b71c1c",
    }.get(direction, "")


def _render_signal_evidence_df(signals: list[dict], *, include_definition: bool = False) -> None:
    df = _signal_evidence_table(signals, include_definition=include_definition)
    if "Direction" in df.columns:
        try:
            styled = df.style.map(_colour_signal_direction, subset=["Direction"])
        except AttributeError:
            styled = df.style.applymap(_colour_signal_direction, subset=["Direction"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_signal_dashboard(signals: list[dict], *, region_row: dict | None = None) -> None:
    st.caption(_signal_confidence_summary(signals))
    # Use 2-column grid regardless of signal count to avoid excessively narrow columns
    n_cols = min(len(signals), 2)
    rows = [signals[i:i + n_cols] for i in range(0, len(signals), n_cols)]
    for row_signals in rows:
        cols = st.columns(n_cols)
        for idx, signal in enumerate(row_signals):
            display_val = _signal_display_value(signal, region_row)
            badge_html = f"{render_badge(signal['bucket'])} {render_evidence_badge(signal['evidence_level'])}"
            confidence_line = f"{_confidence_lead(signal['evidence_level'])} signal | {_direction_label(signal['bucket'])}"
            card_html = f"""
<div style="background:rgba(255,255,255,0.82);border:1px solid #d3dce5;border-radius:14px;
            padding:16px 18px;min-height:170px;margin-bottom:12px;">
  <div style="font-size:0.78rem;color:#51606f;margin-bottom:4px;">{signal['display_name']}</div>
  <div style="font-size:1.08rem;font-weight:600;color:#132238;margin-bottom:8px;line-height:1.35;">
    {display_val}
  </div>
  <div style="margin-bottom:8px;">{badge_html}</div>
  <div style="font-size:0.76rem;color:#51606f;margin-bottom:4px;">{signal['question']}</div>
  <div style="font-size:0.73rem;color:#51606f;">{confidence_line}</div>
  <div style="font-size:0.73rem;color:#51606f;">Internal score: {signal['score']:.0f} / 100</div>
</div>"""
            cols[idx].markdown(card_html, unsafe_allow_html=True)


def _signal_scores_for_chart(signals: list[dict]) -> dict[str, float]:
    return {signal["display_name"]: float(signal["score"]) for signal in signals}


def _signal_evidence_table(signals: list[dict], *, include_definition: bool = False) -> pd.DataFrame:
    rows = []
    for signal in signals:
        row = {
            "Signal": signal["display_name"],
            "Plain-English view": signal["simple_status"],
            "Score": signal["score"],
            "Direction": _direction_label(signal["bucket"]),
            "Evidence": _evidence_label(signal["evidence_level"]),
        }
        if include_definition:
            row["Why it matters"] = signal["definition"]
        rows.append(row)
    return pd.DataFrame(rows)


def _headline_insight(region: str, dashboard: dict, *, institutional: bool = False) -> str:
    positives, negatives = top_signal_contributors(dashboard["signals"])
    valuation = next(
        signal for signal in dashboard["signals"] if signal["id"] == "valuation")
    cycle = next(
        signal for signal in dashboard["signals"] if signal["id"] == "cycle")
    framing = "institutional" if institutional else "household"
    return (
        f"{region}: valuation is {_direction_label(valuation['bucket']).lower()} while cycle conditions are "
        f"{_direction_label(cycle['bucket']).lower()} in the current {framing} research view. "
        f"Main support comes from {positives[0]}; the softest area is {negatives[-1]}."
    )


def _regional_house_view_summary(data, region_row: dict, dashboard: dict) -> str:
    valuation_pct = float(region_row.get(
        "valuation_stretch_percentile", np.nan))
    downside_pct = float(region_row.get(
        "downside_vulnerability_percentile", np.nan))
    cycle_pct = float(region_row.get("cycle_strength_percentile", np.nan))
    valuation_read = (
        "stretched versus local history"
        if np.isfinite(valuation_pct) and valuation_pct >= 75
        else "supportive versus local history"
        if np.isfinite(valuation_pct) and valuation_pct <= 25
        else "around long-run neutral"
    )
    cycle_read = (
        "stabilising"
        if np.isfinite(cycle_pct) and cycle_pct >= 60
        else "still cooling"
        if np.isfinite(cycle_pct) and cycle_pct <= 40
        else "mixed"
    )
    downside_read = (
        "elevated downside vulnerability"
        if np.isfinite(downside_pct) and downside_pct >= 75
        else "contained downside vulnerability"
        if np.isfinite(downside_pct) and downside_pct <= 25
        else "middle-range downside vulnerability"
    )
    return (
        f"Market read: {region_row['region']} looks {valuation_read}, with the cycle {cycle_read}. "
        f"Scenario diagnostics point to {downside_read}. "
        f"This read is anchored on the model panel through {data.metadata['model_panel_coverage_end']} and later market data are descriptive only."
    )


def _topline_market_read(data) -> str:
    table = data.region_table.copy()
    stretched = int((table["valuation_stretch_percentile"] >= 75).sum())
    supportive = int((table["valuation_stretch_percentile"] <= 25).sum())
    vulnerable = int((table["downside_vulnerability_percentile"] >= 75).sum())
    total = len(table)
    latest_context = data.metadata.get(
        "latest_complete_housing_context_end", data.metadata.get("latest_market_data_end", "n/a"))
    return (
        f"**Valuation:** stretched in {stretched}/{total} regions · supportive in {supportive}/{total} regions\n\n"
        f"**Downside vulnerability:** elevated in {vulnerable}/{total} regions (baseline scenario)\n\n"
        f"**Data backbone:** model panel through {data.metadata['model_panel_coverage_end']}; "
        f"descriptive market context to {latest_context} (not re-estimated)"
    )


def _render_secondary_synthesis_box(secondary: dict) -> None:
    with st.expander("Optional signal synthesis", expanded=False):
        st.markdown(
            f"""<div style="background:rgba(255,255,255,0.82);border:1px solid #d3dce5;
                border-radius:14px;padding:12px 16px;margin-bottom:8px;">
              <div style="font-size:0.76rem;color:#51606f;margin-bottom:3px;">Secondary synthesis</div>
              <div style="font-size:1.0rem;font-weight:600;color:#132238;">{secondary['score']:.0f} / 100</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{secondary['bucket']} synthesis of the underlying signals. "
            "This is secondary orientation context rather than the primary analytical frame."
        )


def _render_key_limitations_block(_data=None) -> None:
    st.caption(
        "See the **Limitations** page in the sidebar for the full model limitations register (L1–L22)."
    )


def _render_validation_status(data, *, compact: bool = False) -> None:
    validation = data.metadata.get("validation_summary", {})
    subsample = validation.get("subsample_holdout", {})
    benchmarks = validation.get("forecast_benchmarks", {})
    rate_stability = validation.get("rate_stability", {})
    if not validation:
        return

    model_rmse = benchmarks.get("model", {}).get("rmse")
    zero_rmse = benchmarks.get("zero", {}).get("rmse")
    ar1_rmse = benchmarks.get("ar1", {}).get("rmse")
    all_stable = bool(rate_stability.get("all_regions_stable", False))
    flagged_regions = int(rate_stability.get(
        "flagged_unstable_regions", 0) or 0)
    rate_text = (
        f"{flagged_regions} flagged unstable regions"
        if flagged_regions
        else "0 flagged unstable regions"
    )

    if compact:
        compact_parts = [f"Rate-channel stability: {rate_text}."]
        if model_rmse is not None and ar1_rmse is not None:
            compact_parts.append(
                f"Forecast benchmark: model RMSE {model_rmse:.4f} versus AR(1) {ar1_rmse:.4f}."
            )
        if model_rmse is not None and zero_rmse is not None:
            compact_parts.append(
                f"Zero-growth remains tougher at {zero_rmse:.4f} versus model {model_rmse:.4f}."
            )
        if subsample.get("directional_1m") is not None:
            compact_parts.append(
                f"Directional accuracy is {subsample['directional_1m']:.1%} at 1 month."
            )
        st.info("Validation status: " + " ".join(compact_parts))
        return

    st.subheader("Validation Status")
    metric_cols = st.columns(4)
    _validation_items = [
        ("Rate stability", "All regions stable" if all_stable else f"{flagged_regions} flagged",
         "Whether the mortgage rate channel is stable across all 12 regions. Instability can cause simulation paths to diverge unexpectedly."),
        ("1m directional accuracy", f"{subsample['directional_1m']:.1%}" if subsample.get("directional_1m") is not None else "n/a",
         "Fraction of 1-month-ahead price direction calls (up/down) that matched realised outcomes in the holdout period. >55% is considered economically meaningful."),
        ("Model RMSE", f"{model_rmse:.4f}" if model_rmse is not None else "n/a",
         "Root mean squared error of the OLS model's 1-month-ahead price growth forecast on the holdout panel. Lower is better."),
        ("Zero-growth RMSE", f"{zero_rmse:.4f}" if zero_rmse is not None else "n/a",
         "RMSE of a naive zero-growth benchmark (always predict 0% price change). The model should beat this to add forecasting value."),
    ]
    for col, (label, value, tooltip_text) in zip(metric_cols, _validation_items):
        col.markdown(
            f"""<div style="background:rgba(255,255,255,0.82);border:1px solid #d3dce5;
                border-radius:14px;padding:14px 16px;min-height:90px;" title="{tooltip_text}">
              <div style="font-size:0.78rem;color:#51606f;margin-bottom:4px;">{label}</div>
              <div style="font-size:1.05rem;font-weight:600;color:#132238;line-height:1.3;">{value}</div>
              <div style="font-size:0.70rem;color:#8a9bb0;margin-top:6px;line-height:1.3;">{tooltip_text}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    if model_rmse is not None and zero_rmse is not None and model_rmse <= zero_rmse:
        st.success(
            "Holdout check: the deployed model matches or beats the zero-growth benchmark on RMSE.",
            icon="✅",
        )
    else:
        st.warning(
            "Holdout check: the deployed model now has stable rate-channel behaviour and beats AR(1), but it still trails the zero-growth benchmark on RMSE. Treat it as a governed valuation-and-scenario framework rather than a signed-off forecasting engine.",
            icon="⚠️",
        )

    benchmark_rows = []
    for benchmark_name in ("model", "zero", "mean12", "ar1"):
        benchmark = benchmarks.get(benchmark_name)
        if not benchmark:
            continue
        benchmark_rows.append(
            {
                "Benchmark": benchmark_name,
                "RMSE": benchmark.get("rmse"),
                "MAE": benchmark.get("mae"),
                "Forecasts": benchmark.get("n_forecasts"),
            }
        )
    if benchmark_rows:
        st.dataframe(pd.DataFrame(benchmark_rows),
                     use_container_width=True, hide_index=True)
        st.caption(
            "The model currently outperforms AR(1) on RMSE and MAE, while zero-growth remains the toughest benchmark in the post-rate-shock holdout."
        )


def _render_everyday_summary(region: str, dashboard: dict) -> None:
    st.caption(f"Plain-English read for {region} — does not replace the research signals below.")
    for takeaway in plain_signal_takeaways(dashboard["signals"]):
        st.markdown(f"- {takeaway}")


def _render_simulation_transparency(region: str) -> None:
    distribution = get_region_distribution_comparison(region)
    moments = get_region_moment_comparison(region)
    st.plotly_chart(historical_vs_simulated_distribution_chart(distribution, region), use_container_width=True)
    st.caption(
        "Historical versus simulated 5-year outcome distribution. This is a calibration view for plausibility, not a claim of forecast superiority."
    )
    table = pd.DataFrame(
        [
            ("Historical mean 5y return",
             f"{moments['historical_mean_pct']:.1f}%"),
            ("Historical median 5y return",
             f"{moments['historical_median_pct']:.1f}%"),
            ("Simulated median 5y return",
             f"{moments['simulated_median_pct']:.1f}%"),
            ("Historical P10 / P90",
             f"{moments['historical_p10_pct']:.1f}% / {moments['historical_p90_pct']:.1f}%"),
            ("Simulated P10 / P90",
             f"{moments['simulated_p10_pct']:.1f}% / {moments['simulated_p90_pct']:.1f}%"),
            ("Historical P(loss >10%)",
             f"{moments['historical_prob_loss_10pct']:.0%}"),
            ("Simulated P(loss >10%)",
             f"{moments['simulated_prob_loss_10pct']:.0%}"),
            ("Tail-risk gap", f"{moments['tail_risk_gap']:.2f}"),
        ],
        columns=["Calibration check", "Value"],
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "Sources: canonical model panel and baseline Stage 5 simulation summary.")


def render_home() -> None:
    ensure_state_defaults()
    apply_app_style()
    data = load_dashboard_data()
    config = load_config()

    _render_research_header(data, show_title=False)
    st.markdown("## UK Housing Market Research Overview")
    st.caption(
        "Regional UK housing valuation, vulnerability, and scenario research platform. Valuation and downside context come first; forecasting is secondary."
    )
    st.markdown("### House View Summary")
    st.info(_topline_market_read(data))
    spotlight_region = st.selectbox(
        "Spotlight region",
        config.project.regions,
        index=config.project.regions.index(config.ui.default_region),
        key="home_spotlight_region",
    )
    region_row = get_region_record(spotlight_region)
    region_summary = get_region_simulation_summary(spotlight_region)
    home_dashboard = regional_market_signal_dashboard(
        spotlight_region,
        valuation_signal=float(region_row.get(
            "C1_mispricing", region_row["consumer_score"])),
        cycle_signal=float(region_row.get(
            "C3_regime", region_row.get("C2_reit", region_row["reit_score"]))),
        downside_prob=float(region_row.get("p_terminal_loss_10_avg", 0.0)),
        rent_support_signal=float(region_row.get(
            "rent_support_percentile", 50.0)),
    )
    st.markdown(f"> {_regional_house_view_summary(data, region_row, home_dashboard)}")

    st.markdown("### Market Snapshot")
    _rent_raw = region_row.get("latest_monthly_rent_raw")
    _rent_display = format_currency(float(_rent_raw)) if _rent_raw and float(_rent_raw) > 0 else "n/a"
    _fv_gap_pct = float(region_row.get("fair_value_gap_log", region_row.get("pct_above_pstar", 0.0))) * 100.0
    _fv_gap_str = f"{_fv_gap_pct:+.1f}%" if np.isfinite(_fv_gap_pct) else "n/a"
    snapshot = st.columns(4)
    for _col, (_lbl, _val) in zip(snapshot, [
        ("Spotlight house price", format_currency(float(region_row["latest_house_price"]))),
        ("Spotlight rent (monthly)", _rent_display),
        ("Fair value gap", _fv_gap_str),
        ("Latest descriptive data", data.metadata.get("latest_market_data_end", "n/a")),
    ]):
        _col.markdown(
            f"""<div style="background:rgba(255,255,255,0.82);border:1px solid #d3dce5;
                border-radius:14px;padding:12px 16px;">
              <div style="font-size:0.76rem;color:#51606f;margin-bottom:3px;">{_lbl}</div>
              <div style="font-size:1.0rem;font-weight:600;color:#132238;">{_val}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.caption(
        "The model is estimated on the harmonised panel through the model coverage end date. Later house-price and rent observations are shown here as descriptive context only."
    )
    # --- Persistent data-currency disclosure (L1) ---
    _effective = data.metadata.get(
        "effective_data_as_of",
        data.metadata.get("model_panel_coverage_end", "unknown"),
    )
    try:
        from datetime import date as _date
        _age = (
            _date.today() -
            _date.fromisoformat(str(_effective)[:10])
        ).days
    except (ValueError, TypeError):
        _age = 0
    _currency_msg = (
        f"Signals anchored to **{_effective}**. Panel age: **{_age} days**. "
        "All outputs reflect research conditions at that date, not current prices. "
        "See the **Limitations** page for full disclosure."
    )
    if _age > 180:
        st.error(_currency_msg, icon="🚨")
    elif _age > 90:
        st.warning(_currency_msg, icon="⚠️")
    else:
        st.success(_currency_msg, icon="✅")
    st.markdown("### Key Housing Signals")
    st.caption("Primary analytical view for the selected region. Stronger-evidence signals should carry more weight than descriptive context.")
    st.warning(
        "Valuation and cycle signals use income and supply features estimated from "
        "annual data interpolated to monthly frequency. Timing precision within a "
        "calendar year should not be relied upon. See Model Limitations for detail.",
        icon="⚠️",
    )
    _render_signal_dashboard(home_dashboard["signals"], region_row=region_row)
    with st.expander("Signal breakdown and evidence detail", expanded=False):
        sig_left, sig_right = st.columns([1, 1])
        with sig_left:
            st.plotly_chart(score_breakdown_chart(_signal_scores_for_chart(
                home_dashboard["signals"]), f"{spotlight_region}: signal panel"), use_container_width=True)
            st.caption(
                "Signal summary by component. Valuation remains the strongest empirically supported indicator in the current framework.")
        with sig_right:
            _render_signal_evidence_df(home_dashboard["signals"], include_definition=True)

    st.markdown("### Risk Context")
    st.plotly_chart(scenario_comparison_chart(region_summary), use_container_width=True)
    st.caption(
        "Scenario range for the spotlight region, showing how medium-term price outcomes vary across the canonical scenario set.")
    with st.expander("Simulation calibration view", expanded=False):
        _render_simulation_transparency(spotlight_region)

    with st.expander("Cross-regional orientation", expanded=False):
        st.plotly_chart(regional_rank_chart(data.region_table, "consumer_score",
                  spotlight_region, "Cross-regional synthesis orientation"), use_container_width=True)
        st.caption("Cross-regional synthesis is shown for orientation only. Primary interpretation should come from the underlying signal panel.")
        st.plotly_chart(tail_risk_heatmap(data.simulation), use_container_width=True)
        st.caption(
            "Scenario downside map based on canonical Stage 5 outputs. This is a scenario-risk summary, not a forecast ranking.")

    with st.expander("Coverage and source detail", expanded=False):
        display = data.region_table[
            [
                "region",
                "date",
                "nominal_house_price",
                "latest_house_price_date",
                "latest_house_price",
                "gross_yield_pct",
                "consumer_score",
                "reit_score",
                "consumer_band",
                "reit_band",
            ]
        ].copy()
        display.columns = [
            "Region",
            "Latest harmonised data point",
            "Model-panel average price",
            "Latest raw HPI date",
            "Latest raw HPI price",
            "Gross yield (%)",
            "Consumer synthesis",
            "REIT synthesis",
            "Consumer bucket",
            "REIT bucket",
        ]
        st.dataframe(display, use_container_width=True)
        if data.metadata.get("source_coverage"):
            source_rows = []
            for source_name, source_meta in data.metadata["source_coverage"].items():
                source_rows.append(
                    {
                        "Source": source_name.replace("_", " ").title(),
                        "Observation start": source_meta.get("observation_start"),
                        "Observation end": source_meta.get("observation_end"),
                        "File": source_meta.get("source_file"),
                    }
                )
            st.caption("Source coverage varies by dataset; the harmonised panel ends where the canonical research build stops, not at the freshest single macro series.")
            st.dataframe(pd.DataFrame(source_rows), use_container_width=True)
        _render_validation_status(data, compact=True)

    _render_key_limitations_block(data)
    _render_limitations_notice()


def render_consumer_view() -> None:
    ensure_state_defaults()
    apply_app_style()
    data = load_dashboard_data()
    config = load_config()
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as _fh:
        household_explorer_defaults: dict[str, dict] = json.load(
            _fh).get("household_explorer_defaults", {})

    _render_research_header(data, compact=True)
    st.title("Household / Affordability Explorer")
    st.caption(
        "Plain-English household affordability and downside context. Secondary to the main research dashboard — not a buy/do-not-buy tool. "
        "Outputs are illustrative model estimates and do not constitute financial advice."
    )

    # Pre-fetch region_row so the mortgage rate slider can use the current model rate as default
    _default_region = config.ui.default_region
    region_row = get_region_record(_default_region)

    controls, results = st.columns([1, 1])
    with controls:
        st.markdown("**Property**")
        region = st.selectbox("Region", config.project.regions,
                              index=config.project.regions.index(_default_region))
        region_row = get_region_record(region)
        _region_defaults = household_explorer_defaults.get(region, {})
        _default_income = _region_defaults.get(
            "household_income", config.ui.default_income_gbp)
        _default_price = _region_defaults.get(
            "property_price", config.ui.default_property_price_gbp)
        target_price = st.number_input(
            "Target property price", min_value=50000, max_value=5000000, value=_default_price, step=10000)
        st.markdown("**Mortgage**")
        deposit = st.number_input("Deposit (GBP)", min_value=0, max_value=1000000,
                                  value=config.ui.default_deposit_gbp, step=5000)
        mortgage_term = st.slider("Mortgage term (years)", min_value=15,
                                  max_value=40, value=config.ui.default_mortgage_term_years)
        mortgage_rate_input = st.slider(
            "Mortgage rate (%)",
            min_value=1.0, max_value=10.0,
            value=round(float(region_row["mortgage_rate"]), 2),
            step=0.05,
            help="Current model rate pre-filled. Adjust to stress-test your monthly payment.",
        )
        st.markdown("**Household & analysis**")
        income = st.number_input("Gross annual household income (GBP)",
                                 min_value=15000, max_value=500000, value=_default_income, step=5000)
        risk_tolerance = st.select_slider(
            "Risk tolerance", options=config.ui.risk_tolerance_options, value="Balanced")
        scenario = st.selectbox("Macro scenario", options=data.metadata["scenario_names"], index=data.metadata["scenario_names"].index(
            "Baseline"), help=tooltip("scenario weights"))
        st.caption(
            "Defaults are approximate regional medians from ONS / MHCLG 2024. Adjust to match your situation.")
        inputs_valid = True
        if not (50000 <= target_price <= 5000000):
            st.error("Property price must be between £50,000 and £5,000,000")
            inputs_valid = False
        if not (15000 <= income <= 500000):
            st.error("Annual household income must be between £15,000 and £500,000")
            inputs_valid = False
        if target_price > 0:
            _deposit_pct = deposit / target_price * 100
            if _deposit_pct < 5 or _deposit_pct > 50:
                st.warning(
                    "Typical mortgage deposits range from 5% to 50% of the property price.")

    if not inputs_valid:
        return

    region_summary = get_region_simulation_summary(region)
    selected_summary = _scenario_summary_row(region_summary, scenario)

    history = get_region_history(region, months=84)
    sim_params, sim_meta = _scenario_parameters(
        region_row, scenario, stage5_row=selected_summary, horizon_years=5)
    log_model_event(_logger, "simulation_run", {
                    "region": region, "scenario": scenario, "n_paths": config.controls.default_paths_interactive, "view": "consumer"})
    with st.spinner("Running simulation…"):
        paths = run_monte_carlo(
            start_price=float(region_row["nominal_house_price"]),
            n_months=60,
            n_paths=config.controls.default_paths_interactive,
            params=sim_params,
            seed=config.controls.random_seed,
        )
    summary = summarize_paths(
        paths, percentiles=config.ui.fan_chart_percentiles)
    live_percentiles, live_downside_prob = _live_terminal_percentiles(
        paths, float(region_row["nominal_house_price"]))
    signal_dashboard = consumer_signal_dashboard(
        region,
        income,
        deposit,
        live_percentiles,
        valuation_signal=float(region_row.get(
            "C1_mispricing", region_row["consumer_score"])),
        cycle_signal=float(region_row.get(
            "C2_consumer", region_row["consumer_score"])),
        property_price=target_price or float(
            region_row["nominal_house_price"]),
        mortgage_rate=mortgage_rate_input,
        mortgage_term_years=mortgage_term,
        risk_tolerance=risk_tolerance,
        downside_prob=live_downside_prob,
    )

    st.markdown("### Headline View")
    st.info(_headline_insight(region, signal_dashboard))
    _render_everyday_summary(region, signal_dashboard)

    _loan_amount = max(0, target_price - deposit)
    _ltv_pct = (_loan_amount / target_price * 100.0) if target_price > 0 else 0.0
    _stress_level = signal_dashboard["mortgage_stress"]
    _stress_colour = {"High": "#b71c1c", "Elevated": "#e65100"}.get(_stress_level, "#2e7d32")

    with results:
        st.markdown("### Affordability Summary")
        _card_items = [
            ("Monthly payment", format_currency(signal_dashboard["mortgage_payment_monthly"]), "#132238"),
            ("Loan amount", format_currency(_loan_amount), "#132238"),
            ("LTV", f"{_ltv_pct:.1f}%", "#132238"),
            ("Mortgage rate", format_pct(mortgage_rate_input, precision=2), "#132238"),
            ("Mortgage stress", _stress_level, _stress_colour),
        ]
        meta_cols = st.columns(len(_card_items))
        for _col, (_lbl, _val, _colour) in zip(meta_cols, _card_items):
            _col.markdown(
                f"""<div style="background:rgba(255,255,255,0.82);border:1px solid #d3dce5;
                    border-radius:14px;padding:12px 16px;">
                  <div style="font-size:0.76rem;color:#51606f;margin-bottom:3px;">{_lbl}</div>
                  <div style="font-size:1.0rem;font-weight:600;color:{_colour};">{_val}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        st.markdown(consumer_commentary(region_row, signal_dashboard, scenario))
        latest_hpi_date = region_row.get("latest_house_price_date")
        latest_hpi_value = region_row.get("latest_house_price")
        if pd.notna(latest_hpi_date) and pd.notna(latest_hpi_value):
            st.caption(
                f"Latest raw HPI context: {format_currency(float(latest_hpi_value))} at {pd.to_datetime(latest_hpi_date).date()}. "
                f"Calibrated model signals still anchor on the harmonised {pd.to_datetime(region_row['date']).date()} model panel."
            )
        with st.expander("Technical signal inputs", expanded=False):
            st.caption(
                f"Interactive volatility input: {sim_params['sigma']:.1%} annualised "
                f"(Stage 4 monthly sigma {sim_meta['stage4_sigma_monthly']:.2%}; "
                f"rate channel mode: {sim_meta['rate_channel_mode'].replace('_', ' ')})."
            )
        _render_secondary_synthesis_box(
            signal_dashboard["secondary_composite"])

    st.markdown("### Supporting Signals")
    st.warning(
        "Valuation and cycle signals use income and supply features estimated from "
        "annual data interpolated to monthly frequency. Timing precision within a "
        "calendar year should not be relied upon.",
        icon="⚠️",
    )
    _render_signal_dashboard(signal_dashboard["signals"], region_row=region_row)

    st.markdown("### Supporting Charts")
    st.plotly_chart(fan_chart(history, summary, region, sim_meta["scenario_name"]), use_container_width=True)
    st.caption("Historical path and simulated scenario range for the selected region. The fan chart illustrates plausible dispersion around the calibrated baseline, not a single predicted path.")

    st.plotly_chart(score_breakdown_chart(_signal_scores_for_chart(
        signal_dashboard["signals"]), "Consumer signal dashboard"), use_container_width=True)
    st.caption(
        "Component view of the household signal panel, ordered to support interpretation rather than ranking.")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        _render_signal_evidence_df(signal_dashboard["signals"], include_definition=True)
    with bottom_right:
        st.plotly_chart(scenario_comparison_chart(region_summary), use_container_width=True)
        st.caption("Canonical scenario comparison for the selected region. This helps frame how sensitive the outlook is to different macro paths.")

    with st.expander("Simulation calibration view", expanded=False):
        _render_simulation_transparency(region)

    st.subheader("Key risks and caveats")
    for risk in consumer_risk_flags(region_row, signal_dashboard):
        st.markdown(f"- {risk}")
    st.caption(
        "The signal dashboard is the primary output. Valuation has the strongest historical support; cycle is partial; downside and affordability remain descriptive context."
    )
    _render_key_limitations_block(data)
    _render_limitations_notice()


def render_reit_view() -> None:
    ensure_state_defaults()
    apply_app_style()
    data = load_dashboard_data()
    config = load_config()
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as _fh:
        _ui_ref = json.load(_fh).get("ui_reference_values", {})
    _downside_gfc = _ui_ref.get("downside_prob_gfc_avg", 0.71)
    _downside_2022 = _ui_ref.get("downside_prob_2022_correction_avg", 0.48)

    _render_research_header(data, compact=True)
    st.title("Regional Signal Dashboard")
    st.caption("Analyst-facing regional valuation, cycle, and downside diagnostics. This is the core institutional research view.")
    st.warning(
        "REIT composite weights are expert-prior and have not been optimised against "
        "realised REIT investment outcomes. The yield signal has not been backtested against "
        "realised returns. Composite scores are research bucketing tools, not investment recommendations. "
        "See **Limitations** page for full disclosure.",
        icon="⚠️",
    )

    controls, results = st.columns([1, 1])
    with controls:
        region = st.selectbox("Region", config.project.regions, index=config.project.regions.index(
            config.ui.default_region), key="reit_region")
        scenario = st.selectbox("Scenario focus", options=data.metadata["scenario_names"], index=data.metadata["scenario_names"].index(
            "Baseline"), key="reit_scenario", help=tooltip("scenario weights"))

    region_row = get_region_record(region)
    region_summary = get_region_simulation_summary(region)
    selected_summary = _scenario_summary_row(region_summary, scenario)

    sim_params, sim_meta = _scenario_parameters(
        region_row, scenario, stage5_row=selected_summary, horizon_years=config.ui.default_holding_horizon_years)
    log_model_event(_logger, "simulation_run", {
                    "region": region, "scenario": scenario, "n_paths": config.controls.default_paths_interactive, "view": "reit"})
    with st.spinner("Running simulation…"):
        paths = run_monte_carlo(
            start_price=float(region_row["nominal_house_price"]),
            n_months=config.ui.default_holding_horizon_years * 12,
            n_paths=config.controls.default_paths_interactive,
            params=sim_params,
            seed=config.controls.random_seed,
        )
    history = get_region_history(region, months=84)
    summary = summarize_paths(
        paths, percentiles=config.ui.fan_chart_percentiles)
    live_percentiles, live_downside_prob = _live_terminal_percentiles(
        paths, float(region_row["nominal_house_price"]))
    signal_dashboard = regional_market_signal_dashboard(
        region,
        valuation_signal=float(region_row.get(
            "C1_mispricing", region_row["reit_score"])),
        cycle_signal=float(region_row.get(
            "C3_regime", region_row.get("C2_reit", region_row["reit_score"]))),
        downside_prob=live_downside_prob,
        rent_support_signal=float(region_row.get(
            "rent_support_percentile", 50.0)),
    )

    st.markdown("### Headline View")
    st.info(_regional_house_view_summary(data, region_row, signal_dashboard))

    with results:
        st.markdown("### Key Metrics")
        meta_cols = st.columns(2)
        for _col, (_lbl, _val) in zip(meta_cols, [
            ("Current gross yield", format_pct(float(region_row["gross_yield_pct"]), precision=1)),
            ("Avg. downside risk (10%+ loss)", format_pct(
                float(region_row.get("p_terminal_loss_10_avg", 0.0)) * 100.0, precision=0)),
        ]):
            _col.markdown(
                f"""<div style="background:rgba(255,255,255,0.82);border:1px solid #d3dce5;
                    border-radius:14px;padding:12px 16px;">
                  <div style="font-size:0.76rem;color:#51606f;margin-bottom:3px;">{_lbl}</div>
                  <div style="font-size:1.0rem;font-weight:600;color:#132238;">{_val}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        st.caption(
            f"For context: the model averaged {_downside_gfc:.0%} downside probability across regions "
            f"during the 2008\u201309 financial crisis and {_downside_2022:.0%} during the 2022\u201323 correction."
        )
        st.caption(
            f"{region} — valuation is the most decision-relevant signal, cycle is secondary, "
            f"and downside risk remains scenario-driven under {scenario}. "
            "Treat this as a research framework, not a stand-alone trade signal."
        )
        latest_hpi_date = region_row.get("latest_house_price_date")
        latest_hpi_value = region_row.get("latest_house_price")
        if pd.notna(latest_hpi_date) and pd.notna(latest_hpi_value):
            st.caption(
                f"Latest raw HPI context: {format_currency(float(latest_hpi_value))} at {pd.to_datetime(latest_hpi_date).date()}. "
                f"The calibrated model signals still use the harmonised {pd.to_datetime(region_row['date']).date()} panel state."
            )
        with st.expander("Technical signal inputs", expanded=False):
            st.caption(
                f"Interactive volatility input: {sim_params['sigma']:.1%} annualised "
                f"(Stage 4 monthly sigma {sim_meta['stage4_sigma_monthly']:.2%}; "
                f"rate channel mode: {sim_meta['rate_channel_mode'].replace('_', ' ')})."
            )

    st.markdown("### Supporting Signals")
    st.warning(
        "Valuation and cycle signals use income and supply features estimated from "
        "annual data interpolated to monthly frequency. Timing precision within a "
        "calendar year should not be relied upon. See **Methodology** for detail.",
        icon="⚠️",
    )
    _render_signal_dashboard(signal_dashboard["signals"], region_row=region_row)

    st.markdown("### Supporting Charts")
    st.plotly_chart(fan_chart(history, summary, region, sim_meta["scenario_name"]), use_container_width=True)
    st.caption("Historical path and simulated scenario range for the selected region. The fan chart is a calibrated scenario view, not a deterministic price forecast.")
    if region in ("Wales", "Yorkshire and The Humber"):
        st.warning(
            "IQR reliability note: Simulation fan chart width for this region is compressed relative to "
            "historical volatility (simulated IQR = 20\u201344% of historical). Downside probability estimates "
            "should be treated as indicative only. See Limitations page, L17 and L20."
        )

    st.plotly_chart(score_breakdown_chart(_signal_scores_for_chart(
        signal_dashboard["signals"]), "Regional signal dashboard"), use_container_width=True)
    st.caption(
        "Signal panel summary for the institutional lens. Component evidence should matter more than any blended summary.")

    left, right = st.columns(2)
    with left:
        _render_signal_evidence_df(signal_dashboard["signals"], include_definition=True)
    with right:
        st.plotly_chart(scenario_comparison_chart(region_summary), use_container_width=True)
        st.caption("Canonical scenario comparison for the selected region. Use this to frame sensitivity, not as a probability-weighted forecast table.")

    st.plotly_chart(yield_risk_chart(data.region_table, region), use_container_width=True)
    st.caption(
        "Current yield versus downside tail context across regions, with the selected region highlighted.")

    with st.expander("Simulation calibration view", expanded=False):
        _render_simulation_transparency(region)

    _csv_cols = ["date", "nominal_house_price", "real_house_price",
                 "rental_yield", "fair_value_gap", "downside_probability"]
    _chart_csv = history[[c for c in _csv_cols if c in history.columns]].copy()
    _region_slug = region.lower().replace(" ", "_")
    st.download_button(
        label="\U0001f4e5 Download chart data (CSV)",
        data=_chart_csv.to_csv(index=False).encode("utf-8"),
        file_name=f"{_region_slug}_{datetime.date.today()}_housing_signals.csv",
        mime="text/csv",
    )

    st.subheader("Key risks and caveats")
    for risk in reit_risk_flags(region_row, {"yield_gap_pct": float(region_row["gross_yield_pct"]) - 5.5, "signals": signal_dashboard["signals"]}):
        st.markdown(f"- {risk}")
    st.caption(
        "This page is the core institutional view. Valuation is strongest empirically; cycle is partial; rent support and downside remain contextual research signals."
    )
    _render_key_limitations_block(data)
    _render_limitations_notice()


def render_scenario_lab() -> None:
    ensure_state_defaults()
    apply_app_style()
    data = load_dashboard_data()
    config = load_config()
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as _fh:
        scenario_descriptions: dict[str, str] = json.load(
            _fh).get("scenario_descriptions", {})

    _render_research_header(data, compact=True)
    st.title("Scenario & Simulation Diagnostics")
    st.caption("Scenario sensitivity and simulation calibration diagnostics. This page is for analyst review rather than end-user interpretation.")

    left, right = st.columns([1, 1])
    with left:
        region = st.selectbox(
            "Region", config.project.regions, key="scenario_lab_region")
        preset = st.selectbox("Preset", options=config.ui.scenario_labels,
                              key="scenario_lab_preset", help=tooltip("scenario weights"))
        st.info(f"\U0001f4a1 {scenario_descriptions.get(preset, '')}")
        st.caption(
            "Note: Scenario paths are generated using expert-prior drift and volatility overlays applied to "
            "the baseline Ornstein-Uhlenbeck model. They do not represent re-estimated OLS coefficients "
            "under each macro regime."
        )
        rate_shift = st.slider("Additional rate shock (bps)", min_value=-250, max_value=250, value=int(
            st.session_state["override_rate_shift_bps"]), step=25, key="override_rate_shift_bps", help=tooltip("rate channel"))
        sigma_multiplier = st.slider("Volatility multiplier", min_value=0.5, max_value=2.0, value=float(
            st.session_state["override_sigma_multiplier"]), step=0.05, key="override_sigma_multiplier", help=tooltip("sigma"))
        drift_shift = st.slider("Annual drift override (%)", min_value=-3.0, max_value=3.0, value=float(
            st.session_state["override_drift_shift_pct"]), step=0.25, key="override_drift_shift_pct", help=tooltip("drift"))
        horizon_years = st.slider("Horizon (years)", min_value=2, max_value=15, value=int(
            st.session_state["override_horizon_years"]), key="override_horizon_years")
        n_paths = st.slider("Simulation paths", min_value=250, max_value=config.controls.max_paths_interactive, value=int(
            st.session_state["override_paths"]), step=250, key="override_paths", help=tooltip("downside probability"))
        st.caption(
            "Calibrated defaults: 0 bps rate shift · 1.0× volatility · 0% drift · 5-year horizon. "
            "More paths = smoother fan chart, slower render."
        )
        st.markdown("---")
        st.button("↺ Reset all controls to defaults", on_click=reset_override_state, use_container_width=True)

    region_row = get_region_record(region)
    region_summary = get_region_simulation_summary(region)
    selected_summary = _scenario_summary_row(region_summary, preset)
    params, meta = _scenario_parameters(
        region_row,
        preset,
        stage5_row=selected_summary,
        rate_shift_bps=rate_shift,
        sigma_multiplier=sigma_multiplier,
        drift_shift_pct=drift_shift,
        horizon_years=horizon_years,
    )
    warning_text = override_warning(
        rate_shift, sigma_multiplier, horizon_years)
    if warning_text:
        st.warning(warning_text)
        _logger.warning("Guardrail fired: region=%s, rate_shift_bps=%s, sigma_multiplier=%s, horizon_years=%s",
                        region, rate_shift, sigma_multiplier, horizon_years)

    log_model_event(
        _logger,
        "simulation_run",
        {
            "region": region,
            "scenario": preset,
            "n_paths": n_paths,
            "view": "scenario_lab",
            "rate_shift_bps": rate_shift,
            "sigma_multiplier": sigma_multiplier,
        },
    )
    with st.spinner("Running simulation…"):
        paths = run_monte_carlo(
            start_price=float(region_row["nominal_house_price"]),
            n_months=horizon_years * 12,
            n_paths=n_paths,
            params=params,
            seed=config.controls.random_seed,
        )
    summary = summarize_paths(
        paths, percentiles=config.ui.fan_chart_percentiles)
    history = get_region_history(region, months=84)

    _live_pcts, _live_dp = _live_terminal_percentiles(paths, float(region_row["nominal_house_price"]))

    with right:
        rate_metric_label = "Scenario rate assumption" if meta["rate_channel_mode"] in {
            "rate_change_plus_stress", "rate_change_plus_aux_stress", "rate_gap_plus_aux_stress"} else "Scenario rate anchor"
        _scen_metrics = [
            ("Target terminal fair value", format_currency(meta["target_fair_value"])),
            (rate_metric_label, format_pct(meta["target_rate"], precision=2)),
            ("Annual volatility input", format_pct(params["sigma"] * 100.0, precision=1)),
            ("Live P(loss >10%)", f"{_live_dp:.0%}"),
        ]
        for _row_items in [_scen_metrics[:2], _scen_metrics[2:]]:
            _scen_cols = st.columns(len(_row_items))
            for _col, (_label, _val) in zip(_scen_cols, _row_items):
                _col.markdown(
                    f"""<div style="background:rgba(255,255,255,0.82);border:1px solid #d3dce5;
                        border-radius:14px;padding:12px 14px;margin-bottom:10px;">
                      <div style="font-size:0.76rem;color:#51606f;margin-bottom:3px;">{_label}</div>
                      <div style="font-size:1.0rem;font-weight:600;color:#132238;">{_val}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        st.caption(meta["note"])

    st.plotly_chart(fan_chart(history, summary, region, meta["scenario_name"]), use_container_width=True)
    st.caption("Scenario path for the selected region under the chosen override set. This chart is a stress-testing view, not a forecast recommendation.")

    scenario_rows = []
    _scenario_csv_frames = []
    with st.spinner("Computing scenario comparison…"):
        for preset_name in config.ui.scenario_labels:
            preset_summary_row = _scenario_summary_row(region_summary, preset_name)
            preset_params, preset_meta = _scenario_parameters(
                region_row, preset_name, stage5_row=preset_summary_row, horizon_years=horizon_years)
            preset_paths = run_monte_carlo(
                start_price=float(region_row["nominal_house_price"]),
                n_months=horizon_years * 12,
                n_paths=min(500, n_paths),
                params=preset_params,
                seed=config.controls.random_seed,
            )
            preset_summary = summarize_paths(
                preset_paths, percentiles=[10, 25, 50, 75, 90])
            start_price = float(region_row["nominal_house_price"])
            _preset_terminal = preset_paths.iloc[-1].to_numpy()
            _preset_downside = float(np.mean((_preset_terminal / start_price - 1.0) <= -0.10))
            scenario_rows.append(
                {
                    "Scenario": preset_name,
                    "Median 5y return": format_pct((preset_summary.iloc[-1]["p50"] / start_price - 1.0) * 100.0, signed=True),
                    "P10 5y return": format_pct((preset_summary.iloc[-1]["p10"] / start_price - 1.0) * 100.0, signed=True),
                    "P90 5y return": format_pct((preset_summary.iloc[-1]["p90"] / start_price - 1.0) * 100.0, signed=True),
                    "P(loss >10%)": f"{_preset_downside:.0%}",
                }
            )
            _n_proj = horizon_years * 12
            _start_date = pd.to_datetime(region_row["date"])
            _proj_dates = [(_start_date + pd.DateOffset(months=m)).date()
                           for m in range(1, _n_proj + 1)]
            _scenario_csv_frames.append(pd.DataFrame({
                "scenario_name": preset_name,
                "date": _proj_dates,
                "price_path_p10": preset_summary.iloc[1:]["p10"].values,
                "price_path_p25": preset_summary.iloc[1:]["p25"].values,
                "price_path_p50": preset_summary.iloc[1:]["p50"].values,
                "price_path_p75": preset_summary.iloc[1:]["p75"].values,
                "price_path_p90": preset_summary.iloc[1:]["p90"].values,
            }))
    st.subheader("Scenario Comparison")
    st.caption("All scenarios run at the same horizon and path count as the selected preset above.")
    st.dataframe(pd.DataFrame(scenario_rows), use_container_width=True, hide_index=True)
    _scenario_csv = pd.concat(_scenario_csv_frames, ignore_index=True)
    _region_slug = region.lower().replace(" ", "_")
    st.download_button(
        label="\U0001f4e5 Download scenario paths (CSV)",
        data=_scenario_csv.to_csv(index=False).encode("utf-8"),
        file_name=f"scenario_lab_{_region_slug}_{datetime.date.today()}.csv",
        mime="text/csv",
    )
    with st.expander("Simulation calibration diagnostics", expanded=False):
        _render_simulation_transparency(region)
    _render_key_limitations_block()
    _render_limitations_notice()


def render_model_controls() -> None:
    ensure_state_defaults()
    apply_app_style()
    config = load_config()
    data = load_dashboard_data()

    _render_research_header(data, compact=True)
    st.title("Model Controls")
    st.caption("Calibrated parameters are read-only. User overrides are limited to scenario stress testing and are intentionally separated from the core research calibration.")

    from src.ui.glossary import GLOSSARY

    region = st.selectbox("Inspect region", config.project.regions, index=config.project.regions.index(
        config.ui.default_region), key="controls_region")
    region_row = get_region_record(region)
    sigma_frequency = str(region_row.get("sigma_frequency", "monthly")).lower()
    sigma_label = "sigma_stage4_annual" if sigma_frequency == "annual" else "sigma_stage4_monthly"
    sigma_description = (
        "Stage 4 residual volatility stored at annual frequency"
        if sigma_frequency == "annual"
        else "Stage 4 residual volatility stored at monthly frequency"
    )

    def _fmt_param(v) -> str:
        try:
            fv = float(v)
            if not np.isfinite(fv):
                return "n/a"
            return f"{fv:.4g}"
        except (TypeError, ValueError):
            return "n/a"

    calibrated = pd.DataFrame(
        [
            ("kappa", _fmt_param(region_row["kappa"]), "per month", "Stage 4 mean reversion speed — higher = faster reversion to fair value"),
            (sigma_label, _fmt_param(region_row["sigma"]), "annualised std dev" if sigma_frequency == "annual" else "monthly std dev", sigma_description),
            ("hist_vol_annual", _fmt_param(region_row.get("hist_price_growth_vol_annual", np.nan)), "annualised std dev", "Annualised historical monthly price-growth volatility"),
            ("mu_equilibrium", _fmt_param(region_row["mu_equilibrium"]), "£", "Stage 4 equilibrium price"),
            ("P_star_now", _fmt_param(region_row.get("P_star_now", np.nan)), "£", "Rate-conditional fair value used by Stage 6"),
            ("gamma_annual", _fmt_param(region_row.get("gamma_annual", np.nan)), "% per year", "Annual drift term — long-run expected price growth outside of mean reversion"),
        ],
        columns=["Parameter", "Value", "Units", "Description"],
    )
    st.dataframe(calibrated, use_container_width=True, hide_index=True)
    st.caption("Values are read from the canonical Stage 4 rebuild for the selected region. Use the cross-regional comparison expander below to see the full spread.")

    st.subheader("Interactive override guardrails")
    guardrails = pd.DataFrame(
        [
            ("Rate override warning",
             f"{config.controls.override_warning_rate_bps:.0f} bps"),
            ("Volatility warning",
             f"{config.controls.override_warning_sigma_multiplier:.1f}x calibrated sigma"),
            ("Validated horizon",
             f"<= {config.controls.override_warning_horizon_years:.0f} years"),
            ("Interactive path cap",
             f"{config.controls.max_paths_interactive:,}"),
        ],
        columns=["Guardrail", "Threshold"],
    )
    st.dataframe(guardrails, use_container_width=True)

    with st.expander("Cross-regional parameter comparison", expanded=False):
        _cross_rows = []
        for _r in config.project.regions:
            try:
                _rr = get_region_record(_r)
                _sf = str(_rr.get("sigma_frequency", "monthly")).lower()
                _sigma_ann = float(_rr["sigma"]) if _sf == "annual" else float(_rr["sigma"]) * (12 ** 0.5)
                _cross_rows.append({
                    "Region": _r,
                    "κ (kappa)": f"{float(_rr['kappa']):.4g}",
                    "σ annual": f"{_sigma_ann:.4g}",
                    "μ equilibrium (£)": f"{float(_rr['mu_equilibrium']):,.0f}",
                    "γ annual (%)": f"{float(_rr.get('gamma_annual', 0.0)) * 100.0:.2f}",
                    "Gross yield (%)": f"{float(_rr.get('gross_yield_pct', 0.0)):.1f}",
                })
            except Exception:
                continue
        if _cross_rows:
            st.dataframe(pd.DataFrame(_cross_rows), use_container_width=True, hide_index=True)
            st.caption("All parameters from the Stage 4 canonical rebuild. σ has been annualised for comparability.")

    with st.expander("What do these parameters mean?", expanded=False):
        terms = list(GLOSSARY.items())
        mid = (len(terms) + 1) // 2
        col_left, col_right = st.columns(2)
        with col_left:
            for term, definition in terms[:mid]:
                st.markdown(f"**{term}**")
                st.markdown(definition)
        with col_right:
            for term, definition in terms[mid:]:
                st.markdown(f"**{term}**")
                st.markdown(definition)

    _render_limitations_notice()


def render_methodology() -> None:
    ensure_state_defaults()
    apply_app_style()
    data = load_dashboard_data()

    _render_research_header(data, compact=True)
    st.title("Methodology and Limitations")

    _tab_method, _tab_validation, _tab_backtests, _tab_docs = st.tabs(
        ["Methodology", "Validation & Coverage", "Backtests", "Documentation"]
    )

    with _tab_method:
        st.subheader("What this platform is")
        st.write(
            "A regional UK housing valuation, vulnerability, and scenario research system. "
            "It is stronger as a disciplined research framework than as a high-confidence forecasting engine."
        )

        st.subheader("Data architecture")
        st.write(
            "The repository uses a dual-horizon design. A harmonised model panel is used for regression, "
            "simulation, and validated signal inputs. A separate descriptive market panel extends later and "
            "is used only for current market context. Source horizons differ materially, so the app shows "
            "both windows explicitly."
        )

        st.subheader("Stage 4 — Panel calibration")
        st.write(
            "Estimates a semi-structural fair-value relation for real house prices using as-of earnings, "
            "rent, and mortgage rates, then exports region-level parameters such as mean reversion (`kappa`) "
            "and volatility (`sigma`). The active growth layer uses the lagged mortgage-rate level "
            "(`mortgage_rate_lag3`) because it is more sign-stable across regions and cleaner in "
            "late-sample diagnostics. Financial stress is retained only as an episodic auxiliary term. "
            "Directional sign restrictions, sigma clipping, gamma bounds, and kappa fallback are treated "
            "as explicit guardrails rather than hidden identification claims."
        )

        st.subheader("Stage 5 — Scenario simulation")
        st.write(
            "Converts calibrated parameters into five-year regional price distributions rather than "
            "single-point forecasts. The baseline is calibrated separately from stress overlays, and "
            "the scenario mechanics are parameterised in config rather than buried as code constants."
        )

        st.subheader("Signal layer")
        st.write(
            "The deployed decision layer is a signal dashboard rather than a primary composite. "
            "Valuation is the strongest empirically supported signal. Cycle is partially supported. "
            "Downside, rent support, and household affordability are contextual signals that help "
            "interpretation but do not carry the same evidential weight."
        )

        st.subheader("Simulation transparency")
        st.write(
            "The dashboard compares historical realised 5-year regional return distributions with the "
            "canonical baseline simulation distribution. These views are calibration diagnostics, not "
            "proof of forecast superiority."
        )

        st.subheader("Audience")
        st.write(
            "The research overview, regional signal dashboard, and scenario diagnostics are analyst-facing. "
            "The household explorer is a secondary translation layer for affordability and downside context."
        )

        st.subheader("Current assessment")
        st.warning(
            "The canonical rebuild is materially stronger than the legacy Stage 4 and Stage 5 artifacts, "
            "but it is still not a signed-off bank forecasting model. Outputs should be read as "
            "valuation-and-vulnerability research rather than ground truth.",
            icon="⚠️",
        )

    with _tab_validation:
        _render_validation_status(data, compact=False)
        st.subheader("Current coverage")
        _meta = data.metadata
        _cov_rows = []
        for _src, _info in _meta.get("source_coverage", {}).items():
            _cov_rows.append({
                "Source": _src.replace("_", " ").title(),
                "Start": _info.get("observation_start", "n/a"),
                "End": _info.get("observation_end", "n/a"),
                "File": _info.get("source_file", "n/a"),
            })
        if _cov_rows:
            st.dataframe(pd.DataFrame(_cov_rows), use_container_width=True, hide_index=True)
        _key_dates = {
            "Model panel coverage end": _meta.get("model_panel_coverage_end", "n/a"),
            "Latest house price": _meta.get("latest_house_price_end", "n/a"),
            "Latest rental level": _meta.get("latest_rental_level_end", "n/a"),
            "Effective data as of": _meta.get("effective_data_as_of", "n/a"),
            "Weakest model input end": _meta.get("weakest_model_input_end", "n/a"),
            "Regions covered": len(_meta.get("regions_covered", [])),
            "Scenarios": _meta.get("scenario_count", "n/a"),
            "Last pipeline rebuild": _meta.get("latest_model_run_utc", "n/a"),
        }
        st.dataframe(
            pd.DataFrame(list(_key_dates.items()), columns=["Item", "Value"]),
            use_container_width=True, hide_index=True,
        )
        if _meta.get("cutoff_reason"):
            st.caption(_meta["cutoff_reason"])

    with _tab_backtests:
        st.subheader("Structural Stability Tests")
        try:
            from src.model.structural_breaks import run_chow_tests
            from src.core.paths import CANONICAL_PANEL_PATH
            import pandas as _pd_sb
            _panel = _pd_sb.read_csv(CANONICAL_PANEL_PATH, parse_dates=["date"])
            _chow_df, _chow_summary = run_chow_tests(_panel)
            _p_col = next((c for c in _chow_df.columns if "p" in c.lower() and "val" in c.lower()), None)
            if _p_col:
                def _chow_colour(val):
                    try:
                        return "color: #b94040; font-weight: 600" if float(val) < 0.05 else "color: #2e7d32"
                    except (TypeError, ValueError):
                        return ""
                try:
                    st.dataframe(_chow_df.style.map(_chow_colour, subset=[_p_col]), use_container_width=True)
                except AttributeError:
                    st.dataframe(_chow_df.style.applymap(_chow_colour, subset=[_p_col]), use_container_width=True)
            else:
                st.dataframe(_chow_df, use_container_width=True)
            st.caption(
                "Chow test: red p-values (< 0.05) indicate a structural break; green values indicate stability "
                "across the 2008 financial crisis and 2020 pandemic breakpoints."
            )
            if _chow_summary["any_breaks_detected"]:
                st.warning(
                    "Structural breaks detected at 5% significance in: " +
                    ", ".join(_chow_summary["break_points_detected"]) + ". " +
                    _chow_summary["interpretation"]
                )
            else:
                st.success("No structural breaks detected at 5% significance level.")
        except Exception as _e:
            st.info(f"Structural break tests could not be run: {_e}")

        st.subheader("Signal Backtests")
        try:
            import warnings as _warnings
            from src.scoring.backtest import (
                backtest_c2_downside, backtest_c3_affordability,
                backtest_c4_rental_yield, backtest_c5_cycle,
            )
            from src.core.paths import CANONICAL_PANEL_PATH
            _panel = pd.read_csv(CANONICAL_PANEL_PATH, parse_dates=["date"])
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", UserWarning)
                _c2_df, _c2_summary = backtest_c2_downside(_panel)
            _c3_df, _c3_summary = backtest_c3_affordability(_panel)
            _c4_df, _c4_summary = backtest_c4_rental_yield(_panel)
            _c5_df, _c5_summary = backtest_c5_cycle(_panel)

            _col_c2, _col_c3, _col_c4, _col_c5 = st.columns(4)

            def _backtest_card(col, title, label, value, sub):
                col.markdown(f"**{title}**")
                col.markdown(
                    f"""<div style="background:rgba(255,255,255,0.82);border:1px solid #d3dce5;
                        border-radius:14px;padding:12px 14px;margin-bottom:6px;min-height:90px;
                        display:flex;flex-direction:column;justify-content:center;">
                      <div style="font-size:0.78rem;color:#51606f;margin-bottom:4px;">{label}</div>
                      <div style="font-size:1.05rem;font-weight:600;color:#132238;">{value}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                col.caption(sub)

            with _col_c2:
                _c2_acc = _c2_summary.get("accuracy")
                if _c2_acc != _c2_acc:
                    _backtest_card(_col_c2, "C2 — Downside", "Accuracy (24m)", "N/A",
                                   "Requires `downside_probability` column — not yet in panel.")
                else:
                    _backtest_card(_col_c2, "C2 — Downside", "Accuracy (24m)", f"{_c2_acc:.1%}",
                                   f"Precision: {_c2_summary.get('precision', float('nan')):.1%} | Recall: {_c2_summary.get('recall', float('nan')):.1%}")
            with _col_c3:
                if len(_c3_df) == 0:
                    _backtest_card(_col_c3, "C3 — Affordability", "Accuracy (24m)", "N/A",
                                   "Requires `affordability_gap` or `payment_burden_gap`.")
                else:
                    _backtest_card(_col_c3, "C3 — Affordability", "Accuracy (24m)",
                                   f"{_c3_summary.get('accuracy', 0):.1%}",
                                   f"Dir. IR: {_c3_summary.get('directional_information_ratio', float('nan')):.3f}")
            with _col_c4:
                if len(_c4_df) == 0:
                    _backtest_card(_col_c4, "C4 — Rental Yield", "Accuracy (24m)", "N/A",
                                   "Requires `gross_yield_pct` or `yield_lag1`.")
                else:
                    _backtest_card(_col_c4, "C4 — Rental Yield", "Accuracy (24m)",
                                   f"{_c4_summary.get('accuracy', 0):.1%}",
                                   f"IR proxy: {_c4_summary.get('information_ratio_proxy', float('nan')):.3f}")
            with _col_c5:
                if len(_c5_df) == 0:
                    _backtest_card(_col_c5, "C5 — Cycle / Momentum", "Accuracy (12m)", "N/A",
                                   "Requires a momentum column in the panel.")
                else:
                    _backtest_card(_col_c5, "C5 — Cycle / Momentum", "Accuracy (12m)",
                                   f"{_c5_summary.get('accuracy', 0):.1%}",
                                   f"IR proxy: {_c5_summary.get('information_ratio_proxy', float('nan')):.3f}")
            st.caption(
                "Directional accuracy: fraction of calls that matched the realised direction. "
                ">55% is considered economically meaningful. "
                "Weights adjusted based on backtest results on 2026-03-31."
            )
        except Exception as _e:
            st.info(f"Signal backtests could not be run: {_e}")

    with _tab_docs:
        from src.core.paths import ROOT_DIR
        st.caption("For term definitions, see the **Technical Controls** page → 'What do these parameters mean?' expander.")
        _DOC_GROUPS: list[tuple[str, list[str]]] = [
            ("Architecture & data", [
                "DUAL_HORIZON_ARCHITECTURE_NOTE.md",
                "MODEL_PANEL_COVERAGE_DIAGNOSIS.md",
                "DESCRIPTIVE_PANEL_BUILD_NOTE.md",
                "HOUSE_PRICE_COVERAGE_DIAGNOSIS.md",
                "HOUSE_PRICE_EXTENSION_NOTE.md",
                "RATE_CHANNEL_DIAGNOSIS.md",
            ]),
            ("Validation & calibration", [
                "STAGE4_STAGE5_ASSESSMENT.md",
                "FINANCIAL_STRESS_STABILITY_NOTE.md",
                "ECONOMETRIC_GUARDRAILS_AUDIT.md",
                "PRIORS_AND_GUARDRAILS_NOTE.md",
                "VALIDATION_BOUNDARY_REMEDIATION.md",
                "SIMULATION_RECALIBRATION_NOTE.md",
                "SIMULATION_TRANSPARENCY_NOTE.md",
            ]),
            ("Signal layer", [
                "SIGNAL_DASHBOARD_REDESIGN_NOTE.md",
                "SIGNAL_COMPONENT_DEFINITIONS.md",
                "SIGNAL_PRESENTATION_REFACTOR.md",
                "SCORE_LAYER_EVIDENCE_REVIEW.md",
            ]),
            ("Audit trail & sign-off", [
                "ASSUMPTIONS_REGISTER.md",
                "MODEL_LIMITATIONS.md",
                "AUDIT_STAGE1_TO_STAGE6.md",
                "SIGNOFF_READINESS_CHECKLIST.md",
                "FINAL_SIGNOFF_REVIEW.md",
                "FINAL_RESEARCH_QUALITY_UPGRADE_REPORT.md",
                "MODEL_UPGRADE_REPORT.md",
                "MASTERPIECE_UPGRADE_REPORT.md",
            ]),
            ("Narrative & presentation", [
                "PRODUCT_POSITIONING_NOTE.md",
                "HOME_PAGE_NEUTRALITY_NOTE.md",
                "AUDIENCE_SPLIT_NOTE.md",
                "RESEARCH_NARRATIVE_REFINEMENT.md",
                "CHART_RESEARCH_STYLE_NOTE.md",
                "HOUSE_VIEW_SUMMARY_NOTE.md",
                "UI_RESEARCH_PRESENTATION_NOTE.md",
            ]),
        ]
        for group_name, doc_names in _DOC_GROUPS:
            _group_docs = [(n, ROOT_DIR / n) for n in doc_names if (ROOT_DIR / n).exists()]
            if not _group_docs:
                continue
            st.markdown(f"##### {group_name}")
            for doc_name, path in _group_docs:
                with st.expander(doc_name.replace("_", " ").replace(".md", ""), expanded=False):
                    st.markdown(path.read_text(encoding="utf-8"))

    _render_limitations_notice()


def render_comparison_view() -> None:
    ensure_state_defaults()
    apply_app_style()
    config = load_config()
    data = load_dashboard_data()

    _render_research_header(data, compact=True)
    st.title("Region Comparison")
    st.caption("Compare key housing market metrics across regions over time.")

    all_regions = config.project.regions
    default_regions = [r for r in ["London", "North East",
                                   "Yorkshire and The Humber", "South West"] if r in all_regions]
    selected = st.multiselect(
        "Select regions to compare (up to 4)",
        options=all_regions,
        default=default_regions,
    )
    if len(selected) > 4:
        st.warning(
            "More than 4 regions selected \u2014 only the first 4 will be shown.")
        selected = selected[:4]
    if not selected:
        st.info("Select at least one region to display the comparison.")
        return

    metric = st.selectbox(
        "Metric",
        options=[
            "Nominal House Price Index (2005=100)",
            "Real House Price Index (2005=100)",
            "Fair Value Gap (%)",
            "Downside Probability (%) — simulation only, no historical series",
            "Rental Yield (%)",
        ],
    )
    period = st.selectbox(
        "Time period",
        options=[
            "Full history (2005\u2013present)",
            "Post-GFC (2010\u2013present)",
            "Post-pandemic (2020\u2013present)",
        ],
    )

    _CI_COLS = ["fair_value_gap_lower_90", "fair_value_gap_upper_90"]
    fair_cols = ["region", "date", "fair_value_gap_log"] + _CI_COLS
    available_fair_cols = [
        c for c in fair_cols if c in data.fair_panel.columns]
    panel_df = data.master.merge(
        data.fair_panel[available_fair_cols].drop_duplicates(
            subset=["region", "date"]),
        on=["region", "date"],
        how="left",
    )

    fig = plot_region_comparison(selected, metric, period, panel_df)
    st.plotly_chart(fig, use_container_width=True)
    _rebase_note = " — indices rebased to 100 at period start" if "Index" in metric else ""
    _source_note = "Stage 5 Monte Carlo simulation" if "simulation only" in metric else "Land Registry, ONS"
    st.caption(f"**Period:** {period}{_rebase_note}. Source: {_source_note}.")

    if metric == "Fair Value Gap (%)":
        _has_ci = all(c in panel_df.columns for c in _CI_COLS)
        _latest = (
            panel_df[panel_df["region"].isin(selected)]
            .sort_values("date")
            .groupby("region", sort=False)
            .last()
        )
        _gap_cols = st.columns(4)
        for _i, _r in enumerate((selected + ["", "", "", ""])[:4]):
            if not _r or _r not in _latest.index:
                continue
            _row = _latest.loc[_r]
            if not pd.notna(_row.get("fair_value_gap_log")):
                continue
            _gap_pct = float(_row["fair_value_gap_log"]) * 100.0
            _ci_line = ""
            if _has_ci and pd.notna(_row.get("fair_value_gap_lower_90")):
                _lo = float(_row["fair_value_gap_lower_90"]) * 100.0
                _hi = float(_row["fair_value_gap_upper_90"]) * 100.0
                _sig = (
                    "significant"
                    if (_lo > 0 and _hi > 0) or (_lo < 0 and _hi < 0)
                    else "not significant at 90%"
                )
                _ci_line = f"<div style='font-size:0.70rem;color:#51606f;margin-top:4px;'>90% CI: [{_lo:.1f}%, {_hi:.1f}%] — {_sig}</div>"
            else:
                _ci_line = "<div style='font-size:0.70rem;color:#8a9bb0;margin-top:4px;'>No CI data</div>"
            _gap_cols[_i].markdown(
                f"""<div style="background:rgba(255,255,255,0.82);border:1px solid #d3dce5;
                    border-radius:14px;padding:12px 16px;min-height:100px;">
                  <div style="font-size:0.76rem;color:#51606f;margin-bottom:3px;">{_r}</div>
                  <div style="font-size:1.05rem;font-weight:600;color:#132238;">
                    Fair value gap: {_gap_pct:+.1f}%</div>
                  {_ci_line}
                </div>""",
                unsafe_allow_html=True,
            )

    _render_key_limitations_block()
    _render_limitations_notice()
