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
    current_rate = float(region_row.get("r_current", region_row.get("mortgage_rate", 5.25)))
    target_rate = max(0.5, current_rate + (scenario.rate_shift_bps + rate_shift_bps) / 100.0)
    rate_channel_mode = str(region_row.get("rate_channel_mode", "rate_level_anchor"))
    alpha = float(region_row.get("alpha_used", 0.5))
    r_bar = float(region_row.get("r_bar", 3.54))
    p_star_base = float(region_row.get("P_star_base", region_row.get("mu_equilibrium", start_price)))
    current_fair_value = float(region_row.get("P_star_now", p_star_base))
    if rate_channel_mode in {"rate_change_plus_stress", "rate_change_plus_aux_stress", "rate_gap_plus_aux_stress"}:
        gap_current = float(region_row.get("valuation_gap_current", 0.0))
        gap_closure = VALUATION_GAP_CLOSURE_UI.get(scenario_name, 0.15)
        target_fair_value = start_price * np.exp(-gap_closure * gap_current)
        target_fair_value *= 1.0 + scenario.income_growth_shift_pct / 150.0 - scenario.supply_shift_pct / 400.0
    else:
        target_fair_value = p_star_base * (r_bar / target_rate) ** alpha
        target_fair_value *= 1.0 + scenario.income_growth_shift_pct / 100.0 - scenario.supply_shift_pct / 200.0
    if stage5_row is not None and "median_5yr_growth" in stage5_row:
        stage5_terminal = start_price * (1.0 + float(stage5_row["median_5yr_growth"]) / 100.0)
        if np.isfinite(stage5_terminal) and stage5_terminal > 0:
            blend = 0.65 if rate_channel_mode in {"rate_change_plus_stress", "rate_change_plus_aux_stress", "rate_gap_plus_aux_stress"} else 0.50
            target_fair_value = (1.0 - blend) * target_fair_value + blend * stage5_terminal
    target_fair_value = max(start_price * config.controls.minimum_price_floor_ratio, target_fair_value)

    drift_annual = float(region_row.get("gamma_annual", region_row.get("gamma_annual_pp", 0.0)))
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
    hist_vol_annual = float(region_row.get("hist_price_growth_vol_annual", stage4_sigma_annual))
    if not np.isfinite(hist_vol_annual) or hist_vol_annual <= 0:
        hist_vol_annual = stage4_sigma_annual
    sigma_implied_annual = _implied_annual_sigma_from_stage5(stage5_row, horizon_years, hist_vol_annual)
    sigma_annual = max(stage4_sigma_annual, 0.5 * (hist_vol_annual + sigma_implied_annual))

    horizon_months = int(horizon_years * 12)
    mean_path = np.linspace(current_fair_value, target_fair_value, horizon_months + 1)
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


def _render_research_header(data, *, compact: bool = False) -> None:
    model_start = pd.to_datetime(data.metadata["data_coverage_start"]).year
    model_end = pd.to_datetime(data.metadata["model_panel_coverage_end"]).year
    descriptive_end = pd.to_datetime(data.metadata["latest_market_data_end"]).year
    st.markdown("### Housing Market Research Dashboard")
    st.caption(
        "Model estimated on the harmonised panel. Newer descriptive market data are shown separately and are not used for Stage 4 estimation."
    )
    cols = st.columns(3)
    cols[0].metric("Model coverage window", f"{model_start}-{model_end}")
    cols[1].metric("Latest market data available", f"{descriptive_end}")
    cols[2].metric("Last pipeline rebuild", data.metadata["latest_model_run_utc"][:10])
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
        pct = float(region_row.get("downside_vulnerability_percentile", np.nan))
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


def _render_signal_dashboard(signals: list[dict], *, region_row: dict | None = None) -> None:
    st.caption(_signal_confidence_summary(signals))
    cols = st.columns(len(signals))
    for idx, signal in enumerate(signals):
        cols[idx].metric(signal["display_name"], _signal_display_value(signal, region_row))
        cols[idx].markdown(
            f"{render_badge(signal['bucket'])} {render_evidence_badge(signal['evidence_level'])}",
            unsafe_allow_html=True,
        )
        cols[idx].caption(signal["question"])
        cols[idx].caption(
            f"{_confidence_lead(signal['evidence_level'])} signal | {_direction_label(signal['bucket'])}"
        )
        cols[idx].caption(f"Internal score: {signal['score']:.0f} / 100")


def _signal_scores_for_chart(signals: list[dict]) -> dict[str, float]:
    return {signal["display_name"]: float(signal["score"]) for signal in signals}


def _signal_evidence_table(signals: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Signal": signal["display_name"],
                "Plain-English view": signal["simple_status"],
                "Score": signal["score"],
                "Direction": _direction_label(signal["bucket"]),
                "Evidence": _evidence_label(signal["evidence_level"]),
                "Why it matters": signal["definition"],
            }
            for signal in signals
        ]
    )


def _headline_insight(region: str, dashboard: dict, *, institutional: bool = False) -> str:
    positives, negatives = top_signal_contributors(dashboard["signals"])
    valuation = next(signal for signal in dashboard["signals"] if signal["id"] == "valuation")
    cycle = next(signal for signal in dashboard["signals"] if signal["id"] == "cycle")
    framing = "institutional" if institutional else "household"
    return (
        f"{region}: valuation is {_direction_label(valuation['bucket']).lower()} while cycle conditions are "
        f"{_direction_label(cycle['bucket']).lower()} in the current {framing} research view. "
        f"Main support comes from {positives[0]}; the softest area is {negatives[-1]}."
    )


def _regional_house_view_summary(data, region_row: dict, dashboard: dict) -> str:
    valuation_pct = float(region_row.get("valuation_stretch_percentile", np.nan))
    downside_pct = float(region_row.get("downside_vulnerability_percentile", np.nan))
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
    latest_context = data.metadata.get("latest_complete_housing_context_end", data.metadata.get("latest_market_data_end", "n/a"))
    return (
        f"House view summary: valuation still looks stretched in {stretched} of {len(table)} regions, while {supportive} regions look relatively more supportive versus their own history. "
        f"Downside vulnerability is elevated in {vulnerable} regions on the current baseline scenario read. "
        f"The analytical backbone remains the harmonised model panel through {data.metadata['model_panel_coverage_end']}; market context extends to {latest_context} but is descriptive rather than re-estimated."
    )


def _render_secondary_synthesis_box(secondary: dict) -> None:
    with st.expander("Optional signal synthesis", expanded=False):
        st.metric("Secondary synthesis", f"{secondary['score']:.0f} / 100")
        st.caption(
            f"{secondary['bucket']} synthesis of the underlying signals. "
            "This is secondary orientation context rather than the primary analytical frame."
        )


def _render_key_limitations_block(data) -> None:
    st.subheader("Key Limitations")
    for item in [
        f"Model panel ends {data.metadata['model_panel_coverage_end']}.",
        "Holdout validation beats AR(1) but still does not beat zero-growth RMSE.",
        "Some signals are descriptive rather than empirically validated.",
        "Simulation outputs represent scenario ranges rather than deterministic forecasts.",
    ]:
        st.markdown(f"- {item}")


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
    flagged_regions = int(rate_stability.get("flagged_unstable_regions", 0) or 0)
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
    metric_cols[0].metric(
        "Rate stability",
        "All regions stable" if all_stable else f"{flagged_regions} flagged",
    )
    metric_cols[1].metric(
        "1m directional accuracy",
        f"{subsample['directional_1m']:.1%}" if subsample.get("directional_1m") is not None else "n/a",
    )
    metric_cols[2].metric(
        "Model RMSE",
        f"{model_rmse:.4f}" if model_rmse is not None else "n/a",
    )
    metric_cols[3].metric(
        "Zero-growth RMSE",
        f"{zero_rmse:.4f}" if zero_rmse is not None else "n/a",
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
        st.dataframe(pd.DataFrame(benchmark_rows), use_container_width=True, hide_index=True)
        st.caption(
            "The model currently outperforms AR(1) on RMSE and MAE, while zero-growth remains the toughest benchmark in the post-rate-shock holdout."
        )


def _render_everyday_summary(region: str, dashboard: dict) -> None:
    st.markdown("#### In Plain English")
    st.caption(f"A simple summary for {region}. This does not replace the research signals below.")
    for takeaway in plain_signal_takeaways(dashboard["signals"]):
        st.markdown(f"- {takeaway}")


def _render_simulation_transparency(region: str) -> None:
    distribution = get_region_distribution_comparison(region)
    moments = get_region_moment_comparison(region)
    st.pyplot(historical_vs_simulated_distribution_chart(distribution, region))
    st.caption(
        "Historical versus simulated 5-year outcome distribution. This is a calibration view for plausibility, not a claim of forecast superiority."
    )
    table = pd.DataFrame(
        [
            ("Historical mean 5y return", f"{moments['historical_mean_pct']:.1f}%"),
            ("Historical median 5y return", f"{moments['historical_median_pct']:.1f}%"),
            ("Simulated median 5y return", f"{moments['simulated_median_pct']:.1f}%"),
            ("Historical P10 / P90", f"{moments['historical_p10_pct']:.1f}% / {moments['historical_p90_pct']:.1f}%"),
            ("Simulated P10 / P90", f"{moments['simulated_p10_pct']:.1f}% / {moments['simulated_p90_pct']:.1f}%"),
            ("Historical P(loss >10%)", f"{moments['historical_prob_loss_10pct']:.0%}"),
            ("Simulated P(loss >10%)", f"{moments['simulated_prob_loss_10pct']:.0%}"),
            ("Tail-risk gap", f"{moments['tail_risk_gap']:.2f}"),
        ],
        columns=["Calibration check", "Value"],
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("Sources: canonical model panel and baseline Stage 5 simulation summary.")


def render_home() -> None:
    ensure_state_defaults()
    apply_app_style()
    data = load_dashboard_data()
    config = load_config()

    _render_research_header(data)
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
        valuation_signal=float(region_row.get("C1_mispricing", region_row["consumer_score"])),
        cycle_signal=float(region_row.get("C3_regime", region_row.get("C2_reit", region_row["reit_score"]))),
        downside_prob=float(region_row.get("p_terminal_loss_10_avg", 0.0)),
        rent_support_signal=float(region_row.get("rent_support_percentile", 50.0)),
    )

    st.info(_regional_house_view_summary(data, region_row, home_dashboard))

    st.markdown("### Market Snapshot")
    snapshot = st.columns(4)
    snapshot[0].metric(
        "Model coverage window",
        f"{pd.to_datetime(data.metadata['data_coverage_start']).year}-{pd.to_datetime(data.metadata['model_panel_coverage_end']).year}",
    )
    snapshot[1].metric("Latest descriptive data", data.metadata.get("latest_market_data_end", "n/a"))
    snapshot[2].metric("Spotlight house price", format_currency(float(region_row["latest_house_price"])))
    snapshot[3].metric("Spotlight rent context", format_currency(float(region_row.get("latest_monthly_rent_raw", 0.0))))
    st.caption(
        "The model is estimated on the harmonised panel through the model coverage end date. Later house-price and rent observations are shown here as descriptive context only."
    )
    summary_table = pd.DataFrame(
        [
            ("Spotlight region", spotlight_region),
            ("Model panel date", str(pd.to_datetime(region_row["date"]).date())),
            ("Latest raw HPI date", str(pd.to_datetime(region_row["latest_house_price_date"]).date())),
            ("Current gross yield", format_pct(float(region_row["gross_yield_pct"]), precision=1)),
            ("Research mode", config.ui.historical_mode_label),
        ],
        columns=["Item", "Value"],
    )
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

    # --- Persistent data-currency disclosure (L1) ---
    _effective = data.metadata.get(
        "effective_data_as_of",
        data.metadata.get("model_panel_coverage_end", "unknown"),
    )
    try:
        from datetime import date as _date
        _age = (
            _date.today()
            - _date.fromisoformat(str(_effective)[:10])
        ).days
    except (ValueError, TypeError):
        _age = 0
    st.info(
        f"Signals anchored to **{_effective}**. Panel age: **{_age} days**. "
        "All outputs reflect research conditions at that date, not current prices. "
        "See the **Limitations** page for full disclosure.",
        icon="ℹ️",
    )
    _render_validation_status(data, compact=True)

    st.markdown("### Key Housing Signals")
    st.caption("Primary analytical view for the selected region. Stronger-evidence signals should carry more weight than descriptive context.")
    _render_signal_dashboard(home_dashboard["signals"], region_row=region_row)
    st.caption(
        "⚠ Valuation and cycle signals use income and supply features estimated from "
        "annual data interpolated to monthly frequency. Timing precision within a "
        "calendar year should not be relied upon. See Model Limitations for detail."
    )
    signal_left, signal_right = st.columns([1.05, 0.95])
    with signal_left:
        st.pyplot(score_breakdown_chart(_signal_scores_for_chart(home_dashboard["signals"]), f"{spotlight_region}: signal panel"))
        st.caption("Signal summary by component. Valuation remains the strongest empirically supported indicator in the current framework.")
    with signal_right:
        st.dataframe(_signal_evidence_table(home_dashboard["signals"]), use_container_width=True, hide_index=True)

    st.markdown("### Risk Context")
    risk_left, risk_right = st.columns([1.0, 1.0])
    with risk_left:
        st.pyplot(scenario_comparison_chart(region_summary))
        st.caption("Scenario range for the spotlight region, showing how medium-term price outcomes vary across the canonical scenario set.")
    with risk_right:
        _render_simulation_transparency(spotlight_region)

    with st.expander("Cross-regional orientation", expanded=False):
        st.pyplot(regional_rank_chart(data.region_table, "consumer_score", spotlight_region, "Cross-regional synthesis orientation"))
        st.caption("Cross-regional synthesis is shown for orientation only. Primary interpretation should come from the underlying signal panel.")
        st.pyplot(tail_risk_heatmap(data.simulation))
        st.caption("Scenario downside map based on canonical Stage 5 outputs. This is a scenario-risk summary, not a forecast ranking.")

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

    _render_key_limitations_block(data)
    _render_limitations_notice()


def render_consumer_view() -> None:
    ensure_state_defaults()
    apply_app_style()
    data = load_dashboard_data()
    config = load_config()
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as _fh:
        household_explorer_defaults: dict[str, dict] = json.load(_fh).get("household_explorer_defaults", {})

    _render_research_header(data, compact=True)
    st.title("Household / Affordability Explorer")
    st.caption("Plain-English household affordability and downside context. This page is secondary to the main research dashboard and is not an automated buy / do-not-buy tool.")
    st.caption("Affordability outputs are illustrative model estimates. They do not constitute financial advice and should not be used for mortgage applications.")

    controls, results = st.columns([0.95, 1.05])
    with controls:
        region = st.selectbox("Region", config.project.regions, index=config.project.regions.index(config.ui.default_region))
        _region_defaults = household_explorer_defaults.get(region, {})
        _default_income = _region_defaults.get("household_income", config.ui.default_income_gbp)
        _default_price = _region_defaults.get("property_price", config.ui.default_property_price_gbp)
        income = st.number_input("Gross annual household income (GBP)", min_value=15000, max_value=500000, value=_default_income, step=5000)
        deposit = st.number_input("Deposit (GBP)", min_value=0, max_value=1000000, value=config.ui.default_deposit_gbp, step=5000)
        mortgage_term = st.slider("Mortgage term (years)", min_value=15, max_value=40, value=config.ui.default_mortgage_term_years)
        target_price = st.number_input("Target property price", min_value=50000, max_value=5000000, value=_default_price, step=10000)
        risk_tolerance = st.select_slider("Risk tolerance", options=config.ui.risk_tolerance_options, value="Balanced")
        scenario = st.selectbox("Macro scenario", options=data.metadata["scenario_names"], index=data.metadata["scenario_names"].index("Baseline"), help=tooltip("scenario weights"))
        st.caption("Defaults are approximate regional medians from ONS / MHCLG 2024. Adjust to match your situation.")
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
                st.warning("Typical mortgage deposits range from 5% to 50% of the property price.")

    if not inputs_valid:
        return

    region_row = get_region_record(region)
    region_summary = get_region_simulation_summary(region)
    selected_summary = _scenario_summary_row(region_summary, scenario)

    history = get_region_history(region, months=84)
    sim_params, sim_meta = _scenario_parameters(region_row, scenario, stage5_row=selected_summary, horizon_years=5)
    log_model_event(_logger, "simulation_run", {"region": region, "scenario": scenario, "n_paths": config.controls.default_paths_interactive, "view": "consumer"})
    paths = run_monte_carlo(
        start_price=float(region_row["nominal_house_price"]),
        n_months=60,
        n_paths=config.controls.default_paths_interactive,
        params=sim_params,
        seed=config.controls.random_seed,
    )
    summary = summarize_paths(paths, percentiles=config.ui.fan_chart_percentiles)
    live_percentiles, live_downside_prob = _live_terminal_percentiles(paths, float(region_row["nominal_house_price"]))
    signal_dashboard = consumer_signal_dashboard(
        region,
        income,
        deposit,
        live_percentiles,
        valuation_signal=float(region_row.get("C1_mispricing", region_row["consumer_score"])),
        cycle_signal=float(region_row.get("C2_consumer", region_row["consumer_score"])),
        property_price=target_price or float(region_row["nominal_house_price"]),
        mortgage_rate=float(region_row["mortgage_rate"]),
        mortgage_term_years=mortgage_term,
        risk_tolerance=risk_tolerance,
        downside_prob=live_downside_prob,
    )

    st.markdown("### Headline View")
    st.info(_headline_insight(region, signal_dashboard))
    _render_everyday_summary(region, signal_dashboard)

    with results:
        st.markdown("### Supporting Signals")
        _render_signal_dashboard(signal_dashboard["signals"], region_row=region_row)
        st.caption(
            "⚠ Valuation and cycle signals use income and supply features estimated from "
            "annual data interpolated to monthly frequency. Timing precision within a "
            "calendar year should not be relied upon."
        )
        st.page_link(render_methodology, label="See Model Limitations for detail.")
        meta_cols = st.columns(2)
        meta_cols[0].metric("Monthly payment", format_currency(signal_dashboard["mortgage_payment_monthly"]))
        meta_cols[1].metric("Mortgage stress", signal_dashboard["mortgage_stress"])
        st.write(consumer_commentary(region_row, signal_dashboard, scenario))
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
        _render_secondary_synthesis_box(signal_dashboard["secondary_composite"])

    st.markdown("### Supporting Charts")
    st.pyplot(fan_chart(history, summary, region, sim_meta["scenario_name"]))
    st.caption("Historical path and simulated scenario range for the selected region. The fan chart illustrates plausible dispersion around the calibrated baseline, not a single predicted path.")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.pyplot(score_breakdown_chart(_signal_scores_for_chart(signal_dashboard["signals"]), "Consumer signal dashboard"))
        st.caption("Component view of the household signal panel, ordered to support interpretation rather than ranking.")
        st.dataframe(_signal_evidence_table(signal_dashboard["signals"]), use_container_width=True, hide_index=True)
    with bottom_right:
        st.pyplot(scenario_comparison_chart(region_summary))
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
        "realised REIT investment outcomes. The yield signal is descriptive only. "
        "See Limitations page, L9 and L20.",
        icon="⚠️",
    )

    controls, results = st.columns([0.95, 1.05])
    with controls:
        region = st.selectbox("Region", config.project.regions, index=config.project.regions.index(config.ui.default_region), key="reit_region")
        scenario = st.selectbox("Scenario focus", options=data.metadata["scenario_names"], index=data.metadata["scenario_names"].index("Baseline"), key="reit_scenario", help=tooltip("scenario weights"))

    region_row = get_region_record(region)
    region_summary = get_region_simulation_summary(region)
    selected_summary = _scenario_summary_row(region_summary, scenario)

    sim_params, sim_meta = _scenario_parameters(region_row, scenario, stage5_row=selected_summary, horizon_years=config.ui.default_holding_horizon_years)
    log_model_event(_logger, "simulation_run", {"region": region, "scenario": scenario, "n_paths": config.controls.default_paths_interactive, "view": "reit"})
    paths = run_monte_carlo(
        start_price=float(region_row["nominal_house_price"]),
        n_months=config.ui.default_holding_horizon_years * 12,
        n_paths=config.controls.default_paths_interactive,
        params=sim_params,
        seed=config.controls.random_seed,
    )
    history = get_region_history(region, months=84)
    summary = summarize_paths(paths, percentiles=config.ui.fan_chart_percentiles)
    live_percentiles, live_downside_prob = _live_terminal_percentiles(paths, float(region_row["nominal_house_price"]))
    signal_dashboard = regional_market_signal_dashboard(
        region,
        valuation_signal=float(region_row.get("C1_mispricing", region_row["reit_score"])),
        cycle_signal=float(region_row.get("C3_regime", region_row.get("C2_reit", region_row["reit_score"]))),
        downside_prob=live_downside_prob,
        rent_support_signal=float(region_row.get("rent_support_percentile", 50.0)),
    )

    st.markdown("### Headline View")
    st.info(_regional_house_view_summary(data, region_row, signal_dashboard))
    st.caption(
        "This page is the core analyst-facing regional read. Use valuation and downside context first; treat the cycle signal as supporting evidence rather than a stand-alone forecast."
    )

    with results:
        st.markdown("### Supporting Signals")
        _render_signal_dashboard(signal_dashboard["signals"], region_row=region_row)
        st.caption(
            "⚠ Valuation and cycle signals use income and supply features estimated from "
            "annual data interpolated to monthly frequency. Timing precision within a "
            "calendar year should not be relied upon."
        )
        st.page_link(render_methodology, label="See Model Limitations for detail.")
        meta_cols = st.columns(2)
        meta_cols[0].metric("Current yield", format_pct(float(region_row["gross_yield_pct"]), precision=1))
        meta_cols[1].metric("Scenario downside risk", format_pct(float(region_row.get("p_terminal_loss_10_avg", 0.0)) * 100.0, precision=0))
        st.caption(
            f"For context: the model averaged {_downside_gfc:.0%} downside probability across regions "
            f"during the 2008\u201309 financial crisis and {_downside_2022:.0%} during the 2022\u201323 correction."
        )
        st.write(
            f"{region} remains a valuation-and-vulnerability case study rather than a stand-alone trade. "
            f"Valuation is the most decision-relevant signal, cycle is secondary, and downside risk remains scenario-driven under {scenario}."
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

    st.markdown("### Supporting Charts")
    st.pyplot(fan_chart(history, summary, region, sim_meta["scenario_name"]))
    st.caption("Historical path and simulated scenario range for the selected region. The fan chart is a calibrated scenario view, not a deterministic price forecast.")
    if region in ("Wales", "Yorkshire and The Humber"):
        st.warning(
            "IQR reliability note: Simulation fan chart width for this region is compressed relative to "
            "historical volatility (simulated IQR = 20\u201344% of historical). Downside probability estimates "
            "should be treated as indicative only. See Limitations > R020/R021."
        )

    left, right = st.columns(2)
    with left:
        st.pyplot(score_breakdown_chart(_signal_scores_for_chart(signal_dashboard["signals"]), "Regional signal dashboard"))
        st.caption("Signal panel summary for the institutional lens. Component evidence should matter more than any blended summary.")
        st.dataframe(_signal_evidence_table(signal_dashboard["signals"]), use_container_width=True, hide_index=True)
        st.pyplot(yield_risk_chart(data.region_table, region))
        st.caption("Current yield versus downside tail context across regions, with the selected region highlighted.")
    with right:
        st.pyplot(scenario_comparison_chart(region_summary))
        st.caption("Canonical scenario comparison for the selected region. Use this to frame sensitivity, not as a probability-weighted forecast table.")

    with st.expander("Simulation calibration view", expanded=False):
        _render_simulation_transparency(region)

    _csv_cols = ["date", "nominal_house_price", "real_house_price", "rental_yield", "fair_value_gap", "downside_probability"]
    _chart_csv = history[[c for c in _csv_cols if c in history.columns]].copy()
    _region_slug = region.lower().replace(" ", "_")
    st.download_button(
        label="\U0001f4e5 Download chart data (CSV)",
        data=_chart_csv.to_csv(index=False).encode("utf-8"),
        file_name=f"{_region_slug}_{datetime.date.today()}_housing_signals.csv",
        mime="text/csv",
    )

    st.subheader("Research-style commentary")
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
        scenario_descriptions: dict[str, str] = json.load(_fh).get("scenario_descriptions", {})

    _render_research_header(data, compact=True)
    st.title("Scenario & Simulation Diagnostics")
    st.caption("Scenario sensitivity and simulation calibration diagnostics. This page is for analyst review rather than end-user interpretation.")

    left, right = st.columns([0.85, 1.15])
    with left:
        region = st.selectbox("Region", config.project.regions, key="scenario_lab_region")
        preset = st.selectbox("Preset", options=config.ui.scenario_labels, key="scenario_lab_preset", help=tooltip("scenario weights"))
        st.info(f"\U0001f4a1 {scenario_descriptions.get(preset, '')}")
        st.caption(
            "Note: Scenario paths are generated using expert-prior drift and volatility overlays applied to "
            "the baseline Ornstein-Uhlenbeck model. They do not represent re-estimated OLS coefficients "
            "under each macro regime."
        )
        rate_shift = st.slider("Additional rate shock (bps)", min_value=-250, max_value=250, value=int(st.session_state["override_rate_shift_bps"]), step=25, key="override_rate_shift_bps", help=tooltip("rate channel"))
        sigma_multiplier = st.slider("Volatility multiplier", min_value=0.5, max_value=2.0, value=float(st.session_state["override_sigma_multiplier"]), step=0.05, key="override_sigma_multiplier", help=tooltip("sigma"))
        drift_shift = st.slider("Annual drift override (%)", min_value=-3.0, max_value=3.0, value=float(st.session_state["override_drift_shift_pct"]), step=0.25, key="override_drift_shift_pct", help=tooltip("drift"))
        horizon_years = st.slider("Horizon (years)", min_value=2, max_value=15, value=int(st.session_state["override_horizon_years"]), key="override_horizon_years")
        n_paths = st.slider("Simulation paths", min_value=250, max_value=config.controls.max_paths_interactive, value=int(st.session_state["override_paths"]), step=250, key="override_paths", help=tooltip("downside probability"))
        if st.button("Reset controls"):
            reset_override_state()
            st.rerun()

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
    warning_text = override_warning(rate_shift, sigma_multiplier, horizon_years)
    if warning_text:
        st.warning(warning_text)
        _logger.warning("Guardrail fired: region=%s, rate_shift_bps=%s, sigma_multiplier=%s, horizon_years=%s", region, rate_shift, sigma_multiplier, horizon_years)

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
    paths = run_monte_carlo(
        start_price=float(region_row["nominal_house_price"]),
        n_months=horizon_years * 12,
        n_paths=n_paths,
        params=params,
        seed=config.controls.random_seed,
    )
    summary = summarize_paths(paths, percentiles=config.ui.fan_chart_percentiles)
    history = get_region_history(region, months=84)

    with right:
        st.metric("Target terminal fair value", format_currency(meta["target_fair_value"]), help=tooltip("fair value gap"))
        rate_metric_label = "Scenario rate assumption" if meta["rate_channel_mode"] in {"rate_change_plus_stress", "rate_change_plus_aux_stress", "rate_gap_plus_aux_stress"} else "Scenario rate anchor"
        st.metric(rate_metric_label, format_pct(meta["target_rate"], precision=2), help=tooltip("rate channel"))
        st.metric("Annual volatility input", format_pct(params["sigma"] * 100.0, precision=1), help=tooltip("sigma"))
        st.caption(meta["note"])
        st.pyplot(fan_chart(history, summary, region, meta["scenario_name"]))
        st.caption("Scenario path for the selected region under the chosen override set. This chart is a stress-testing view, not a forecast recommendation.")

    scenario_rows = []
    _scenario_csv_frames = []
    for preset_name in config.ui.scenario_labels:
        preset_summary_row = _scenario_summary_row(region_summary, preset_name)
        preset_params, preset_meta = _scenario_parameters(region_row, preset_name, stage5_row=preset_summary_row, horizon_years=horizon_years)
        preset_paths = run_monte_carlo(
            start_price=float(region_row["nominal_house_price"]),
            n_months=horizon_years * 12,
            n_paths=min(500, n_paths),
            params=preset_params,
            seed=config.controls.random_seed,
        )
        preset_summary = summarize_paths(preset_paths, percentiles=[10, 25, 50, 75, 90])
        start_price = float(region_row["nominal_house_price"])
        scenario_rows.append(
            {
                "Preset": preset_name,
                "Median 5y return": format_pct((preset_summary.iloc[-1]["p50"] / start_price - 1.0) * 100.0, signed=True),
                "P10 5y return": format_pct((preset_summary.iloc[-1]["p10"] / start_price - 1.0) * 100.0, signed=True),
                "P90 5y return": format_pct((preset_summary.iloc[-1]["p90"] / start_price - 1.0) * 100.0, signed=True),
            }
        )
        _n_proj = horizon_years * 12
        _start_date = pd.to_datetime(region_row["date"])
        _proj_dates = [(_start_date + pd.DateOffset(months=m)).date() for m in range(1, _n_proj + 1)]
        _scenario_csv_frames.append(pd.DataFrame({
            "scenario_name": preset_name,
            "date": _proj_dates,
            "price_path_p10": preset_summary.iloc[1:]["p10"].values,
            "price_path_p25": preset_summary.iloc[1:]["p25"].values,
            "price_path_p50": preset_summary.iloc[1:]["p50"].values,
            "price_path_p75": preset_summary.iloc[1:]["p75"].values,
            "price_path_p90": preset_summary.iloc[1:]["p90"].values,
        }))
    st.dataframe(pd.DataFrame(scenario_rows), use_container_width=True)
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


def render_model_controls() -> None:
    ensure_state_defaults()
    apply_app_style()
    config = load_config()
    data = load_dashboard_data()

    _render_research_header(data, compact=True)
    st.title("Model Controls")
    st.caption("Calibrated parameters are read-only. User overrides are limited to scenario stress testing and are intentionally separated from the core research calibration.")

    from src.ui.glossary import GLOSSARY

    with st.expander("ℹ️ What do these parameters mean?", expanded=False):
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

    region = st.selectbox("Inspect region", config.project.regions, index=config.project.regions.index(config.ui.default_region), key="controls_region")
    region_row = get_region_record(region)
    sigma_frequency = str(region_row.get("sigma_frequency", "monthly")).lower()
    sigma_label = "sigma_stage4_annual" if sigma_frequency == "annual" else "sigma_stage4_monthly"
    sigma_description = (
        "Stage 4 residual volatility stored at annual frequency"
        if sigma_frequency == "annual"
        else "Stage 4 residual volatility stored at monthly frequency"
    )

    calibrated = pd.DataFrame(
        [
            ("kappa", region_row["kappa"], "Stage 4 mean reversion speed"),
            (sigma_label, region_row["sigma"], sigma_description),
            ("hist_vol_annual", region_row.get("hist_price_growth_vol_annual", np.nan), "Annualised historical monthly price-growth volatility"),
            ("mu_equilibrium", region_row["mu_equilibrium"], "Stage 4 equilibrium price"),
            ("P_star_now", region_row.get("P_star_now", np.nan), "Rate-conditional fair value used by Stage 6"),
            ("gamma_annual", region_row.get("gamma_annual", np.nan), "Annual drift term"),
        ],
        columns=["Parameter", "Value", "Description"],
    )
    st.dataframe(calibrated, use_container_width=True)

    st.subheader("Interactive override guardrails")
    guardrails = pd.DataFrame(
        [
            ("Rate override warning", f"{config.controls.override_warning_rate_bps:.0f} bps"),
            ("Volatility warning", f"{config.controls.override_warning_sigma_multiplier:.1f}x calibrated sigma"),
            ("Validated horizon", f"<= {config.controls.override_warning_horizon_years:.0f} years"),
            ("Interactive path cap", f"{config.controls.max_paths_interactive:,}"),
        ],
        columns=["Guardrail", "Threshold"],
    )
    st.dataframe(guardrails, use_container_width=True)
    _render_limitations_notice()


def render_methodology() -> None:
    ensure_state_defaults()
    apply_app_style()
    data = load_dashboard_data()

    _render_research_header(data, compact=True)
    st.title("Methodology and Limitations")
    st.markdown(
        """
        **Product framing**: this platform is best understood as a regional UK housing valuation, vulnerability, and scenario research system. It is stronger as a disciplined research framework than as a high-confidence forecasting engine.

        **Data**: the repository now has a dual-horizon design. A harmonised model panel is used for regression, simulation, and validated signal inputs. A separate descriptive market panel extends later and is used only for current market context. Source horizons differ materially, so the app shows both windows explicitly.

        **Stage 4 role**: panel calibration estimates a semi-structural fair-value relation for real house prices using as-of earnings, rent, and mortgage rates, then exports region-level parameters such as mean reversion (`kappa`) and volatility (`sigma`). The active growth layer now uses the lagged mortgage-rate level (`mortgage_rate_lag3`) rather than the earlier mortgage-rate-gap term because it is more sign-stable across regions and cleaner in late-sample diagnostics. Financial stress is retained only as an episodic auxiliary term, not as a stable core narrative coefficient. Directional sign restrictions, sigma clipping, gamma bounds, and kappa fallback are treated as explicit guardrails rather than hidden identification claims.

        **Stage 5 role**: scenario simulations convert those calibrated parameters into five-year regional price distributions rather than single-point forecasts. The baseline is calibrated separately from stress overlays, and the scenario mechanics are now parameterised in config rather than buried as code constants.

        **Current assessment**: the canonical rebuild is materially stronger than the legacy Stage 4 and Stage 5 artifacts, but it is still not a signed-off bank forecasting model. The app therefore presents the outputs as valuation-and-vulnerability research rather than hidden ground truth.

        **Signal layer**: the deployed decision layer is a signal dashboard rather than a primary composite. Valuation is the strongest empirically supported signal. Cycle is partially supported. Downside, rent support, and household affordability are contextual signals that help interpretation but do not carry the same evidential weight.

        **Simulation transparency**: the dashboard compares historical realised 5-year regional return distributions with the canonical baseline simulation distribution. These views are calibration diagnostics, not proof of forecast superiority.

        **Audience split**: the research overview, regional signal dashboard, and scenario diagnostics are analyst-facing. The household explorer is a secondary translation layer for affordability and downside context.
        """
    )
    _render_validation_status(data, compact=False)
    st.subheader("Current coverage")
    st.json(data.metadata)

    # -----------------------------------------------------------------------
    # Structural Stability Tests
    # -----------------------------------------------------------------------
    st.subheader("Structural Stability Tests")
    try:
        from src.model.structural_breaks import run_chow_tests
        from src.ui.loaders import get_region_history
        from src.core.paths import CANONICAL_PANEL_PATH

        import pandas as _pd_sb
        _panel = _pd_sb.read_csv(CANONICAL_PANEL_PATH, parse_dates=["date"])
        _chow_df, _chow_summary = run_chow_tests(_panel)

        st.dataframe(_chow_df, use_container_width=True)
        st.caption(
            "Chow test checks whether regression coefficients are stable across the "
            "2008 financial crisis and 2020 pandemic. "
            "A significant result (p < 0.05) suggests the full-sample coefficient "
            "may not hold in the current regime."
        )

        if _chow_summary["any_breaks_detected"]:
            st.warning(
                "Structural breaks detected at 5% significance in: "
                + ", ".join(_chow_summary["break_points_detected"])
                + ". "
                + _chow_summary["interpretation"]
            )
        else:
            st.success("No structural breaks detected at 5% significance level.")
    except Exception as _e:
        st.info(f"Structural break tests could not be run: {_e}")

    # -----------------------------------------------------------------------
    # Signal Backtests
    # -----------------------------------------------------------------------
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

        with _col_c2:
            st.markdown("**C2 — Downside**")
            _c2_acc = _c2_summary.get("accuracy")
            if _c2_acc != _c2_acc:  # NaN check
                st.metric("Accuracy (24m)", "N/A")
                st.caption("Requires `downside_probability` column — not yet in panel.")
            else:
                st.metric("Accuracy (24m)", f"{_c2_acc:.1%}")
                st.caption(f"Precision: {_c2_summary.get('precision', float('nan')):.1%} | Recall: {_c2_summary.get('recall', float('nan')):.1%}")

        with _col_c3:
            st.markdown("**C3 — Affordability**")
            if len(_c3_df) == 0:
                st.metric("Accuracy (24m)", "N/A")
                st.caption("Requires `affordability_gap` or `payment_burden_gap`.")
            else:
                st.metric("Accuracy (24m)", f"{_c3_summary.get('accuracy', 0):.1%}")
                st.caption(f"Dir. IR: {_c3_summary.get('directional_information_ratio', float('nan')):.3f}")

        with _col_c4:
            st.markdown("**C4 — Rental Yield**")
            if len(_c4_df) == 0:
                st.metric("Accuracy (24m)", "N/A")
                st.caption("Requires `gross_yield_pct` or `yield_lag1`.")
            else:
                st.metric("Accuracy (24m)", f"{_c4_summary.get('accuracy', 0):.1%}")
                st.caption(f"IR proxy: {_c4_summary.get('information_ratio_proxy', float('nan')):.3f}")

        with _col_c5:
            st.markdown("**C5 — Cycle / Momentum**")
            if len(_c5_df) == 0:
                st.metric("Accuracy (12m)", "N/A")
                st.caption("Requires a momentum column in the panel.")
            else:
                st.metric("Accuracy (12m)", f"{_c5_summary.get('accuracy', 0):.1%}")
                st.caption(f"IR proxy: {_c5_summary.get('information_ratio_proxy', float('nan')):.3f}")

        st.caption(
            "Directional accuracy: fraction of calls that matched the realised direction. "
            ">55% is considered economically meaningful. "
            "Weights adjusted based on backtest results on 2026-03-31."
        )
    except Exception as _e:
        st.info(f"Signal backtests could not be run: {_e}")

    from src.core.paths import ROOT_DIR

    st.subheader("Project documentation")
    for doc_name in [
        "ASSUMPTIONS_REGISTER.md",
        "MODEL_LIMITATIONS.md",
        "AUDIT_STAGE1_TO_STAGE6.md",
        "STAGE4_STAGE5_ASSESSMENT.md",
        "RATE_CHANNEL_DIAGNOSIS.md",
        "DUAL_HORIZON_ARCHITECTURE_NOTE.md",
        "MODEL_PANEL_COVERAGE_DIAGNOSIS.md",
        "DESCRIPTIVE_PANEL_BUILD_NOTE.md",
        "HOUSE_PRICE_COVERAGE_DIAGNOSIS.md",
        "HOUSE_PRICE_EXTENSION_NOTE.md",
        "FINANCIAL_STRESS_STABILITY_NOTE.md",
        "SIGNAL_DASHBOARD_REDESIGN_NOTE.md",
        "SIGNAL_COMPONENT_DEFINITIONS.md",
        "PRODUCT_POSITIONING_NOTE.md",
        "SIGNAL_PRESENTATION_REFACTOR.md",
        "HOME_PAGE_NEUTRALITY_NOTE.md",
        "ECONOMETRIC_GUARDRAILS_AUDIT.md",
        "PRIORS_AND_GUARDRAILS_NOTE.md",
        "VALIDATION_BOUNDARY_REMEDIATION.md",
        "AUDIENCE_SPLIT_NOTE.md",
        "RESEARCH_NARRATIVE_REFINEMENT.md",
        "CHART_RESEARCH_STYLE_NOTE.md",
        "HOUSE_VIEW_SUMMARY_NOTE.md",
        "MASTERPIECE_UPGRADE_REPORT.md",
        "SIGNOFF_READINESS_CHECKLIST.md",
        "SIMULATION_RECALIBRATION_NOTE.md",
        "SIMULATION_TRANSPARENCY_NOTE.md",
        "SCORE_LAYER_EVIDENCE_REVIEW.md",
        "UI_RESEARCH_PRESENTATION_NOTE.md",
        "MODEL_UPGRADE_REPORT.md",
        "FINAL_SIGNOFF_REVIEW.md",
        "FINAL_RESEARCH_QUALITY_UPGRADE_REPORT.md",
    ]:
        path = ROOT_DIR / doc_name
        if path.exists():
            with st.expander(doc_name.replace("_", " ").replace(".md", ""), expanded=False):
                st.markdown(path.read_text(encoding="utf-8"))

    # -----------------------------------------------------------------------
    # Glossary
    # -----------------------------------------------------------------------
    from src.ui.glossary import GLOSSARY

    st.subheader("Glossary")
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


def render_comparison_view() -> None:
    ensure_state_defaults()
    apply_app_style()
    config = load_config()
    data = load_dashboard_data()

    _render_research_header(data, compact=True)
    st.title("Region Comparison")
    st.caption("Compare key housing market metrics across regions over time.")

    all_regions = config.project.regions
    default_regions = [r for r in ["London", "North East", "Yorkshire and The Humber", "South West"] if r in all_regions]
    selected = st.multiselect(
        "Select regions to compare (up to 4)",
        options=all_regions,
        default=default_regions,
    )
    if len(selected) > 4:
        st.warning("More than 4 regions selected \u2014 only the first 4 will be shown.")
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
            "Downside Probability (%)",
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
    available_fair_cols = [c for c in fair_cols if c in data.fair_panel.columns]
    panel_df = data.master.merge(
        data.fair_panel[available_fair_cols].drop_duplicates(subset=["region", "date"]),
        on=["region", "date"],
        how="left",
    )

    fig = plot_region_comparison(selected, metric, period, panel_df)
    st.pyplot(fig)
    st.caption("Indices rebased to 100 at start of selected period. Source: Land Registry, ONS.")

    if metric == "Fair Value Gap (%)":
        _has_ci = all(c in panel_df.columns for c in _CI_COLS)
        _latest = (
            panel_df[panel_df["region"].isin(selected)]
            .sort_values("date")
            .groupby("region", sort=False)
            .last()
        )
        _gap_cols = st.columns(min(len(selected), 4))
        for _i, _r in enumerate(selected[:4]):
            if _r not in _latest.index:
                continue
            _row = _latest.loc[_r]
            if not pd.notna(_row.get("fair_value_gap_log")):
                continue
            _gap_pct = float(_row["fair_value_gap_log"]) * 100.0
            _gap_cols[_i].metric(f"{_r}", f"{_gap_pct:+.1f}%", help="Current fair value gap (log-scale approximation to %)")
            if _has_ci and pd.notna(_row.get("fair_value_gap_lower_90")):
                _lo = float(_row["fair_value_gap_lower_90"]) * 100.0
                _hi = float(_row["fair_value_gap_upper_90"]) * 100.0
                _sig = (
                    "statistically significant"
                    if (_lo > 0 and _hi > 0) or (_lo < 0 and _hi < 0)
                    else "not statistically significant at 90% level"
                )
                _gap_cols[_i].caption(f"90% CI: [{_lo:.1f}%, {_hi:.1f}%] — gap is {_sig}")
            else:
                _gap_cols[_i].caption("Run pipeline to compute confidence intervals.")
