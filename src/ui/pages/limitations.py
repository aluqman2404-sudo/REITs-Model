"""
Limitations and Disclosures page — Stage 7 Streamlit dashboard.

Surfaces all disclosures across three tiers:
  • 4 material risks   (always visible, st.error)
  • 10 significant limitations  (st.expander, st.warning)
  • 8 technical notes  (st.expander, st.info)

The data-currency block reads effective_data_as_of from
canonical_panel_metadata.json and alerts users when the panel is stale.
"""

from __future__ import annotations

import json
from datetime import date

import streamlit as st

from src.core.paths import CANONICAL_PANEL_METADATA_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_panel_metadata() -> dict:
    if not CANONICAL_PANEL_METADATA_PATH.exists():
        return {}
    with CANONICAL_PANEL_METADATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Page render function
# ---------------------------------------------------------------------------

def render_limitations() -> None:
    """Render the Limitations & Disclosures page."""

    meta = _load_panel_metadata()
    panel_start = meta.get("panel_start", "2005-01-01")
    panel_end = meta.get("panel_end",   "2025-10-01")
    effective = meta.get("effective_data_as_of", panel_end)

    # -----------------------------------------------------------------------
    # Title and subtitle
    # -----------------------------------------------------------------------
    st.title("Model Limitations & Disclosures")
    st.caption(
        f"v0.2.0 \u2014 Panel coverage {panel_start} to {panel_end}"
        " \u2014 Read before using model outputs"
    )

    # -----------------------------------------------------------------------
    # Summary line
    # -----------------------------------------------------------------------
    st.markdown(
        "<span style='color:#d32f2f; font-weight:bold;'>4 material risks</span>"
        " \u00b7 "
        "<span style='color:#f57c00; font-weight:bold;'>10 significant limitations</span>"
        " \u00b7 "
        "<span style='color:#616161;'>8 technical notes</span>",
        unsafe_allow_html=True,
    )

    import pandas as _pd_lim
    _LREF = [
        ("L1",  "Material", "Data currency / publication-lag risk"),
        ("L2",  "Significant", "Source series have unequal coverage endpoints"),
        ("L3",  "Significant", "Publication lags in earnings and population data"),
        ("L4",  "Significant", "Model does not beat zero-growth RMSE in 2022–2025 holdout"),
        ("L5",  "Technical", "Directional accuracy is modest (56–62% at 1–6 months)"),
        ("L6",  "Material", "No transactions volume data"),
        ("L7",  "Significant", "C1 valuation is the only empirically validated signal"),
        ("L8",  "Technical", "Cycle signal (C2) has only partial empirical support"),
        ("L9",  "Significant", "Downside, affordability, and yield signals are descriptive only"),
        ("L10", "Significant", "12-month score signal is weak and non-monotonic"),
        ("L11", "Technical", "Score IR cannot be verified in current market conditions"),
        ("L12", "Technical", "Regression on regional averages; property-level factors not captured"),
        ("L13", "Significant", "Annual data interpolated to monthly — timing within year unreliable"),
        ("L14", "Material", "Northern Ireland calibration gap; Wales/Scotland data sparser"),
        ("L15", "Technical", "Residual autocorrelation present in 9 of 12 regions"),
        ("L16", "Technical", "Not tested against structural breaks outside training window"),
        ("L17", "Technical", "Simulated baseline median understates historical 5-year return"),
        ("L18", "Significant", "Scenario paths conditional on assumed macro trajectories"),
        ("L19", "Significant", "Scenario Lab is a perturbation tool, not a full model re-run"),
        ("L20", "Significant", "Scores are research indicators, not investment advice"),
        ("L21", "Technical", "Score labels can create false precision without component decomposition"),
        ("L22", "Material", "Held-out validation remains below forecasting sign-off standard"),
    ]
    with st.expander("Quick-reference index (all L-numbers)", expanded=False):
        _tier_colours = {"Material": "#d32f2f", "Significant": "#f57c00", "Technical": "#616161"}
        st.dataframe(
            _pd_lim.DataFrame(_LREF, columns=["L-number", "Tier", "Short title"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Expand the sections below to read each limitation in full.")

    # -----------------------------------------------------------------------
    # Tier 1 — Material Risks (always visible, no expander)
    # -----------------------------------------------------------------------
    st.error(
        "**L1 \u2014 Data currency / publication-lag risk.** "
        "Core signals are anchored to the effective data date and may lag current market conditions materially; see data currency status below.",
        icon="\U0001f6a8",
    )
    st.error(
        "**L6 \u2014 No transactions volume data.** "
        "Model cannot distinguish thin-market price moves from genuine demand shifts.",
        icon="\U0001f6a8",
    )
    st.error(
        "**L14 \u2014 Northern Ireland calibration gap.** "
        "NI median sits 19\u2009pp above model estimate; gap is gamma/kappa-driven, "
        "not resolvable by sigma adjustment. "
        "Wales, Scotland, and Northern Ireland have sparser supply and "
        "affordability data; model confidence is lower for these regions.",
        icon="\U0001f6a8",
    )
    st.error(
        "**L22 \u2014 Held-out validation remains below sign-off standard.** "
        "The model has now been tested out of sample and beats AR(1), but it does not yet beat zero-growth RMSE in the post-rate-shock holdout. This blocks full forecasting-model sign-off.",
        icon="\U0001f6a8",
    )

    # -----------------------------------------------------------------------
    # Data currency status block (placed immediately after Tier 1 cards)
    # -----------------------------------------------------------------------
    try:
        effective_date = date.fromisoformat(str(effective)[:10])
    except (ValueError, TypeError):
        effective_date = date.today()

    age_days = (date.today() - effective_date).days
    age_msg = (
        f"Panel is **{age_days} days old**. "
        f"Outputs reflect conditions as at **{effective}**, "
        "not current market prices. "
        "Refresh the panel before relying on any signal."
    )

    if age_days > 180:
        st.error(age_msg, icon="\U0001f6a8")
    elif age_days > 90:
        st.warning(age_msg, icon="\u26a0\ufe0f")
    else:
        st.success(f"Panel is current ({age_days} days old).", icon="\u2705")

    st.divider()

    # -----------------------------------------------------------------------
    # Tier 2 — Significant Limitations
    # -----------------------------------------------------------------------
    st.markdown("#### Significant Limitations")
    _TIER2 = [
        ("L2", f"Source series have unequal coverage endpoints", f"""
**L2 \u2014 Source series have unequal coverage endpoints.**
The canonical model panel stops at the weakest OLS model input (HMRC
transactions, {meta.get("weakest_model_input_end", "2025-10-01")}). Rental
yield is forward-filled from its last quarterly observation (within the
18-month governance limit; see R025). Land Registry house prices extend to
{meta.get("latest_house_price_end", "2025-12-01")} and are descriptive context
only. Rental levels end {meta.get("latest_rental_level_end", "2025-02-01")}.
"""),
        ("L3", "Publication lags in earnings and population data", f"""
**L3 \u2014 Publication lags mean that even the endpoint reflects data
available somewhat later.**
Annual earnings data (ASHE) has a publication lag of approximately six months.
Regional population data has a lag of approximately twelve months. The
`effective_data_as_of` field in the panel metadata (**{effective}**) reports
the date at which all series would have been simultaneously available to a
real-time analyst.
"""),
        ("L4", "Model does not beat zero-growth RMSE in 2022\u20132025 holdout", """
**L4 \u2014 The model does not reliably beat zero-growth on RMSE in the
2022\u20132025 holdout period.**
In out-of-sample testing across the 2022\u20132025 period \u2014 which
coincides with the UK\u2019s most severe mortgage repricing shock since 2008
\u2014 the model\u2019s RMSE at all forecast horizons does not improve on the
naive zero-growth baseline. This is a period structurally different from the
2005\u20132021 training window. Forecasts should not be read as superior to a
simple market-consensus view during regime changes.
"""),
        ("L7", "C1 valuation is the only empirically validated signal", """
**L7 \u2014 The valuation signal (C1 mispricing) is the only empirically
validated signal.**
The valuation signal, derived from the Stage 4 log fair-value gap, shows a
statistically meaningful forward-return spread at 36-month and 60-month
horizons (36-percentage-point spread between Supportive and Cautious regions
at 60 months, based on the historical backtest). This is the primary evidence
base for the model\u2019s usefulness as a research tool.
"""),
        ("L9", "Downside, affordability, and yield signals are descriptive only", """
**L9 \u2014 Downside, affordability, and yield signals are descriptive
only.**
These signals compress scenario outcomes and user-specific financial metrics
into scores. They have not been backtested against realised investment
outcomes. They are useful for framing the risk environment and for
user-specific affordability analysis, but they should not be presented as
validated forward-return predictors.
"""),
        ("L10", "12-month score signal is weak and non-monotonic", """
**L10 \u2014 The 12-month score signal is weak and non-monotonic.**
Backtest evidence shows that Supportive-vs-Cautious return spread at 12 months
is not reliable and is not monotonic across the score range. The threshold
calibration is anchored to 36-month evidence. Users should not rely on signals
for short-horizon decisions without this qualification.
"""),
        ("L13", "Annual data interpolated to monthly — timing within year unreliable", """
**L13 \u2014 Annual data series are interpolated to monthly frequency; timing
precision within a calendar year is unreliable.**
Earnings (ASHE), housing supply (MHCLG), population (ONS mid-year estimates),
and affordability ratios are collected annually. These are carried forward as
as-of values between annual releases. Valuation and cycle signals that depend
on these features \u2014 including `earnings_growth_12m_lag1` and supply-side
features \u2014 should not be used to draw conclusions about timing within a
single calendar year. Flag columns `earnings_growth_12m_lag1_is_interpolated`
and `net_additions_monthly_is_interpolated` in the canonical panel and Stage 4
output identify affected observations.
"""),
        ("L18", "Scenario paths conditional on assumed macro trajectories", """
**L18 \u2014 Scenario paths are conditional on assumed macro trajectories.**
Each of the five scenario presets (Baseline, Soft_Landing, Rate_Shock,
Stagflation, Recovery_Boom) embeds expert-designed assumptions about rate
shifts, income growth, stress overlays, and regime probabilities. These
assumptions are internally consistent but are not derived from a macro model.
The model cannot tell users which scenario is more likely to occur.
"""),
        ("L19", "Scenario Lab is a perturbation tool, not a full model re-run", """
**L19 \u2014 The interactive Scenario Lab is a calibrated perturbation tool,
not a full model re-run.**
The live simulation in the dashboard perturbs fair-value anchors and volatility
around calibrated values. It does not re-estimate OLS coefficients or recompute
P&#42;. Extreme user-specified overrides (rate shifts > 150\u00a0bps, volatility
multipliers > 1.5\u00d7) trigger a warning in the UI but are not prevented.
"""),
        ("L20", "Scores are research indicators, not investment advice", """
**L20 \u2014 Scores are descriptive research indicators, not investment
advice.**
Supportive, Mixed, and Cautious labels are research bucketing tools calibrated
to the historical valuation signal. They are not recommendations to buy, hold,
or sell any property or security. The composite consumer and REIT scores blend
empirically-supported and descriptive-only signals using expert weights; they
have not been backtested as trading signals and should not be used as such.
"""),
    ]
    for _lnum, _title, _body in _TIER2:
        with st.expander(f"{_lnum} — {_title}", expanded=False):
            st.warning(_body)

    # -----------------------------------------------------------------------
    # Tier 3 — Technical Notes
    # -----------------------------------------------------------------------
    st.markdown("#### Technical Notes")
    _TIER3 = [
        ("L5", "Directional accuracy is modest", """
**L5 \u2014 Directional accuracy is modest.**
The model correctly calls the direction of monthly price movements 56\u201362%
of the time at 1\u20136 month horizons, and approximately 50% (coin-flip) at
12 months. This is better than random, but it is not a high-confidence timing
signal. The model is more valuable as a relative valuation framework and
scenario organiser than as a short-horizon price direction predictor.
"""),
        ("L8", "Cycle signal (C2) has only partial empirical support", """
**L8 \u2014 The cycle and momentum signal (C2) has only partial empirical
support.**
The cycle signal is informed by the OLS growth equation and the scenario
simulation path. It has directional plausibility but weaker validated
return-predictive power than the valuation signal at any tested horizon. It
should be treated as cycle context rather than a timing rule.
"""),
        ("L11", "Score IR cannot be verified in current market conditions", """
**L11 \u2014 Score information ratio cannot be computed in current
conditions.**
As of the model observation date, all 12 regions score \u2265\u202045, leaving
no Cautious contrast group. The historical backtest IR of 0.68 at 36 months
remains the primary evidence, but this cannot be verified against current
market conditions until market repricing produces regions scoring below 45.
"""),
        ("L12", "Regression on regional averages — property-level factors not captured", """
**L12 \u2014 Regression operates on regional averages; property-level factors
are not captured.**
The model does not observe individual borrower credit quality, mortgage product
mix, property condition, floor area, local micro-market dynamics, or tax
position. All signals apply to a notional regional average property. Actual
returns on specific properties will differ materially.
"""),
        ("L15", "Residual autocorrelation present in 9 of 12 regions", """
**L15 \u2014 Residual autocorrelation is present in 9 of 12 regions.**
Breusch-Godfrey order-3 tests confirm autocorrelated residuals in most
regions. This reflects UK macroeconomic cycle persistence rather than a
correctable model misspecification, but it means HAC (Newey-West) standard
errors must be used for all inference, and the model does not fully capture
serial dependencies in price dynamics.
"""),
        ("L16", "Not tested against structural breaks outside training window", """
**L16 \u2014 The model has not been tested against structural breaks outside
the training window.**
The calibration period (2005\u20132025) includes the 2008 global financial
crisis and the 2022\u20132024 rate shock, but does not include a sustained
housing market crash of the magnitude seen in Ireland or the US in
2008\u20132012, or pre-2005 UK episodes. Behaviour under conditions outside
the training distribution is unknown.
"""),
        ("L17", "Simulated baseline median understates historical 5-year return", """
**L17 \u2014 The simulation baseline median substantially underestimates the
historical 5-year return distribution.**
Independent validation shows the simulated baseline median 5-year return is
3.07% against a historical mean of 18.05% across UK regions. The simulation
is useful for scenario ordering (which scenarios produce better/worse outcomes
relative to each other) and for tail-risk assessment, but the absolute level
of the simulated paths should not be read as a calibrated price forecast.
"""),
        ("L21", "Score labels can create false precision without component decomposition", """
**L21 \u2014 Score labels can create false precision if read without the
component decomposition.**
A single composite score compresses four or more signal components into one
number. The component-level breakdown and the evidence classification for each
signal are essential context. Reviewers and users should always examine the
full signal panel, not only the composite headline score.
"""),
    ]
    for _lnum, _title, _body in _TIER3:
        with st.expander(f"{_lnum} — {_title}", expanded=False):
            st.info(_body)

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------
    st.divider()
    st.caption(
        "These disclosures are reproduced from SIGN_OFF_CHECKLIST.md "
        "\u00a7\u00a73.1\u20133.6 (v0.2.0). "
        "Any change to model assumptions or guardrails requires this page "
        "to be updated in the same commit."
    )
