# UI Section Guide

This document explains what each section of the Stage 7 Streamlit UI is for,
what the user should use it for, and how each page should be interpreted.

Primary navigation is defined in `src/ui/app.py`. The sidebar pages are:

1. `Research Overview`
2. `Regional Signals`
3. `Region Comparison`
4. `Scenario & Diagnostics`
5. `Household Explorer`
6. `Methodology`
7. `Technical Controls`
8. `Limitations`

## 1. Research Overview

Purpose:
Provide the main top-level research read for the UK housing market and a single
selected region.

Audience:
General reviewer, supervisor, bank-style research reviewer, or first-time user.

What this page contains:
- `House View Summary`: a plain summary of how stretched or supportive the
  current cross-regional market looks.
- `Market Snapshot`: the selected region's current model date, latest raw house
  price context, gross yield, and research mode.
- `Validation status`: a compact statement of the current validation state,
  including rate stability and benchmark performance.
- `Key Housing Signals`: the main signal dashboard for the selected region.
- `Risk Context`: scenario ranges and simulation calibration views.
- `Cross-regional orientation`: optional rankings and downside heatmap.
- `Coverage and source detail`: source windows and region-level coverage table.

How to interpret it:
- This is the best starting page for understanding the project.
- Valuation and downside context should be read before any forecast-like use.
- Later descriptive data are shown as context only and are not silently folded
  back into the calibrated Stage 4 model.

What this page is not:
- Not a trading screen.
- Not a real-time market dashboard.
- Not a property-specific recommendation tool.

## 2. Regional Signals

Purpose:
Provide the main analyst-facing regional dashboard for institutional-style
review of one region at a time.

Audience:
Research analyst, marker, dissertation reviewer, or bank-style model reviewer.

What this page contains:
- `Headline View`: the current read on valuation, cycle, and downside.
- `Supporting Signals`: the signal panel with evidence tags.
- `Supporting Charts`: fan chart, score breakdown, scenario comparison, and
  yield-versus-risk context.
- `Research-style commentary`: the region's main risks in plain language.

Core signals shown here:
- `Valuation`: strongest evidence base.
- `Cycle`: partially supported.
- `Downside`: scenario/risk context.
- `Rent support / yield`: contextual rather than fully validated.

How to interpret it:
- This is the core institutional page.
- The component signals matter more than the blended score.
- A reviewer should focus first on valuation, then downside, then cycle.

What this page is not:
- Not a probability-weighted forecast table.
- Not a REIT backtest engine.

## 3. Region Comparison

Purpose:
Compare a small set of regions over time on the same chart.

Audience:
Anyone trying to understand regional dispersion rather than a single-region
case study.

User inputs:
- Up to 4 regions.
- Comparison metric.
- Time window.

Metrics available:
- Nominal House Price Index.
- Real House Price Index.
- Fair Value Gap.
- Downside Probability.
- Rental Yield.

How to interpret it:
- This page is for relative comparison.
- It is especially useful for seeing whether London and the regions are moving
  together or diverging.
- When `Fair Value Gap (%)` is selected, the page also shows the latest gap and
  confidence-interval context where available.

What this page is not:
- Not the main decision page.
- Not the place to judge model validation.

## 4. Scenario & Diagnostics

Purpose:
Stress-test a region under alternative macro assumptions and inspect the
simulation behaviour.

Audience:
Analyst or technically engaged reviewer.

User inputs:
- Region.
- Scenario preset.
- Additional rate shock.
- Volatility multiplier.
- Drift override.
- Horizon.
- Number of simulation paths.

What this page contains:
- Scenario-specific fair-value and rate assumptions.
- Fan chart under the selected stress setup.
- Scenario comparison table across presets.
- Downloadable scenario path CSV.
- Simulation calibration diagnostics.

How to interpret it:
- This is a perturbation and stress-testing tool.
- It does not re-estimate the econometric model live.
- It helps answer: "What happens if the macro path is worse, better, or more
  volatile than baseline?"

What this page is not:
- Not a model retraining page.
- Not proof that a given macro scenario is the most likely one.

## 5. Household Explorer

Purpose:
Translate the research system into an affordability and downside view for a
household-level example.

Audience:
Non-technical user, student reviewer, or someone wanting to see consumer-side
implications.

User inputs:
- Region.
- Household income.
- Deposit.
- Mortgage term.
- Target property price.
- Risk tolerance.
- Macro scenario.

What this page contains:
- `Headline View`: plain-language summary for the chosen household setup.
- `Supporting Signals`: affordability, valuation, cycle, and downside context.
- `Monthly payment` and `Mortgage stress` outputs.
- Fan chart and scenario comparison.
- Key risks and caveats.

How to interpret it:
- This page is secondary to the core regional research dashboard.
- It is useful for illustrating affordability pressure and downside exposure.
- It should not be used as mortgage advice or as a buy / do-not-buy decision
  engine.

What this page is not:
- Not financial advice.
- Not lender underwriting.

## 6. Methodology

Purpose:
Explain how the model works, what data it uses, and what the current validation
position is.

Audience:
Marker, technical reviewer, supervisor, or model-risk reviewer.

What this page contains:
- High-level explanation of the pipeline design.
- The distinction between model-panel data and descriptive data.
- Current validation status.
- Structural stability tests.
- Signal backtest summaries.
- Links to the wider project documentation.
- Glossary of terms.

How to interpret it:
- This is the main "how it works" page.
- It should be read together with the `Limitations` page.
- It is the best page for defending the project academically or institutionally.

## 7. Technical Controls

Purpose:
Expose calibrated parameters and override guardrails without allowing users to
change the core model calibration.

Audience:
Technical reviewer or analyst.

What this page contains:
- Region-level calibrated parameters such as `kappa`, `sigma`,
  `mu_equilibrium`, `P_star_now`, and `gamma_annual`.
- Interactive guardrails for scenario overrides.
- Glossary-backed parameter explanations.

How to interpret it:
- This page improves transparency and auditability.
- It shows what the model is actually using under the hood.
- The values are read-only and are not meant to be hand-tuned here.

What this page is not:
- Not a calibration editor.
- Not a research notebook.

## 8. Limitations

Purpose:
State the model risks, validation weaknesses, and practical constraints
explicitly.

Audience:
Everyone. This page is mandatory reading before serious use.

What this page contains:
- `4 material risks`
- `10 significant limitations`
- `8 technical notes`
- Data-currency warning based on `effective_data_as_of`

Key themes covered:
- publication lag and stale-data risk;
- uneven source coverage;
- holdout validation weakness versus zero-growth;
- regional calibration issues;
- descriptive versus validated signal boundaries;
- structural break and simulation limitations.

How to interpret it:
- This page defines the safe boundary of the project.
- If the `Methodology` page explains how the system works, this page explains
  where it can fail.

## Recommended Reading Order

For a first-time reviewer:

1. `Research Overview`
2. `Regional Signals`
3. `Methodology`
4. `Limitations`
5. `Scenario & Diagnostics`

For a bank-style or dissertation-style assessment:

1. `Methodology`
2. `Limitations`
3. `Research Overview`
4. `Regional Signals`
5. `Technical Controls`

## One-Sentence Summary

The UI is structured so that the user starts with a high-level regional housing
research view, drills into region-level signals and scenarios, and then uses
the methodology and limitations pages to understand exactly how much weight the
outputs should carry.
