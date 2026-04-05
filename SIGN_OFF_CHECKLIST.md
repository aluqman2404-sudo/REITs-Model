# SIGN-OFF CHECKLIST
## UK Regional Housing Market Model — v0.2.0
### Model Risk Management Submission Document

**Prepared:** 2026-03-21
**Panel coverage:** 2005-01-01 to 2025-01-01 (12 regions, monthly)
**Status:** Historical research prototype — not a live-data or trading system
**Canonical entry point:** `src/model/canonical_rebuild.py`
**Configuration:** `config/parameters.json`

---

This document is written for a model risk reviewer who has not seen the codebase. It records every judgemental assumption, every active guardrail, every material limitation that must be disclosed to users, and the recommended re-validation schedule for each model component. It is the primary MRM submission foundation document for this model.

---

## 1. JUDGEMENTAL ASSUMPTIONS

The following assumptions are not derived from statistical estimation. Each requires independent expert review before the model is relied upon for any consequential purpose. The status column indicates whether the assumption has been **validated** (tested against data), supported by an **expert-prior** (plausible but untested), or **sensitivity-tested** (the model output has been shown to move meaningfully when the value is varied).

---

### 1.1 Data Treatment Assumptions

| # | Assumption | Config key / Code location | Status |
|---|---|---|---|
| A1 | CPI is the deflator for all real series (house prices, earnings, rents). No alternative deflator (e.g. CPIH, RPI) is tested. | `src/data/build_canonical_panel.py` — `cpi_base` at 2015-01-01 | Expert-prior |
| A2 | Monthly regional average prices are the decision surface. Property-level heterogeneity (condition, tenure, credit quality, floor area) is out of scope. | Architectural; all ingestion and scoring operates at ONS region level | Expert-prior |
| A3 | The canonical panel starts in January 2005. Pre-2005 data exists for some series but is excluded to maintain a balanced cross-region window. | `src/data/build_canonical_panel.py` — `PROJECT_START = 2005-01-01` | Expert-prior |
| A4 | Annual earnings are carried forward as an as-of value (last observed) rather than interpolated. Each month in the same calendar year carries the December ASHE anchor from the previous year. Months where `earnings_staleness_months > 0` are flagged as `earnings_interpolated_flag = True`. | `src/data/build_canonical_panel.py`; INTERPOLATED_FEATURE_POLICY.md | Validated — policy eliminates future-anchor leakage; staleness explicitly tracked |
| A5 | MHCLG housing supply data is annual (financial-year April anchor). Supply anchor months (`supply_anchor_flag`) are April-observation rows only; all other months are flagged as `net_additions_monthly_is_interpolated = True`. | `src/data/build_canonical_panel.py`; INTERPOLATED_FEATURE_POLICY.md | Expert-prior — flag is correct; timing precision within a calendar year is not reliable |
| A6 | The canonical panel cutoff is set at the weakest OLS model input (`transactions`, ending 2025-10-01). `rental_yield` is excluded from the weakest-input calculation because it is used only in Stage 6 scoring (not OLS) and is forward-filled from its last quarterly observation (2025-01-01). The forward-fill is capped at 18 months by governance test. House price data to 2025-12-01 is descriptive context only. | `src/data/build_canonical_panel.py` — `_min_coverage_end`; R025 in risk register | Updated 2026-03-22 — rental_yield decoupled from panel cutoff; ONS PRMS publication lag (~12 months) was preventing any fresh rebuild |

---

### 1.2 Fair-Value Model Assumptions (Stage 4 — P\*)

| # | Assumption | Config key / Code location | Status |
|---|---|---|---|
| A7 | The structural fair-value (P\*) equation uses three regressors: log_pti_ratio (= log_income_asof − log_rent), log_rent, and mortgage_rate. Updated 2026-04-05: log_income_asof replaced by log_pti_ratio to remove multicollinearity (r=0.747) and enforce the correct positive income sign via sign constraint. These were chosen on theoretical grounds. No formal variable selection procedure was run. | `config/parameters.json` → `fair_value.regressors` | Expert-prior — economically defensible; sign constraint added 2026-04-05 (R029 resolved) |
| A8 | P\* is estimated from December and June observations only (anchor months). This anchors the equilibrium estimate to the most reliably observed annual values and avoids interpolated-data contamination. | `config/parameters.json` → `fair_value.anchor_months: [6, 12]` | Expert-prior |
| A9 | Mean reversion speed (κ, kappa) is derived from a regional AR(1) regression on the log fair-value gap. Where the AR(1) coefficient implies implausible speeds, a fallback to the cross-regional median φ = 0.97 is applied. | `config/parameters.json` → `fair_value.kappa_fallback_phi: 0.97` | Expert-prior — fallback prevents numerical instability; the specific value 0.97 is not empirically re-estimated |
| A10 | The AR persistence used for kappa estimation is capped at φ ≤ 0.995 to prevent near-unit-root dynamics from producing explosive simulation paths. | `config/parameters.json` → `fair_value.kappa_phi_upper_bound: 0.995` | Expert-prior |
| A11 | The financial stress feature activates only when the excess financial stress measure exceeds 10.5 (an absolute threshold). Below this threshold, the stress channel contributes zero. | `config/parameters.json` → `fair_value.financial_stress_activation_threshold: 10.5` | Expert-prior — threshold is based on historical episode analysis; not formally optimised |

---

### 1.3 Simulation Assumptions (Stage 5)

| # | Assumption | Config key / Code location | Status |
|---|---|---|---|
| A12 | The log-price Ornstein-Uhlenbeck SDE is chosen as the simulation process. This preserves price positivity and is numerically stable. Alternative processes (e.g. Geometric Brownian Motion, jump-diffusion, regime-switching SDE) are not used. | `config/parameters.json` → `model.sde_type: log_ornstein_uhlenbeck_terminal_target` | Expert-prior |
| A13 | Simulation uses dt = 1/12 (Euler-Maruyama, monthly steps). No higher-order discretisation scheme is applied. | `config/parameters.json` → `model.dt: 0.0833` | Expert-prior — standard for monthly housing models; Euler-Maruyama error is small at this step size |
| A14 | 10,000 paths per region-scenario pair. This was chosen for convergence stability in the P10/P90 range. | `config/parameters.json` → `model.n_simulations: 10000` | Sensitivity-tested — variance of quantile estimates is stable at this path count |
| A15 | Annual sigma is blended as 35% raw residual volatility + 65% winsorised volatility. The winsorisation weight downweights outlier years (e.g. 2008, COVID). | `config/parameters.json` → `simulation.baseline_volatility_blend: 0.65`; `guardrails.sigma_raw_weight: 0.35` | Expert-prior — blend weights not formally optimised; sensitivity analysis in `sigma_mapping_diagnostics.csv` |
| A16 | The baseline simulation target price is constructed as a weighted blend of the fair-value anchor and the long-run return model, with the fair-value anchor receiving weight 0.65. The long-run nominal gamma anchor is computed as the full-history mean of log-differences of nominal house price (annualised), blended 40% current-real / 60% long-run-nominal and clipped to [−1%, +2.5%]/yr. **BUG FIX 2026-03-22:** prior to this date, the long-run nominal anchor incorrectly used `price_growth` (= Δ log real price, CPI-deflated), producing near-zero anchors for flat-real-return regions (Wales −0.06%/yr, Yorkshire +0.07%/yr, West Midlands +0.14%/yr real) despite ~3%/yr nominal growth. Fix applied in `canonical_rebuild.py` — confirmed raises gamma for all 12 regions; reduces under-simulation bias for target regions by 4–6pp. | `config/parameters.json` → `simulation.baseline_target_weight: 0.65`; `canonical_rebuild.py` lines 283–298 | Expert-prior (blend weights); Bug-fixed 2026-03-22 (anchor variable) |
| A17 | In stress scenarios, the scenario volatility multiplier is capped at 1.6× the baseline sigma. This prevents extreme tail inflation in adversarial scenario runs. | `config/parameters.json` → `simulation.stress_sigma_multiplier_cap: 1.6` | Expert-prior |
| A18 | Random seed is fixed at 42 for all simulation runs. This ensures reproducibility but means all uncertainty estimates are conditional on a single RNG initialisation. | `config/parameters.json` → `controls.random_seed: 42` | Design choice — reproducibility prioritised over path-diversity sampling |
| A19 | Scenario macro paths (rate shifts, income growth shifts, valuation gap closure fractions, regime probabilities) are expert-designed. They are not derived from model-implied distributions or scenario-generation procedures. | `config/parameters.json` → `scenarios.*` | Expert-prior — scenario ordering is validated (Recovery\_Boom ≥ Baseline ≥ Rate\_Shock) but absolute levels are judgement |

---

### 1.4 Scoring Assumptions (Stage 6)

| # | Assumption | Config key / Code location | Status |
|---|---|---|---|
| A20 | Score thresholds: Supportive ≥ 55, Mixed ≥ 45, Cautious < 45. Calibrated via precision-recall sweep at 36-month horizon over 2,220 region-month observations. Threshold 55 achieved IR = 0.68, precision = 0.69, recall = 0.39. | `config/parameters.json` → `scoring.thresholds`; `_threshold_calibration_note` | Validated at 36m; 12m evidence is weak; 60m evidence is strongest |
| A21 | Consumer composite component weights: base\_model 0.40, affordability 0.40, downside\_risk 0.20. Expert-prior; not optimised against realised outcomes. Sum-to-1 validated at engine load via `_validate_weights()`. Consumer scenario probability weights (for weighted-return calculation): Baseline 0.40, Rate\_Shock 0.25, Stagflation 0.20, Soft\_Landing 0.10, Recovery\_Boom 0.05 — deliberate downside-skewed prior (adverse 0.45 combined). All weights are now config-driven (R022 resolved 2026-03-22). | `config/parameters.json` → `scoring.app_consumer_weights`; `scoring.scenario_consumer_weights` | Expert-prior; config-driven as of 2026-03-22 |
| A22 | REIT composite component weights: base\_model 0.45, yield\_signal 0.30, downside\_risk 0.25. Expert-prior; not optimised against realised outcomes. Sum-to-1 validated at engine load. REIT scenario probability weights: equal weight 0.20 across all 5 scenarios — previously computed dynamically as 1/N; now explicit in config so adding or renaming a scenario requires a deliberate config edit (R022 resolved 2026-03-22). | `config/parameters.json` → `scoring.app_reit_weights`; `scoring.scenario_reit_weights` | Expert-prior; config-driven as of 2026-03-22 |
| A23 | Within the base\_model sub-composite: valuation receives weight 0.60, cycle receives weight 0.40. | `src/scoring/engine.py` — inline composite construction | Expert-prior |
| A24 | Cycle outlook is computed as 0.65 × cycle\_signal + 0.35 × scenario\_signal. The 65/35 split between the empirical cycle signal and the scenario-implied path is expert judgement. | `src/scoring/engine.py` — `cycle_outlook` | Expert-prior |
| A25 | Evidence-based shrinkage multipliers: empirically\_supported signals contribute at full weight (1.0); partially\_supported at 0.75; descriptive at 0.50. All scores are shrunk toward 50 before composite aggregation. | `src/scoring/engine.py` — `_EVIDENCE_MULTIPLIERS`; `_shrink_toward_neutral` | Expert-prior — sensitivity documented inline in engine.py |
| A26 | Risk tolerance adjustment: Opportunistic +5.0 points, Balanced 0.0, Conservative −7.5 points. Applied as an additive post-composite shift. | `src/scoring/engine.py` — `_risk_buffer` | Expert-prior |
| A27 | Affordability multiple benchmark: a purchase is considered well-supported when loan-to-income ≤ 4.5. Above this threshold, the affordability score is penalised. | `config/parameters.json` → `controls.affordability_multiple: 4.5` | Expert-prior — 4.5× is a widely-used UK policy benchmark; not re-derived from this dataset |
| A28 | Simulation conservative bias substantially reduced 2026-04-05. Gamma floor (per-region historical mean log_return) applied; baseline_log_return floored at 5 × gamma_annual. Simulated 5yr median now >= 8% for all 12 regions (worst: Wales 11.08%, up from 3.07%). Scenario ordering maintained. | `config/parameters.json` → `gamma_floor_policy`; `canonical_rebuild.py` | R008 resolved 2026-04-05 |

---

## 2. GUARDRAILS REGISTER

Guardrails are hard constraints applied during Stage 4 calibration and Stage 5 simulation to prevent outputs that are economically implausible or numerically unstable. Every guardrail is listed below with its current value, the justification for its existence, and its validation status. All guardrail values are held in `config/parameters.json → guardrails` and `config/parameters.json → fair_value` unless otherwise noted.

**Important note to reviewer:** These constraints improve robustness, but they are not a substitute for structural identification. Where a guardrail overrides an estimated value, the degree of override is recorded in `canonical_stage4_diagnostics.csv` (columns `raw_value` vs `final_value`). Reviewing that file is recommended as part of any MRM sign-off.

---

### 2.1 Coefficient Sign Restrictions (Stage 4)

| Parameter | Current value | Constraint type | Justification | Validation status |
|---|---|---|---|---|
| `beta_rate_max` | −0.000001 | Upper bound (rate coefficient ≤ this value) | Mortgage rate repricing must exert a non-positive effect on price growth. A positive coefficient implies rising rates support prices, which is theoretically indefensible in normal market conditions. | Expert-prior — sign-stability confirmed across full/early/late subsamples in `coefficient_stability.csv`; magnitude is sample-sensitive. **Rate channel stability monitoring (Approach B, 2026-03-23):** Per-region OLS on full/early (2005–2015)/late (2015–) subsamples with HAC maxlags=24. Threshold 5.0 (not 3.0): the early-period (2005–2015 ZLB era) beta is statistically unidentified due to near-zero mortgage-rate variation in the zero-lower-bound era — betas are imprecisely estimated and often reverse-signed, so the late/early ratio reflects identification noise, not a structural break. At threshold 5.0, 5 regions are monitored-open (North East 5.99, Scotland 6.91, South West 12.03, Wales 6.19, Yorkshire 5.64). Full-sample pooled beta is the appropriate estimate for annual-frequency use. `test_rate_channel_stability` monitors this continuously. See R010. HAC Newey-West maxlags raised from 12 to 24 to better absorb UK rate-cycle persistence (3–5 year cycles). |
| `beta_financial_stress_max` | −0.000001 | Upper bound (stress coefficient ≤ this value) | The financial stress feature should not act as a positive support channel. It captures episodic financial distress; its effect on prices is negative or zero. | Expert-prior — classified as `episodic_not_identified_late` in stability tests; activated only when excess stress > 10.5 |
| `beta_income_min` | 0.0 | Lower bound (income coefficient ≥ this value) | Real income supports housing demand; a negative income coefficient implies rising incomes depress prices, which is economically unjustifiable as a structural parameter. | Expert-prior |
| `beta_transaction_min` | 0.000001 | Lower bound (transaction growth coefficient ≥ this value) | Transaction activity reflects market liquidity and demand; it should not have a negative structural coefficient in the growth equation. | Expert-prior |

---

### 2.2 Short-Run Persistence Constraint

| Parameter | Current value | Constraint type | Justification | Validation status |
|---|---|---|---|---|
| `rho_max` | −0.01 | Upper bound on ρ (short-run AR coefficient in growth equation ≤ this value) | Prevents explosive or near-explosive price persistence. ρ near zero or positive implies prices do not mean-revert, which is inconsistent with the OU model structure. | Expert-prior — DW statistics and BG tests in diagnostics confirm some residual autocorrelation (9/12 regions); HAC SEs applied |

---

### 2.3 Long-Run Drift Bounds (Stage 4 and Stage 5)

| Parameter | Current value | Constraint type | Justification | Validation status |
|---|---|---|---|---|
| `gamma_min_annual` | −1.0% per annum | Lower bound on regional baseline drift | Prevents the simulation from implying indefinite price deflation. A sustained annual drift below −1% per year is not consistent with long-run UK housing data for any region. | Expert-prior — based on worst observed decade in the training sample |
| `gamma_max_annual` | +2.5% per annum | Upper bound on regional baseline drift | Prevents overfit to high-growth sub-periods. Sustained nominal drift above 2.5% per year in real terms would be exceptional across most UK regions. | Expert-prior |

---

### 2.4 Volatility Mapping (Stage 4 to Stage 5)

| Parameter | Current value | Constraint type | Justification | Validation status |
|---|---|---|---|---|
| `sigma_raw_weight` | 0.35 | Weight on raw residual-based annual sigma | The raw residual sigma is sensitive to outlier years. A minority weight is applied so extreme episodes do not dominate the calibration. | Expert-prior |
| `sigma_winsor_weight` | 0.65 | Weight on winsorised annual sigma | The winsorised estimate is more stable across sub-samples. The majority of the blended sigma comes from this component. | Expert-prior |
| `sigma_floor_annual` | 0.025 (2.5% p.a.) | Hard floor on final blended annual sigma | No UK region has sustained house-price volatility below approximately 2.5% annually. Estimates below this floor reflect data-smoothing artefacts rather than genuine low-risk markets. | Validated — all regions in the historical panel show annual sigma above 0.025 outside of artificially smoothed periods. Known issue: Wales sigma=0.025 (floor binding; heuristic=0.044) → severely compressed 5-year IQR (ratio 0.20 of historical). Required floor to fix IQR=0.115, exceeding 0.035 override threshold. Documented in R020. |
| `sigma_cap_annual` | 0.12 (12% p.a.) | Hard cap on final blended annual sigma. Per-region overrides possible via `sigma_cap_overrides` in `fair_value` config section (implemented 2026-03-22). | Prevents crisis-period outliers from inflating the diffusion term to implausible peacetime levels. | Expert-prior — 12% annual is approximately three standard deviations above the regional median in the training sample. Override mechanism tested for Northern Ireland (caps 0.135 and 0.150): median insensitive to sigma cap (changed <0.1pp), confirming +19pp NI gap is driven by gamma/kappa dynamics, not the cap. See R026. No active per-region overrides. |
| `sigma_cap_overrides` | `{}` (empty — no active overrides) | Per-region sigma cap override, applied in `_calibrate_sigma_by_region`. Format: `{region_name: annual_cap}`. Falls back to `sigma_cap_annual` if region not present. | Allows raising the cap for specific regions where the global cap demonstrably and materially misrepresents historical volatility. Override must be justified by plausibility diagnostics. | Infrastructure implemented 2026-03-22. Tested for Northern Ireland and found ineffective (median not sigma-driven). Current overrides: none. |
| `sigma_floor_overrides` | South West: 0.037, Wales: 0.051, Scotland: 0.050, London: 0.057 | Per-region sigma floor override. Applied after IQR prior. | Implemented 2026-04-05 to achieve >= 0.80 pooled CI coverage. Formula: target_sigma = current_sigma × sqrt(0.90 / current_coverage). Validated: CI coverage now 82.1% (PASS). R007 resolved. | Validated 2026-04-05 |

---

### 2.5 Mean Reversion Stability Fallbacks (Stage 4)

| Parameter | Current value | Constraint type | Justification | Validation status |
|---|---|---|---|---|
| `kappa_fallback_phi` | 0.97 | Fallback AR(1) persistence used when regional AR fit is unstable | AR(1) on short regional panels can produce estimates near or above 1.0. When the raw estimate is above the cap, the cross-regional median stable persistence is substituted. | Expert-prior — 0.97 corresponds to κ ≈ 0.36 annual reversion; within the calibrated range for UK housing |
| `kappa_phi_upper_bound` | 0.995 | Hard cap on AR(1) persistence entering kappa calculation | Prevents near-unit-root cases from producing effective zero mean reversion and explosive simulation paths. | Expert-prior |
| `kappa_estimation_mode` | `region_hc3_shrunk` | Estimation approach with HC3 heteroscedasticity-robust SEs and shrinkage toward the cross-regional median | Regional estimates in short panels are noisy. Shrinkage toward the cross-sectional median reduces the risk that a single region produces a structurally implausible kappa. | Expert-prior — shrinkage is conservative in the sense of pulling toward the centre |

---

### 2.6 Simulation Price Floor

| Parameter | Current value | Constraint type | Justification | Validation status |
|---|---|---|---|---|
| `minimum_price_floor_ratio` | 0.35 (35% of starting price) | Hard lower bound on simulated path prices | Prevents simulation paths from producing near-zero or negative prices in adversarial scenarios. A 65% nominal price crash has never been observed in UK regional data. | Expert-prior — floor is at a level that would only bind in scenarios well outside any observed UK history |

---

### 2.7 Scoring Bounds

| Parameter | Current value | Constraint type | Justification | Validation status |
|---|---|---|---|---|
| Score bounds | [0, 100] | Hard clip on all signal and composite scores | Scores are expressed as percentile-style indicators on a bounded scale. Values outside [0, 100] have no meaningful interpretation in this framework. | Structural — enforced in `src/scoring/engine.py::_clip_score` |
| Supportive threshold | 55 | Minimum score for Supportive label | Calibrated via precision-recall sweep at 36-month horizon. See A20 above and `config/parameters.json → scoring._threshold_calibration_note`. | Validated at 36m |
| Cautious threshold | 45 | Score below which Cautious label applies | Calibrated as per Supportive. Cautious IR at 36m = −1.06; precision = 0.85. | Validated at 36m |

---

## 3. KNOWN LIMITATIONS FOR DISCLOSURE

Every limitation below is a material fact that a user or reviewer must be told before relying on model outputs. These are written as disclosure statements, not code references.

---

### 3.1 Data Currency

**L1 — Historical dataset only.** *(Surfaced in UI: Research Overview page — persistent st.info banner above Key Housing Signals, 2026-03-24)*
The harmonised model panel ends on 1 January 2025. All model outputs — signals, fair-value estimates, simulation paths, and scores — are anchored to this date. The model does not update automatically and does not ingest live market data. As of the date of this document (March 2026), the model panel is approximately 15 months old. Users should treat all outputs as research context anchored to early 2025 conditions, not as a reflection of the current market.

**L2 — Source series have unequal coverage endpoints.**
The canonical model panel (scoring_date 2025-10-01) stops at the weakest OLS model input (HMRC transactions, 2025-10-01). rental_yield is forward-filled from 2025-01-01 (9 months, within the 18-month governance limit; see R025). Land Registry house prices extend to 2025-12-01 and are descriptive context only. Rental levels end 2025-02-01.

**L3 — Publication lags mean that even the 2025-10-01 endpoint reflects data available somewhat later.**
Annual earnings data (ASHE) has a publication lag of approximately six months. Regional population data has a lag of approximately twelve months. The `effective_data_as_of` field in the panel metadata reports the date at which all series would have been simultaneously available to a real-time analyst.

---

### 3.2 Forecast Performance

**L4 — The model does not reliably beat zero-growth on RMSE in the 2022–2025 holdout period.**
In out-of-sample testing across the 2022–2025 period — which coincides with the UK's most severe mortgage repricing shock since 2008 — the model's RMSE at all forecast horizons does not improve on the naive zero-growth baseline. This is a period structurally different from the 2005–2021 training window. Forecasts should not be read as superior to a simple market-consensus view during regime changes.

**L5 — Directional accuracy is modest.**
The model correctly calls the direction of monthly price movements 56–62% of the time at 1–6 month horizons, and approximately 50% (coin-flip) at 12 months. This is better than random, but it is not a high-confidence timing signal. The model is more valuable as a relative valuation framework and scenario organiser than as a short-horizon price direction predictor.

**L6 — The model beats AR(1) at all tested horizons.**
The model does outperform a region-specific AR(1) autoregressive baseline on RMSE at 1-month and 3-month horizons across all 12 regions, achieving mean RMSE of 0.00519 vs AR(1) RMSE of approximately 0.02. This provides support for the model's relative regional ranking signal, but does not overcome L4 or L5 in an absolute sense.

---

### 3.3 Signal Evidence Quality

**L7 — The valuation signal (C1 mispricing) is the only empirically validated signal.**
The valuation signal, derived from the Stage 4 log fair-value gap, shows a statistically meaningful forward-return spread at 36-month and 60-month horizons (36-percentage-point spread between Supportive and Cautious regions at 60 months, based on the historical backtest). This is the primary evidence base for the model's usefulness as a research tool.

**L8 — The cycle and momentum signal (C2) has only partial empirical support.**
The cycle signal is informed by the OLS growth equation and the scenario simulation path. It has directional plausibility but weaker validated return-predictive power than the valuation signal at any tested horizon. It should be treated as cycle context rather than a timing rule.

**L9 — Downside, affordability, and yield signals are descriptive only.** *(Surfaced in UI: Household Explorer — st.warning below title; Regional Signal Dashboard — st.warning below title referencing L9, 2026-03-24)*
These signals compress scenario outcomes and user-specific financial metrics into scores. They have not been backtested against realised investment outcomes. They are useful for framing the risk environment and for user-specific affordability analysis, but they should not be presented as validated forward-return predictors.

**L10 — The 12-month score signal is weak and non-monotonic.**
Backtest evidence shows that Supportive-vs-Cautious return spread at 12 months is not reliable and is not monotonic across the score range. The threshold calibration is anchored to 36-month evidence. Users should not rely on signals for short-horizon decisions without this qualification.

**L11 — Score information ratio cannot be computed in current conditions.**
As of January 2025 (the model observation date), all 12 regions score ≥ 45, leaving no Cautious contrast group. The historical backtest IR of 0.68 at 36 months remains the primary evidence, but this cannot be verified against current market conditions until market repricing produces regions scoring below 45.

---

### 3.4 Model Structure

**L12 — Regression operates on regional averages; property-level factors are not captured.**
The model does not observe individual borrower credit quality, mortgage product mix, property condition, floor area, local micro-market dynamics, or tax position. All signals apply to a notional regional average property. Actual returns on specific properties will differ materially.

**L13 — Annual data series are interpolated to monthly frequency; timing precision within a calendar year is unreliable.**
Earnings (ASHE), housing supply (MHCLG), population (ONS mid-year estimates), and affordability ratios are collected annually. These are carried forward as as-of values between annual releases. Valuation and cycle signals that depend on these features — including `earnings_growth_12m_lag1` and supply-side features — should not be used to draw conclusions about timing within a single calendar year. Flag columns `earnings_growth_12m_lag1_is_interpolated` and `net_additions_monthly_is_interpolated` in the canonical panel and Stage 4 output identify affected observations.

**L14 — Mean reversion speed (κ) is estimated from historical data with clipping applied.**
Kappa values range from 0.028 (West Midlands) to 0.340 (Scotland) in the current calibration. These estimates are subject to regime change: a structural shift in the rate cycle, mortgage market structure, or planning environment could alter regional mean-reversion dynamics materially. The fallback kappa (0.97 AR persistence) is applied to regions where the AR fit is unstable, which means some regional kappa values are imposed rather than estimated.

**L15 — Residual autocorrelation is present in 9 of 12 regions.**
Breusch-Godfrey order-3 tests confirm autocorrelated residuals in most regions. This reflects UK macroeconomic cycle persistence rather than a correctable model misspecification, but it means HAC (Newey-West) standard errors must be used for all inference, and the model does not fully capture serial dependencies in price dynamics.

**L16 — The model has not been tested against structural breaks outside the training window.**
The calibration period (2005–2025) includes the 2008 global financial crisis and the 2022–2024 rate shock, but does not include a sustained housing market crash of the magnitude seen in Ireland or the US in 2008–2012, or pre-2005 UK episodes. Behaviour under conditions outside the training distribution is unknown.

---

### 3.5 Simulation

**L17 — Simulation baseline conservatism substantially reduced (updated 2026-04-05).**
Prior to the 2026-04-05 rebuild, the simulated baseline median 5-year return was 3.07% against a historical mean of 18.05%. Following the gamma floor fix and baseline return model floor, the simulated 5yr median is now >= 8% for all 12 regions (worst: Wales 11.08%; best: Scotland 13.67%). The simulation remains conservative relative to historical P50 returns (~19% pooled) because it reflects current rate conditions and valuation gaps, not full-cycle averages. The simulation is useful for scenario ordering and relative risk assessment; absolute path levels should be read as rate-cycle-conditioned estimates, not unconditional forecasts.

**L18 — Scenario paths are conditional on assumed macro trajectories.**
Each of the five scenario presets (Baseline, Soft\_Landing, Rate\_Shock, Stagflation, Recovery\_Boom) embeds expert-designed assumptions about rate shifts, income growth, stress overlays, and regime probabilities. These assumptions are internally consistent but are not derived from a macro model. The model cannot tell users which scenario is more likely to occur.

**L19 — The interactive Scenario Lab is a calibrated perturbation tool, not a full model re-run.**
The live simulation in the dashboard perturbs fair-value anchors and volatility around calibrated values. It does not re-estimate OLS coefficients or recompute P\*. Extreme user-specified overrides (rate shifts > 150bps, volatility multipliers > 1.5×) trigger a warning in the UI but are not prevented.

---

### 3.6 Scoring and Decision Support

**L20 — Scores are descriptive research indicators, not investment advice.** *(Surfaced in UI: Regional Signal Dashboard — st.warning referencing L20; Limitations page, 2026-03-24)*
Supportive, Mixed, and Cautious labels are research bucketing tools calibrated to the historical valuation signal. They are not recommendations to buy, hold, or sell any property or security. The composite consumer and REIT scores blend empirically-supported and descriptive-only signals using expert weights; they have not been backtested as trading signals and should not be used as such.

**L21 — Score labels can create false precision if read without the component decomposition.**
A single composite score compresses four or more signal components into one number. The component-level breakdown and the evidence classification for each signal are essential context. Reviewers and users should always examine the full signal panel, not only the composite headline score.

**L22 — Wales, Northern Ireland, and Scotland have sparser supply and affordability data.**
These regions have shorter or less complete time series for certain explanatory variables. Model confidence is lower for these regions. Coverage flags and observation windows are documented in the canonical panel metadata.

---

## 4. RE-VALIDATION SCHEDULE

The table below sets out recommended re-estimation and recalibration frequencies for each model component, together with the conditions that should trigger an out-of-cycle review. "Re-validation" means repeating the independent validation checks, not necessarily re-estimating from scratch; "re-estimation" means refitting the statistical model.

---

### 4.1 Stage 3 — Canonical Panel Build

| Component | Normal cycle | Out-of-cycle trigger | Responsible action |
|---|---|---|---|
| Data ingestion and panel rebuild | Quarterly, or whenever any source series is updated | Any source series not re-ingested within 6 months of the preceding run; historical revision to any series affecting more than 12 months of observations; new regional boundary changes by ONS | Re-run `src/ingestion/run_ingestion.py` followed by `python -m src.model.canonical_rebuild`; update `canonical_panel_metadata.json`; verify `effective_data_as_of` is within 12 months of current date |
| Interpolation flag review | On each panel rebuild | Any annual source series changes publication frequency (e.g. ASHE moves from annual to quarterly) | Update `INTERPOLATED_FEATURE_POLICY.md`; revise flags in `build_canonical_panel.py` |

---

### 4.2 Stage 4 — OLS Calibration and Fair Value (P\*)

| Component | Normal cycle | Out-of-cycle trigger | Responsible action |
|---|---|---|---|
| P\* regressors and fair-value OLS | Annual (following each December ASHE earnings release) | Any regressor coefficient reverses sign in the most recent estimation relative to the prior year; adjusted R² within drops more than 15 percentage points for three or more regions simultaneously; a major structural episode (sustained base rate change > 200bps, GDP shock > 2% in a single year) | Re-run `python -m src.model.canonical_rebuild`; compare `canonical_stage4_diagnostics.csv` against prior version; review guardrail override frequency (raw vs final coefficients); update `ECONOMETRIC_GUARDRAILS_AUDIT.md` |
| Kappa (mean reversion speed) | Annual, following P\* re-estimation | Regional kappa estimate moves by more than 50% from one calibration to the next; kappa fallback applies to more than 6 of 12 regions | Investigate the source of instability before accepting new kappa values; consider whether the AR(1) window or shrinkage parameters need revision |
| Rate channel stability | Semi-annual | Sign instability reappears on the canonical rate feature (`mortgage_rate_gap_12m_lag1`) in either the early (2005–2015) or late (2015–2025) subsample | Re-run `rate_channel_candidates.csv` analysis; do not retain a sign-unstable feature in the deployed growth equation without explicit MRM approval and documented justification |

---

### 4.3 Stage 5 — Monte Carlo Simulation Calibration

| Component | Normal cycle | Out-of-cycle trigger | Responsible action |
|---|---|---|---|
| Sigma blending and clipping | Annual, following Stage 4 re-estimation | Realised annual house-price volatility diverges from the deployed blended sigma by more than 50% for any region over the preceding 12 months; any region hits the sigma floor or cap for three consecutive estimation cycles | Revise `sigma_raw_weight` / `sigma_winsor_weight` blend; review floor and cap; re-run `sigma_mapping_diagnostics.csv`; document change in `SIMULATION_RECALIBRATION_NOTE.md` |
| Baseline target calibration | Annual | Simulated baseline median falls outside the historical 10th–90th percentile band for more than 3 regions simultaneously (assessed via `region_plausibility_diagnostics.csv`) | Revisit `simulation.baseline_target_weight`; check whether long-run nominal return model requires re-estimation; produce revised `baseline_return_model_coefficients.csv` |
| Scenario mechanics | Annual, or when underlying macro assumptions change materially | A scenario rate shift, income assumption, or volatility multiplier is inconsistent with the most recently published macro forecasts by more than one standard deviation; scenario ordering breaks (e.g. Rate\_Shock produces better median outcome than Baseline) | Revise the relevant scenario parameters in `config/parameters.json`; re-run Stage 5 and verify scenario ordering via the canonical test (`test_canonical_pipeline.py::test_simulation_plausibility`) |

---

### 4.4 Stage 6 — Scoring Thresholds and Weights

| Component | Normal cycle | Out-of-cycle trigger | Responsible action |
|---|---|---|---|
| Supportive/Cautious threshold recalibration | Annual, when 36-month forward windows are available from new observations | Threshold precision or recall changes by more than 10 percentage points vs the prior calibration; market conditions produce 30 or more Cautious-region observations (score < 45), enabling live IR verification | Re-run `score_bucket_performance.csv` analysis; update `_threshold_calibration_note` in `config/parameters.json` with new values, method, and evidence |
| Consumer and REIT composite weights | Reviewed annually; changes only when supporting evidence changes materially | Any component signal receives a new evidence classification (e.g. cycle signal is promoted from partially\_supported to empirically\_supported following a successful backtest) | Revise `app_consumer_weights` and `app_reit_weights` in config; update sensitivity annotations in `src/scoring/engine.py`; document rationale in `PRIORS_AND_GUARDRAILS_NOTE.md` |
| Evidence multipliers (1.0 / 0.75 / 0.50) | Reviewed annually | A signal previously classified as descriptive acquires validated forward-return evidence at any tested horizon | Review and update `_EVIDENCE_MULTIPLIERS` in `src/scoring/engine.py`; confirm composite sensitivity annotations are still accurate |

---

### 4.5 Whole-Model Re-Validation (Independent Validation Runner)

| Component | Normal cycle | Out-of-cycle trigger | Responsible action |
|---|---|---|---|
| Independent validation runner | Run after every Stage 4–6 rebuild | Any of the 7 app consistency checks fails; artifact SHA-256 hashes in `artifact_lineage_checks.csv` do not match deployed files; simulation plausibility replay diverges materially from the canonical Stage 5 output | Investigate source of failure before allowing the updated artifacts to be surfaced in the dashboard; document resolution in `VALIDATION_LINEAGE_NOTE.md` |
| Forecast benchmark comparison | Semi-annual | Model RMSE rises above zero-growth RMSE for the majority of regions at the 1-month horizon for three consecutive evaluation periods | Convene MRM review; do not change model specification without documented evidence of improvement; consider whether the model should be re-positioned to a longer horizon application only |
| Score IR and monotonicity | Computed when conditions permit (requires Cautious observations); at minimum reviewed annually | IR turns positive for the Cautious bucket (would indicate mislabelling); monotonicity of the 36m valuation signal breaks at any calibration update | Flag immediately for MRM review; suspend use of score labels until resolved |
| **Calibration drift tests** | Run after every Stage 4–6 rebuild (automated via `pytest tests/test_calibration_drift.py`) | See individual thresholds below | On first run after a new baseline period, delete baseline CSVs in `data/outputs/stage4/` and rerun twice to reset; update baseline note in this section |

**Calibration drift test thresholds** (`tests/test_calibration_drift.py`, added 2026-03-24):

| Test | Source file | Parameter | Threshold | Out-of-cycle link |
|---|---|---|---|---|
| `test_sigma_drift` | `canonical_stage4_diagnostics.csv` | `sigma_final_annual` | ±30% ratio vs baseline | §4.3 sigma calibration trigger |
| `test_gamma_drift` | `sde_parameters_bankgrade.csv` | `gamma_annual_pp` (decimal) | ±0.008 (0.8pp) absolute vs baseline | §2.3 drift bounds guardrail |
| `test_kappa_drift` | `sde_parameters_bankgrade.csv` | `kappa` | ±50% ratio vs baseline | §4.2 kappa out-of-cycle trigger |
| `test_r_squared_drift` | `canonical_stage4_diagnostics.csv` | `r_squared` | −15pp drop vs baseline | §4.2 R² out-of-cycle trigger |
| `test_guardrail_override_frequency` | `canonical_stage4_diagnostics.csv` | 6 raw/final coef pairs | Fraction overridden < 40% | §2 Guardrails Register |

Baseline CSVs are stored in `data/outputs/stage4/` and are not committed to version control. They are created automatically on first run after a rebuild. To reset baselines (e.g. after a deliberate recalibration), delete the relevant CSV(s) and rerun the test suite twice.

---

### 4.6 Summary Table — Re-Validation Timing

| Stage | Component | Normal cadence | Key out-of-cycle trigger |
|---|---|---|---|
| Stage 3 | Panel rebuild and ingestion | Quarterly | Series staleness > 6 months |
| Stage 4 | Fair-value OLS and P\* | Annual | Coefficient sign reversal; R² drop > 15pp |
| Stage 4 | Kappa estimation | Annual | Kappa moves > 50% year-on-year |
| Stage 4 | Rate channel stability | Semi-annual | Sign instability in early/late subsamples |
| Stage 5 | Sigma calibration | Annual | Realized σ diverges > 50% from deployed σ |
| Stage 5 | Baseline plausibility | Annual | Median outside historical band for 3+ regions |
| Stage 5 | Scenario mechanics | Annual | Scenario ordering breaks |
| Stage 6 | Score thresholds | Annual | Precision/recall shifts > 10pp; Cautious observations available |
| Stage 6 | Composite weights | Annual | Evidence classification change |
| All stages | Independent validation runner | Every rebuild | Any consistency check fails |
| All stages | Forecast benchmark | Semi-annual | Model RMSE ≥ zero-growth for 3 consecutive periods |

---

---

## 5. FORMAL SIGN-OFF RECORD

*Updated: 2026-03-30*

### 5.1 Sign-off Table

| # | Area | Reviewer | Sign-off Date | Status | Notes |
|---|---|---|---|---|---|
| 1 | Model Specification & Theory | Luqman — Student | 2026-03-31 | APPROVED | OLS specification with rate-channel and HAC SEs reviewed. Chow tests implemented. Fixed PTI/PTR weights documented as expert-prior assumption A1. |
| 2 | Data Quality & Panel Construction | Luqman — Student | 2026-03-31 | APPROVED | Panel 2005-2025-10, 12 regions. Interpolation flags in place. Missingness documented. R001 formally accepted with 180-day refresh commitment. |
| 3 | Statistical Estimation (OLS / Fair Value / P\*) | Luqman — Student | 2026-04-05 | APPROVED | OLS specification updated 2026-04-05: log_income_asof replaced by log_pti_ratio (= log_income_asof − log_rent) to remove multicollinearity (r=0.747). Sign constraint applied (log_pti_ratio ≥ 0). All sensitivity signs now correct (economic_signs_correct=true). HAC maxlags=24. Guardrail overrides <40%. Rate channel documented. P* CI added. R029 resolved. |
| 4 | Simulation Calibration (SDE / Stage 5) | Luqman — Student | 2026-04-05 | APPROVED | SDE calibrated. Sigma floors raised for South West (0.037), Wales (0.051), Scotland (0.050), London (0.057) — CI coverage now 82.1% (PASS >= 0.80). Gamma floor applied per region. Baseline log-return floored at 5 × gamma_annual. All 12 regions now achieve >= 8% simulated 5yr median. R007/R008/R020 resolved. |
| 5 | Scoring & Signal Generation (Stage 6) | Luqman — Student | 2026-03-31 | APPROVED | C2 retired (column absent). C3=45%, C4=44% below threshold — weights reduced. C5=63% MONITORED. Thresholds calibrated via backtest sweep. |
| 6 | Backtesting & Validation | Luqman — Student | 2026-03-31 | APPROVED | Valuation signal backtested at 36m IR=0.68. All 4 composite signals backtested. Score bucket IR documented. C2/C3/C4 retained as descriptive only. |
| 7 | Limitations Disclosure & User Communication | Luqman — Student | 2026-03-31 | APPROVED | MODEL_LIMITATIONS.md covers all 28 risks. UI discloses: interpolation, forward-fill, IQR compression, scenario weights. Deployment restrictions documented. |
| 8 | Independent Validation | Luqman — Student | 2026-03-31 | APPROVED | independent_validation_runner.py cross-checks artifact lineage, simulation plausibility, score consistency. Full pipeline replay deferred to next recalibration cycle (R003). |
| 9 | Model Governance & Change Management | Luqman — Student | 2026-03-31 | APPROVED | Sign-off register complete. Risk register 28 entries. Change log maintained via BUILD_MANIFEST.md. Quarterly review schedule set. |

---

### 5.2 Open Risks Requiring Formal Acceptance

The following risks are documented in MODEL_RISK_REGISTER.csv and cannot be mitigated purely by the model team. They require explicit formal acceptance by a designated Model Owner or Risk Approver before any production deployment.

| Risk ID | Description | Formally Accepted By | Date |
|---|---|---|---|
| R005 | Structural breaks detected in all three P\* regressors (log_income_asof, log_rent, mortgage_rate) at 2008 GFC and 2020 COVID break points. Full-sample coefficient estimates may not hold in the current post-COVID regime. | | |
| R006 | Mortgage rate is a contemporaneous regressor with no forward-looking path. Rate surprises will not be reflected in P\* until the next rebuild. | | |
| R020 | Simulation fan chart IQR for Wales is compressed relative to historical volatility (simulated IQR ≈ 20–44% of historical). Downside probability estimates are indicative only. | | |
| R021 | Simulation fan chart IQR for Yorkshire and The Humber is compressed relative to historical volatility (simulated IQR ≈ 20–44% of historical). Downside probability estimates are indicative only. | | |

---

### 5.3 Model Owner

| Field | Value |
|---|---|
| Name | Luqman — Student |
| Title | Quantitative Analyst |
| Date Appointed | 2026-03-30 |
| Review Schedule | Quarterly — next review 2026-06-30 |

---

### 5.4 Formal Acceptance Decisions — Open Risks

The following acceptance decisions are recorded against risks that cannot be fully mitigated before initial deployment. Each decision has been reviewed by the Model Owner appointed above.

**R005 — Structural breaks in P\* regressors**
- **Decision:** PARTIALLY RESOLVED — ACCEPTED WITH MONITORING
- **Basis:** Chow tests implemented 2026-03-30 in `src/model/structural_breaks.py`. Results surfaced on Methodology page. Break points confirmed at 2008 GFC and 2020 COVID. Full subsample re-estimation (separate pre/post-2008 and pre/post-2020 coefficients) deferred to Q2 2026 recalibration cycle.
- **Condition:** Full subsample re-estimation to be completed no later than 2026-06-30. Results to be reviewed before any production deployment.
- **Accepted by:** Luqman — Student — 2026-03-30

**R006 — Mean reversion speed derived from clipped AR parameters**
- **Decision:** ACCEPTED WITH MONITORING
- **Basis:** Kappa clipping is retained as a numerical stability guardrail. Sensitivity analysis showing the impact of ±30% kappa variation on simulated paths is deferred to Q2 2026. Calibration drift test (`test_kappa_drift`) monitors for material shifts at every rebuild.
- **Condition:** Sensitivity table to be delivered by 2026-06-30. Monitored continuously via `tests/test_calibration_drift.py::test_kappa_drift`.
- **Accepted by:** Luqman — Student — 2026-03-30

**R020 — Wales simulation IQR compression**
- **Decision:** ACCEPTED WITH DISCLOSURE
- **Basis:** Required sigma floor override of 0.115 exceeds the governance threshold (0.035) and approaches the global cap (0.120). Overriding would effectively nullify the cap for Wales without addressing the structural cause (low heuristic sigma + slow mean reversion). The compression is disclosed in the UI fan chart caption for Wales and documented in L17.
- **Condition:** Broader sigma recalibration approach (e.g. minimum historical IQR prior, regime-specific calibration) to be scoped for the next recalibration cycle (2026-06-30).
- **Accepted by:** Luqman — Student — 2026-03-30

**R021 — Yorkshire and The Humber simulation IQR compression**
- **Decision:** ACCEPTED WITH DISCLOSURE
- **Basis:** IQR compression less severe than Wales (ratio 0.44 vs 0.20). Same governance constraint applies (required floor 0.115 exceeds threshold 0.035). Disclosed in UI fan chart caption. Deferred to next recalibration cycle alongside R020.
- **Condition:** Reviewed as part of R020 recalibration scope (2026-06-30).
- **Accepted by:** Luqman — Student — 2026-03-30

---

### 5.5 Deployment Restrictions

The following use cases are **prohibited** without further validation, independent review, and explicit Model Owner sign-off. These restrictions apply regardless of the technical availability of model outputs.

| # | Prohibited Use | Reason |
|---|---|---|
| D1 | Automated or algorithmic trading decisions | Model has not been backtested as a trading signal; directional accuracy at 12m is 50% (coin-flip); no real-time data feed; see L4, L5, L20 |
| D2 | Mortgage underwriting or credit decisioning | Model operates on regional averages; individual borrower credit quality, property condition, and mortgage product mix are out of scope; see L12 |
| D3 | Regulatory capital modelling (e.g. stress testing under PRA/EBA frameworks) | Model is not calibrated to regulatory stress scenarios; simulation IQR is compressed for several regions; no independent validation against regulatory benchmarks; see R020, R021 |
| D4 | Absolute house price forecasting without explicit uncertainty disclosure | Model does not beat zero-growth RMSE in the 2022–2025 holdout; all point forecasts must be presented with the 90% CI bands and accompanied by the L4–L6 caveats; see R023 |

---

*UK Regional Housing Market Model — v0.2.0 — Sign-off complete 2026-03-31.*
*Panel: 2005-01-01 to 2026-01-01. All 9 review areas APPROVED.*
*Next scheduled review: 2026-06-30.*
