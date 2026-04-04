"""
Plain-English glossary for the UK Regional Housing Market Model dashboard.

Definitions are written for a sophisticated investor audience — technically
precise but not assuming familiarity with the specific model implementation.
Used in the Methodology page glossary section, the Technical Controls expander,
and as help= text on individual widgets via the tooltip() helper.
"""

from __future__ import annotations

GLOSSARY: dict[str, str] = {
    "kappa": (
        "Mean-reversion speed in the Ornstein-Uhlenbeck process: how quickly the "
        "simulated price path is pulled back toward the estimated fair value. A higher "
        "kappa implies faster reversion — prices correct more aggressively after a "
        "mispricing episode. Values range from ~0.03 (slow, West Midlands) to ~0.34 "
        "(fast, Scotland) in the current calibration."
    ),
    "sigma": (
        "Annual volatility parameter controlling the width of the simulation fan chart. "
        "It is calibrated from the residual standard deviation of the Stage 4 OLS growth "
        "equation, blended 35% raw and 65% winsorised to reduce the influence of outlier "
        "years such as 2008 and COVID. A higher sigma produces wider P10–P90 bands; the "
        "floor is 2.5% p.a. and the cap is 12% p.a."
    ),
    "gamma": (
        "Long-run drift anchor: the baseline annual price growth assumption embedded in "
        "the simulation. It is a 40/60 blend of the current real growth rate and the "
        "full-history nominal anchor, clipped to [−1%, +2.5%] per annum. A higher gamma "
        "shifts all scenario paths upward; it does not affect the width of the fan or "
        "the mean-reversion pull."
    ),
    "drift": (
        "Baseline annual price growth assumption built into the simulation paths before "
        "any scenario overlay is applied. Drift is the deterministic component of the "
        "Ornstein-Uhlenbeck process and reflects long-run nominal house price appreciation "
        "for each region. It is closely related to gamma and is overridden by scenario-"
        "specific drift adjustments in stress tests."
    ),
    "OU process": (
        "Ornstein-Uhlenbeck stochastic differential equation: the mathematical engine "
        "driving Monte Carlo simulations. It combines a mean-reverting pull toward the "
        "fair-value anchor (controlled by kappa) with a random diffusion term (controlled "
        "by sigma) and a long-run drift (gamma). The log-price variant is used here to "
        "preserve price positivity and produce lognormally distributed terminal prices."
    ),
    "HAC standard errors": (
        "Heteroskedasticity- and autocorrelation-consistent standard errors (Newey-West). "
        "Monthly house-price data exhibits serial correlation — this month's growth partly "
        "predicts next month's — which causes ordinary OLS standard errors to be too small "
        "and t-statistics to be overstated. HAC corrections inflate the standard errors "
        "appropriately; the model uses maxlags=12 to cover a full annual cycle."
    ),
    "ZLB era": (
        "Zero Lower Bound era (approximately 2009–2021): the period when Bank Rate was "
        "held at or near 0.1–0.5%, removing meaningful variation in the mortgage rate "
        "series. Regression estimates of the rate channel are statistically unreliable "
        "for data from this period because near-zero variance in the regressor inflates "
        "coefficient uncertainty. The model flags ZLB-era instability via the rate-"
        "channel stability diagnostic."
    ),
    "rate channel": (
        "The model component that translates changes in mortgage rates into house price "
        "growth. A 1 percentage-point rise in the mortgage rate is estimated to reduce "
        "annual price growth by approximately 0.4–0.7pp across UK regions, with the "
        "effect entering with a 12-month lag. The channel is sign-stable in the full "
        "sample but imprecisely estimated in the ZLB era due to limited rate variation."
    ),
    "fair value gap": (
        "The difference between the current regional house price and the model's "
        "estimated equilibrium price (P*), expressed as a log-ratio. A positive gap "
        "means prices are above fair value (overvalued); a negative gap means they are "
        "below (undervalued). P* is derived from a semi-structural regression on real "
        "income, rental yield, and mortgage rates. The gap is the primary input to the "
        "valuation signal (C1)."
    ),
    "downside probability": (
        "The fraction of the 10,000 Monte Carlo simulation paths that end below the "
        "starting price at the chosen horizon (typically 5 years). It is a tail-risk "
        "measure rather than a central forecast: a downside probability of 20% means "
        "that in 2,000 of 10,000 scenarios the investor holds a nominal loss at exit. "
        "It is sensitive to both sigma and the scenario rate assumption."
    ),
    "scenario weights": (
        "The relative importance assigned to each macro scenario when computing "
        "scenario-weighted composite signals. Consumer and REIT views use different "
        "default weights: the consumer view applies a deliberately downside-skewed prior "
        "(Baseline 40%, Rate Shock 25%, Stagflation 20%) while the REIT view applies "
        "equal weights (20% per scenario). Both weight sets are expert-prior and "
        "configurable in parameters.json."
    ),
    "rental yield": (
        "Annual rental income as a percentage of property value, used in this model as "
        "a valuation anchor alongside income and mortgage rates when estimating the "
        "fair-value equilibrium price (P*). A rising yield — either from rent growth or "
        "price falls — reduces the fair value gap, all else equal. The series is sourced "
        "from ONS Private Rental Market Statistics and has a ~12-month publication lag."
    ),
    "IQR": (
        "Interquartile range: the spread between the 25th and 75th percentile of the "
        "simulation distribution at a given horizon. It is used in this model as a "
        "volatility calibration check — if the simulated IQR is materially narrower "
        "than the historical IQR of rolling 5-year returns, the simulation is under-"
        "dispersed and sigma may need upward revision. Currently Wales and Yorkshire "
        "show IQR compression due to the sigma floor binding."
    ),
    "guardrail": (
        "A hard constraint applied during Stage 4 calibration or Stage 5 simulation to "
        "prevent outputs that are economically implausible or numerically unstable. "
        "Examples include the sigma floor (2.5% p.a.), the sigma cap (12% p.a.), the "
        "kappa stability fallback (AR persistence capped at 0.995), and the directional "
        "sign restrictions on beta coefficients. Guardrails are explicit modelling "
        "choices, not hidden identification claims, and are logged in "
        "canonical_stage4_diagnostics.csv."
    ),
}


def tooltip(term: str) -> str:
    """Return a Streamlit-compatible help string for st.metric / st.selectbox help= param."""
    return GLOSSARY.get(term, "")
