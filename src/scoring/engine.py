"""Transparent Stage 6 signal dashboard helpers used by the UI and tests."""

from __future__ import annotations

import warnings
from dataclasses import asdict
from math import isfinite

from src.core.config import load_config


CONFIG = load_config()
_THRESHOLDS = CONFIG.scoring.thresholds
_CONSUMER_W = CONFIG.scoring.app_consumer_weights
_REIT_W = CONFIG.scoring.app_reit_weights
_AFFORDABILITY_MULTIPLE = CONFIG.controls.affordability_multiple


def _validate_weights(weights: object, name: str) -> None:
    """Raise ValueError if weights do not sum to 1.0 (tolerance 1e-9).

    Accepts either a dataclass (converted via asdict) or a plain dict.
    Note keys beginning with '_' are excluded automatically.
    """
    if hasattr(weights, "__dataclass_fields__"):
        w_dict = asdict(weights)
    else:
        w_dict = dict(weights)
    w_numeric = {k: v for k, v in w_dict.items() if not str(k).startswith("_")}
    total = sum(w_numeric.values())
    if not abs(total - 1.0) < 1e-9:
        raise ValueError(
            f"{name} weights sum to {total:.10f}, expected 1.0. "
            f"Offending weights: {w_numeric}"
        )


_validate_weights(_CONSUMER_W, "app_consumer_weights")
_validate_weights(_REIT_W, "app_reit_weights")
# WEIGHT: 1.0  (empirically_supported evidence multiplier)
# JUSTIFICATION: expert-prior — judgement-based, no backtested basis
# SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.11 points
#   (signal already near 50 cross-sectionally; shrinkage change has small composite effect)
# LAST_REVIEWED: 2026-03-21
#
# WEIGHT: 0.75  (partially_supported evidence multiplier)
# JUSTIFICATION: expert-prior — judgement-based, no backtested basis
# SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.08 points
#   (cycle signal is already compressed; residual deviation from 50 is modest)
# LAST_REVIEWED: 2026-03-21
#
# WEIGHT: 0.50  (descriptive evidence multiplier)
# JUSTIFICATION: expert-prior — judgement-based, no backtested basis
# SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.60 points
#   (descriptive signals — downside, affordability — carry 0.60 of total composite weight;
#   largest sensitivity of the three multipliers)
# LAST_REVIEWED: 2026-03-21
_EVIDENCE_MULTIPLIERS = {
    "empirically_supported": 1.0,
    "partially_supported": 0.75,
    "descriptive": 0.50,
}


def _clip_score(value: float) -> float:
    return float(max(0.0, min(100.0, value)))


def _weighted_average(component_scores: dict[str, float], weights: dict[str, float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    raw = sum(component_scores[name] * weights[name]
              for name in weights) / total
    return _clip_score(raw)


def _mortgage_payment(loan_amount: float, mortgage_rate: float, mortgage_term_years: int) -> float:
    if loan_amount <= 0:
        return 0.0
    monthly_rate = mortgage_rate / 100.0 / 12.0
    n_periods = mortgage_term_years * 12
    if monthly_rate == 0:
        return loan_amount / n_periods
    return loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** (-n_periods))


def _downside_score(simulation_percentiles: dict, downside_prob: float | None = None) -> float:
    if downside_prob is not None and isfinite(downside_prob):
        return _clip_score(100.0 * (1.0 - downside_prob))
    p10 = float(simulation_percentiles.get("p10", 0.0))
    return _clip_score(60.0 + 2.0 * p10)


def _scenario_signal_score(simulation_percentiles: dict) -> float:
    p50 = float(simulation_percentiles.get("p50", 0.0))
    return _clip_score(50.0 + 2.0 * p50)


def _risk_buffer(risk_tolerance: str) -> float:
    # WEIGHT: +5.0  (Opportunistic risk buffer, added to composite before clipping)
    # JUSTIFICATION: expert-prior — judgement-based, no backtested basis
    # SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.10 points
    #   (direct additive shift; effect is linear except at 0/100 clip boundaries)
    # LAST_REVIEWED: 2026-03-21
    #
    # WEIGHT: 0.0  (Balanced risk buffer)
    # JUSTIFICATION: expert-prior — judgement-based, no backtested basis
    # SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.10 points
    # LAST_REVIEWED: 2026-03-21
    #
    # WEIGHT: -7.5  (Conservative risk buffer)
    # JUSTIFICATION: expert-prior — judgement-based, no backtested basis
    # SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.10 points
    #   (largest magnitude buffer; users at Conservative already see depressed composites,
    #   so clipping at 0 is more likely to limit the full effect)
    # LAST_REVIEWED: 2026-03-21
    lookup = {"Conservative": -7.5, "Balanced": 0.0, "Opportunistic": 5.0}
    return lookup.get(risk_tolerance, 0.0)


def label_score(score: float) -> str:
    """Map a 0-100 value to descriptive research buckets, not action advice."""
    if score >= _THRESHOLDS.supportive:
        return "Supportive"
    if score >= _THRESHOLDS.mixed:
        return "Mixed"
    return "Cautious"


def _simple_status(signal_id: str, bucket: str) -> str:
    mappings = {
        "valuation": {
            "Supportive": "Looks better value than usual",
            "Mixed": "Looks around fair value",
            "Cautious": "Looks expensive for the area",
        },
        "cycle": {
            "Supportive": "Market trend is improving",
            "Mixed": "Market trend is mixed",
            "Cautious": "Market trend is weakening",
        },
        "downside": {
            "Supportive": "Chance of a large fall looks more contained",
            "Mixed": "Downside risk looks balanced",
            "Cautious": "Chance of a large fall looks elevated",
        },
        "affordability": {
            "Supportive": "Payments look manageable",
            "Mixed": "Payments look tight",
            "Cautious": "Payments look stretched",
        },
        "yield": {
            "Supportive": "Yield looks supportive",
            "Mixed": "Yield looks adequate",
            "Cautious": "Yield looks thin",
        },
    }
    return mappings.get(signal_id, {}).get(bucket, bucket)


def _evidence_multiplier(evidence_level: str) -> float:
    return float(_EVIDENCE_MULTIPLIERS.get(evidence_level, 0.5))


def _shrink_toward_neutral(score: float, evidence_level: str) -> float:
    multiplier = _evidence_multiplier(evidence_level)
    return _clip_score(50.0 + multiplier * (score - 50.0))


def _signal_card(
    signal_id: str,
    display_name: str,
    value: float,
    *,
    question: str,
    evidence_level: str,
    definition: str,
    note: str,
) -> dict:
    score = _clip_score(value)
    bucket = label_score(score)
    return {
        "id": signal_id,
        "display_name": display_name,
        "question": question,
        "score": round(score, 1),
        "bucket": bucket,
        "evidence_level": evidence_level,
        "evidence_multiplier": _evidence_multiplier(evidence_level),
        "definition": definition,
        "note": note,
        "simple_status": _simple_status(signal_id, bucket),
    }


def consumer_signal_dashboard(
    region: str,
    income: float,
    deposit: float,
    simulation_percentiles: dict,
    *,
    valuation_signal: float,
    cycle_signal: float,
    property_price: float | None = None,
    mortgage_rate: float = 5.25,
    mortgage_term_years: int = 30,
    risk_tolerance: str = "Balanced",
    downside_prob: float | None = None,
) -> dict:
    if downside_prob is None:
        warnings.warn(
            "C2 (downside_probability) is None — signal weight has been zeroed in config. "
            "Composite continues without C2 contribution.",
            UserWarning, stacklevel=2
        )
    if income <= 0:
        raise ValueError("income must be positive")
    if deposit < 0:
        raise ValueError("deposit must be non-negative")
    property_price = float(
        property_price or simulation_percentiles.get("start_price", 300000.0))
    if property_price <= 0:
        raise ValueError("property_price must be positive")
    if mortgage_term_years <= 0:
        raise ValueError("mortgage_term_years must be positive")

    loan_amount = max(property_price - deposit, 0.0)
    payment = _mortgage_payment(
        loan_amount, mortgage_rate, mortgage_term_years)
    payment_ratio = (payment * 12.0) / income
    loan_to_income = loan_amount / income
    ltv = loan_amount / property_price if property_price > 0 else 0.0

    affordability_score = _clip_score(
        100.0 -
        max(0.0, loan_to_income - _AFFORDABILITY_MULTIPLE) * 18.0 -
        max(0.0, payment_ratio - 0.30) * 140.0 -
        max(0.0, ltv - 0.80) * 120.0
    )
    # WEIGHT: 0.65  (cycle_signal contribution to cycle_outlook blend)
    # JUSTIFICATION: expert-prior — not-yet-empirically-optimised
    # SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.09 points
    #   (cycle_outlook feeds into base_model via 0.4 weight; dampened through two blending stages)
    # LAST_REVIEWED: 2026-03-21
    #
    # WEIGHT: 0.35  (scenario_signal contribution to cycle_outlook blend)
    # JUSTIFICATION: expert-prior — not-yet-empirically-optimised
    # SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.72 points
    #   (scenario signal has wider cross-sectional spread than cycle_signal;
    #   higher sensitivity despite identical blend weight)
    # LAST_REVIEWED: 2026-03-21
    cycle_outlook = _clip_score(
        0.65 * cycle_signal + 0.35 * _scenario_signal_score(simulation_percentiles))
    downside_resilience = _downside_score(
        simulation_percentiles, downside_prob)

    signals = [
        _signal_card(
            "valuation",
            "Valuation",
            valuation_signal,
            question="Is this area expensive relative to local incomes and rents?",
            evidence_level="empirically_supported",
            definition="Relative valuation signal derived from the Stage 4 fair-value gap.",
            note="Strongest historical support is at 36-60 month horizons.",
        ),
        _signal_card(
            "cycle",
            "Cycle / Momentum",
            cycle_outlook,
            question="Is the market still cooling or stabilising?",
            evidence_level="partially_supported",
            definition="Cycle-sensitive outlook combining the regional regime view with the live scenario path.",
            note="Useful as context, but weaker than valuation as a stand-alone ranking signal.",
        ),
        _signal_card(
            "downside",
            "Downside Resilience",
            downside_resilience,
            question="How exposed is this area in weaker scenarios?",
            evidence_level="descriptive",
            definition="Resilience implied by the selected simulation path and downside probability.",
            note="Scenario-plausibility context rather than a validated forecast edge.",
        ),
        _signal_card(
            "affordability",
            "Affordability",
            affordability_score,
            question="Do your income and deposit make the purchase manageable?",
            evidence_level="descriptive",
            definition="User-specific burden metric based on income, deposit, mortgage rate, term, and LTV.",
            note="Important for the buyer, but not backtested as a return-predictive signal.",
        ),
    ]
    signal_scores = {signal["id"]: float(
        signal["score"]) for signal in signals}
    adjusted_signal_scores = {
        signal["id"]: _shrink_toward_neutral(
            float(signal["score"]), signal["evidence_level"])
        for signal in signals
    }
    secondary_composite = _clip_score(
        _weighted_average(
            {
                # WEIGHT: 0.60  (valuation within base_model internal blend)
                # JUSTIFICATION: expert-prior — not-yet-empirically-optimised
                # SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.08 points
                #   (valuation signal is tightly shrunk toward 50; cross-sectional spread is low)
                # LAST_REVIEWED: 2026-03-21
                #
                # WEIGHT: 0.40  (cycle_outlook within base_model internal blend)
                # JUSTIFICATION: expert-prior — not-yet-empirically-optimised
                # SENSITIVITY: changing by ±0.10 moves composite score by approximately 2.15 points
                #   (cycle is more volatile cross-sectionally; largest hardcoded-weight sensitivity
                #   in the consumer composite)
                # LAST_REVIEWED: 2026-03-21
                "base_model": adjusted_signal_scores["valuation"] * 0.6 + adjusted_signal_scores["cycle"] * 0.4,
                "affordability": adjusted_signal_scores["affordability"],
                "downside_risk": adjusted_signal_scores["downside"],
            },
            {
                # WEIGHT: 0.40  (_CONSUMER_W.base_model — defined in config: scoring.app_consumer_weights.base_model)
                # JUSTIFICATION: expert-prior — not-yet-empirically-optimised
                # SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.19 points
                # LAST_REVIEWED: 2026-03-21
                "base_model": _CONSUMER_W.base_model,
                # WEIGHT: 0.40  (_CONSUMER_W.affordability — defined in config: scoring.app_consumer_weights.affordability)
                # JUSTIFICATION: expert-prior — not-yet-empirically-optimised
                # SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.42 points
                # LAST_REVIEWED: 2026-03-21
                "affordability": _CONSUMER_W.affordability,
                # WEIGHT: 0.20  (_CONSUMER_W.downside_risk — defined in config: scoring.app_consumer_weights.downside_risk)
                # JUSTIFICATION: expert-prior — not-yet-empirically-optimised
                # SENSITIVITY: changing by ±0.10 moves composite score by approximately 1.10 points
                #   (downside signal diverges most from neutral; small weight change has
                #   disproportionate composite effect relative to its 0.20 nominal weight)
                # LAST_REVIEWED: 2026-03-21
                "downside_risk": _CONSUMER_W.downside_risk,
            },
        ) +
        _risk_buffer(risk_tolerance)
    )

    if payment_ratio >= 0.45:
        stress = "High"
    elif payment_ratio >= 0.35:
        stress = "Elevated"
    else:
        stress = "Contained"

    return {
        "region": region,
        "signals": signals,
        "signal_scores": {k: round(v, 1) for k, v in signal_scores.items()},
        "secondary_composite": {
            "score": round(secondary_composite, 1),
            "bucket": label_score(secondary_composite),
            "label_mode": "secondary_synthesis_only",
            "method": "evidence_adjusted_secondary_synthesis",
        },
        "mortgage_payment_monthly": round(payment, 0),
        "payment_to_income_ratio": round(payment_ratio, 3),
        "loan_to_income": round(loan_to_income, 2),
        "loan_to_value": round(ltv, 3),
        "mortgage_stress": stress,
    }


def reit_signal_dashboard(
    region: str,
    gross_yield: float,
    simulation_percentiles: dict,
    *,
    valuation_signal: float,
    cycle_signal: float,
    target_yield: float = 5.5,
    risk_tolerance: str = "Balanced",
    downside_prob: float | None = None,
) -> dict:
    if gross_yield < 0:
        raise ValueError("gross_yield must be non-negative")

    yield_signal = _clip_score(50.0 + (gross_yield - target_yield) * 18.0)
    # WEIGHT: 0.65  (cycle_signal contribution to cycle_outlook blend)
    # JUSTIFICATION: expert-prior — not-yet-empirically-optimised
    # SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.09 points
    #   (cycle_outlook feeds into base_model via 0.4 weight; dampened through two blending stages)
    # LAST_REVIEWED: 2026-03-21
    #
    # WEIGHT: 0.35  (scenario_signal contribution to cycle_outlook blend)
    # JUSTIFICATION: expert-prior — not-yet-empirically-optimised
    # SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.72 points
    #   (scenario signal has wider cross-sectional spread than cycle_signal;
    #   higher sensitivity despite identical blend weight)
    # LAST_REVIEWED: 2026-03-21
    cycle_outlook = _clip_score(
        0.65 * cycle_signal + 0.35 * _scenario_signal_score(simulation_percentiles))
    downside_resilience = _downside_score(
        simulation_percentiles, downside_prob)

    signals = [
        _signal_card(
            "valuation",
            "Valuation",
            valuation_signal,
            question="Is this area expensive relative to local incomes and rents?",
            evidence_level="empirically_supported",
            definition="Relative valuation signal derived from the Stage 4 fair-value gap.",
            note="Strongest historical support is at medium horizons.",
        ),
        _signal_card(
            "cycle",
            "Cycle / Momentum",
            cycle_outlook,
            question="Is the market still cooling or stabilising?",
            evidence_level="partially_supported",
            definition="Regional cycle outlook informed by the scenario path and Stage 6 regime layer.",
            note="Useful context, but not a clean stand-alone timing rule.",
        ),
        _signal_card(
            "downside",
            "Downside Resilience",
            downside_resilience,
            question="How exposed is this area in weaker scenarios?",
            evidence_level="descriptive",
            definition="Scenario-based resilience to adverse simulated price outcomes.",
            note="Simulation context rather than hard forecast validation.",
        ),
        _signal_card(
            "yield",
            "Yield Support",
            yield_signal,
            question="Do current rents provide support for the market?",
            evidence_level="descriptive",
            definition="Current gross yield versus the selected hurdle rate.",
            note="Relevant for allocation context, but not validated as a complete alpha signal.",
        ),
    ]
    signal_scores = {signal["id"]: float(
        signal["score"]) for signal in signals}
    adjusted_signal_scores = {
        signal["id"]: _shrink_toward_neutral(
            float(signal["score"]), signal["evidence_level"])
        for signal in signals
    }
    secondary_composite = _clip_score(
        _weighted_average(
            {
                # WEIGHT: 0.60  (valuation within base_model internal blend)
                # JUSTIFICATION: expert-prior — not-yet-empirically-optimised
                # SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.08 points
                #   (valuation signal is tightly shrunk toward 50; cross-sectional spread is low)
                # LAST_REVIEWED: 2026-03-21
                #
                # WEIGHT: 0.40  (cycle_outlook within base_model internal blend)
                # JUSTIFICATION: expert-prior — not-yet-empirically-optimised
                # SENSITIVITY: changing by ±0.10 moves composite score by approximately 2.15 points
                #   (cycle is more volatile cross-sectionally; largest hardcoded-weight sensitivity
                #   in the REIT composite)
                # LAST_REVIEWED: 2026-03-21
                "base_model": adjusted_signal_scores["valuation"] * 0.6 + adjusted_signal_scores["cycle"] * 0.4,
                "yield_signal": adjusted_signal_scores["yield"],
                "downside_risk": adjusted_signal_scores["downside"],
            },
            {
                # WEIGHT: 0.45  (_REIT_W.base_model — defined in config: scoring.app_reit_weights.base_model)
                # JUSTIFICATION: expert-prior — not-yet-empirically-optimised
                # SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.24 points
                # LAST_REVIEWED: 2026-03-21
                "base_model": _REIT_W.base_model,
                # WEIGHT: 0.30  (_REIT_W.yield_signal — defined in config: scoring.app_reit_weights.yield_signal)
                # JUSTIFICATION: expert-prior — not-yet-empirically-optimised
                # SENSITIVITY: changing by ±0.10 moves composite score by approximately 0.51 points
                # LAST_REVIEWED: 2026-03-21
                "yield_signal": _REIT_W.yield_signal,
                # WEIGHT: 0.25  (_REIT_W.downside_risk — defined in config: scoring.app_reit_weights.downside_risk)
                # JUSTIFICATION: expert-prior — not-yet-empirically-optimised
                # SENSITIVITY: changing by ±0.10 moves composite score by approximately 1.01 points
                #   (downside signal diverges most from neutral; effect is amplified relative
                #   to its 0.25 nominal weight)
                # LAST_REVIEWED: 2026-03-21
                "downside_risk": _REIT_W.downside_risk,
            },
        ) +
        _risk_buffer(risk_tolerance)
    )

    return {
        "region": region,
        "signals": signals,
        "signal_scores": {k: round(v, 1) for k, v in signal_scores.items()},
        "secondary_composite": {
            "score": round(secondary_composite, 1),
            "bucket": label_score(secondary_composite),
            "label_mode": "secondary_synthesis_only",
            "method": "evidence_adjusted_secondary_synthesis",
        },
        "yield_gap_pct": round(gross_yield - target_yield, 2),
        "target_yield": round(target_yield, 2),
    }


def regional_market_signal_dashboard(
    region: str,
    *,
    valuation_signal: float,
    cycle_signal: float,
    downside_prob: float,
    rent_support_signal: float,
) -> dict:
    downside_vulnerability = _clip_score(100.0 * downside_prob)
    signals = [
        _signal_card(
            "valuation",
            "Valuation",
            valuation_signal,
            question="Is this area expensive relative to local incomes and rents?",
            evidence_level="empirically_supported",
            definition="Relative valuation signal derived from the Stage 4 fair-value gap.",
            note="Best treated as a medium-horizon valuation anchor rather than a short-term timing rule.",
        ),
        _signal_card(
            "cycle",
            "Cycle State",
            cycle_signal,
            question="Is the market still cooling or stabilising?",
            evidence_level="partially_supported",
            definition="Regional cycle context informed by activity and the baseline scenario path.",
            note="Useful as cycle context, but weaker than valuation as a stand-alone ranking signal.",
        ),
        _signal_card(
            "downside",
            "Downside Vulnerability",
            downside_vulnerability,
            question="How exposed is this area in weaker scenarios?",
            evidence_level="descriptive",
            definition="Cross-scenario downside exposure based on the current simulation layer.",
            note="Scenario-risk context rather than a validated forecast edge.",
        ),
        _signal_card(
            "rental_support",
            "Rent Support",
            rent_support_signal,
            question="Do rents provide support for this market?",
            evidence_level="descriptive",
            definition="Current rental-income context using the region's cross-sectional gross-yield standing.",
            note="Descriptive market context, not a direct return-predictive signal.",
        ),
    ]
    return {
        "region": region,
        "signals": signals,
        "signal_scores": {signal["id"]: float(signal["score"]) for signal in signals},
    }


def score_consumer(
    region: str,
    income: float,
    deposit: float,
    simulation_percentiles: dict,
    *,
    property_price: float | None = None,
    mortgage_rate: float = 5.25,
    mortgage_term_years: int = 30,
    risk_tolerance: str = "Balanced",
    base_model_score: float | None = None,
    downside_prob: float | None = None,
) -> dict:
    dashboard = consumer_signal_dashboard(
        region,
        income,
        deposit,
        simulation_percentiles,
        valuation_signal=float(base_model_score if base_model_score is not None else simulation_percentiles.get(
            "base_model_score", 50.0)),
        cycle_signal=float(base_model_score if base_model_score is not None else simulation_percentiles.get(
            "base_model_score", 50.0)),
        property_price=property_price,
        mortgage_rate=mortgage_rate,
        mortgage_term_years=mortgage_term_years,
        risk_tolerance=risk_tolerance,
        downside_prob=downside_prob,
    )
    return {
        "region": region,
        "total_score": dashboard["secondary_composite"]["score"],
        "label": dashboard["secondary_composite"]["bucket"],
        "label_mode": "descriptive_signal",
        "component_scores": {
            "base_model": round(0.6 * dashboard["signal_scores"]["valuation"] + 0.4 * dashboard["signal_scores"]["cycle"], 1),
            "affordability": dashboard["signal_scores"]["affordability"],
            "downside_risk": dashboard["signal_scores"]["downside"],
        },
        "mortgage_payment_monthly": dashboard["mortgage_payment_monthly"],
        "payment_to_income_ratio": dashboard["payment_to_income_ratio"],
        "loan_to_income": dashboard["loan_to_income"],
        "loan_to_value": dashboard["loan_to_value"],
        "mortgage_stress": dashboard["mortgage_stress"],
    }


def score_reit(
    region: str,
    gross_yield: float,
    simulation_percentiles: dict,
    *,
    target_yield: float = 5.5,
    risk_tolerance: str = "Balanced",
    base_model_score: float | None = None,
    downside_prob: float | None = None,
) -> dict:
    dashboard = reit_signal_dashboard(
        region,
        gross_yield,
        simulation_percentiles,
        valuation_signal=float(base_model_score if base_model_score is not None else simulation_percentiles.get(
            "base_model_score", 50.0)),
        cycle_signal=float(base_model_score if base_model_score is not None else simulation_percentiles.get(
            "base_model_score", 50.0)),
        target_yield=target_yield,
        risk_tolerance=risk_tolerance,
        downside_prob=downside_prob,
    )
    return {
        "region": region,
        "total_score": dashboard["secondary_composite"]["score"],
        "label": dashboard["secondary_composite"]["bucket"],
        "label_mode": "descriptive_signal",
        "component_scores": {
            "base_model": round(0.6 * dashboard["signal_scores"]["valuation"] + 0.4 * dashboard["signal_scores"]["cycle"], 1),
            "yield_signal": dashboard["signal_scores"]["yield"],
            "downside_risk": dashboard["signal_scores"]["downside"],
        },
        "yield_gap_pct": dashboard["yield_gap_pct"],
        "target_yield": dashboard["target_yield"],
    }
