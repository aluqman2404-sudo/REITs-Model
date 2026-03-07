"""
Stage 6: Scoring engine.
Transforms Monte Carlo output into a 0–100 Buy / Hold / Don't Buy score
for both consumer and REIT views.
"""

from config.settings import load_parameters

PARAMS = load_parameters()
_THRESHOLDS = PARAMS["scoring"]["thresholds"]
_CONSUMER_W = PARAMS["scoring"]["consumer_weights"]
_REIT_W     = PARAMS["scoring"]["reit_weights"]


def score_consumer(
    region: str,
    income: float,
    deposit: float,
    simulation_percentiles: dict,
) -> dict:
    """
    Compute the consumer Buy / Hold / Don't Buy score.

    Args:
        region:                  ONS region name
        income:                  annual household income (£)
        deposit:                 deposit amount (£)
        simulation_percentiles:  dict with keys p10, p50, p90 from Monte Carlo

    Returns:
        dict with keys: total_score, label, component_scores
    """
    # TODO (Stage 6): implement affordability-weighted consumer score
    raise NotImplementedError("score_consumer not yet implemented")


def score_reit(
    region: str,
    gross_yield: float,
    simulation_percentiles: dict,
) -> dict:
    """
    Compute the REIT yield + cycle positioning score.

    Returns:
        dict with keys: total_score, label, component_scores
    """
    # TODO (Stage 6): implement REIT yield/cycle score
    raise NotImplementedError("score_reit not yet implemented")


def label_score(score: float) -> str:
    """Map a 0–100 score to Buy / Hold / Don't Buy."""
    if score >= _THRESHOLDS["buy"]:
        return "Buy"
    elif score >= _THRESHOLDS["hold"]:
        return "Hold"
    return "Don't Buy"
