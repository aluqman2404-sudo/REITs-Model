"""Plain-English signal explanations and risk notes."""

from __future__ import annotations


def top_signal_contributors(signals: list[dict]) -> tuple[list[str], list[str]]:
    ordered = sorted(signals, key=lambda item: item["score"], reverse=True)
    positives = [
        f"{item['display_name']} ({item['score']:.0f}/100)" for item in ordered[:2]]
    negatives = [
        f"{item['display_name']} ({item['score']:.0f}/100)" for item in ordered[-2:]]
    return positives, negatives


def plain_signal_takeaways(signals: list[dict]) -> list[str]:
    ordered = sorted(signals, key=lambda item: item["score"], reverse=True)
    strongest = ordered[0]
    weakest = ordered[-1]
    takeaways = [
        f"Best-looking area right now: {strongest['display_name'].lower()} because it {strongest['simple_status'].lower()}.",
        f"Main pressure point: {weakest['display_name'].lower()} because it {weakest['simple_status'].lower()}.",
    ]
    for signal in signals:
        if signal["evidence_level"] == "empirically_supported":
            takeaways.append(
                f"The strongest research-backed indicator is {signal['display_name'].lower()}, which currently {signal['simple_status'].lower()}."
            )
            break
    return takeaways


def scenario_sensitivity_text(spread: float) -> str:
    if spread >= 18:
        return "High scenario sensitivity: the outlook changes materially across macro paths."
    if spread >= 8:
        return "Moderate scenario sensitivity: the central view matters, but the ranking is fairly stable."
    return "Low scenario sensitivity: structural valuation dominates the short-run macro regime."


def consumer_commentary(region_row: dict, signal_dashboard: dict, scenario_name: str) -> str:
    positives, negatives = top_signal_contributors(signal_dashboard["signals"])
    secondary = signal_dashboard["secondary_composite"]
    sensitivity = scenario_sensitivity_text(float(region_row.get("scenario_spread", 0.0)))
    return (
        f"**{region_row['region']} — {scenario_name} scenario**\n\n"
        f"- **Strengths:** {', '.join(positives)}\n"
        f"- **Pressures:** {', '.join(negatives)}\n"
        f"- **Secondary composite:** {secondary['score']:.0f}/100 ({secondary['bucket']})\n"
        f"- **Scenario sensitivity:** {sensitivity}"
    )


def reit_commentary(region_row: dict, signal_dashboard: dict, scenario_name: str) -> str:
    positives, negatives = top_signal_contributors(signal_dashboard["signals"])
    secondary = signal_dashboard["secondary_composite"]
    sensitivity = scenario_sensitivity_text(float(region_row.get("scenario_spread", 0.0)))
    loss_prob = float(region_row.get("p_terminal_loss_10_avg", 0.0))
    return (
        f"**{region_row['region']} — {scenario_name} institutional view**\n\n"
        f"- **Strengths:** {', '.join(positives)}\n"
        f"- **Pressures:** {', '.join(negatives)}\n"
        f"- **Secondary composite:** {secondary['score']:.0f}/100 ({secondary['bucket']})\n"
        f"- **Scenario sensitivity:** {sensitivity}\n"
        f"- **Tail risk:** {loss_prob:.0%} avg probability of a 10%+ terminal loss across scenarios"
    )


def consumer_risk_flags(region_row: dict, signal_dashboard: dict) -> list[str]:
    risks = []
    if signal_dashboard["mortgage_stress"] in {"High", "Elevated"}:
        risks.append(
            "Mortgage servicing burden is elevated relative to gross income.")
    if float(region_row.get("pct_above_pstar", 0.0)) > 15:
        risks.append(
            "Regional prices remain above the model's current fair-value estimate.")
    if float(region_row.get("wtd_return_consumer", 0.0)) < 0:
        risks.append(
            "The weighted 5-year return outlook is still negative across core scenarios.")
    if float(region_row.get("unemployment_lag1", region_row.get("unemployment_rate", 0.0))) > 5.5:
        risks.append(
            "Local labour-market slack is elevated relative to the pre-2020 range.")
    if float(region_row.get("approvals_growth", 0.0)) < -0.10:
        risks.append(
            "Mortgage approvals remain weak, which is a headwind for demand momentum.")
    return risks or ["Model risk remains material; the signal dashboard is decision support only."]


def reit_risk_flags(region_row: dict, signal_dashboard: dict) -> list[str]:
    risks = []
    if signal_dashboard["yield_gap_pct"] < 0:
        risks.append(
            "Current regional yield is below the selected hurdle rate.")
    if float(region_row.get("p_terminal_loss_10_avg", 0.0)) > 0.40:
        risks.append("Downside tail risk remains high in adverse scenarios.")
    if float(region_row.get("scenario_spread", 0.0)) > 18:
        risks.append("Scenario ranking is highly sensitive to the macro path.")
    if float(region_row.get("transaction_growth", 0.0)) < -0.10:
        risks.append(
            "Transaction momentum is weak, which can impair exit liquidity.")
    return risks or ["The main risk is standard model uncertainty rather than a single dominant macro variable."]


def override_warning(rate_shift_bps: float, sigma_multiplier: float, horizon_years: float) -> str | None:
    warnings = []
    if abs(rate_shift_bps) >= 150:
        warnings.append("rate override exceeds 150 bps")
    if sigma_multiplier >= 1.5:
        warnings.append(
            "volatility override is materially above calibrated levels")
    if horizon_years > 10:
        warnings.append(
            "forecast horizon exceeds the validated research window")
    if not warnings:
        return None
    return "User override warning: " + "; ".join(warnings) + "."
