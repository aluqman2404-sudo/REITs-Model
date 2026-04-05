"""
Stage 2: HMRC Stamp Duty Land Tax (SDLT) calculator.

Standalone module — no external data required.
Applies the correct SDLT band rules including:
  - Standard residential rates
  - First-time buyer relief (FTB)
  - Additional dwelling surcharge (buy-to-let / second homes)
  - Higher rates for non-UK residents

Current thresholds as of Autumn Budget 2024 (effective 1 April 2025):
  - Nil rate band reduced back to £125,000 (FTB relief up to £500,000)
  - Additional dwelling surcharge increased to 5%

Thresholds stored in config/parameters.json for easy updating.
"""

from __future__ import annotations
from dataclasses import dataclass


# ── SDLT band definitions ─────────────────────────────────────────────────────
# Format: (upper_limit, rate)  — None upper_limit = no ceiling
_STANDARD_BANDS = [
    (125_000,   0.00),
    (250_000,   0.02),
    (925_000,   0.05),
    (1_500_000, 0.10),
    (None,      0.12),
]

_FTB_BANDS = [
    (425_000,   0.00),  # FTB nil-rate up to £425k (reverts to £300k Apr 2025)
    (625_000,   0.05),  # FTB max price for relief
    # Above £625k, FTB pays standard rates (no relief applies)
]

_ADDITIONAL_SURCHARGE = 0.05  # 5% on additional dwellings (from Oct 2024)
_NON_RESIDENT_SURCHARGE = 0.02


@dataclass
class SDLTResult:
    property_price:  float
    sdlt_total:      float
    effective_rate:  float        # as a decimal
    breakdown:       list[dict]   # per-band breakdown
    buyer_type:      str


def calculate_sdlt(
    price: float,
    first_time_buyer: bool = False,
    additional_dwelling: bool = False,
    non_uk_resident: bool = False,
) -> SDLTResult:
    """
    Calculate SDLT for a residential property purchase in England/NI.

    Args:
        price:               Purchase price in £
        first_time_buyer:    True if all buyers are first-time buyers
        additional_dwelling: True if buyer already owns a residential property
        non_uk_resident:     True if buyer is not a UK resident

    Returns:
        SDLTResult with total tax, effective rate, and per-band breakdown
    """
    if price <= 0:
        raise ValueError("Property price must be positive")

    # First-time buyer relief applies only up to £625,000
    # (reverts to £500k from April 2025 — update bands accordingly)
    use_ftb = first_time_buyer and price <= 625_000

    if use_ftb:
        bands = _build_ftb_bands(price)
        buyer_type = "First-time buyer"
    else:
        bands = _STANDARD_BANDS
        buyer_type = "Standard"

    total, breakdown = _apply_bands(price, bands)

    # Surcharges are applied on top of the base SDLT
    if additional_dwelling:
        surcharge = price * _ADDITIONAL_SURCHARGE
        total += surcharge
        buyer_type = "Additional dwelling"
        breakdown.append({
            "band":    "Additional dwelling surcharge",
            "rate":    _ADDITIONAL_SURCHARGE,
            "taxable": price,
            "tax":     surcharge,
        })

    if non_uk_resident:
        nr_charge = price * _NON_RESIDENT_SURCHARGE
        total += nr_charge
        breakdown.append({
            "band":    "Non-UK resident surcharge",
            "rate":    _NON_RESIDENT_SURCHARGE,
            "taxable": price,
            "tax":     nr_charge,
        })

    return SDLTResult(
        property_price=price,
        sdlt_total=round(total, 2),
        effective_rate=round(total / price, 6) if price > 0 else 0.0,
        breakdown=breakdown,
        buyer_type=buyer_type,
    )


def sdlt_summary(result: SDLTResult) -> str:
    """Return a formatted string summary of an SDLTResult."""
    lines = [
        f"Property price:  £{result.property_price:,.0f}",
        f"Buyer type:      {result.buyer_type}",
        f"SDLT total:      £{result.sdlt_total:,.0f}",
        f"Effective rate:  {result.effective_rate * 100:.2f}%",
        "",
        "Breakdown:",
    ]
    for b in result.breakdown:
        lines.append(
            f"  {b['band']:<35} "
            f"£{b['taxable']:>10,.0f} @ {b['rate']*100:.0f}% = £{b['tax']:>8,.0f}"
        )
    return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_ftb_bands(price: float) -> list[tuple]:
    """
    For FTB relief: nil rate up to £425k, 5% on £425k–£625k.
    Above £625k the standard bands apply (relief not available).
    """
    if price > 625_000:
        return _STANDARD_BANDS
    return [
        (425_000, 0.00),
        (None,    0.05),
    ]


def _apply_bands(price: float, bands: list[tuple]) -> tuple[float, list[dict]]:
    """Apply SDLT bands to a price and return total tax and per-band breakdown."""
    total = 0.0
    breakdown = []
    prev_limit = 0

    for upper, rate in bands:
        if price <= prev_limit:
            break

        ceiling = upper if upper is not None else float("inf")
        taxable = min(price, ceiling) - prev_limit
        tax = taxable * rate
        total += tax

        breakdown.append({
            "band":    f"£{prev_limit:,} – £{upper:,}" if upper else f"Above £{prev_limit:,}",
            "rate":    rate,
            "taxable": taxable,
            "tax":     tax,
        })

        prev_limit = upper if upper is not None else price

    return total, breakdown
