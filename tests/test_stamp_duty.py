"""
Tests for the SDLT calculator — no network calls required.
Run: pytest tests/test_stamp_duty.py -v
"""

import pytest
from src.ingestion.stamp_duty import calculate_sdlt, SDLTResult


def test_zero_sdlt_below_threshold():
    result = calculate_sdlt(120_000)
    assert result.sdlt_total == 0.0


def test_standard_purchase_250k():
    # £125k @ 0% + £125k @ 2% = £2,500
    result = calculate_sdlt(250_000)
    assert result.sdlt_total == 2_500.0


def test_standard_purchase_500k():
    # £125k @ 0% + £125k @ 2% + £250k @ 5% = £15,000
    result = calculate_sdlt(500_000)
    assert result.sdlt_total == 15_000.0


def test_ftb_relief_below_425k():
    # FTB: nil rate up to £425k
    result = calculate_sdlt(400_000, first_time_buyer=True)
    assert result.sdlt_total == 0.0
    assert result.buyer_type == "First-time buyer"


def test_ftb_relief_between_425k_and_625k():
    # FTB: nil up to £425k, 5% on remainder
    result = calculate_sdlt(500_000, first_time_buyer=True)
    expected = (500_000 - 425_000) * 0.05  # = £3,750
    assert result.sdlt_total == pytest.approx(expected)


def test_ftb_no_relief_above_625k():
    # Above £625k, FTB pays standard rates
    result_ftb      = calculate_sdlt(700_000, first_time_buyer=True)
    result_standard = calculate_sdlt(700_000, first_time_buyer=False)
    assert result_ftb.sdlt_total == result_standard.sdlt_total


def test_additional_dwelling_surcharge():
    # Standard SDLT + 5% surcharge on full price
    standard = calculate_sdlt(300_000)
    additional = calculate_sdlt(300_000, additional_dwelling=True)
    assert additional.sdlt_total == pytest.approx(standard.sdlt_total + 300_000 * 0.05)
    assert additional.buyer_type == "Additional dwelling"


def test_non_resident_surcharge():
    standard     = calculate_sdlt(500_000)
    non_resident = calculate_sdlt(500_000, non_uk_resident=True)
    assert non_resident.sdlt_total == pytest.approx(standard.sdlt_total + 500_000 * 0.02)


def test_effective_rate():
    result = calculate_sdlt(500_000)
    assert result.effective_rate == pytest.approx(result.sdlt_total / 500_000)


def test_returns_sdlt_result_type():
    result = calculate_sdlt(300_000)
    assert isinstance(result, SDLTResult)


def test_invalid_price_raises():
    with pytest.raises(ValueError):
        calculate_sdlt(0)
    with pytest.raises(ValueError):
        calculate_sdlt(-100_000)
