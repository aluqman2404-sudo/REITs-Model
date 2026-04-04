from pathlib import Path

from src.scoring.engine import regional_market_signal_dashboard


ROOT = Path(__file__).resolve().parents[1]


def test_independent_validation_runner_has_no_ui_dependency():
    source = (ROOT / "src/model/independent_validation_runner.py").read_text(encoding="utf-8")

    assert "src.ui." not in source


def test_views_do_not_hardcode_display_dates():
    source = (ROOT / "src/ui/views.py").read_text(encoding="utf-8")

    assert "2005-2023" not in source
    assert 'Latest market data available", "2025"' not in source


def test_home_page_has_no_hidden_household_defaults():
    source = (ROOT / "src/ui/views.py").read_text(encoding="utf-8")
    home_section = source.split("def render_home() -> None:")[1].split("def render_consumer_view() -> None:")[0]

    assert "consumer_signal_dashboard(" not in home_section
    assert "default_income_gbp" not in home_section
    assert "default_deposit_gbp" not in home_section
    assert "default_mortgage_term_years" not in home_section


def test_regional_market_dashboard_is_region_only():
    dashboard = regional_market_signal_dashboard(
        "London",
        valuation_signal=40,
        cycle_signal=55,
        downside_prob=0.30,
        rent_support_signal=65,
    )

    assert {signal["display_name"] for signal in dashboard["signals"]} == {
        "Valuation",
        "Cycle State",
        "Downside Vulnerability",
        "Rent Support",
    }
    assert all("question" in signal for signal in dashboard["signals"])
