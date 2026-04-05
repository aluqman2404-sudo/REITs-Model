"""Formatting and small UI helpers."""

from __future__ import annotations


import streamlit as st

from src.core.paths import UI_ASSETS_DIR


BAND_COLOURS = {
    "Supportive": "#1f7a45",
    "Mixed": "#b26a00",
    "Cautious": "#a63f3f",
}

EVIDENCE_COLOURS = {
    "empirically_supported": "#12355b",
    "partially_supported": "#8a5a00",
    "descriptive": "#52606d",
}


def format_currency(value: float, precision: int = 0) -> str:
    return f"GBP {value:,.{precision}f}"


def format_pct(value: float, precision: int = 1, signed: bool = False) -> str:
    sign = "+" if signed else ""
    return f"{value:{sign},.{precision}f}%"


def format_number(value: float, precision: int = 1) -> str:
    return f"{value:,.{precision}f}"


def band_colour(label: str) -> str:
    return BAND_COLOURS.get(label, "#4f5b66")


def _pill_html(label: str, colour: str) -> str:
    return (
        f"<span style='display:inline-block;padding:0.2rem 0.55rem;"
        f"border-radius:999px;background:{colour};color:white;font-weight:600;'>"
        f"{label}</span>"
    )


def render_badge(label: str) -> str:
    return _pill_html(label, band_colour(label))


def render_evidence_badge(evidence_level: str) -> str:
    label = {
        "empirically_supported": "Validated",
        "partially_supported": "Partial",
        "descriptive": "Context Only",
    }.get(evidence_level, evidence_level.replace("_", " ").title())
    return _pill_html(label, EVIDENCE_COLOURS.get(evidence_level, "#4f5b66"))


def apply_app_style() -> None:
    """Inject the project CSS bundle once per page render."""
    css_path = UI_ASSETS_DIR / "style.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
