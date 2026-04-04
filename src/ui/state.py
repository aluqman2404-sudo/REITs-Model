"""Session-state helpers for Streamlit pages."""

from __future__ import annotations

import streamlit as st

from src.core.config import load_config


def ensure_state_defaults() -> None:
    """Populate session state with stable defaults."""
    config = load_config()
    default_preset = "Baseline" if "Baseline" in config.ui.scenario_labels else config.ui.scenario_labels[0]
    defaults = {
        "scenario_lab_region": config.ui.default_region,
        "scenario_lab_preset": default_preset,
        "override_rate_shift_bps": 0.0,
        "override_sigma_multiplier": 1.0,
        "override_drift_shift_pct": 0.0,
        "override_horizon_years": config.ui.default_holding_horizon_years,
        "override_paths": config.controls.default_paths_interactive,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state["scenario_lab_preset"] not in config.ui.scenario_labels:
        st.session_state["scenario_lab_preset"] = default_preset


def reset_override_state() -> None:
    """Reset Scenario Lab / controls overrides to defaults."""
    config = load_config()
    default_preset = "Baseline" if "Baseline" in config.ui.scenario_labels else config.ui.scenario_labels[0]
    st.session_state["scenario_lab_preset"] = default_preset
    st.session_state["override_rate_shift_bps"] = 0.0
    st.session_state["override_sigma_multiplier"] = 1.0
    st.session_state["override_drift_shift_pct"] = 0.0
    st.session_state["override_horizon_years"] = config.ui.default_holding_horizon_years
    st.session_state["override_paths"] = config.controls.default_paths_interactive
