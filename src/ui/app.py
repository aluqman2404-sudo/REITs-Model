"""Stage 7 entrypoint for the multipage Streamlit dashboard."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import load_config
from src.core.logging_utils import configure_monitoring, get_logger, log_model_event
from src.ui.pages.limitations import render_limitations
from src.ui.views import (
    render_comparison_view,
    render_consumer_view,
    render_home,
    render_methodology,
    render_model_controls,
    render_reit_view,
    render_scenario_lab,
)

# Configure monitoring before any Streamlit calls so the startup event is captured.
configure_monitoring()

import streamlit as st

_logger = get_logger(__name__)


def main() -> None:
    config = load_config()

    # Log startup with scoring_date from stage6 handoff metadata
    try:
        import json as _json
        from src.core.paths import STAGE6_OUTPUT_DIR
        _handoff_meta_path = STAGE6_OUTPUT_DIR / "canonical_rebuild_metadata.json"
        _meta = _json.loads(_handoff_meta_path.read_text(encoding="utf-8"))
        _scoring_date = _meta.get("stage6_path", "unknown")
        # Prefer the scoring_date field from the handoff CSV if available
        import pandas as _pd
        _handoff_path = STAGE6_OUTPUT_DIR / load_config().artifacts.stage6_handoff_file
        if _handoff_path.exists():
            _hdf = _pd.read_csv(_handoff_path, nrows=1)
            _scoring_date = str(_hdf["scoring_date"].iloc[0]) if "scoring_date" in _hdf.columns else "unknown"
    except Exception:
        _scoring_date = "unknown"
    log_model_event(
        _logger,
        "APP_STARTUP",
        {
            "version": "0.2.0",
            "scoring_date": _scoring_date,
            "env": os.getenv("ENV", "dev"),
        },
    )

    st.set_page_config(
        page_title=config.project.name,
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pages = {
        "Housing Research Platform": [
            st.Page(render_home, title="Research Overview", icon=":material/home:"),
            st.Page(render_reit_view, title="Regional Signals", icon=":material/analytics:"),
            st.Page(render_comparison_view, title="Region Comparison", icon=":material/compare:"),
            st.Page(render_scenario_lab, title="Scenario & Diagnostics", icon=":material/tune:"),
            st.Page(render_consumer_view, title="Household Explorer", icon=":material/person:"),
            st.Page(render_methodology, title="Methodology", icon=":material/menu_book:"),
            st.Page(render_model_controls, title="Technical Controls", icon=":material/settings:"),
            st.Page(render_limitations, title="Limitations", icon=":material/warning:"),
        ]
    }
    nav = st.navigation(pages, position="sidebar")
    _logger.info("Page loaded: %s", nav.title)
    nav.run()


if __name__ == "__main__":
    main()
