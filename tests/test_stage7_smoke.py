import importlib
import os
from pathlib import Path

from streamlit.testing.v1 import AppTest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
ROOT = Path(__file__).resolve().parents[1]


def test_stage7_legacy_entrypoint_smoke():
    at = AppTest.from_file(str(ROOT / "src/ui/app.py"))
    at.run(timeout=30)

    assert not at.exception
    markdown_values = [element.value for element in at.markdown]
    assert "Housing Market Research Dashboard" in " ".join(markdown_values)


def test_limitations_page_imports():
    """
    The limitations module must import without error.

    Verifies that src.ui.pages.limitations is importable in a non-running
    Streamlit context (no top-level st.* calls at import time) and that the
    render_limitations callable is exposed.
    """
    mod = importlib.import_module("src.ui.pages.limitations")
    assert callable(getattr(mod, "render_limitations", None)), (
        "render_limitations is not a callable in src.ui.pages.limitations"
    )


def test_scenario_lab_render_smoke():
    at = AppTest.from_string(
        """
from src.ui.views import render_scenario_lab

render_scenario_lab()
"""
    )
    at.run(timeout=30)

    assert not at.exception


def test_limitations_page_has_22_disclosures():
    """
    The limitations source file must contain exactly disclosure labels L1–L22.

    This guards against the numbering drift that can happen when disclosures
    are edited over time. The page claims 22 disclosures, so the raw source
    should expose labels L1 through L22 and nothing higher.
    """
    import re

    source = (ROOT / "src" / "ui" / "pages" / "limitations.py").read_text(
        encoding="utf-8"
    )
    labels = sorted({int(match) for match in re.findall(r"L(\d+)", source)})
    assert labels == list(range(1, 23)), (
        f"Expected disclosure labels L1-L22 exactly, got {labels}."
    )
