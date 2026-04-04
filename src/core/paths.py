"""Portable project path helpers."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "outputs"
METADATA_OUTPUT_DIR = OUTPUT_DATA_DIR / "metadata"
STAGE4_OUTPUT_DIR = OUTPUT_DATA_DIR / "stage4_final"
STAGE5_OUTPUT_DIR = OUTPUT_DATA_DIR / "stage5c"
STAGE6_OUTPUT_DIR = OUTPUT_DATA_DIR / "stage6"
STAGE7_OUTPUT_DIR = OUTPUT_DATA_DIR / "stage7"
CANONICAL_PANEL_PATH = PROCESSED_DATA_DIR / "master_dataset_canonical.csv"
DESCRIPTIVE_PANEL_PATH = PROCESSED_DATA_DIR / "descriptive_market_panel.csv"
CANONICAL_PANEL_METADATA_PATH = METADATA_OUTPUT_DIR / "canonical_panel_metadata.json"
DESCRIPTIVE_PANEL_METADATA_PATH = METADATA_OUTPUT_DIR / "descriptive_panel_metadata.json"
UI_DIR = ROOT_DIR / "src" / "ui"
UI_ASSETS_DIR = UI_DIR / "assets"
ARCHIVE_DIR = ROOT_DIR / "archive"
LEGACY_APP_DIR = ARCHIVE_DIR / "legacy_app"
LEGACY_MODEL_DIR = ARCHIVE_DIR / "legacy_model"
TESTS_DIR = ROOT_DIR / "tests"


def ensure_directory(path: Path) -> Path:
    """Create `path` if required and return it for fluent usage."""
    path.mkdir(parents=True, exist_ok=True)
    return path
