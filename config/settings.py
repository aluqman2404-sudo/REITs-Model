"""Backward-compatible settings wrapper around the typed config layer."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.core.config import DEFAULT_CONFIG_PATH, load_config, reset_config_cache
from src.core.paths import CONFIG_DIR, DATA_DIR, OUTPUT_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ROOT_DIR


load_dotenv()

DATA_RAW = RAW_DATA_DIR
DATA_PROCESSED = PROCESSED_DATA_DIR
DATA_OUTPUTS = OUTPUT_DATA_DIR
PARAMS_PATH = DEFAULT_CONFIG_PATH


def load_parameters() -> dict:
    """Return the config as a plain dictionary for legacy modules."""
    return load_config().model_dump()


def save_parameters(params: dict) -> None:
    """Persist parameters.json and clear the typed config cache."""
    with Path(PARAMS_PATH).open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)
    reset_config_cache()


PARAMS = load_parameters()

ONS_API_KEY = os.getenv("ONS_API_KEY", "")
WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID", "")
