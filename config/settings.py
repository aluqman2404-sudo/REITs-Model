"""
Central settings — loads parameters.json and environment variables.
All modules import from here instead of hardcoding paths or values.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_RAW       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
DATA_OUTPUTS   = ROOT_DIR / "data" / "outputs"
CONFIG_DIR     = ROOT_DIR / "config"

PARAMS_PATH = CONFIG_DIR / "parameters.json"

def load_parameters() -> dict:
    with open(PARAMS_PATH, "r") as f:
        return json.load(f)

def save_parameters(params: dict) -> None:
    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)

PARAMS = load_parameters()

# API keys — set these in a .env file, never hardcode
ONS_API_KEY     = os.getenv("ONS_API_KEY", "")
WOLFRAM_APP_ID  = os.getenv("WOLFRAM_APP_ID", "")
