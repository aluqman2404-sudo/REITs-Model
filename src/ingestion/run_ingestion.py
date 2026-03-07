"""
Stage 2: Master ingestion runner.
Run this script to download all raw data sources in sequence.

Usage:
    cd housing_model/
    python -m src.ingestion.run_ingestion

All files are saved to data/raw/ with a YYYYMM timestamp suffix.
Re-running this script will create a fresh download without overwriting
existing files (new timestamp = new filename).
"""

import sys
import traceback
from datetime import datetime

from src.ingestion.land_registry import fetch_land_registry
from src.ingestion.ons import fetch_ons_cpi, fetch_ons_earnings, fetch_ons_population
from src.ingestion.boe import fetch_boe_rate


def run_all(stop_on_error: bool = False) -> dict:
    """
    Run all ingestion functions and return a status report.

    Args:
        stop_on_error: If True, raise on first failure. If False, continue and report.

    Returns:
        dict mapping source name → 'ok' | error message
    """
    sources = {
        "Land Registry HPI": fetch_land_registry,
        "ONS CPI":           fetch_ons_cpi,
        "ONS Earnings":      fetch_ons_earnings,
        "ONS Population":    fetch_ons_population,
        "BoE Base Rate":     fetch_boe_rate,
    }

    results = {}
    print(f"\n{'='*55}")
    print(f"  UK Housing Model — Data Ingestion")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*55}\n")

    for name, fn in sources.items():
        print(f"── {name} {'─'*(40 - len(name))}")
        try:
            fn(save=True)
            results[name] = "ok"
            print(f"   OK\n")
        except Exception as e:
            msg = f"FAILED: {e}"
            results[name] = msg
            print(f"   {msg}\n")
            if stop_on_error:
                traceback.print_exc()
                raise

    # Summary
    print(f"\n{'='*55}")
    print("  Ingestion Summary")
    print(f"{'='*55}")
    for name, status in results.items():
        icon = "✓" if status == "ok" else "✗"
        print(f"  {icon}  {name:<30} {status}")

    n_ok = sum(1 for s in results.values() if s == "ok")
    print(f"\n  {n_ok}/{len(sources)} sources downloaded successfully.")
    print(f"{'='*55}\n")

    return results


if __name__ == "__main__":
    stop = "--strict" in sys.argv
    results = run_all(stop_on_error=stop)
    failed = [k for k, v in results.items() if v != "ok"]
    sys.exit(1 if failed else 0)
