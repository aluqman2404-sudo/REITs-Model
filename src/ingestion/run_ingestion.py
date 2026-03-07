"""
Stage 2: Master ingestion runner.
Downloads all 8 raw data sources for the UK Housing Market Model.

Usage:
    cd housing_model/
    python -m src.ingestion.run_ingestion

    # Stop on first error:
    python -m src.ingestion.run_ingestion --strict

All files saved to data/raw/ with YYYYMM timestamp suffix.
"""

import sys
import time
import traceback
from datetime import datetime

from src.ingestion.land_registry    import fetch_land_registry
from src.ingestion.ons              import fetch_ons_cpi, fetch_ons_earnings, fetch_ons_population
from src.ingestion.boe              import fetch_boe_rate
from src.ingestion.boe_mortgage_rate import fetch_boe_mortgage_rate
from src.ingestion.mhclg            import fetch_mhclg_supply
from src.ingestion.ons_rental       import fetch_ons_rental_index, fetch_ons_rental_levels


# Maps: display name → (function, pause_before_seconds)
# Pause avoids ONS rate-limiting (429) on consecutive requests
SOURCES = [
    ("Land Registry HPI",        fetch_land_registry,        0),
    ("ONS CPI",                  fetch_ons_cpi,               1),
    ("ONS Earnings",             fetch_ons_earnings,          2),
    ("ONS Population",           fetch_ons_population,        3),
    ("BoE Base Rate",            fetch_boe_rate,              1),
    ("BoE Mortgage Rate",        fetch_boe_mortgage_rate,     1),
    ("MHCLG Housing Supply",     fetch_mhclg_supply,          1),
    ("ONS Rental Index (IPHRP)", fetch_ons_rental_index,      2),
    ("ONS Rental Levels (PRMS)", fetch_ons_rental_levels,     2),
]


def run_all(stop_on_error: bool = False) -> dict:
    """
    Run all ingestion functions. Returns status dict: source → 'ok' | error.
    """
    results = {}

    print(f"\n{'='*57}")
    print(f"  UK Housing Market Model — Data Ingestion")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  {len(SOURCES)} sources | 6 model variables")
    print(f"{'='*57}\n")

    for name, fn, pause in SOURCES:
        if pause:
            time.sleep(pause)

        print(f"── {name} {'─'*(42 - len(name))}")
        try:
            fn(save=True)
            results[name] = "ok"
            print(f"   ✓ OK\n")
        except Exception as e:
            msg = f"FAILED: {e}"
            results[name] = msg
            print(f"   ✗ {msg}\n")
            if stop_on_error:
                traceback.print_exc()
                raise

    # Summary table
    print(f"\n{'='*57}")
    print("  Ingestion Summary")
    print(f"{'='*57}")
    for name, status in results.items():
        icon = "✓" if status == "ok" else "✗"
        truncated = status if len(status) < 45 else status[:42] + "..."
        print(f"  {icon}  {name:<32} {truncated}")

    n_ok = sum(1 for s in results.values() if s == "ok")
    print(f"\n  {n_ok}/{len(SOURCES)} sources downloaded successfully.")

    if n_ok < len(SOURCES):
        print("\n  For failed sources, download manually:")
        print("  Land Registry: https://www.gov.uk/government/collections/uk-house-price-index-reports")
        print("  MHCLG supply:  https://www.gov.uk/government/statistical-data-sets/live-tables-on-house-building")
        print("  ONS PRMS:      https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/privaterentalmarketsummarystatisticsengland")

    print(f"{'='*57}\n")
    return results


if __name__ == "__main__":
    stop = "--strict" in sys.argv
    results = run_all(stop_on_error=stop)
    failed  = [k for k, v in results.items() if v != "ok"]
    sys.exit(1 if failed else 0)
