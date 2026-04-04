"""
Quarterly refresh runner — Stage 2.

Downloads all raw data sources in dependency order, validates each output,
updates JSON release-provenance sidecars, then runs the downstream pipeline
(Stages 3–6) and the independent validation suite.

Usage:
    cd housing_model/
    python -m src.ingestion.run_ingestion

    # Stop on first ingestion error:
    python -m src.ingestion.run_ingestion --strict

    # Skip downstream pipeline (ingestion only):
    python -m src.ingestion.run_ingestion --no-pipeline

Outputs:
    data/logs/ingestion_YYYYMMDD.log    — append-mode log (safe to re-run)
    stdout JSON summary                 — refresh_date, new_panel_end,
                                          staleness_days, series_updated,
                                          tests_passed, tests_failed

Idempotent: running multiple times the same day appends to the same log
file and only overwrites raw data if the fetched data is newer.
"""

from __future__ import annotations

import importlib
import json
import logging
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from src.core.paths import RAW_DATA_DIR, ensure_directory
from src.ingestion.release_metadata import read_release_metadata, write_release_metadata

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "data" / "logs"
STAGE6_OUTPUT_DIR = ROOT / "data" / "outputs" / "stage6"
CANONICAL_METADATA_PATH = ROOT / "data" / "outputs" / "metadata" / "canonical_panel_metadata.json"

# ---------------------------------------------------------------------------
# IngestionSource dataclass
# ---------------------------------------------------------------------------

@dataclass
class IngestionSource:
    """Specification for one raw data series."""

    name: str
    """Short machine-readable key (matches release-metadata sidecar series name)."""

    display_name: str
    """Human-readable label used in logs and the JSON summary."""

    fetch_fn: Callable[..., pd.DataFrame]
    """Function to call.  Must accept ``save=True`` unless ``fn_args_from_results``
    is set, in which case positional DataFrames are passed first."""

    series: str
    """Series identifier written to the release-metadata sidecar."""

    source_url: str
    """Canonical source URL written to the sidecar."""

    publication_lag_months: int
    """Typical months between reference-period end and publication date."""

    required_columns: Sequence[str]
    """Columns that must exist in the returned DataFrame."""

    key_columns: Sequence[str]
    """Subset of required_columns that must have no null values."""

    min_years: int = 10
    """Minimum number of distinct years expected in the date column."""

    pause_before_seconds: int = 1
    """Pause before calling this source to avoid rate-limiting."""

    fn_args_from_results: Sequence[str] = field(default_factory=list)
    """If non-empty, pass the DataFrames stored under these keys from the
    *results* dict as positional arguments to ``fetch_fn`` before ``save``."""


# ---------------------------------------------------------------------------
# Source catalogue  (dependency order)
# ---------------------------------------------------------------------------

def _build_sources() -> list[IngestionSource]:
    # Lazy imports so the module can be tested without importing everything at
    # module level (monkeypatching is simpler when imports are deferred).
    from src.ingestion.land_registry    import fetch_land_registry
    from src.ingestion.boe              import fetch_boe_rate
    from src.ingestion.boe_mortgage_rate import fetch_boe_mortgage_rate
    from src.ingestion.ons              import (
        fetch_ons_cpi,
        fetch_ons_earnings,
        fetch_ons_earnings_regional,
        fetch_ons_population,
    )
    from src.ingestion.mhclg            import fetch_mhclg_supply
    from src.ingestion.ons_rental       import (
        fetch_ons_rental_index,
        fetch_ons_rental_levels,
        compute_rental_yield,
    )

    return [
        IngestionSource(
            name="land_registry_hpi",
            display_name="Land Registry HPI",
            fetch_fn=fetch_land_registry,
            series="land_registry_hpi",
            source_url="https://landregistry.data.gov.uk/app/ukhpi",
            publication_lag_months=2,
            required_columns=["date", "region", "average_price"],
            key_columns=["date", "region", "average_price"],
            min_years=10,
            pause_before_seconds=0,
        ),
        IngestionSource(
            name="boe_base_rate",
            display_name="BoE Base Rate",
            fetch_fn=fetch_boe_rate,
            series="boe_base_rate",
            source_url="https://www.bankofengland.co.uk/boeapps/database/",
            publication_lag_months=0,
            required_columns=["date", "base_rate"],
            key_columns=["date", "base_rate"],
            min_years=10,
            pause_before_seconds=1,
        ),
        IngestionSource(
            name="boe_mortgage_rate",
            display_name="BoE Mortgage Rate",
            fetch_fn=fetch_boe_mortgage_rate,
            series="boe_mortgage_rate",
            source_url="https://www.bankofengland.co.uk/boeapps/database/",
            publication_lag_months=1,
            required_columns=["date", "mortgage_rate"],
            key_columns=["date", "mortgage_rate"],
            min_years=10,
            pause_before_seconds=1,
        ),
        IngestionSource(
            name="ons_cpi",
            display_name="ONS CPI",
            fetch_fn=fetch_ons_cpi,
            series="ons_cpi",
            source_url="https://www.ons.gov.uk/economy/inflationandpriceindices",
            publication_lag_months=1,
            required_columns=["date", "cpi"],
            key_columns=["date", "cpi"],
            min_years=10,
            pause_before_seconds=2,
        ),
        IngestionSource(
            name="ons_earnings",
            display_name="ONS Earnings (national)",
            fetch_fn=fetch_ons_earnings,
            series="ons_earnings",
            source_url="https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours",
            publication_lag_months=2,
            required_columns=["date", "weekly_earnings"],
            key_columns=["date", "weekly_earnings"],
            min_years=10,
            pause_before_seconds=2,
        ),
        IngestionSource(
            name="ons_earnings_regional",
            display_name="ONS Earnings (regional)",
            fetch_fn=fetch_ons_earnings_regional,
            series="ons_earnings_regional",
            source_url="https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/regionalestimates",
            publication_lag_months=3,
            required_columns=["date", "region", "weekly_earnings"],
            key_columns=["date", "region"],
            min_years=10,
            pause_before_seconds=2,
        ),
        IngestionSource(
            name="ons_population",
            display_name="ONS Population",
            fetch_fn=fetch_ons_population,
            series="ons_population",
            source_url="https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration",
            publication_lag_months=12,
            required_columns=["date", "region", "population"],
            key_columns=["date", "region"],
            min_years=10,
            pause_before_seconds=2,
        ),
        IngestionSource(
            name="mhclg_supply",
            display_name="MHCLG Housing Supply",
            fetch_fn=fetch_mhclg_supply,
            series="mhclg_supply",
            source_url="https://www.gov.uk/government/statistical-data-sets/live-tables-on-housing-supply",
            publication_lag_months=6,
            required_columns=["date", "region", "completions"],
            key_columns=["date", "region"],
            min_years=10,
            pause_before_seconds=1,
        ),
        IngestionSource(
            name="ons_rental_index",
            display_name="ONS Rental Index (PIPR)",
            fetch_fn=fetch_ons_rental_index,
            series="ons_rental_index",
            source_url="https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/priceindexofprivaterentsukhistoricalseries",
            publication_lag_months=1,
            required_columns=["date", "region", "rental_index"],
            key_columns=["date", "region"],
            min_years=10,
            pause_before_seconds=2,
        ),
        IngestionSource(
            name="ons_rental_levels",
            display_name="ONS Rental Levels (PRMS)",
            fetch_fn=fetch_ons_rental_levels,
            series="ons_rental_levels",
            source_url="https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/privaterentalmarketsummarystatisticsengland",
            publication_lag_months=12,
            required_columns=["date", "region", "median_monthly_rent"],
            key_columns=["date", "region"],
            min_years=5,
            pause_before_seconds=2,
        ),
        IngestionSource(
            name="rental_yield",
            display_name="Rental Yield (computed)",
            fetch_fn=compute_rental_yield,
            series="rental_yield",
            source_url="computed:ons_rental_levels+land_registry_hpi",
            publication_lag_months=12,
            required_columns=["date", "region", "gross_yield_pct"],
            key_columns=["date", "region", "gross_yield_pct"],
            min_years=5,
            pause_before_seconds=0,
            fn_args_from_results=["ons_rental_levels", "land_registry_hpi"],
        ),
    ]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_output(df: pd.DataFrame, source: IngestionSource, log: logging.Logger) -> None:
    """Raise ValueError if df fails basic structural checks."""
    if df is None or df.empty:
        raise ValueError(f"{source.display_name}: fetch returned empty DataFrame")

    missing_cols = [c for c in source.required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{source.display_name}: missing required columns {missing_cols}"
        )

    for col in source.key_columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            raise ValueError(
                f"{source.display_name}: key column '{col}' has {null_count} null values"
            )

    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError(f"{source.display_name}: no parseable dates in 'date' column")

    years = dates.dt.year.nunique()
    if years < source.min_years:
        raise ValueError(
            f"{source.display_name}: only {years} distinct years in data "
            f"(expected >= {source.min_years})"
        )

    log.debug(
        "%s: validated OK — %d rows, date range %s to %s",
        source.display_name,
        len(df),
        dates.min().strftime("%Y-%m"),
        dates.max().strftime("%Y-%m"),
    )


def _check_coverage_regression(
    df: pd.DataFrame,
    source: IngestionSource,
    raw_data_dir: Path,
    log: logging.Logger,
) -> None:
    """Raise ValueError if the new data's max date is earlier than the prior sidecar."""
    prior = read_release_metadata(source.series, raw_data_dir)
    if prior is None:
        log.debug("%s: no prior sidecar — regression check skipped", source.display_name)
        return

    prior_end_str = prior.get("observation_period_end", "")
    if not prior_end_str:
        return

    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    new_end = dates.max().to_period("M").to_timestamp()
    prior_end = pd.Period(prior_end_str, freq="M").to_timestamp()

    if new_end < prior_end:
        raise ValueError(
            f"Coverage regression for {source.display_name}: "
            f"new data ends {new_end.strftime('%Y-%m')} but prior sidecar "
            f"recorded {prior_end_str}. Refusing to overwrite."
        )

    log.debug(
        "%s: coverage OK — prior end %s, new end %s",
        source.display_name,
        prior_end_str,
        new_end.strftime("%Y-%m"),
    )


def _call_fetch(
    source: IngestionSource,
    results: dict[str, pd.DataFrame],
    log: logging.Logger,
) -> pd.DataFrame:
    """Call source.fetch_fn, passing any required prior results as positional args."""
    if source.fn_args_from_results:
        missing = [k for k in source.fn_args_from_results if k not in results]
        if missing:
            raise ValueError(
                f"{source.display_name}: required prior results missing: {missing}"
            )
        positional = [results[k] for k in source.fn_args_from_results]
        log.debug("%s: calling with positional args from %s", source.display_name, list(source.fn_args_from_results))
        return source.fetch_fn(*positional, save=True)

    return source.fetch_fn(save=True)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(logs_dir: Path) -> tuple[logging.Logger, Path]:
    """Create a file+console logger.  Returns (logger, log_path)."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"ingestion_{date.today().strftime('%Y%m%d')}.log"

    fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # File handler — append so re-runs on the same day accumulate
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt))

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt))

    log = logging.getLogger("ingestion.run")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    log.addHandler(fh)
    log.addHandler(ch)
    log.propagate = False

    return log, log_path


# ---------------------------------------------------------------------------
# Downstream pipeline
# ---------------------------------------------------------------------------

def _run_downstream_pipeline(log: logging.Logger) -> None:
    """Run Stages 3–6 via canonical_rebuild after ingestion completes."""
    log.info("Running downstream pipeline (Stages 3–6)…")
    result = subprocess.run(
        [sys.executable, "-m", "src.model.canonical_rebuild"],
        cwd=ROOT,
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"canonical_rebuild exited with code {result.returncode}"
        )
    log.info("Downstream pipeline completed successfully")


def _run_tests(log: logging.Logger) -> tuple[int, int]:
    """Run the pytest suite. Returns (passed, failed) counts."""
    log.info("Running test suite…")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr

    # Parse pytest summary line: "N passed, M failed"
    passed = failed = 0
    for line in output.splitlines():
        if "passed" in line or "failed" in line or "error" in line:
            import re
            m_passed = re.search(r"(\d+) passed", line)
            m_failed = re.search(r"(\d+) failed", line)
            m_error  = re.search(r"(\d+) error", line)
            if m_passed:
                passed = int(m_passed.group(1))
            if m_failed:
                failed += int(m_failed.group(1))
            if m_error:
                failed += int(m_error.group(1))

    log.info("Tests: %d passed, %d failed", passed, failed)
    if result.returncode != 0:
        log.warning("Pytest exited with code %d — review test output", result.returncode)
        # Log last 30 lines of output for diagnostics
        for line in output.splitlines()[-30:]:
            log.debug("pytest: %s", line)

    return passed, failed


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_all(
    stop_on_error: bool = False,
    run_pipeline: bool = True,
    logs_dir: Path | None = None,
) -> dict:
    """
    Run all ingestion sources then (optionally) the downstream pipeline.

    Returns a JSON-serialisable summary dict.
    """
    if logs_dir is None:
        logs_dir = LOGS_DIR
    log, log_path = _setup_logging(logs_dir)

    raw_data_dir = ensure_directory(RAW_DATA_DIR)
    sources = _build_sources()

    log.info("=" * 72)
    log.info("Quarterly refresh started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("%d sources configured", len(sources))
    log.info("Log: %s", log_path)

    results_dfs: dict[str, pd.DataFrame] = {}
    series_updated: list[str] = []
    source_errors: dict[str, str] = {}

    for source in sources:
        if source.pause_before_seconds:
            log.debug("Pausing %ds before %s", source.pause_before_seconds, source.display_name)
            time.sleep(source.pause_before_seconds)

        log.info("Fetching %s", source.display_name)
        try:
            df = _call_fetch(source, results_dfs, log)
            _validate_output(df, source, log)
            _check_coverage_regression(df, source, raw_data_dir, log)

            # Write/update release-metadata sidecar
            # Find the raw file by globbing for the most recently modified CSV
            raw_csvs = sorted(
                raw_data_dir.glob(f"*{source.series}*.*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            raw_file = raw_csvs[0] if raw_csvs else raw_data_dir / f"{source.series}.csv"

            write_release_metadata(
                series=source.series,
                raw_file_path=raw_file,
                source_url=source.source_url,
                publication_lag_months_estimated=source.publication_lag_months,
                df=df,
            )

            results_dfs[source.name] = df
            series_updated.append(source.display_name)
            log.info("OK — %s (%d rows)", source.display_name, len(df))

        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            source_errors[source.display_name] = msg
            log.error("FAILED — %s: %s", source.display_name, msg)
            if stop_on_error:
                traceback.print_exc()
                raise

    # ------------------------------------------------------------------
    # Downstream pipeline (Stages 3–6) + tests
    # ------------------------------------------------------------------
    pipeline_ok = False
    tests_passed = tests_failed = 0

    if run_pipeline:
        try:
            _run_downstream_pipeline(log)
            pipeline_ok = True
        except Exception as exc:
            log.error("Downstream pipeline FAILED: %s", exc)
            source_errors["_pipeline"] = str(exc)

        tests_passed, tests_failed = _run_tests(log)

    # ------------------------------------------------------------------
    # Derive summary values from rebuilt metadata
    # ------------------------------------------------------------------
    new_panel_end: str | None = None
    staleness_days: int | None = None
    effective_data_as_of: str | None = None

    try:
        if CANONICAL_METADATA_PATH.exists():
            meta = json.loads(CANONICAL_METADATA_PATH.read_text(encoding="utf-8"))
            new_panel_end = meta.get("panel_end")
            effective_data_as_of = meta.get("effective_data_as_of")
            if new_panel_end:
                staleness_days = (date.today() - date.fromisoformat(new_panel_end)).days
    except Exception as exc:
        log.warning("Could not read canonical metadata for summary: %s", exc)

    summary = {
        "refresh_date": date.today().isoformat(),
        "new_panel_end": new_panel_end,
        "staleness_days": staleness_days,
        "effective_data_as_of": effective_data_as_of,
        "series_updated": series_updated,
        "series_errors": source_errors,
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "pipeline_ran": pipeline_ok,
        "log_file": str(log_path),
    }

    log.info("=" * 72)
    log.info("Refresh complete. Panel end: %s, staleness: %s days", new_panel_end, staleness_days)
    log.info("%d/%d sources updated; %d errors", len(series_updated), len(sources), len(source_errors))

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    stop = "--strict" in sys.argv
    no_pipeline = "--no-pipeline" in sys.argv

    summary = run_all(stop_on_error=stop, run_pipeline=not no_pipeline)
    print(json.dumps(summary, indent=2))

    has_errors = bool(summary.get("series_errors")) or summary.get("tests_failed", 0) > 0
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
