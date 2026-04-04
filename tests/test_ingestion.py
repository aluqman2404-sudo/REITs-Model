"""
Tests for src/ingestion/run_ingestion.py
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingestion.run_ingestion import (
    IngestionSource,
    _build_sources,
    _check_coverage_regression,
    _validate_output,
    run_all,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal DataFrames that pass _validate_output
# ---------------------------------------------------------------------------

def _make_df(n_years: int = 12, extra_cols: dict | None = None) -> pd.DataFrame:
    """Return a minimal DataFrame with a date column spanning n_years."""
    dates = pd.date_range("2005-01-01", periods=n_years * 12, freq="MS")
    df = pd.DataFrame({"date": dates, "region": "London", "average_price": 200_000})
    if extra_cols:
        for col, val in extra_cols.items():
            df[col] = val
    return df


# ---------------------------------------------------------------------------
# test_run_ingestion_dry_run
# ---------------------------------------------------------------------------

class TestRunIngestionDryRun:
    """Verify that run_all() calls fetch functions in the declared dependency
    order, writes a log file, and returns the expected summary structure."""

    def test_call_order_and_summary_structure(self, tmp_path, monkeypatch):
        """Each source's fetch function is called exactly once in SOURCES order,
        the log file is created, and the summary has the required keys."""

        sources = _build_sources()
        call_order: list[str] = []

        # Build a mock fetch function for every source
        def _make_mock_fetch(name: str, df: pd.DataFrame):
            def mock_fetch(*args, save=True):  # noqa: ARG001
                call_order.append(name)
                return df
            return mock_fetch

        # Patch every fetch_fn on the IngestionSource objects
        patched_sources: list[IngestionSource] = []
        for src in sources:
            df = _make_df(extra_cols=_minimal_extra_cols(src))
            patched = IngestionSource(
                name=src.name,
                display_name=src.display_name,
                fetch_fn=_make_mock_fetch(src.name, df),
                series=src.series,
                source_url=src.source_url,
                publication_lag_months=src.publication_lag_months,
                required_columns=src.required_columns,
                key_columns=src.key_columns,
                min_years=src.min_years,
                pause_before_seconds=0,  # no sleeping in tests
                fn_args_from_results=src.fn_args_from_results,
            )
            patched_sources.append(patched)

        # Monkeypatch _build_sources so run_all uses our patched list
        monkeypatch.setattr(
            "src.ingestion.run_ingestion._build_sources",
            lambda: patched_sources,
        )

        # Patch downstream pipeline and tests — not exercised here
        monkeypatch.setattr(
            "src.ingestion.run_ingestion._run_downstream_pipeline",
            lambda log: None,
        )
        monkeypatch.setattr(
            "src.ingestion.run_ingestion._run_tests",
            lambda log: (46, 0),
        )

        # Patch write_release_metadata so no real files are written
        monkeypatch.setattr(
            "src.ingestion.run_ingestion.write_release_metadata",
            lambda **kwargs: tmp_path / "dummy_sidecar.json",
        )

        # Patch read_release_metadata to return None (no prior sidecar)
        monkeypatch.setattr(
            "src.ingestion.run_ingestion.read_release_metadata",
            lambda series, raw_data_dir: None,
        )

        # Redirect raw data dir and logs to tmp_path
        monkeypatch.setattr(
            "src.ingestion.run_ingestion.RAW_DATA_DIR",
            tmp_path / "raw",
        )
        (tmp_path / "raw").mkdir()

        # Patch canonical metadata path to return a minimal JSON
        meta_path = tmp_path / "canonical_panel_metadata.json"
        meta_path.write_text(
            json.dumps({"panel_end": "2025-10-01", "effective_data_as_of": "2025-08-01"}),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "src.ingestion.run_ingestion.CANONICAL_METADATA_PATH",
            meta_path,
        )

        summary = run_all(run_pipeline=True, logs_dir=tmp_path / "logs")

        # --- Call order must match SOURCES list order ---
        expected_order = [s.name for s in patched_sources]
        assert call_order == expected_order, (
            f"Call order mismatch.\n  Expected: {expected_order}\n  Got:      {call_order}"
        )

        # --- Log file must exist ---
        log_files = list((tmp_path / "logs").glob("ingestion_*.log"))
        assert len(log_files) == 1, "Expected exactly one log file"
        log_content = log_files[0].read_text(encoding="utf-8")
        for src in patched_sources:
            assert src.display_name in log_content, (
                f"Display name '{src.display_name}' not found in log"
            )

        # --- Summary structure ---
        required_keys = {
            "refresh_date",
            "new_panel_end",
            "staleness_days",
            "effective_data_as_of",
            "series_updated",
            "tests_passed",
            "tests_failed",
        }
        assert required_keys.issubset(summary.keys()), (
            f"Missing summary keys: {required_keys - summary.keys()}"
        )
        assert summary["tests_passed"] == 46
        assert summary["tests_failed"] == 0
        assert summary["new_panel_end"] == "2025-10-01"


def _minimal_extra_cols(src: IngestionSource) -> dict:
    """Return extra column values needed for src.required_columns beyond
    date/region/average_price (which _make_df already provides)."""
    extra = {}
    for col in src.required_columns:
        if col not in ("date", "region", "average_price"):
            extra[col] = 1.0
    return extra


# ---------------------------------------------------------------------------
# test_sidecar_regression_guard
# ---------------------------------------------------------------------------

class TestSidecarRegressionGuard:
    """Verify that _check_coverage_regression raises ValueError when the
    fetched data covers fewer months than the existing sidecar."""

    def test_raises_on_coverage_regression(self, tmp_path):
        source = IngestionSource(
            name="land_registry_hpi",
            display_name="Land Registry HPI",
            fetch_fn=MagicMock(),
            series="land_registry_hpi",
            source_url="https://example.com",
            publication_lag_months=2,
            required_columns=["date", "region", "average_price"],
            key_columns=["date", "region", "average_price"],
        )

        # Prior sidecar claims data up to 2025-12
        prior_meta = {
            "series": "land_registry_hpi",
            "observation_period_end": "2025-12",
            "data_pulled_at": "2026-01-01T00:00:00+00:00",
            "source_url": "https://example.com",
            "publication_lag_months_estimated": 2,
        }

        import logging
        log = logging.getLogger("test")

        with patch(
            "src.ingestion.run_ingestion.read_release_metadata",
            return_value=prior_meta,
        ):
            # New data only goes up to 2024-12 — a regression
            df = pd.DataFrame({
                "date": pd.date_range("2005-01-01", "2024-12-01", freq="MS"),
                "region": "London",
                "average_price": 200_000,
            })

            with pytest.raises(ValueError, match="Coverage regression"):
                _check_coverage_regression(df, source, tmp_path, log)

    def test_no_error_when_coverage_advances(self, tmp_path):
        source = IngestionSource(
            name="land_registry_hpi",
            display_name="Land Registry HPI",
            fetch_fn=MagicMock(),
            series="land_registry_hpi",
            source_url="https://example.com",
            publication_lag_months=2,
            required_columns=["date", "region", "average_price"],
            key_columns=["date", "region", "average_price"],
        )

        prior_meta = {
            "series": "land_registry_hpi",
            "observation_period_end": "2025-10",
        }

        import logging
        log = logging.getLogger("test")

        with patch(
            "src.ingestion.run_ingestion.read_release_metadata",
            return_value=prior_meta,
        ):
            # New data extends to 2025-12 — no regression
            df = pd.DataFrame({
                "date": pd.date_range("2005-01-01", "2025-12-01", freq="MS"),
                "region": "London",
                "average_price": 200_000,
            })
            # Should not raise
            _check_coverage_regression(df, source, tmp_path, log)

    def test_no_error_when_no_prior_sidecar(self, tmp_path):
        source = IngestionSource(
            name="land_registry_hpi",
            display_name="Land Registry HPI",
            fetch_fn=MagicMock(),
            series="land_registry_hpi",
            source_url="https://example.com",
            publication_lag_months=2,
            required_columns=["date", "region", "average_price"],
            key_columns=["date", "region", "average_price"],
        )

        import logging
        log = logging.getLogger("test")

        with patch(
            "src.ingestion.run_ingestion.read_release_metadata",
            return_value=None,
        ):
            df = _make_df()
            # Should not raise — no prior sidecar to compare against
            _check_coverage_regression(df, source, tmp_path, log)
