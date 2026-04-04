"""Shared helper — writes JSON release-provenance sidecar alongside each raw data file.

Each ingestion function calls write_release_metadata() after saving its CSV.
The sidecar is named:
    <series>_release_metadata.json
in the same directory as the raw data file.

Fields written:
    series                        — short identifier (e.g. "land_registry_hpi")
    observation_period_end        — "YYYY-MM" of the last observation in the file
    data_pulled_at                — ISO-8601 UTC timestamp of the download
    source_url                    — primary URL the data was fetched from
    publication_lag_months_estimated — typical months between reference-period end
                                      and the agency's publication date

The build_canonical_panel step reads these sidecars to compute:
    max_publication_lag_months    — worst-case lag across all series used
    effective_data_as_of          — panel_end minus that lag (look-ahead bias check)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def write_release_metadata(
    *,
    series: str,
    raw_file_path: Path,
    source_url: str,
    publication_lag_months_estimated: int,
    df: pd.DataFrame | None = None,
    observation_period_end: str | None = None,
) -> Path:
    """Write a JSON release-provenance sidecar alongside *raw_file_path*.

    Args:
        series: Short identifier for the data series (e.g. "land_registry_hpi").
        raw_file_path: Path to the CSV file just saved by the ingestion function.
        source_url: Primary URL the data was fetched from.
        publication_lag_months_estimated: Typical months from reference-period end
            to the agency's publication date.
        df: DataFrame with a ``date`` column.  Used to derive
            *observation_period_end* when the argument is not given directly.
        observation_period_end: ``"YYYY-MM"`` string.  Supply when *df* is not
            available or when the series uses a non-standard date column.

    Returns:
        Path to the sidecar file written.
    """
    if observation_period_end is None:
        if df is None:
            raise ValueError(
                "Either df or observation_period_end must be supplied "
                "to write_release_metadata()."
            )
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if dates.empty:
            raise ValueError(
                f"DataFrame passed for series '{series}' has no parseable dates."
            )
        observation_period_end = dates.max().strftime("%Y-%m")

    payload: dict = {
        "series": series,
        "observation_period_end": observation_period_end,
        "data_pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": source_url,
        "publication_lag_months_estimated": publication_lag_months_estimated,
    }

    sidecar_path = raw_file_path.parent / f"{series}_release_metadata.json"
    sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sidecar_path


def read_release_metadata(series: str, raw_data_dir: Path) -> dict | None:
    """Read the sidecar for *series* from *raw_data_dir*, or return None.

    Silently returns None rather than raising if the file is absent so that
    build_canonical_panel can degrade gracefully when a sidecar has not been
    written yet (e.g. data was ingested before this feature was added).
    """
    path = raw_data_dir / f"{series}_release_metadata.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
