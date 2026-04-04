"""Build app-ready Stage 7 cache files from validated Stage 1-6 outputs."""

from __future__ import annotations

from src.core.logging_utils import get_logger
from src.ui.loaders import load_dashboard_data, write_dashboard_cache


logger = get_logger("stage7.cache")


def main() -> None:
    data = load_dashboard_data()
    written = write_dashboard_cache(data)
    for label, path in written.items():
        logger.info("Wrote %s -> %s", label, path)


if __name__ == "__main__":
    main()
