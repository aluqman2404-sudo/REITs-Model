"""Shared logging configuration for research pipelines."""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
from pathlib import Path

from src.core.paths import ensure_directory

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
)


def configure_monitoring(app_name: str = "uk-housing-model") -> logging.Logger:
    """Configure the root logger with the destination set by LOG_DESTINATION.

    Supported LOG_DESTINATION values: stdout (default), cloudwatch, file, datadog.
    stdout handler is always kept regardless of LOG_DESTINATION.
    Optional cloud destinations fall back to stdout if the required package is
    not installed.
    """
    import warnings as _warnings

    destination = os.environ.get("LOG_DESTINATION", "stdout").lower()
    level = getattr(logging, _LOG_LEVEL, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Always ensure at least one stdout handler
    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(
        h, logging.FileHandler) for h in root.handlers)
    if not has_stream:
        _sh = logging.StreamHandler()
        _sh.setFormatter(_formatter)
        root.addHandler(_sh)

    if destination == "cloudwatch":
        try:
            import boto3  # noqa: F401
            from watchtower import CloudWatchLogHandler

            log_group = os.environ.get(
                "CLOUDWATCH_LOG_GROUP", "/uk-housing-model/app")
            _cwh = CloudWatchLogHandler(log_group=log_group)
            _cwh.setFormatter(_formatter)
            root.addHandler(_cwh)
            root.info(
                "configure_monitoring: CloudWatch handler added (log_group=%s)", log_group)
        except ImportError:
            _warnings.warn(
                "configure_monitoring: LOG_DESTINATION=cloudwatch but watchtower/boto3 is not installed. "
                "Falling back to stdout. Install with: pip install watchtower boto3",
                UserWarning,
                stacklevel=2,
            )

    elif destination == "file":
        _log_dir = Path("logs")
        _log_dir.mkdir(parents=True, exist_ok=True)
        _log_path = _log_dir / "app.log"
        _rfh = logging.handlers.RotatingFileHandler(
            _log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        _rfh.setFormatter(_formatter)
        root.addHandler(_rfh)
        root.info(
            "configure_monitoring: rotating file handler added (path=%s)", _log_path)

    elif destination == "datadog":
        try:
            # type: ignore[import]
            from datadog_logger import DatadogLogHandler

            _ddh = DatadogLogHandler()
            _ddh.setFormatter(_formatter)
            root.addHandler(_ddh)
            root.info("configure_monitoring: Datadog handler added")
        except ImportError:
            _warnings.warn(
                "configure_monitoring: LOG_DESTINATION=datadog but datadog_logger is not installed. "
                "Falling back to stdout. Install with: pip install datadog-lambda",
                UserWarning,
                stacklevel=2,
            )

    return root


def log_model_event(logger: logging.Logger, event_type: str, details: dict) -> None:
    """Emit a structured JSON governance event at INFO level."""
    payload = {"event": event_type, **details}
    logger.info(json.dumps(payload, default=str))


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Return a module logger with a consistent formatter."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        ensure_directory(log_file.parent)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
