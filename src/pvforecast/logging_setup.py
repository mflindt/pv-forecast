"""
Logging setup for entry-point scripts.
"""

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from pvforecast.config import LOG_DIR

# Reduce noise from third-party loggers in DEBUG output
_NOISY_LOGGERS = ("urllib3", "requests")

# Env var set by Makefile to aggregate logs from multi-step runs into one file
_LOG_FILE_ENV = "PVFORECAST_LOG_FILE"


def setup_logging(name: str) -> Path:
    """Configure root logging with one log file per run.

    Uses PVFORECAST_LOG_FILE if set (append mode), otherwise creates a timestamped file.
    Each run writes a header for easier log separation.
    """
    override = os.environ.get(_LOG_FILE_ENV)
    if override:
        log_file = Path(override)
        mode = "a"
    else:
        log_file = LOG_DIR / f"{name}_{datetime.now(timezone.utc):%Y-%m-%d_%H%M%S}.log"
        mode = "w"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fmt.converter = time.gmtime

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file = logging.FileHandler(log_file, mode=mode, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file)

    # Keep own messages readable in the DEBUG file.
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("--- %s ---", name.upper())
    logger.debug("Run '%s' gestartet (Logdatei: %s)", name, log_file)

    return log_file
