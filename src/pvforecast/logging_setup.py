"""
Logging setup for entry-point scripts.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from pvforecast.config import LOG_DIR

# Third-party loggers that would otherwise flood the DEBUG file.
_NOISY_LOGGERS = ("urllib3", "requests")


def setup_logging(name: str) -> Path:
    """Configure root logging"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{name}_{datetime.now(timezone.utc):%Y-%m-%d_%H%M%S}.log"

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fmt.converter = time.gmtime

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file = logging.FileHandler(log_file, encoding="utf-8")
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

    logging.getLogger(__name__).debug(
        "Run '%s' gestartet (Logdatei: %s)", name, log_file
    )

    return log_file
