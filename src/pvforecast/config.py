"""Project paths, constants, and run configuration."""

import logging
from datetime import UTC, datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
# Figures from the notebooks
FIGURES_DIR = REPORTS_DIR / "eda"

CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "config.yaml"
LOG_DIR = PROJECT_ROOT / "logs"

# Raw files written by ingest.py
PV_RAW = RAW_DIR / "smard_pv_realized_quarterhour_2015-2026.csv"
PV_RAW_HOURLY = RAW_DIR / "smard_pv_realized_hour_2015-2026.csv"
FORECAST_RAW = RAW_DIR / "smard_pv_forecast_dayahead_quarterhour_2015-2026.csv"
WEATHER_RAW = RAW_DIR / "weather_openmeteo_era5_2015-2026.csv"
CAPACITY_RAW = RAW_DIR / "capacity_energycharts_solar_2002-2026.csv"

# Derived tables written by preprocessing.py
PV_HOURLY = INTERIM_DIR / "pv_hourly.parquet"
MODEL_INPUT = PROCESSED_DIR / "pv_weather_hourly.parquet"

# Five ERA5 cells across Germany
SITES = (
    ("nord", 53.0, 9.5, 46.0),
    ("ost", 52.3, 13.0, 79.0),
    ("west", 51.0, 7.5, 186.0),
    ("suedwest", 48.6, 9.0, 570.0),
    ("suedost", 48.8, 11.8, 363.0),
)

RADIATION_VARS = ["shortwave_radiation", "direct_radiation", "diffuse_radiation"]
INSTANT_VARS = [
    "temperature_2m",
    "cloud_cover",
    "relative_humidity_2m",
    "wind_speed_10m",
]
HOURLY_VARS = RADIATION_VARS + INSTANT_VARS

# GHI = direct + diffuse, so only two can be used together
REDUNDANT_VAR = "direct_radiation"
MODEL_VARS = [name for name in HOURLY_VARS if name != REDUNDANT_VAR]

# # 2015–2025; year boundaries are in UTC
PERIOD_START = "2014-12-31 23:00"
PERIOD_END = "2025-12-31 23:00"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    """Read a run configuration."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config nicht gefunden: {path}")

    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    logger.info(f"Config: {path.name}")
    return cfg


def setup_logging(name: str = "pvforecast", level: int = logging.INFO) -> Path:
    """Log to the console and to one timestamped file per run under logs/."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{name}_{datetime.now(UTC):%Y-%m-%d_%H%M%S}.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return log_file
