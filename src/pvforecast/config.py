"""Project paths, constants, and run configuration."""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
# Figures from the notebooks
FIGURES_DIR = EVALUATION_DIR / "eda"

CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "config.yaml"

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
HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")


def resolve_config(path: Path | str) -> Path:
    """Accept a path or the bare name of a file in configs/."""
    path = Path(path)
    if not path.is_file() and not path.suffix:
        path = CONFIG_DIR / f"{path.name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config nicht gefunden: {path}")
    return path.resolve()


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    """Read a run configuration; `extends` fills in the keys it does not override."""
    path = resolve_config(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = cfg.pop("extends", None)
    if parent is not None:
        cfg = {**load_config(CONFIG_DIR / parent), **cfg}

    logger.info(f"Config: {path.name}")
    return cfg


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Log to the console, and to a file once the run has a directory."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
