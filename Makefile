.PHONY: all data dataset weather

# One log file per `make` run; multi-step runs (e.g. `make all`) write into sections
export PVFORECAST_LOG_FILE := logs/run_$(shell date -u +%Y-%m-%d_%H%M%S).log

# Full reproduction from the SMARD API to the processed parquet.
all: data dataset

# Pull the raw quarter-hour + hour series from SMARD into data/raw.
data:
	uv run python scripts/fetch_smard.py

# Build the clean hourly series from the cached raw data into data/processed.
dataset:
	uv run python scripts/build_dataset.py

# Pull the raw ERA5 hourly weather series from Open-Meteo into data/raw.
# Not wired into `all` yet: no step consumes the weather data so far.
weather:
	uv run python scripts/fetch_weather.py
