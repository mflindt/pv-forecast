.PHONY: all data dataset weather join

# One log file per `make` run; multi-step runs (e.g. `make all`) write into sections
export PVFORECAST_LOG_FILE := logs/run_$(shell date -u +%Y-%m-%d_%H%M%S).log

# Full reproduction: SMARD + Open-Meteo down to the joined model input.
all: data dataset weather join

# Pull the raw quarter-hour + hour series from SMARD into data/raw.
data:
	uv run python scripts/fetch_smard.py

# Build the clean hourly series from the cached raw data into data/interim.
dataset:
	uv run python scripts/build_dataset.py

# Pull the raw ERA5 hourly weather series from Open-Meteo into data/raw.
weather:
	uv run python scripts/fetch_weather.py

# Join the clean PV series with the weather series into data/processed.
join:
	uv run python scripts/build_model_input.py
