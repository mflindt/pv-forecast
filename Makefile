.PHONY: all data forecast dataset weather capacity join references

# One log file per `make` run; multi-step runs (e.g. `make all`) write into sections
export PVFORECAST_LOG_FILE := logs/run_$(shell date -u +%Y-%m-%d_%H%M%S).log

# Full reproduction: SMARD + Open-Meteo down to the joined model input.
all: data forecast dataset weather capacity join

# Pull the raw quarter-hour + hour series from SMARD into data/raw.
data:
	uv run python scripts/fetch_smard.py

# Pull the TSO day-ahead PV forecast from SMARD into data/raw (reference R4).
forecast:
	uv run python scripts/fetch_smard_forecast.py

# Build the clean hourly series from the cached raw data into data/interim.
dataset:
	uv run python scripts/build_dataset.py

# Pull the raw ERA5 hourly weather series from Open-Meteo into data/raw.
weather:
	uv run python scripts/fetch_weather.py

# Pull the monthly installed PV capacity from Energy-Charts into data/raw.
capacity:
	uv run python scripts/fetch_capacity.py

# Join the clean PV series with the weather series into data/processed.
join:
	uv run python scripts/build_model_input.py

# Run the reference forecasts over the rolling-origin folds into reports/results.
references:
	uv run python scripts/run_references.py
