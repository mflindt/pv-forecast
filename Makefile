.PHONY: data run test lint

# Load raw data
data:
	uv run python -m pvforecast.ingest
	uv run python -m pvforecast.preprocessing

# Full run
run:
	uv run python main.py

test:
	uv run pytest -q

lint:
	uv run ruff format --check . && uv run ruff check .
