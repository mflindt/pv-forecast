# pv-forecast

## Forschungsfrage

## Daten

## Setup

Voraussetzung: [uv](https://docs.astral.sh/uv/)

```bash
git clone <url> && cd pv-forecast-de
uv sync          # erzeugt .venv exakt aus uv.lock
```

## Nutzung

## Projektstruktur

```text
pv-forecast/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── data/
│   ├── raw/           # Originaldaten (unverändert)
│   ├── interim/       # Zwischenergebnisse
│   └── processed/     # Aufbereitete Daten
├── notebooks/         # Explorative Analysen
└── src/               # Quellcode (wird schrittweise erweitert)
```


## Lizenz
MIT – siehe [LICENSE](LICENSE).