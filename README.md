# pv-forecast

Day-ahead-Prognose der deutschen PV-Einspeisung.

Dieses Repository enthält den Code zu meinem Forschungsbeleg. Die Arbeit untersucht,
wie sich die Prognosegüte eines Tabular Foundation Model gegenüber hyperparameter-
optimiertem Gradient Boosting mit zunehmender Kontextgröße verändert.

Der Datensatz umfasst die stündliche PV-Einspeisung Deutschlands von 2016 bis 2024
aus SMARD, dazu stündliche Wetterdaten aus der ERA5-Reanalyse über fünf Gitterpunkte
und die monatliche installierte Leistung von Energy-Charts. Bewertet wird gegen vier
naive Referenzen und gegen die auf SMARD veröffentlichte Day-ahead-Prognose der
Übertragungsnetzbetreiber.

---

## Inhalt

- [Projektstruktur](#projektstruktur)
- [Quick Start](#quick-start)
- [Dokumentation](#dokumentation)
- [Tests](#tests)
- [Lizenz](#lizenz)

---

## Projektstruktur

```
pv-forecast/
├── main.py                 # Einstiegspunkt, ruft die vier Schritte der Reihe nach auf
├── Makefile
├── configs/                # Einstellungen je Lauf
├── data/
│   ├── raw/                # Originaldaten, unverändert
│   ├── interim/            # bereinigte Stundenreihe
│   └── processed/          # Modell-Input aus PV und Wetter
├── docs/                   # Forschungsfrage, Gliederung, Zeitplan
├── notebooks/              # explorative Analysen, nicht Teil der Pipeline
├── reports/
│   ├── eda/                # Abbildungen aus den Notebooks
│   ├── latest -> ...       # Verweis auf den jüngsten Lauf
│   └── <zeitstempel>/      # ein Ordner je Lauf, Tabellen und Abbildungen
├── src/pvforecast/         # sieben Module, siehe Dokumentation
└── tests/
```

`data/interim/` und `data/processed/` liegen nicht im Repository. Beide entstehen mit
`make data` aus `data/raw/`.

---

## Quick Start

### Voraussetzungen

Das Projekt nutzt [uv](https://docs.astral.sh/uv/) für die Abhängigkeiten.

```bash
git clone <url> && cd pv-forecast
uv sync
```

### Einen Lauf ausführen

```bash
make run
```

Das rechnet Preprocessing, Training, Evaluation und Visualization am Stück und legt
alle Tabellen und Abbildungen in `reports/<zeitstempel>/` ab.

Für einen schnellen Probelauf genügt ein einzelner Fold:

```bash
uv run python main.py --folds 1
```

### Daten neu beschaffen

Die Rohdaten liegen im Repository, dieser Schritt ist also nur nötig, wenn der
Zeitraum erweitert werden soll. Er braucht eine Internetverbindung.

```bash
make data
```

---

## Dokumentation

### Die vier Schritte der Pipeline

Der Ablauf steht vollständig in `main.py`. Jeder Schritt ist ein Modul.

| Schritt | Datei | Inhalt |
|---|---|---|
| Preprocessing | [`src/pvforecast/preprocessing.py`](src/pvforecast/preprocessing.py) | Bereinigung der Viertelstundenreihe, Join mit dem Wetter, Zielgröße, Features, Feature-Stufen S1 bis S3 |
| Training | [`src/pvforecast/training.py`](src/pvforecast/training.py) | Rolling-Origin-Folds, innere Validierung, Random Search, Fold-Schleife |
| Evaluation | [`src/pvforecast/evaluation.py`](src/pvforecast/evaluation.py) | Metriken, Skill Score, Fehler nach Schichten, Signifikanztest |
| Visualization | [`src/pvforecast/visualization.py`](src/pvforecast/visualization.py) | die Abbildungen eines Laufs |

### Weitere Module

| Datei | Inhalt |
|---|---|
| [`src/pvforecast/config.py`](src/pvforecast/config.py) | alle Pfade und festen Konstanten des Projekts, Laden der Config |
| [`src/pvforecast/models.py`](src/pvforecast/models.py) | alle Prognoseverfahren: Referenzen R0 bis R3, Ridge, LightGBM und TabPFN-3 |
| [`src/pvforecast/ingest.py`](src/pvforecast/ingest.py) | Download von SMARD, Open-Meteo und Energy-Charts nach `data/raw/` |

### Modelle und Referenzen

| Name | Beschreibung |
|---|---|
| `R0_climatology` | mittlerer Kapazitätsfaktor je Monat und Stunde |
| `R1_persistence` | Kapazitätsfaktor derselben Stunde 48 Stunden zuvor |
| `R2_clearsky_persistence` | Clear-Sky-Persistenz, kalibriert über die Trainingsdaten |
| `R3_combined` | konvexe Kombination aus R0 und R1, Referenz des Skill Scores |
| `R4_tso_dayahead` | veröffentlichte Day-ahead-Prognose der Übertragungsnetzbetreiber |
| `ridge` | Ridge-Regression auf standardisierten Features |
| `lightgbm` | Gradient Boosting mit Early Stopping auf der inneren Validierung |
| `tabpfn3` | Tabular Foundation Model, ungetunt; braucht das `gpu`-Extra |

R1 greift auf 48 statt 24 Stunden zurück, weil ein Rückgriff auf den Vortag den
Prognosezeitpunkt von 10:00 UTC überschreiten würde.

`ridge` ist deterministisch und wird deshalb nur mit dem ersten Seed gerechnet; `seeds`
wirkt auf die übrigen lernenden Modelle.

### Notebooks

Die Notebooks dokumentieren die explorative Arbeit an den Daten und sind nicht Teil
der Pipeline. Ihre Abbildungen liegen in `reports/eda/`.

| Notebook | Inhalt |
|---|---|
| `01_ingestion.ipynb` | erster Zugriff auf die SMARD-API |
| `02_cleaning_resampling.ipynb` | Aggregation auf Stunden, Abgleich mit der SMARD-Stundenreihe |
| `03_eda.ipynb` | univariate Exploration der Zielgröße |
| `04_weather_ingestion.ipynb` | Aufbau und Prüfung des Open-Meteo-Abrufs |
| `05_kapazitaet_drift.ipynb` | Nennleistung gegen rollierende empirische Kapazität |
| `06_colab_tabpfn.ipynb` | TabPFN-3 auf einer Colab-GPU, schreibt `predictions.parquet` |

---

## Lizenz

MIT, siehe [LICENSE](LICENSE).
