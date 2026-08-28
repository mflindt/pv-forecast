"""A complete run from start to finish."""

import argparse
import json
import logging
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
import yaml

from pvforecast import evaluation, models, preprocessing, training, visualization
from pvforecast.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config, setup_logging

logger = logging.getLogger(__name__)

# Keys a run configuration has to carry.
REQUIRED_KEYS = (
    "seed",
    "seeds",
    "splits",
    "tuning",
    "evaluation",
    "models",
    "featuresets",
    "references",
    "include_tso",
    "output_dir",
)

# Packages whose version changes a result; recorded with every run.
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "pvlib",
    "tabpfn",
)

# Date columns of folds.csv; unparsed they break the split figure.
FOLD_DATES = ["train_start", "train_end", "test_start", "test_end"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimentlauf aus einer Config")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Pfad zur Config"
    )
    parser.add_argument(
        "--stage",
        choices=("all", "train", "evaluate"),
        default="all",
        # Split so the fold loop can run on a GPU and the scoring at home.
        help="Nur trainieren, nur auswerten, oder beides",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Auszuwertende Laufordner unter reports/ (nur mit --stage evaluate)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=None,
        # Shortens the run without moving the fold boundaries: a smoke test.
        help="Probelauf: nur die ersten N Folds rechnen",
    )
    return parser.parse_args()


def validate(cfg: dict) -> dict:
    """Fail on an unusable config before the data load, not an hour into the run."""
    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"Config-Schlüssel fehlen: {missing}")

    for name in cfg["models"]:
        models.spec(name)
    for name in cfg["references"]:
        if name not in models.REFERENCES:
            raise ValueError(f"Unbekannte Referenz: {name!r}")
    for stage in cfg["featuresets"]:
        if stage not in preprocessing.STAGES:
            raise ValueError(f"Unbekannte Feature-Stufe: {stage!r}")

    if not cfg["models"] and not cfg["references"] and not cfg["include_tso"]:
        raise ValueError("Config rechnet nichts: keine Modelle, Referenzen oder ÜNB")
    if cfg["evaluation"]["normaliser"] not in evaluation.NORMALISERS:
        raise ValueError(f"Unbekannte Normierung: {cfg['evaluation']['normaliser']!r}")
    if not cfg["seeds"]:
        raise ValueError("seeds darf nicht leer sein")

    for size in cfg.get("contexts") or []:
        if size is not None and (not isinstance(size, int) or size < 1):
            raise ValueError(f"Ungültige Kontextgröße: {size!r}")
    return cfg


def git_commit() -> str:
    """The commit the run was made from, marked when the tree was dirty."""
    try:
        run = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return run.stdout.strip() + ("-dirty" if status.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unbekannt"


def make_run_dir(cfg: dict) -> Path:
    """Timestamped directory for this run; 'latest' is repointed to it."""
    parent = PROJECT_ROOT / cfg["output_dir"]
    out = parent / f"{datetime.now(UTC):%Y-%m-%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    latest = parent / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(out.name)
    return out


def environment() -> dict:
    """Versions and hardware of the run; a Colab result has to stay traceable."""
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
    try:
        import torch

        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return env


def write_config(out: Path, cfg: dict, commit: str, **extra) -> None:
    """The resolved configuration: everything a rerun would need."""
    resolved = {
        **cfg,
        "run_id": out.name,
        "git_commit": commit,
        "created_utc": f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}",
        "search_spaces": {name: models.spec(name).space for name in cfg["models"]},
        "environment": environment(),
        **extra,
    }
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def write_train(
    out: Path,
    predictions: pd.DataFrame,
    hyperparams: list[dict] | None,
    spans: pd.DataFrame,
) -> None:
    """What the fold loop produces; the only artefact that travels between machines."""
    predictions.to_parquet(out / "predictions.parquet", index=False)
    spans.to_csv(out / "folds.csv", index=False)
    # None after a merge: the tuning happened in the runs this one reads.
    if hyperparams is not None:
        (out / "hyperparams.json").write_text(
            json.dumps(hyperparams, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    logger.info(f"Prognosen geschrieben: {out}")


def read_folds(cfg: dict, run: str) -> pd.DataFrame:
    """The fold layout of a finished run."""
    path = PROJECT_ROOT / cfg["output_dir"] / run / "folds.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Keine Fold-Tabelle in {run}: {path}")
    return pd.read_csv(path, parse_dates=FOLD_DATES)


def write_evaluation(out: Path, cfg: dict, results: dict[str, pd.DataFrame]) -> None:
    """Every table the thesis reads."""
    for name, frame in results.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "summary.md").write_text(
        evaluation.summary_table(cfg, results["metrics_agg"], out.name),
        encoding="utf-8",
    )
    logger.info(f"Auswertung geschrieben: {out}")


def load_runs(cfg: dict, names: list[str]) -> pd.DataFrame:
    """Read the prediction frames of finished training runs and merge them."""
    parent = PROJECT_ROOT / cfg["output_dir"]
    frames = []
    for name in names:
        path = parent / name / "predictions.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Kein Prognose-Frame in {name}: {path}")
        frame = pd.read_parquet(path)
        logger.info(f"{len(frame)} Prognosezeilen geladen: {name}")
        frames.append(frame)
    return evaluation.merge_predictions(frames)


def run_pipeline(cfg: dict, stage: str = "all", runs: list[str] | None = None) -> Path:
    """Preprocessing -> Training -> Evaluation -> Visualization."""
    validate(cfg)
    commit = git_commit()
    if commit.endswith("-dirty"):
        logger.warning(
            f"Uncommittete Änderungen: Lauf aus {commit} nicht reproduzierbar"
        )

    sources: dict = {}
    hyperparams: list[dict] | None = None

    if stage == "evaluate":
        if not runs:
            raise ValueError("--stage evaluate braucht --runs")
        predictions = load_runs(cfg, runs)
        # The fold layout is identical across merged runs; keep the run self-contained.
        spans = read_folds(cfg, runs[0])
        sources = {"sources": list(runs)}
    else:
        X, y, meta, tso = preprocessing.build_dataset(cfg)
        predictions, hyperparams, spans = training.run_folds(cfg, X, y, meta, tso)

    # Only now, so a failed run leaves no empty directory behind.
    out = make_run_dir(cfg)
    write_train(out, predictions, hyperparams, spans)
    write_config(out, cfg, commit, **sources)
    if stage == "train":
        print(f"Prognosen: {out}")
        return out

    results = evaluation.score(cfg, predictions)
    write_evaluation(out, cfg, results)
    visualization.make_all(out, cfg, predictions, results, spans)

    print("\n" + evaluation.summary_table(cfg, results["metrics_agg"], out.name))
    print(f"Ergebnisse: {out}")
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.folds is not None:
        cfg["max_folds"] = args.folds

    setup_logging("run")
    run_pipeline(cfg, args.stage, args.runs)


if __name__ == "__main__":
    main()
