"""A complete run from start to finish."""

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimentlauf aus einer Config")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Pfad zur Config"
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


def write_results(
    out: Path,
    cfg: dict,
    predictions: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    hyperparams: list[dict],
    spans: pd.DataFrame,
    commit: str,
) -> None:
    """Write every table the thesis reads, plus what a rerun would need."""
    predictions.to_parquet(out / "predictions.parquet", index=False)
    spans.to_csv(out / "folds.csv", index=False)
    for name, frame in results.items():
        frame.to_csv(out / f"{name}.csv", index=False)

    (out / "hyperparams.json").write_text(
        json.dumps(hyperparams, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "summary.md").write_text(
        evaluation.summary_table(cfg, results["metrics_agg"], out.name),
        encoding="utf-8",
    )

    resolved = {
        **cfg,
        "run_id": out.name,
        "git_commit": commit,
        "created_utc": f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}",
        "search_spaces": {name: models.spec(name).space for name in cfg["models"]},
    }
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    logger.info(f"Artefakte geschrieben: {out}")


def run_pipeline(cfg: dict) -> Path:
    """Preprocessing -> Training -> Evaluation -> Visualization."""
    validate(cfg)
    commit = git_commit()
    if commit.endswith("-dirty"):
        logger.warning(
            f"Uncommittete Änderungen: Lauf aus {commit} nicht reproduzierbar"
        )

    # 1 - Preprocessing
    X, y, meta, tso = preprocessing.build_dataset(cfg)

    # 2 - Training
    predictions, hyperparams, spans = training.run_folds(cfg, X, y, meta, tso)

    # 3 - Evaluation
    results = evaluation.score(cfg, predictions)

    # 4 - Visualization
    out = make_run_dir(cfg)
    write_results(out, cfg, predictions, results, hyperparams, spans, commit)
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
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
