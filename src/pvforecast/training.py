"""Rolling-origin folds and model evaluation."""

import itertools
import logging
import math
import random
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from pvforecast import models, preprocessing
from pvforecast.config import HOLDOUT_START
from pvforecast.evaluation import daylight_mask
from pvforecast.preprocessing import to_energy

logger = logging.getLogger(__name__)

HOURS_PER_DAY = 24

# References read named columns instead of a feature stage, and carry no seed.
REFERENCE_FEATURESET = "-"
NO_SEED = 0

# Whole training window
FULL_CONTEXT = 0

# What each forecast knows about the target hour.
HISTORY_ONLY = "history_only"
PERFECT_PROG = "perfect_prog"
OPERATIONAL = "operational"


def set_seed(seed: int) -> None:
    """Set the global random seed."""
    if seed < 0:
        raise ValueError(f"Seed muss >= 0 sein, ist {seed}")
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Seed gesetzt: {seed}")


def timed(call: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Run a call and return its result together with the seconds it took."""
    start = time.perf_counter()
    result = call(*args, **kwargs)
    return result, time.perf_counter() - start


def complete_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Unique UTC days of the index; a partially covered day is an error."""
    counts = index.normalize().value_counts().sort_index()
    partial = counts[counts != HOURS_PER_DAY]
    if not partial.empty:
        raise ValueError(
            f"{len(partial)} unvollständige UTC-Tage im Index, "
            f"z. B. {partial.index[0]:%Y-%m-%d} mit {partial.iloc[0]} Stunden"
        )
    return pd.DatetimeIndex(counts.index)


def hours_of(index: pd.DatetimeIndex, days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """All timestamps of the index that fall on one of the given days."""
    return index[index.normalize().isin(days)]


def rolling_origin_days(
    index: pd.DatetimeIndex,
    n_folds: int = 12,
    test_days: int = 90,
    gap_hours: int = 48,
    mode: str = "expanding",
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Build rolling-origin folds of whole UTC days, oldest fold first."""
    if mode not in ("expanding", "sliding"):
        raise ValueError(f"Unbekannter mode: {mode!r}")
    if n_folds < 1 or test_days < 1:
        raise ValueError("n_folds und test_days müssen >= 1 sein")
    if gap_hours < 0:
        raise ValueError("gap_hours darf nicht negativ sein")

    usable = index[index < HOLDOUT_START]
    if usable.empty:
        raise ValueError(f"Kein Index-Anteil vor dem Hold-out {HOLDOUT_START:%Y-%m-%d}")

    days = complete_days(usable)
    gap_days = math.ceil(gap_hours / HOURS_PER_DAY)

    first_test_pos = len(days) - n_folds * test_days
    if first_test_pos - gap_days < 1:
        raise ValueError(
            f"{len(days)} Tage reichen nicht für {n_folds} Folds à {test_days} Tage "
            f"plus {gap_days} Tage Gap"
        )

    train_len = first_test_pos - gap_days
    folds = []
    for i in range(n_folds):
        test_start = first_test_pos + i * test_days
        train_end = test_start - gap_days
        train_start = 0 if mode == "expanding" else train_end - train_len

        train_days = days[train_start:train_end]
        test_block = days[test_start : test_start + test_days]
        folds.append((hours_of(usable, train_days), hours_of(usable, test_block)))

    logger.info(
        f"{n_folds} Rolling-Origin-Folds ({mode}), Test je {test_days} Tage, "
        f"Gap {gap_days} Tage, Hold-out ab {HOLDOUT_START:%Y-%m-%d} ausgeschlossen"
    )
    return folds


def holdout_fold(
    index: pd.DatetimeIndex, gap_hours: int = 48
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """The confirmation fold: fit on everything before the hold-out, test on it."""
    if gap_hours < 0:
        raise ValueError("gap_hours darf nicht negativ sein")

    test = index[index >= HOLDOUT_START]
    if test.empty:
        raise ValueError(f"Kein Index-Anteil ab dem Hold-out {HOLDOUT_START:%Y-%m-%d}")

    history = index[index < HOLDOUT_START]
    if history.empty:
        raise ValueError(f"Kein Index-Anteil vor dem Hold-out {HOLDOUT_START:%Y-%m-%d}")

    days = complete_days(history)
    gap_days = math.ceil(gap_hours / HOURS_PER_DAY)
    if len(days) - gap_days < 1:
        raise ValueError(
            f"{len(days)} Tage Historie reichen nicht für {gap_days} Tage Gap"
        )

    train = hours_of(history, days[: len(days) - gap_days])
    return train, hours_of(test, complete_days(test))


def build_folds(
    index: pd.DatetimeIndex, holdout: bool = False, **splits
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """The fold layout of a run: the rolling origin, or the single hold-out fold."""
    if not holdout:
        return rolling_origin_days(index, **splits)

    ignored = sorted(key for key in splits if key != "gap_hours")
    if ignored:
        logger.warning(f"Hold-out-Modus: {', '.join(ignored)} bleibt ohne Wirkung")

    gap = {"gap_hours": splits["gap_hours"]} if "gap_hours" in splits else {}
    train, test = holdout_fold(index, **gap)
    logger.warning(
        f"Hold-out-Modus: Test ist {HOLDOUT_START:%Y} "
        f"({test.min():%Y-%m-%d} bis {test.max():%Y-%m-%d}), "
        f"Training bis {train.max():%Y-%m-%d} — dieser Lauf ist einmalig vorgesehen"
    )
    return [(train, test)]


def inner_split(
    train: pd.DatetimeIndex, validation_days: int = 90, gap_hours: int = 48
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split training data for tuning."""
    if validation_days < 1:
        raise ValueError("validation_days muss >= 1 sein")
    if gap_hours < 0:
        raise ValueError("gap_hours darf nicht negativ sein")

    days = complete_days(train)
    gap_days = math.ceil(gap_hours / HOURS_PER_DAY)
    core_len = len(days) - validation_days - gap_days
    if core_len < 1:
        raise ValueError(
            f"{len(days)} Trainingstage reichen nicht für {validation_days} Tage "
            f"innere Validierung plus {gap_days} Tage Gap"
        )

    core = hours_of(train, days[:core_len])
    validation = hours_of(train, days[-validation_days:])
    return core, validation


def limit_context(
    index: pd.DatetimeIndex,
    elevation: pd.Series,
    context_rows: int | None = None,
    daylight_only: bool = False,
) -> pd.DatetimeIndex:
    """Restrict a fit window to daylight hours and to its most recent rows."""
    if daylight_only:
        index = index[daylight_mask(elevation.loc[index]).to_numpy()]
    if context_rows is not None:
        if context_rows < 1:
            raise ValueError(f"context_rows muss >= 1 sein, ist {context_rows}")
        # Silently taking fewer rows would label a smaller fit with the larger size.
        if len(index) < context_rows:
            raise ValueError(
                f"Kontext {context_rows} verlangt mehr als die {len(index)} Zeilen "
                "des Fensters"
            )
        index = index[-context_rows:]
    if index.empty:
        raise ValueError("Kontextfenster ist nach der Einschränkung leer")
    return index


def sample_configs(space: dict, budget: int, rng: np.random.Generator) -> list[dict]:
    """Draw distinct configurations; a budget beyond the grid enumerates it fully."""
    if budget < 1:
        raise ValueError(f"Budget muss >= 1 sein, ist {budget}")
    if not space:
        return [{}]

    keys = list(space)
    grid_size = math.prod(len(values) for values in space.values())
    if budget >= grid_size:
        return [
            dict(zip(keys, values, strict=True))
            for values in itertools.product(*space.values())
        ]

    seen: set[tuple[int, ...]] = set()
    configs = []
    while len(configs) < budget:
        drawn = tuple(int(rng.integers(len(space[key]))) for key in keys)
        if drawn in seen:
            continue
        seen.add(drawn)
        configs.append({key: space[key][i] for key, i in zip(keys, drawn, strict=True)})
    return configs


def random_search(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    core: pd.DatetimeIndex,
    validation: pd.DatetimeIndex,
    seed: int,
) -> tuple[dict, float | None, int]:
    """Random hyperparameter search on the inner validation block."""
    if not core.intersection(validation).empty:
        raise ValueError("Fit- und Validierungsblock überschneiden sich")

    model_spec = models.spec(name)
    # An untuned model skips the inner fit; searching an empty space costs a pass.
    if not model_spec.space:
        logger.info(f"{name}: kein Suchraum, ungetunt")
        return {}, None, 0

    configs = sample_configs(
        model_spec.space, model_spec.budget, np.random.default_rng(seed)
    )
    mask = daylight_mask(X.loc[validation, "sun_elevation"]).to_numpy()

    X_core, y_core = X.loc[core], y.loc[core]
    X_val, y_val = X.loc[validation], y.loc[validation]
    if not mask.any():
        raise ValueError("Keine Tagstunden in der inneren Validierung")

    best_params: dict = {}
    best_score = math.inf
    for params in configs:
        estimator = models.build(name, params, seed)
        if getattr(estimator, "uses_validation", False):
            estimator.fit(X_core, y_core, validation=(X_val, y_val))
        else:
            estimator.fit(X_core, y_core)

        error = estimator.predict(X_val).to_numpy()[mask] - y_val.to_numpy()[mask]
        # In cf space the MAE on daylight rows is exactly nMAE under cap_roll.
        score = float(np.abs(error).mean())
        if not math.isfinite(score):
            raise ValueError(f"Innere Validierung ohne endlichen Score bei {params}")

        if score < best_score:
            frozen = estimator.freeze() if hasattr(estimator, "freeze") else {}
            best_params, best_score = {**params, **frozen}, score

    logger.info(
        f"{name}: {len(configs)} Konfigurationen, beste innere nMAE {best_score:.5f} "
        f"bei {best_params}"
    )
    return best_params, best_score, len(configs)


def seeds_for(name: str, seeds: list[int]) -> list[int]:
    """Repeats of one model; a deterministic fit gives the same forecast every time."""
    if not seeds:
        raise ValueError("seeds darf nicht leer sein")
    return seeds[:1] if models.spec(name).deterministic else seeds


def information_set(featureset: str) -> str:
    """Whether a feature stage carries target-hour weather, and is thus perfect prog."""
    if featureset == REFERENCE_FEATURESET:
        return HISTORY_ONLY
    weather = set(preprocessing.WEATHER) & set(preprocessing.STAGES[featureset])
    return PERFECT_PROG if weather else HISTORY_ONLY


def _block(
    index: pd.DatetimeIndex,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    pred_mwh: pd.Series,
    model: str,
    featureset: str,
    seed: int,
    fold: int,
    context_rows: int,
    fit_seconds: float,
    predict_seconds: float,
    info_set: str | None = None,
) -> pd.DataFrame:
    """One block of the long-format prediction frame."""
    return pd.DataFrame(
        {
            "time": index,
            "fold": fold,
            "model": model,
            "featureset": featureset,
            "information_set": info_set or information_set(featureset),
            "context_rows": context_rows,
            "seed": seed,
            "y_true_mwh": meta.loc[index, "pv_mwh"].to_numpy(),
            "y_pred_mwh": pred_mwh.to_numpy(),
            "cap_roll_mwh": meta.loc[index, "cap_roll_mwh"].to_numpy(),
            "cap_ac_mw": meta.loc[index, "cap_ac_mw"].to_numpy(),
            "sun_elevation": X.loc[index, "sun_elevation"].to_numpy(),
            "kt": X.loc[index, "kt"].to_numpy(),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        }
    )


def run_folds(
    cfg: dict,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    tso: pd.Series,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Fit all models on each fold and collect predictions."""
    set_seed(cfg["seed"])
    folds = build_folds(X.index, **cfg["splits"])
    if cfg.get("max_folds"):
        folds = folds[: cfg["max_folds"]]
        logger.warning(f"Probelauf: nur die ersten {len(folds)} Folds")

    # None means the whole training window; a list of sizes sweeps the context.
    contexts = cfg.get("contexts") or [None]
    daylight_training = cfg.get("daylight_training", False)
    elevation = X["sun_elevation"]
    if len(contexts) > 1:
        logger.info(f"Kontextsweep über {contexts}")

    blocks: list[pd.DataFrame] = []
    hyperparams: list[dict] = []
    spans: list[dict] = []

    for fold, (train, test) in enumerate(folds, start=1):
        logger.info(
            f"Fold {fold}/{len(folds)}: Test {test.min():%Y-%m-%d} "
            f"bis {test.max():%Y-%m-%d}"
        )
        spans.append(
            {
                "fold": fold,
                "train_start": train.min(),
                "train_end": train.max(),
                "test_start": test.min(),
                "test_end": test.max(),
                "train_days": len(train) // HOURS_PER_DAY,
                "test_days": len(test) // HOURS_PER_DAY,
            }
        )
        capacity = meta.loc[test, "cap_roll_mwh"]

        # References keep the full window: they are the yardstick of the sweep.
        for name in cfg["references"]:
            reference, fit_seconds = timed(
                models.build(name).fit, X.loc[train], y.loc[train]
            )
            cf, predict_seconds = timed(reference.predict, X.loc[test])
            blocks.append(
                _block(
                    test,
                    X,
                    meta,
                    to_energy(cf, capacity),
                    model=name,
                    featureset=REFERENCE_FEATURESET,
                    seed=NO_SEED,
                    fold=fold,
                    context_rows=FULL_CONTEXT,
                    fit_seconds=fit_seconds,
                    predict_seconds=predict_seconds,
                )
            )

        # R4 is observed, not fitted: it enters as MWh and needs no back-transform.
        if cfg["include_tso"]:
            if tso.empty:
                raise ValueError("include_tso gesetzt, aber keine ÜNB-Prognose geladen")
            blocks.append(
                _block(
                    test,
                    X,
                    meta,
                    tso.loc[test],
                    model=models.TSO_NAME,
                    featureset=REFERENCE_FEATURESET,
                    seed=NO_SEED,
                    fold=fold,
                    context_rows=FULL_CONTEXT,
                    # Published, not fitted here: there is no runtime to record.
                    fit_seconds=0.0,
                    predict_seconds=0.0,
                    info_set=OPERATIONAL,
                )
            )

        core, validation = inner_split(train, **cfg["tuning"])
        # The validation block is a fixed yardstick, so only the fit sets shrink.
        fit_validation = limit_context(validation, elevation, None, daylight_training)

        for stage in cfg["featuresets"]:
            X_stage = preprocessing.select(X, stage)

            for context in contexts:
                fit_core = limit_context(core, elevation, context, daylight_training)
                fit_train = limit_context(train, elevation, context, daylight_training)

                for name in cfg["models"]:
                    (params, inner_nmae, n_configs), search_seconds = timed(
                        random_search,
                        name,
                        X_stage,
                        y,
                        fit_core,
                        fit_validation,
                        cfg["seed"],
                    )
                    hyperparams.append(
                        {
                            "model": name,
                            "featureset": stage,
                            "fold": fold,
                            "context_rows": context or FULL_CONTEXT,
                            "fit_rows": len(fit_train),
                            "params": params,
                            "inner_nmae": inner_nmae,
                            "n_configs": n_configs,
                            # The price of the tuning budget, 0 for an untuned model.
                            "search_seconds": round(search_seconds, 3),
                        }
                    )

                    for seed in seeds_for(name, cfg["seeds"]):
                        estimator = models.build(name, params, seed)
                        _, fit_seconds = timed(
                            estimator.fit, X_stage.loc[fit_train], y.loc[fit_train]
                        )
                        cf, predict_seconds = timed(
                            estimator.predict, X_stage.loc[test]
                        )
                        blocks.append(
                            _block(
                                test,
                                X,
                                meta,
                                to_energy(cf, capacity),
                                model=name,
                                featureset=stage,
                                seed=seed,
                                fold=fold,
                                context_rows=context or FULL_CONTEXT,
                                fit_seconds=fit_seconds,
                                predict_seconds=predict_seconds,
                            )
                        )

    predictions = pd.concat(blocks, ignore_index=True)
    logger.info(
        f"{len(predictions)} Prognosezeilen für {predictions['model'].nunique()} "
        f"Modelle über {len(folds)} Folds"
    )
    return predictions, hyperparams, pd.DataFrame(spans)
