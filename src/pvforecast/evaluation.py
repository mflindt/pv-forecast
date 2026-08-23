"""
Forecast metrics for the day-ahead PV evaluation.

Headline numbers are daylight only and normalised per timestamp; both choices are
argued in docs/arbeitsplan.md.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Astronomical, so the mask leaks nothing into the metric.
DAYLIGHT_ELEVATION_DEG = 5.0

# Both divisors are per hour, so MWh per hour and MW are the same number.
NORMALISERS = {"cap_roll": "cap_roll_mwh", "cap_ac": "cap_ac_mw"}

# Long-format contract every experiment writes and every table reads.
PREDICTION_COLUMNS = (
    "time",
    "split",
    "fold",
    "model",
    "featureset",
    "seed",
    "y_true_mwh",
    "y_pred_mwh",
    "cap_roll_mwh",
    "cap_ac_mw",
    "sun_elevation",
    "kt",
)

# Cloud regimes for the stratified error tables.
KT_EDGES = (0.0, 0.35, 0.75, np.inf)
KT_LABELS = ("bedeckt", "teilbewölkt", "klar")


def check_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a prediction frame against the long-format contract."""
    missing = [col for col in PREDICTION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Spalten fehlen im Prognose-Frame: {missing}")

    holes = df[list(PREDICTION_COLUMNS)].isna().sum()
    if holes.any():
        raise ValueError(f"NaN-Werte im Prognose-Frame:\n{holes[holes > 0]}")

    for name in NORMALISERS.values():
        if (df[name] <= 0).any():
            raise ValueError(f"Normierung {name} enthält Werte <= 0")

    return df


def daylight_mask(sun_elevation: pd.Series) -> pd.Series:
    """Hours the sun is high enough for the forecast task to be non-trivial."""
    return sun_elevation > DAYLIGHT_ELEVATION_DEG


def kt_bins(kt: pd.Series) -> pd.Series:
    """Bin the clear-sky index into overcast / partly cloudy / clear."""
    return pd.cut(kt, bins=list(KT_EDGES), labels=list(KT_LABELS), right=False)


def point_metrics(
    y_true: pd.Series, y_pred: pd.Series, capacity: pd.Series
) -> dict[str, float | int]:
    """Absolute and capacity-normalised error metrics for one forecast series."""
    if not (len(y_true) == len(y_pred) == len(capacity)):
        raise ValueError("y_true, y_pred und capacity müssen gleich lang sein")
    if len(y_true) == 0:
        raise ValueError("Leere Auswertungsmenge")

    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    relative = error / np.asarray(capacity, dtype=float)

    return {
        "n": len(error),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "mbe": float(error.mean()),
        "nmae": float(np.abs(relative).mean()),
        "nrmse": float(np.sqrt((relative**2).mean())),
        "nmbe": float(relative.mean()),
    }


def evaluate(
    predictions: pd.DataFrame,
    groupby: tuple[str, ...] = ("model", "featureset", "split", "fold"),
    normaliser: str = "cap_roll",
    daylight_only: bool = True,
) -> pd.DataFrame:
    """Metrics per group from a long-format prediction frame."""
    if normaliser not in NORMALISERS:
        raise ValueError(
            f"Unbekannte Normierung: {normaliser!r} (bekannt: {list(NORMALISERS)})"
        )
    check_predictions(predictions)

    df = predictions
    if daylight_only:
        df = df[daylight_mask(df["sun_elevation"])]
        if df.empty:
            raise ValueError("Keine Tagstunden in der Auswertungsmenge")

    capacity_col = NORMALISERS[normaliser]
    rows = []
    for key, group in df.groupby(list(groupby), observed=True, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groupby, keys, strict=True))
        row.update(
            point_metrics(group["y_true_mwh"], group["y_pred_mwh"], group[capacity_col])
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    # Columns, not attrs: a CSV has to say what it normalised by.
    out["normaliser"] = normaliser
    out["daylight_only"] = daylight_only

    logger.info(
        f"{len(out)} Metrikzeilen ({', '.join(groupby)}), Normierung {normaliser}, "
        f"{'nur Tagstunden' if daylight_only else 'alle 24 Stunden'}"
    )
    return out


def aggregate_folds(
    metrics: pd.DataFrame, groupby: tuple[str, ...] = ("model", "featureset")
) -> pd.DataFrame:
    """Mean and spread of the fold metrics -- the spread is the uncertainty measure."""
    if "fold" not in metrics.columns:
        raise ValueError("aggregate_folds braucht eine 'fold'-Spalte")

    scores = ["nmae", "nrmse", "nmbe", "mae", "rmse", "mbe"]
    grouped = metrics.groupby(list(groupby), observed=True)

    agg = grouped[scores].agg(["mean", "std"])
    agg.columns = [f"{score}_{stat}" for score, stat in agg.columns]

    # Distinct folds, not rows: several seeds per fold must not inflate the count.
    return agg.join(grouped["fold"].nunique().rename("n_folds")).reset_index()


def add_skill(
    metrics: pd.DataFrame,
    reference: str,
    score: str = "nmae",
    by: tuple[str, ...] = ("split", "fold"),
) -> pd.DataFrame:
    """Skill score SS = 1 - score_model / score_reference.

    Matched per fold when the metrics carry fold columns, against the single
    reference row when they do not. The pooled value is the headline number.
    """
    if reference not in set(metrics["model"]):
        raise ValueError(f"Referenzmodell {reference!r} fehlt in den Metriken")

    ref = metrics[metrics["model"] == reference]
    keys = [col for col in by if col in metrics.columns]
    out = metrics.copy()

    if not keys:
        if len(ref) != 1:
            raise ValueError(
                f"Referenz {reference!r} ist nicht eindeutig ({len(ref)} Zeilen)"
            )
        out["ref"] = float(ref[score].iloc[0])
    else:
        if ref.duplicated(keys).any():
            raise ValueError(f"Referenz {reference!r} ist je {keys} nicht eindeutig")
        out = out.join(ref.set_index(keys)[score].rename("ref"), on=keys)
        if out["ref"].isna().any():
            raise ValueError(f"Referenz {reference!r} deckt nicht alle {keys} ab")

    out["skill"] = 1.0 - out[score] / out["ref"]
    return out.drop(columns="ref")


def stratify(
    predictions: pd.DataFrame,
    by: str = "month",
    normaliser: str = "cap_roll",
    daylight_only: bool = True,
) -> pd.DataFrame:
    """Metrics per model and stratum: month, cloud regime or hour of day."""
    check_predictions(predictions)
    df = predictions.copy()

    if by == "month":
        df["stratum"] = pd.to_datetime(df["time"], utc=True).dt.month
    elif by == "kt_bin":
        df["stratum"] = kt_bins(df["kt"])
    elif by == "hour":
        df["stratum"] = pd.to_datetime(df["time"], utc=True).dt.hour
    else:
        raise ValueError(f"Unbekannte Schichtung: {by!r} (month, kt_bin, hour)")

    return evaluate(
        df,
        groupby=("model", "featureset", "stratum"),
        normaliser=normaliser,
        daylight_only=daylight_only,
    )
