"""
Forecast metrics for the day-ahead PV evaluation.

Headline numbers are daylight only and normalised per timestamp; both choices are
argued in docs/arbeitsplan.md.
"""

import logging
import math

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
    "information_set",
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

# Yield regimes: the aggregate nMBE averages a positive bias at low feed-in against a
# strong negative one at high feed-in, so it has to be reported conditionally too.
CF_QUANTILE_LABELS = ("Q1", "Q2", "Q3", "Q4", "Q5")

STRATA = ("month", "kt_bin", "hour", "cf_quantile")

# Below this a HAC variance on daily differentials is not worth reporting.
MIN_TEST_DAYS = 30


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
    """Mean and spread over the folds, plus the separate spread over the repeats.

    The repeats of a stochastic model are collapsed to one value per fold first.
    Taking the spread over the raw rows instead would mix two sources of variation,
    deflate the estimate and claim three times the degrees of freedom it has.

    `<score>_std` is the spread over the folds. The folds are seasonally confounded
    by construction, so it is a seasonal range, not a standard error of the mean.
    `<score>_sd_seed` is the seed spread within a fold, averaged over folds -- that
    is the model uncertainty table T4 asks for.
    """
    if "fold" not in metrics.columns:
        raise ValueError("aggregate_folds braucht eine 'fold'-Spalte")

    scores = ["nmae", "nrmse", "nmbe", "mae", "rmse", "mbe"]
    keys = list(groupby)

    by_fold = metrics.groupby(keys + ["fold"], observed=True)[scores]
    grouped = by_fold.mean().groupby(keys, observed=True)

    agg = grouped.agg(["mean", "std"])
    agg.columns = [f"{score}_{stat}" for score, stat in agg.columns]
    agg["n_folds"] = grouped.size()

    if "seed" in metrics.columns:
        # NaN where a fold ran once -- a deterministic fit has no seed spread.
        seed_sd = by_fold.std().groupby(keys, observed=True).mean()
        agg = agg.join(seed_sd.add_suffix("_sd_seed"))

    return agg.reset_index()


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
    elif by == "cf_quantile":
        # Binned on the evaluated rows only: over all 24 hours the lower quintiles
        # would collapse onto the night zeros. Rows outside carry a NaN stratum and
        # drop out of the groupby.
        rows = daylight_mask(df["sun_elevation"]) if daylight_only else slice(None)
        cf = df.loc[rows, "y_true_mwh"] / df.loc[rows, NORMALISERS[normaliser]]
        # Ranks rather than qcut: the edges stay distinct even when the target has
        # long ties, where qcut would raise on duplicate bin edges.
        edges = np.linspace(0.0, 1.0, len(CF_QUANTILE_LABELS) + 1)
        df["stratum"] = pd.cut(
            cf.rank(pct=True),
            bins=edges,
            labels=CF_QUANTILE_LABELS,
            include_lowest=True,
        )
    else:
        raise ValueError(f"Unbekannte Schichtung: {by!r} ({', '.join(STRATA)})")

    return evaluate(
        df,
        groupby=("model", "featureset", "stratum"),
        normaliser=normaliser,
        daylight_only=daylight_only,
    )


def _hac_variance(x: np.ndarray, lag: int) -> float:
    """Newey-West long-run variance with a Bartlett kernel."""
    variance = float(np.var(x, ddof=0))
    for step in range(1, lag + 1):
        gamma = float(np.cov(x[step:], x[:-step], ddof=0)[0, 1])
        variance += 2.0 * (1.0 - step / (lag + 1)) * gamma
    # Truncation can push the estimate negative on short, weakly dependent samples.
    return max(variance, 0.0)


def _normal_two_sided(t: float) -> float:
    """Two-sided p-value of a standard normal statistic."""
    return math.erfc(abs(t) / math.sqrt(2.0))


def _daily_loss(
    predictions: pd.DataFrame, normaliser: str, daylight_only: bool
) -> pd.DataFrame:
    """Mean normalised absolute loss per model and UTC day, averaged over repeats."""
    df = predictions
    if daylight_only:
        df = df[daylight_mask(df["sun_elevation"])]
    loss = (df["y_pred_mwh"] - df["y_true_mwh"]).abs() / df[NORMALISERS[normaliser]]
    day = pd.to_datetime(df["time"], utc=True).dt.floor("D")
    return (
        df.assign(loss=loss, day=day)
        .groupby(["model", "featureset", "day"], observed=True)["loss"]
        .mean()
        .unstack(["model", "featureset"])
    )


def loss_differential_test(
    predictions: pd.DataFrame,
    reference: str,
    normaliser: str = "cap_roll",
    daylight_only: bool = True,
) -> pd.DataFrame:
    """Giacomini-White test of every model against one reference forecast.

    The loss differential is aggregated to whole UTC days first: hourly errors are
    strongly autocorrelated within a day, and a test on hourly rows would treat
    ~11 correlated hours as independent evidence. The residual day-to-day dependence
    is absorbed by a Newey-West HAC variance with the usual automatic bandwidth.

    Giacomini & White (2006), Econometrica 74:1545 rather than Diebold-Mariano:
    the models are re-estimated on every fold, which is exactly the rolling-scheme
    case DM does not cover. The statistic is the same, the justification is not.

    Holm-corrected across the comparisons, so the family-wise error rate holds.
    """
    check_predictions(predictions)
    daily = _daily_loss(predictions, normaliser, daylight_only)

    columns = [key for key in daily.columns if key[0] == reference]
    if len(columns) != 1:
        raise ValueError(
            f"Referenz {reference!r} ist nicht eindeutig ({len(columns)} Spalten)"
        )
    base = daily[columns[0]]

    rows = []
    for key in daily.columns:
        if key == columns[0]:
            continue
        diff = (daily[key] - base).dropna().to_numpy()
        n = len(diff)
        if n < MIN_TEST_DAYS:
            raise ValueError(f"{n} gemeinsame Tage reichen nicht für {key}")

        lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        standard_error = math.sqrt(_hac_variance(diff, lag) / n)
        mean = float(diff.mean())

        if standard_error > 0:
            t = mean / standard_error
        else:
            # Two forecasts with an identical daily loss are no evidence either way.
            t = 0.0 if mean == 0.0 else math.inf

        rows.append(
            {
                "model": key[0],
                "featureset": key[1],
                "reference": reference,
                "n_days": n,
                "mean_loss_diff": mean,
                "hac_se": standard_error,
                "hac_lag": lag,
                "t": t,
                "p_value": _normal_two_sided(t),
            }
        )

    out = pd.DataFrame(rows).sort_values("mean_loss_diff").reset_index(drop=True)
    out["p_holm"] = _holm(out["p_value"].to_numpy())
    out["normaliser"] = normaliser
    out["daylight_only"] = daylight_only

    logger.info(
        f"Giacomini-White gegen {reference}: {len(out)} Vergleiche über "
        f"{int(out['n_days'].max())} Tage, {int((out['p_holm'] < 0.05).sum())} "
        f"davon signifikant nach Holm (5 %)"
    )
    return out


def _holm(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down correction; monotone and never above 1."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted
