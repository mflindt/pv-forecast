"""Day-ahead PV forecast metrics."""

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Astronomical, so the mask leaks nothing into the metric.
DAYLIGHT_ELEVATION_DEG = 5.0

# MWh per hour equals MW.
NORMALISERS = {"cap_roll": "cap_roll_mwh", "cap_ac": "cap_ac_mw"}

# Shared long-format output for all experiments and tables
PREDICTION_COLUMNS = (
    "time",
    "fold",
    "model",
    "featureset",
    "information_set",
    "context_rows",
    "seed",
    "y_true_mwh",
    "y_pred_mwh",
    "cap_roll_mwh",
    "cap_ac_mw",
    "sun_elevation",
    "kt",
    "fit_seconds",
    "predict_seconds",
)

# What makes one forecast series distinct; a context sweep varies the last key.
SERIES_KEYS = ("model", "featureset", "context_rows")

SCORES = ["nmae", "nrmse", "nmbe", "mae", "rmse", "mbe"]

# Cloud and yield regimes for error tables.
KT_EDGES = (0.0, 0.35, 0.75, np.inf)
KT_LABELS = ("bedeckt", "teilbewölkt", "klar")
CF_QUANTILE_LABELS = ("Q1", "Q2", "Q3", "Q4", "Q5")

STRATA = ("month", "kt_bin", "hour", "cf_quantile")

# Below this, HAC variance is not worth reporting.
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


def merge_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine the prediction frames of several runs, e.g. a local and a GPU run."""
    if not frames:
        raise ValueError("Keine Prognose-Frames zum Zusammenführen")

    merged = pd.concat(
        [check_predictions(frame) for frame in frames], ignore_index=True
    )

    # Differing truth on the same hour means the runs saw different data.
    truth = merged.groupby(["time", "fold"], observed=True)["y_true_mwh"].nunique()
    if (truth > 1).any():
        raise ValueError(
            f"{int((truth > 1).sum())} Stunden mit abweichender Zielgröße: "
            "die Läufe stammen aus verschiedenen Datenständen"
        )

    keys = [*SERIES_KEYS, "seed"]
    duplicates = int(merged.duplicated([*keys, "fold", "time"]).sum())
    if duplicates:
        raise ValueError(
            f"{duplicates} doppelte Prognosezeilen: ein Modell wurde in mehreren "
            "Läufen gerechnet (Referenzen im zweiten Lauf abschalten)"
        )

    # Unequal coverage would compare models over different test sets.
    covered = merged.groupby(keys, observed=True)["time"].count()
    if covered.nunique() > 1:
        raise ValueError(
            f"Prognosen decken unterschiedlich viele Stunden ab "
            f"({covered.min()} bis {covered.max()}): die Läufe haben nicht "
            "dieselben Folds gerechnet"
        )

    logger.info(f"{len(frames)} Läufe zu {len(merged)} Prognosezeilen zusammengeführt")
    return merged


def runtime(predictions: pd.DataFrame) -> pd.DataFrame:
    """Seconds per fit and per prediction; comparable only within one machine."""
    check_predictions(predictions)

    # One block per fit, so the two costs repeat over all rows of that block.
    per_fit = predictions.groupby([*SERIES_KEYS, "fold", "seed"], observed=True)[
        ["fit_seconds", "predict_seconds"]
    ].first()

    out = (
        per_fit.groupby(list(SERIES_KEYS), observed=True)
        .agg(
            n_fits=("fit_seconds", "size"),
            fit_seconds=("fit_seconds", "mean"),
            fit_seconds_total=("fit_seconds", "sum"),
            predict_seconds=("predict_seconds", "mean"),
        )
        .round(4)
        .reset_index()
        .sort_values("fit_seconds_total", ascending=False)
    )

    logger.info(
        f"Laufzeit: {out['fit_seconds_total'].sum():.1f} s für "
        f"{int(out['n_fits'].sum())} Fits"
    )
    return out.reset_index(drop=True)


def daylight_mask(sun_elevation: pd.Series) -> pd.Series:
    """Hours the sun is high enough for the forecast task to be non-trivial."""
    return sun_elevation > DAYLIGHT_ELEVATION_DEG


def kt_bins(kt: pd.Series) -> pd.Series:
    """Bin the clear-sky index into overcast / partly cloudy / clear."""
    return pd.cut(kt, bins=list(KT_EDGES), labels=list(KT_LABELS), right=False)


def point_metrics(y_true, y_pred, capacity) -> dict[str, float | int]:
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
    groupby: tuple[str, ...] = (*SERIES_KEYS, "fold"),
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
    return out


def aggregate_folds(
    metrics: pd.DataFrame, groupby: tuple[str, ...] = SERIES_KEYS
) -> pd.DataFrame:
    """Mean and spread across folds and seeds."""
    if "fold" not in metrics.columns:
        raise ValueError("aggregate_folds braucht eine 'fold'-Spalte")

    keys = list(groupby)
    by_fold = metrics.groupby(keys + ["fold"], observed=True)[SCORES]
    grouped = by_fold.mean().groupby(keys, observed=True)

    agg = grouped.agg(["mean", "std"])
    agg.columns = [f"{score}_{stat}" for score, stat in agg.columns]
    agg["n_folds"] = grouped.size()

    if "seed" in metrics.columns:
        # NaN where a fold ran once -- a deterministic fit has no seed spread.
        agg = agg.join(
            by_fold.std().groupby(keys, observed=True).mean().add_suffix("_sd_seed")
        )
    return agg.reset_index()


def add_skill(
    metrics: pd.DataFrame,
    reference: str,
    score: str = "nmae",
    by: tuple[str, ...] = ("fold",),
) -> pd.DataFrame:
    """Skill score relative to the reference model."""
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
    by: str,
    normaliser: str = "cap_roll",
    daylight_only: bool = True,
) -> pd.DataFrame:
    """Metrics per model and stratum: month, cloud regime, hour or yield quintile."""
    check_predictions(predictions)
    df = predictions.copy()
    time = pd.to_datetime(df["time"], utc=True)

    if by == "month":
        df["stratum"] = time.dt.month
    elif by == "hour":
        df["stratum"] = time.dt.hour
    elif by == "kt_bin":
        df["stratum"] = kt_bins(df["kt"])
    elif by == "cf_quantile":
        # Binned on evaluated rows only; night zeros would distort the bins.
        rows = daylight_mask(df["sun_elevation"]) if daylight_only else slice(None)
        cf = df.loc[rows, "y_true_mwh"] / df.loc[rows, NORMALISERS[normaliser]]
        # Ranks rather than qcut: the edges stay distinct even under long ties.
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
        groupby=(*SERIES_KEYS, "stratum"),
        normaliser=normaliser,
        daylight_only=daylight_only,
    )


def stratify_all(
    predictions: pd.DataFrame, normaliser: str = "cap_roll", daylight_only: bool = True
) -> pd.DataFrame:
    """All four stratifications in one long frame with a `stratum_type` column."""
    frames = []
    for key in STRATA:
        frame = stratify(predictions, key, normaliser, daylight_only)
        frame.insert(0, "stratum_type", key)
        # Mixed types across the four keys; one string column keeps the CSV honest.
        frame["stratum"] = frame["stratum"].astype(str)
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    logger.info(f"{len(out)} Metrikzeilen über {len(STRATA)} Schichtungen")
    return out


def _hac_variance(x: np.ndarray, lag: int) -> float:
    """Newey-West long-run variance with a Bartlett kernel."""
    variance = float(np.var(x, ddof=0))
    for step in range(1, lag + 1):
        gamma = float(np.cov(x[step:], x[:-step], ddof=0)[0, 1])
        variance += 2.0 * (1.0 - step / (lag + 1)) * gamma
    # Truncation can push the estimate negative on short, weakly dependent samples.
    return max(variance, 0.0)


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


def daily_loss(
    predictions: pd.DataFrame, normaliser: str = "cap_roll", daylight_only: bool = True
) -> pd.DataFrame:
    """Daily mean normalised loss per model."""
    check_predictions(predictions)

    df = predictions
    if daylight_only:
        df = df[daylight_mask(df["sun_elevation"])]

    loss = (df["y_pred_mwh"] - df["y_true_mwh"]).abs() / df[NORMALISERS[normaliser]]
    day = pd.to_datetime(df["time"], utc=True).dt.floor("D")
    return (
        df.assign(loss=loss, day=day)
        .groupby([*SERIES_KEYS, "day"], observed=True)["loss"]
        .mean()
        .unstack(list(SERIES_KEYS))
    )


def significance_test(
    predictions: pd.DataFrame,
    reference: str,
    normaliser: str = "cap_roll",
    daylight_only: bool = True,
) -> pd.DataFrame:
    """Giacomini-White test of each model against the reference forecast."""
    daily = daily_loss(predictions, normaliser, daylight_only)

    columns = [key for key in daily.columns if key[0] == reference]
    if len(columns) != 1:
        raise ValueError(f"Referenz {reference!r} ist nicht eindeutig ({len(columns)})")
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
                **dict(zip(SERIES_KEYS, key, strict=True)),
                "reference": reference,
                "n_days": n,
                "mean_loss_diff": mean,
                "hac_se": standard_error,
                "hac_lag": lag,
                "t": t,
                "p_value": math.erfc(abs(t) / math.sqrt(2.0)),
            }
        )

    out = pd.DataFrame(rows).sort_values("mean_loss_diff").reset_index(drop=True)
    out["p_holm"] = _holm(out["p_value"].to_numpy())

    logger.info(
        f"Giacomini-White gegen {reference}: {len(out)} Vergleiche über "
        f"{int(out['n_days'].max())} Tage, {int((out['p_holm'] < 0.05).sum())} "
        f"davon signifikant nach Holm (5 %)"
    )
    return out


def score(cfg: dict, predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Metrics per fold, pooled, per stratum and the significance test."""
    settings = {
        "normaliser": cfg["evaluation"]["normaliser"],
        "daylight_only": cfg["evaluation"]["daylight_only"],
    }
    reference = cfg["evaluation"]["skill_reference"]

    per_fold = evaluate(
        predictions,
        groupby=(*SERIES_KEYS, "information_set", "fold", "seed"),
        **settings,
    )
    pooled = evaluate(
        predictions, groupby=(*SERIES_KEYS, "information_set"), **settings
    )

    if reference in set(predictions["model"]):
        per_fold = add_skill(per_fold, reference)
        pooled = add_skill(pooled, reference)
    else:
        logger.warning(f"Skill-Referenz {reference!r} fehlt im Lauf, kein Skill Score")

    spread = aggregate_folds(per_fold, groupby=SERIES_KEYS)
    pooled = pooled.merge(spread, on=list(SERIES_KEYS), how="left")

    results = {
        "metrics_fold": per_fold,
        "metrics_agg": pooled.sort_values("nmae").reset_index(drop=True),
        "strata": stratify_all(predictions, **settings),
        "runtime": runtime(predictions),
    }
    if cfg["evaluation"].get("significance") and reference in set(predictions["model"]):
        results["tests"] = significance_test(predictions, reference, **settings)
    return results


def _markdown_table(df: pd.DataFrame, decimals: int = 4) -> str:
    """Render a frame as a Markdown table without pulling in a formatting library."""

    def cell(value) -> str:
        if isinstance(value, float):
            return "" if pd.isna(value) else f"{value:.{decimals}f}"
        return str(value)

    header = list(df.columns)
    rows = [[cell(value) for value in row] for row in df.itertuples(index=False)]
    widths = [
        max(len(str(header[i])), *(len(row[i]) for row in rows))
        if rows
        else len(header[i])
        for i in range(len(header))
    ]

    def line(values: list[str]) -> str:
        return (
            "| "
            + " | ".join(v.ljust(w) for v, w in zip(values, widths, strict=True))
            + " |"
        )

    return "\n".join(
        [
            line([str(h) for h in header]),
            "|" + "|".join("-" * (w + 2) for w in widths) + "|",
        ]
        + [line(row) for row in rows]
    )


def summary_table(cfg: dict, pooled: pd.DataFrame, run_id: str = "") -> str:
    """The headline table of the run as Markdown, ready to paste into the thesis."""
    columns = {
        "model": "Modell",
        "featureset": "Features",
        "context_rows": "Kontext",
        "information_set": "Informationsstand",
        "nmae": "nMAE",
        "nmae_std": "sd Fold",
        "nmae_sd_seed": "sd Seed",
        "nrmse": "nRMSE",
        "nmbe": "nMBE",
        "mae": "MAE (MWh)",
        "skill": "Skill",
    }
    present = {key: label for key, label in columns.items() if key in pooled.columns}
    table = pooled.sort_values("nmae")[list(present)].rename(columns=present)

    evaluation = cfg["evaluation"]
    hours = "Tagstunden" if evaluation["daylight_only"] else "alle 24 Stunden"
    folds = int(pooled["n_folds"].max())

    return "\n".join(
        [
            "# Ergebnisse" + (f" {run_id}" if run_id else ""),
            "",
            f"{folds} {'Fold' if folds == 1 else 'Folds'}, ausgewertet über {hours}, "
            f"Normierung `{evaluation['normaliser']}`.",
            f"Skill Score gegen `{evaluation['skill_reference']}`.",
            "`perfect_prog` (ERA5) und `operational` (ÜNB) sind nicht vergleichbar.",
            "",
            _markdown_table(table),
            "",
        ]
    )
