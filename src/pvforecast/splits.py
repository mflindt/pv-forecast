"""Rolling-origin backtesting splits on whole UTC days.

Twelve folds of 90 days cover three full seasonal cycles and still leave six years
in the smallest training window.
"""

import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)

# 2025 is the untouched hold-out; no rolling-origin fold may see it.
HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")

HOURS_PER_DAY = 24

Fold = tuple[pd.DatetimeIndex, pd.DatetimeIndex]


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


def _hours_of(index: pd.DatetimeIndex, days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """All timestamps of the index that fall on one of the given days."""
    return index[index.normalize().isin(days)]


def rolling_origin_days(
    index: pd.DatetimeIndex,
    n_folds: int = 12,
    test_days: int = 90,
    gap_hours: int = 48,
    mode: str = "expanding",
) -> list[Fold]:
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

    folds: list[Fold] = []
    for i in range(n_folds):
        test_start = first_test_pos + i * test_days
        train_end = test_start - gap_days
        train_start = 0 if mode == "expanding" else train_end - train_len

        train_days = days[train_start:train_end]
        test_block = days[test_start : test_start + test_days]
        folds.append((_hours_of(usable, train_days), _hours_of(usable, test_block)))

        logger.debug(
            f"Fold {i + 1}/{n_folds}: Train {train_days[0]:%Y-%m-%d}..."
            f"{train_days[-1]:%Y-%m-%d} ({len(train_days)} Tage) | "
            f"Test {test_block[0]:%Y-%m-%d}...{test_block[-1]:%Y-%m-%d}"
        )

    logger.info(
        f"{n_folds} Rolling-Origin-Folds ({mode}), Test je {test_days} Tage, "
        f"Gap {gap_days} Tage, Hold-out ab {HOLDOUT_START:%Y-%m-%d} ausgeschlossen"
    )
    return folds
