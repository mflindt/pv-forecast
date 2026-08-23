"""Tests for the rolling-origin backtesting splits in pvforecast.splits."""

import pandas as pd
import pytest

from pvforecast import splits


def _index(
    start: str = "2016-01-01", end: str = "2025-12-31 23:00"
) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", tz="UTC")


def test_gap_between_train_and_test():
    folds = splits.rolling_origin_days(_index(), gap_hours=48)

    assert len(folds) == 12
    for train, test in folds:
        assert train.max() + pd.Timedelta(hours=48) <= test.min()


def test_test_folds_are_complete_utc_days():
    folds = splits.rolling_origin_days(_index(), test_days=90)

    for _, test in folds:
        assert len(test) == 90 * 24
        assert test.min().time() == pd.Timestamp("00:00").time()
        assert test.max().time() == pd.Timestamp("23:00").time()
        # Every day carries all 24 hours.
        assert (test.normalize().value_counts() == 24).all()


def test_default_folds_cover_three_seasonal_cycles():
    """Two years of test data would leave the fold spread resting on two winters."""
    folds = splits.rolling_origin_days(_index())
    covered = pd.DatetimeIndex([]).append([test for _, test in folds])

    assert covered.max() - covered.min() > pd.Timedelta(days=1000)
    # The smallest training window still holds several years.
    assert len(folds[0][0]) / 24 / 365 > 5


def test_holdout_year_is_never_touched():
    folds = splits.rolling_origin_days(_index())

    for train, test in folds:
        assert train.max() < splits.HOLDOUT_START
        assert test.max() < splits.HOLDOUT_START
    # The most recent fold reaches the hold-out edge, so nothing is wasted.
    assert folds[-1][1].max() == pd.Timestamp("2024-12-31 23:00", tz="UTC")


def test_test_blocks_are_disjoint_and_ordered():
    folds = splits.rolling_origin_days(_index())

    for (_, earlier), (_, later) in zip(folds[:-1], folds[1:], strict=True):
        assert earlier.max() < later.min()


def test_expanding_grows_while_sliding_stays_constant():
    expanding = splits.rolling_origin_days(_index(), mode="expanding")
    sliding = splits.rolling_origin_days(_index(), mode="sliding")

    exp_len = [len(train) for train, _ in expanding]
    slid_len = [len(train) for train, _ in sliding]

    assert exp_len == sorted(exp_len) and exp_len[0] < exp_len[-1]
    assert len(set(slid_len)) == 1
    # Both start from the same earliest fold.
    assert expanding[0][0].equals(sliding[0][0])


def test_complete_days_raises_on_partial_day():
    idx = _index("2016-01-01", "2016-03-01 12:00")

    with pytest.raises(ValueError, match="unvollständige UTC-Tage"):
        splits.complete_days(idx)


def test_raises_on_unknown_mode():
    with pytest.raises(ValueError, match="Unbekannter mode"):
        splits.rolling_origin_days(_index(), mode="random")


def test_raises_when_history_is_too_short():
    idx = _index("2024-06-01", "2024-12-31 23:00")

    with pytest.raises(ValueError, match="reichen nicht"):
        splits.rolling_origin_days(idx, n_folds=8, test_days=90)


def test_raises_when_index_lies_entirely_in_the_holdout():
    idx = _index("2025-01-01", "2025-12-31 23:00")

    with pytest.raises(ValueError, match="Hold-out"):
        splits.rolling_origin_days(idx)
