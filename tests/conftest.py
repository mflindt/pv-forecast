"""Shared synthetic test data and fixtures."""

import numpy as np
import pandas as pd
import pytest

from pvforecast import preprocessing
from pvforecast.config import INSTANT_VARS


def make_joined_frame(
    start: str = "2020-01-01", end: str = "2022-12-31 23:00"
) -> pd.DataFrame:
    """A PV + weather table with the shape build_features expects."""
    index = pd.date_range(start, end, freq="h", tz="UTC", name="time")
    rng = np.random.default_rng(0)

    cs_ghi = preprocessing.solar_geometry(index)["cs_ghi"].to_numpy()
    kt = np.clip(0.8 + 0.2 * rng.standard_normal(len(index)), 0.05, 1.2)
    ghi = cs_ghi * kt
    # A varying diffuse share; a fixed one would make the columns collinear.
    diffuse = ghi * rng.uniform(0.2, 0.9, len(index))

    frame = pd.DataFrame(
        {
            # Like the real SMARD series, the feed-in never drops to exactly zero.
            "pv_mwh": ghi * 40.0 + rng.normal(0, 50, len(index)).clip(0) + 0.2,
            "shortwave_radiation": ghi,
            "direct_radiation": ghi - diffuse,
            "diffuse_radiation": diffuse,
        },
        index=index,
    )
    for name in INSTANT_VARS:
        frame[name] = rng.uniform(0, 50, len(index))
    return frame


@pytest.fixture(scope="session")
def joined_frame() -> pd.DataFrame:
    return make_joined_frame()


@pytest.fixture(scope="session")
def dataset(joined_frame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Feature matrix, target and meta, built the way build_dataset builds them."""
    X, y, meta = preprocessing.build_features(joined_frame)
    X = pd.concat([X, preprocessing.lag_features(X["kt"], lags=(48,))], axis=1)

    # Drop the capacity warm-up so the matrix is complete, as in the real pipeline.
    start = y.first_valid_index() + pd.Timedelta(hours=max(preprocessing.LAG_HOURS))
    start = start.normalize() + pd.Timedelta(days=1)
    X, y, meta = X.loc[start:], y.loc[start:], meta.loc[start:]

    meta = meta.copy()
    meta["cap_ac_mw"] = meta["cap_roll_mwh"] * 1.1
    return X, y, meta


@pytest.fixture
def prediction_frame() -> pd.DataFrame:
    """A minimal frame that satisfies the long-format contract."""
    index = pd.date_range("2024-01-01", periods=24 * 40, freq="h", tz="UTC")
    elevation = np.tile(np.linspace(-40, 55, 24), 40)
    rng = np.random.default_rng(1)

    blocks = []
    for name, error in (("R3_combined", 200.0), ("lightgbm", 60.0)):
        truth = np.clip(elevation, 0, None) * 100
        blocks.append(
            pd.DataFrame(
                {
                    "time": index,
                    "fold": np.repeat([1, 2], 24 * 20),
                    "model": name,
                    "featureset": "-" if name.startswith("R") else "S3",
                    "information_set": "history_only"
                    if name.startswith("R")
                    else "perfect_prog",
                    "context_rows": 0,
                    "seed": 0 if name.startswith("R") else 42,
                    "y_true_mwh": truth,
                    "y_pred_mwh": truth + rng.normal(0, error, len(index)),
                    "cap_roll_mwh": 6000.0,
                    "cap_ac_mw": 7000.0,
                    "sun_elevation": elevation,
                    "kt": np.clip(rng.uniform(0, 1.2, len(index)), 0, None),
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


@pytest.fixture
def small_config() -> dict:
    return {
        "seed": 42,
        "seeds": [42],
        "splits": {"n_folds": 1, "test_days": 40, "gap_hours": 48, "mode": "expanding"},
        "tuning": {"validation_days": 40, "gap_hours": 48},
        "evaluation": {
            "normaliser": "cap_roll",
            "daylight_only": True,
            "skill_reference": "R3_combined",
            "significance": True,
        },
        "models": ["ridge"],
        "featuresets": ["S3"],
        "references": ["R0_climatology", "R1_persistence", "R3_combined"],
        "include_tso": False,
        "output_dir": "reports",
    }
