"""
Reference forecasts for the day-ahead PV evaluation.

Every reference follows the sklearn fit/predict interface, predicts the capacity
factor and reads the columns it needs out of the feature matrix. R4, the published
TSO forecast, is observed rather than fitted and enters the prediction frame directly.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Capacity factors are non-negative by construction; clipping keeps that true.
CF_FLOOR = 0.0


class Climatology:
    """R0 -- mean capacity factor per month and hour, fitted on the training block."""

    name = "R0_climatology"

    def __init__(self):
        self.table_: pd.Series | None = None
        self.fallback_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Climatology":
        key = [y.index.month, y.index.hour]
        self.table_ = y.groupby(key, observed=True).mean()
        self.table_.index.names = ["month", "hour"]
        self.fallback_ = float(y.mean())

        logger.debug(f"R0: {len(self.table_)} Monat-Stunde-Zellen gefittet")
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.table_ is None:
            raise ValueError("Climatology wurde nicht gefittet")

        key = pd.MultiIndex.from_arrays([X.index.month, X.index.hour])
        # A month-hour cell unseen in training falls back to the overall mean.
        pred = self.table_.reindex(key).to_numpy()
        pred = np.where(np.isnan(pred), self.fallback_, pred)
        return pd.Series(pred, index=X.index, name=self.name).clip(CF_FLOOR)


class Persistence:
    """R1 -- the capacity factor of the same hour 48 h earlier.

    Stateless: the lag is already a column. Two days back, not one, because t-24h
    reaches past the gate.
    """

    name = "R1_persistence"

    def __init__(self, column: str = "cf_lag48h"):
        self.column = column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Persistence":
        self._require(X)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self._require(X)
        return X[self.column].clip(CF_FLOOR).rename(self.name)

    def _require(self, X: pd.DataFrame) -> None:
        if self.column not in X.columns:
            raise ValueError(f"R1 braucht die Spalte {self.column!r}")


class ClearSkyPersistence:
    """R2 -- smart persistence: cf(t) = beta * kt(t - 48h) * cs_ghi(t).

    beta is fitted by least squares through the origin. Even calibrated the
    reference is weak at a 48 h lag, so it is not the skill baseline.
    """

    name = "R2_clearsky_persistence"

    def __init__(self, kt_column: str = "kt_lag48h", cs_column: str = "cs_ghi"):
        self.kt_column = kt_column
        self.cs_column = cs_column
        self.beta_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClearSkyPersistence":
        raw = self._raw(X)
        denominator = float((raw**2).sum())
        if denominator <= 0:
            raise ValueError("R2: Trainingsfenster enthält keine Einstrahlung")

        self.beta_ = float((raw * y).sum() / denominator)
        logger.debug(f"R2: beta = {self.beta_:.3e} (1/(W/m2))")
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.beta_ is None:
            raise ValueError("ClearSkyPersistence wurde nicht gefittet")
        return (self.beta_ * self._raw(X)).clip(CF_FLOOR).rename(self.name)

    def _raw(self, X: pd.DataFrame) -> pd.Series:
        missing = [c for c in (self.kt_column, self.cs_column) if c not in X.columns]
        if missing:
            raise ValueError(f"R2 braucht fehlende Spalten: {missing}")
        return X[self.kt_column] * X[self.cs_column]


class CombinedReference:
    """R3 -- convex combination of climatology and persistence.

    The skill baseline: by construction at least as good as either component, so it
    cannot flatter a model the way a conveniently weak reference would.
    """

    name = "R3_combined"

    def __init__(self, column: str = "cf_lag48h"):
        self.column = column
        self.climatology_ = Climatology()
        self.weight_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CombinedReference":
        self.climatology_.fit(X, y)

        climate = self.climatology_.predict(X)
        persist = X[self.column]

        # Least squares in the one free parameter, kept inside [0, 1].
        spread = climate - persist
        denominator = float((spread**2).sum())
        weight = (
            0.5
            if denominator <= 0
            else float((spread * (y - persist)).sum() / denominator)
        )
        self.weight_ = float(np.clip(weight, 0.0, 1.0))

        logger.debug(f"R3: Gewicht Klimatologie = {self.weight_:.3f}")
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.weight_ is None:
            raise ValueError("CombinedReference wurde nicht gefittet")

        climate = self.climatology_.predict(X)
        combined = self.weight_ * climate + (1.0 - self.weight_) * X[self.column]
        return combined.clip(CF_FLOOR).rename(self.name)


# Name -> factory, so configs and the experiment loop address references by string.
REFERENCES = {
    Climatology.name: Climatology,
    Persistence.name: Persistence,
    ClearSkyPersistence.name: ClearSkyPersistence,
    CombinedReference.name: CombinedReference,
}

# All skill scores in this project use R3 as the reference (see docs/arbeitsplan.md).
SKILL_REFERENCE = CombinedReference.name

# The published TSO forecast; not fitted, joined into the prediction frame.
TSO_NAME = "R4_tso_dayahead"


def build(name: str):
    """Instantiate a reference by name."""
    if name not in REFERENCES:
        raise ValueError(f"Unbekannte Referenz: {name!r} (bekannt: {list(REFERENCES)})")
    return REFERENCES[name]()
