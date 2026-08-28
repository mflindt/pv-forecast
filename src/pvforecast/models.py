"""Naive references and learning models."""

import logging
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Capacity factors are non-negative by construction; clipping keeps that true.
CF_FLOOR = 0.0

DEFAULT_SEED = 42

# The treatment in the model comparison: every tuned model shares this budget.
TUNED_BUDGET = 40

# Upper bound; early stopping on the inner validation picks the actual count.
MAX_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 100

LGBM_DEFAULTS = {
    "objective": "regression",
    "n_estimators": MAX_ROUNDS,
    # Without a bagging frequency the subsample fraction has no effect.
    "subsample_freq": 1,
    "verbose": -1,
    "n_jobs": -1,
}


class Climatology:
    """R0 -- mean capacity factor per month and hour, fitted on the training block."""

    name = "R0_climatology"

    def __init__(self, seed: int = DEFAULT_SEED):
        self.table_: pd.Series | None = None
        self.fallback_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Climatology":
        self.table_ = y.groupby([y.index.month, y.index.hour], observed=True).mean()
        self.table_.index.names = ["month", "hour"]
        self.fallback_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.table_ is None:
            raise ValueError("R0 wurde nicht gefittet")

        key = pd.MultiIndex.from_arrays([X.index.month, X.index.hour])
        # A month-hour cell unseen in training falls back to the overall mean.
        pred = self.table_.reindex(key).to_numpy()
        pred = np.where(np.isnan(pred), self.fallback_, pred)
        return pd.Series(pred, index=X.index, name=self.name).clip(CF_FLOOR)


class Persistence:
    """R1 capacity factor from 48 hours earlier."""

    name = "R1_persistence"
    column = "cf_lag48h"

    def __init__(self, seed: int = DEFAULT_SEED):
        pass

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
    """R2 smart persistence using kt from 48 hours earlier."""

    name = "R2_clearsky_persistence"
    kt_column = "kt_lag48h"
    cs_column = "cs_ghi"

    def __init__(self, seed: int = DEFAULT_SEED):
        self.beta_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClearSkyPersistence":
        raw = self._raw(X)
        denominator = float((raw**2).sum())
        if denominator <= 0:
            raise ValueError("R2: Trainingsfenster enthält keine Einstrahlung")

        self.beta_ = float((raw * y).sum() / denominator)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.beta_ is None:
            raise ValueError("R2 wurde nicht gefittet")
        return (self.beta_ * self._raw(X)).clip(CF_FLOOR).rename(self.name)

    def _raw(self, X: pd.DataFrame) -> pd.Series:
        missing = [c for c in (self.kt_column, self.cs_column) if c not in X.columns]
        if missing:
            raise ValueError(f"R2 braucht fehlende Spalten: {missing}")
        return X[self.kt_column] * X[self.cs_column]


class CombinedReference:
    """R3 convex combination of climatology and persistence."""

    name = "R3_combined"
    column = "cf_lag48h"

    def __init__(self, seed: int = DEFAULT_SEED):
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
            raise ValueError("R3 wurde nicht gefittet")

        climate = self.climatology_.predict(X)
        combined = self.weight_ * climate + (1.0 - self.weight_) * X[self.column]
        return combined.clip(CF_FLOOR).rename(self.name)


class Ridge:
    """Ridge regression on standardised features."""

    name = "ridge"

    def __init__(self, alpha: float = 1.0, seed: int = DEFAULT_SEED):
        self.alpha = alpha
        self.pipeline_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Ridge":
        self.pipeline_ = make_pipeline(StandardScaler(), SklearnRidge(alpha=self.alpha))
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.pipeline_ is None:
            raise ValueError("Ridge wurde nicht gefittet")
        pred = self.pipeline_.predict(X)
        return pd.Series(pred, index=X.index, name=self.name).clip(CF_FLOOR)


class LightGBM:
    """LightGBM with early stopping on the inner validation block."""

    name = "lightgbm"
    uses_validation = True

    def __init__(self, seed: int = DEFAULT_SEED, **params: Any):
        self.seed = seed
        self.params = params
        self.model_ = None
        self.best_iteration_: int | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, validation=None) -> "LightGBM":
        self.model_ = lgb.LGBMRegressor(
            **{**LGBM_DEFAULTS, **self.params, "random_state": self.seed}
        )

        eval_set, callbacks = None, []
        if validation is not None:
            eval_set = [validation]
            callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]

        self.model_.fit(X, y, eval_set=eval_set, eval_metric="l1", callbacks=callbacks)
        self.best_iteration_ = self.model_.best_iteration_ or self.model_.n_estimators_
        return self

    def freeze(self) -> dict:
        """Round count for the refit on the full window."""
        if self.best_iteration_ is None:
            raise ValueError("LightGBM wurde nicht gefittet")
        return {"n_estimators": int(self.best_iteration_)}

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise ValueError("LightGBM wurde nicht gefittet")
        pred = self.model_.predict(X)
        return pd.Series(pred, index=X.index, name=self.name).clip(CF_FLOOR)


class TabPFN3:
    """TabPFN-3, a tabular foundation model; fitting only stores the context."""

    name = "tabpfn3"

    def __init__(self, seed: int = DEFAULT_SEED, **params: Any):
        self.seed = seed
        self.params = params
        self.model_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabPFN3":
        # Imported late so a run without the gpu extra never needs torch.
        from tabpfn import TabPFNRegressor

        self.model_ = TabPFNRegressor(random_state=self.seed, **self.params)
        self.model_.fit(X, y)
        logger.debug(f"TabPFN-3: Kontext aus {len(X)} Zeilen")
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise ValueError("TabPFN-3 wurde nicht gefittet")
        pred = self.model_.predict(X)
        return pd.Series(pred, index=X.index, name=self.name).clip(CF_FLOOR)


@dataclass(frozen=True)
class ModelSpec:
    """A learning model with its search space and tuning budget."""

    factory: type
    space: dict[str, list[Any]]
    budget: int
    library: str
    preprocessing: str


REFERENCES: dict[str, type] = {
    Climatology.name: Climatology,
    Persistence.name: Persistence,
    ClearSkyPersistence.name: ClearSkyPersistence,
    CombinedReference.name: CombinedReference,
}

MODELS: dict[str, ModelSpec] = {
    Ridge.name: ModelSpec(
        factory=Ridge,
        # One hyperparameter over seven decades; the budget enumerates the grid.
        space={"alpha": [float(a) for a in np.logspace(-3, 4, 10)]},
        budget=10,
        library="scikit-learn",
        preprocessing="StandardScaler",
    ),
    LightGBM.name: ModelSpec(
        factory=LightGBM,
        space={
            "learning_rate": [0.01, 0.02, 0.05, 0.1],
            "num_leaves": [15, 31, 63, 127, 255],
            "min_child_samples": [20, 50, 100, 200],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "subsample": [0.6, 0.8, 1.0],
            "reg_lambda": [0.0, 0.1, 1.0, 10.0],
        },
        budget=TUNED_BUDGET,
        library="lightgbm",
        preprocessing="keine",
    ),
    TabPFN3.name: ModelSpec(
        factory=TabPFN3,
        # Empty on purpose: the untuned model is the treatment, not an oversight.
        space={},
        budget=1,
        library="tabpfn",
        preprocessing="keine",
    ),
}

# All skill scores in this project use R3 as the reference.
SKILL_REFERENCE = CombinedReference.name

# The published TSO forecast; not fitted, joined into the prediction frame.
TSO_NAME = "R4_tso_dayahead"


def spec(name: str) -> ModelSpec:
    """Look up a learning model by name."""
    if name not in MODELS:
        raise ValueError(f"Unbekanntes Modell: {name!r} (bekannt: {list(MODELS)})")
    return MODELS[name]


def build(name: str, params: dict | None = None, seed: int = DEFAULT_SEED):
    """Instantiate a reference or a learning model by name."""
    if name in REFERENCES:
        return REFERENCES[name](seed=seed)
    return spec(name).factory(seed=seed, **(params or {}))
