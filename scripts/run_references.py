"""Entry point for the reference forecasts (E1) on the rolling-origin folds."""

import logging

import pandas as pd

from pvforecast import baselines, evaluation
from pvforecast.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR
from pvforecast.data.capacity import load_installed_power
from pvforecast.data.smard_forecast import align_to_target, build_hourly_forecast
from pvforecast.features import DATA_START, baseline_inputs, build_features, to_energy
from pvforecast.logging_setup import setup_logging
from pvforecast.seeds import set_seed
from pvforecast.splits import rolling_origin_days

logger = logging.getLogger(__name__)

MODEL_INPUT = PROCESSED_DIR / "pv_weather_hourly.parquet"
FORECAST_RAW = RAW_DIR / "smard_pv_forecast_dayahead_quarterhour_2015-2026.csv"
CAPACITY_RAW = RAW_DIR / "capacity_energycharts_solar_2002-2026.csv"

RESULTS_DIR = REPORTS_DIR / "results" / "e1_referenzen"

# No feature stage here: the references read named columns, not a matrix.
FEATURESET = "-"


def load_inputs() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Feature matrix, target, meta columns and the TSO forecast, all on one index."""
    df = pd.read_parquet(MODEL_INPUT)
    X, y, meta = build_features(df)
    X = pd.concat([X, baseline_inputs(X)], axis=1)

    X, y, meta = X.loc[DATA_START:], y.loc[DATA_START:], meta.loc[DATA_START:]

    forecast = build_hourly_forecast(FORECAST_RAW, X.index.min(), X.index.max())
    tso = align_to_target(forecast, X.index)

    # Monthly nameplate power, held constant within the month it was reported.
    capacity = load_installed_power(CAPACITY_RAW)["solar_ac_gw"] * 1000
    meta["cap_ac_mw"] = capacity.reindex(X.index.union(capacity.index)).ffill()[X.index]

    return X, y, meta, tso


def predict_folds(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, tso: pd.Series):
    """Fit every reference per fold and collect predictions in long format."""
    folds = rolling_origin_days(X.index)
    rows = []

    for i, (train, test) in enumerate(folds, start=1):
        predictions = {}
        for name in baselines.REFERENCES:
            model = baselines.build(name).fit(X.loc[train], y.loc[train])
            predictions[name] = model.predict(X.loc[test])

        # R4 is observed, not fitted: it enters as MWh and needs no back-transform.
        frames = {
            name: to_energy(cf, meta.loc[test, "cap_roll_mwh"])
            for name, cf in predictions.items()
        }
        frames[baselines.TSO_NAME] = tso.loc[test]

        for name, pred in frames.items():
            rows.append(
                pd.DataFrame(
                    {
                        "time": test,
                        "split": "cv",
                        "fold": i,
                        "model": name,
                        "featureset": FEATURESET,
                        "seed": 0,
                        "y_true_mwh": meta.loc[test, "pv_mwh"].to_numpy(),
                        "y_pred_mwh": pred.to_numpy(),
                        "cap_roll_mwh": meta.loc[test, "cap_roll_mwh"].to_numpy(),
                        "cap_ac_mw": meta.loc[test, "cap_ac_mw"].to_numpy(),
                        "sun_elevation": X.loc[test, "sun_elevation"].to_numpy(),
                        "kt": X.loc[test, "kt"].to_numpy(),
                    }
                )
            )

    return pd.concat(rows, ignore_index=True)


def main():
    """Run the reference forecasts over the rolling-origin folds and report them."""
    set_seed()
    X, y, meta, tso = load_inputs()
    predictions = predict_folds(X, y, meta, tso)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(RESULTS_DIR / "predictions.parquet", index=False)

    per_fold = evaluation.add_skill(
        evaluation.evaluate(predictions), baselines.SKILL_REFERENCE
    )
    per_fold.to_csv(RESULTS_DIR / "metrics_fold.csv", index=False)

    # Pooled over all folds: the headline numbers. The fold spread joins them below.
    pooled = evaluation.add_skill(
        evaluation.evaluate(predictions, groupby=("model",)), baselines.SKILL_REFERENCE
    )
    pooled.to_csv(RESULTS_DIR / "metrics_agg.csv", index=False)

    logger.info(f"Ergebnisse gespeichert: {RESULTS_DIR}")
    report(pooled, per_fold)


def report(pooled: pd.DataFrame, per_fold: pd.DataFrame) -> None:
    """Print the first error table of the project."""
    spread = evaluation.aggregate_folds(per_fold, groupby=("model",))
    table = (
        pooled.set_index("model")[["nmae", "nrmse", "mae", "skill"]]
        .join(spread.set_index("model")["nmae_std"])
        .sort_values("nmae")
    )

    n_folds = int(per_fold["fold"].nunique())
    print(
        f"\nReferenzen, {n_folds} Rolling-Origin-Folds, Tagstunden, Normierung cap_roll"
    )
    print("Gepoolt über alle Folds; sd = Streuung über die Folds")
    print(f"Skill Score gegen {baselines.SKILL_REFERENCE}\n")
    print(
        table.rename(
            columns={
                "nmae": "nMAE",
                "nmae_std": "nMAE sd",
                "nrmse": "nRMSE",
                "mae": "MAE MWh",
                "skill": "Skill",
            }
        )[["nMAE", "nMAE sd", "nRMSE", "MAE MWh", "Skill"]].to_string(
            float_format=lambda v: f"{v:.4f}"
        )
    )


if __name__ == "__main__":
    setup_logging("references")
    main()
