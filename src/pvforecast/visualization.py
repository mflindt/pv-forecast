"""Plotting functions for the run outputs."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvforecast.evaluation import (  # noqa: E402
    KT_LABELS,
    NORMALISERS,
    daily_loss,
    daylight_mask,
)

logger = logging.getLogger(__name__)

DPI = 200

# Checked with the dataviz validator; gray is used for context only.
ACCENTS = ("#2a78d6", "#eb6834", "#4a3aa7")
CONTEXT = "#6b6b6b"
CONTEXT_SOFT = "#a8a8a8"

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e3e0"
SURFACE = "#ffffff"

# Second visual encoding besides color.
FOCUS_MARKERS = ("o", "s", "D")
CONTEXT_STYLES = (("--", "^"), (":", "v"), ("-.", "X"), ((0, (3, 1, 1, 1)), "P"))

LABELS = {
    "R0_climatology": "R0 Klimatologie",
    "R1_persistence": "R1 Persistenz",
    "R2_clearsky_persistence": "R2 Clear-Sky-Persistenz",
    "R3_combined": "R3 Kombiniert",
    "R4_tso_dayahead": "R4 ÜNB D-1",
    "ridge": "Ridge",
    "lightgbm": "LightGBM",
}

INFO_SHORT = {
    "history_only": "Historie",
    "perfect_prog": "Perfect Prog",
    "operational": "operativ",
}

MONTHS = (
    "Jan",
    "Feb",
    "Mär",
    "Apr",
    "Mai",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Okt",
    "Nov",
    "Dez",
)

REFERENCE_FEATURESET = "-"
TSO_MODEL = "R4_tso_dayahead"


def label(name: str) -> str:
    """Readable model name."""
    return LABELS.get(name, name)


def _style() -> None:
    """Clean, borderless base style with subtle grid lines."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "medium",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": INK,
            "font.size": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
        }
    )


def _save(fig, out_dir: Path, name: str) -> Path:
    """Save and close a figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Abbildung geschrieben: {path.name}")
    return path


def model_order(pooled: pd.DataFrame) -> list[str]:
    """Models ranked by pooled nMAE."""
    return pooled.sort_values("nmae")["model"].drop_duplicates().tolist()


def focus_models(pooled: pd.DataFrame, cfg: dict) -> list[str]:
    """Select up to three key series for the plot."""
    available = list(pooled["model"].drop_duplicates())
    chosen: list[str] = []

    if TSO_MODEL in available:
        chosen.append(TSO_MODEL)

    learners = pooled[pooled["featureset"] != REFERENCE_FEATURESET].sort_values("nmae")
    if not learners.empty:
        chosen.append(learners["model"].iloc[0])

    reference = cfg.get("evaluation", {}).get("skill_reference")
    if reference in available and reference not in chosen:
        chosen.append(reference)

    for name in model_order(pooled):
        if len(chosen) >= 3:
            break
        if name not in chosen:
            chosen.append(name)
    return chosen[:3]


def series_styles(order: list[str], focus: list[str]) -> dict[str, dict]:
    """Style each model with color, line style, and marker."""
    styles: dict[str, dict] = {}
    for i, name in enumerate(focus):
        styles[name] = {
            "color": ACCENTS[i % len(ACCENTS)],
            "linestyle": "-",
            "marker": FOCUS_MARKERS[i % len(FOCUS_MARKERS)],
            "linewidth": 2.2,
            "zorder": 3,
        }

    rest = [name for name in order if name not in focus]
    for i, name in enumerate(rest):
        linestyle, marker = CONTEXT_STYLES[i % len(CONTEXT_STYLES)]
        styles[name] = {
            "color": CONTEXT if i % 2 == 0 else CONTEXT_SOFT,
            "linestyle": linestyle,
            "marker": marker,
            "linewidth": 1.3,
            "zorder": 2,
        }
    return styles


def _below(ax, inches: float) -> float:
    """Axis fraction below the plot in inches."""
    height = ax.get_position().height * ax.figure.get_figheight()
    return -inches / max(height, 0.5)


def _legend_below(ax, handles, names, ncols: int = 4, inches: float = 0.58) -> None:
    """Place the legend below the plot."""
    ax.legend(
        handles,
        names,
        loc="upper center",
        bbox_to_anchor=(0.5, _below(ax, inches)),
        ncols=min(ncols, len(names)),
        frameon=False,
        handlelength=2.6,
        columnspacing=1.6,
    )


def _note(ax, text: str, inches: float = 0.62) -> None:
    """Add a footnote below the plot."""
    ax.text(
        0.5,
        _below(ax, inches),
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color=INK_SOFT,
    )


def _relative_luminance(rgb) -> float:
    """Relative luminance according to WCAG 2.2."""
    channels = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3]
    ]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _readable_ink(background) -> str:
    """Choose the text color with better contrast."""
    background_luminance = _relative_luminance(background)

    def contrast(ink_luminance: float) -> float:
        high, low = (
            max(ink_luminance, background_luminance),
            min(ink_luminance, background_luminance),
        )
        return (high + 0.05) / (low + 0.05)

    return (
        "#ffffff"
        if contrast(1.0) > contrast(_relative_luminance((0.04, 0.04, 0.04)))
        else INK
    )


def _first_seed(predictions: pd.DataFrame) -> pd.DataFrame:
    """Use the first repeat for each model."""
    first = predictions.groupby("model")["seed"].transform("min")
    return predictions[predictions["seed"] == first]


def plot_splits(spans: pd.DataFrame, out_dir: Path) -> Path:
    """Training and test windows for each fold."""
    fig, ax = plt.subplots(figsize=(9, 0.40 * len(spans) + 2.0))

    for _, row in spans.iterrows():
        y = row["fold"]
        ax.barh(
            y,
            row["train_end"] - row["train_start"],
            left=row["train_start"],
            height=0.62,
            color=ACCENTS[0],
            label="Training",
        )
        ax.barh(
            y,
            row["test_end"] - row["test_start"],
            left=row["test_start"],
            height=0.62,
            color=ACCENTS[1],
            label="Test",
        )

    handles, names = ax.get_legend_handles_labels()
    unique = dict(zip(names, handles, strict=True))
    _legend_below(ax, list(unique.values()), list(unique.keys()), ncols=2)

    ax.set_yticks(spans["fold"])
    ax.set_yticklabels([f"Fold {f}" for f in spans["fold"]])
    ax.invert_yaxis()
    ax.set_xlabel("Zeit (UTC)")
    ax.set_title("Rolling-Origin-Splits")
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)

    gap_days = (
        spans["test_start"] - spans["train_end"]
    ).dt.total_seconds().max() / 86400
    _note(ax, f"Gap {gap_days:.0f} Tage. Hold-out 2025 ausgeschlossen.", inches=1.02)
    return _save(fig, out_dir, "00_splits.png")


def plot_skill(pooled: pd.DataFrame, cfg: dict, out_dir: Path) -> Path:
    """Skill score by model relative to the reference."""
    reference = cfg["evaluation"]["skill_reference"]
    if "skill" not in pooled.columns:
        raise ValueError("Kein Skill Score im Lauf")

    df = pooled.sort_values("skill").reset_index(drop=True)
    positions = np.arange(len(df))
    # Diverging colors with a neutral midpoint at zero.
    colors = [ACCENTS[0] if value > 0 else ACCENTS[1] for value in df["skill"]]
    colors = [
        CONTEXT_SOFT if m == reference else c
        for m, c in zip(df["model"], colors, strict=True)
    ]

    fig, ax = plt.subplots(figsize=(9, 0.46 * len(df) + 2.6))
    ax.barh(positions, df["skill"], height=0.62, color=colors)
    ax.axvline(0.0, color=INK_SOFT, lw=1.0, zorder=4)

    low, high = float(df["skill"].min()), float(df["skill"].max())
    span = max(high - low, 0.05)
    for pos, value in zip(positions, df["skill"], strict=True):
        offset = span * 0.02
        ax.text(
            value + (offset if value >= 0 else -offset),
            pos,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
            color=INK,
        )

    names = [
        f"{label(m)}  ({INFO_SHORT.get(i, i)})"
        for m, i in zip(df["model"], df["information_set"], strict=True)
    ]
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    # Space for value labels at both ends.
    ax.set_xlim(min(low, 0.0) - span * 0.18, max(high, 0.0) + span * 0.18)
    ax.set_xlabel(f"Skill Score gegen {label(reference)}")
    ax.set_title("Skill Score je Modell")
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)

    _note(
        ax,
        "SS = 1 − nMAE / nMAE(Referenz). Perfect Prog und operativ sind nicht "
        "vergleichbar.",
        inches=0.66,
    )
    return _save(fig, out_dir, "01_skill.png")


def plot_error_distribution(
    predictions: pd.DataFrame, pooled: pd.DataFrame, cfg: dict, out_dir: Path
) -> Path:
    """Daily error distribution by model."""
    settings = {
        "normaliser": cfg["evaluation"]["normaliser"],
        "daylight_only": cfg["evaluation"]["daylight_only"],
    }
    daily = daily_loss(predictions, **settings)
    order = [m for m in model_order(pooled) if m in {key[0] for key in daily.columns}][
        ::-1
    ]

    data, names = [], []
    for name in order:
        column = next(key for key in daily.columns if key[0] == name)
        data.append(daily[column].dropna().to_numpy())
        names.append(label(name))

    fig, ax = plt.subplots(figsize=(9, 0.50 * len(order) + 2.4))
    box = ax.boxplot(
        data,
        orientation="horizontal",
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": INK, "linewidth": 1.6},
        whiskerprops={"color": INK_SOFT, "linewidth": 1.0},
        capprops={"color": INK_SOFT, "linewidth": 1.0},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(ACCENTS[0])
        patch.set_alpha(0.28)
        patch.set_edgecolor(ACCENTS[0])
        patch.set_linewidth(1.4)

    for i, values in enumerate(data, start=1):
        ax.text(
            np.median(values),
            i + 0.36,
            f"Median {np.median(values):.3f}",
            ha="center",
            fontsize=8,
            color=INK_SOFT,
        )

    ax.set_yticklabels(names)
    ax.set_xlabel("nMAE eines Tages")
    ax.set_xlim(left=0)
    ax.set_title("Verteilung des Tagesfehlers")
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)

    _note(ax, f"{len(data[0])} Tage je Modell, Ausreißer ausgeblendet.", inches=0.66)
    return _save(fig, out_dir, "02_fehlerverteilung.png")


def plot_rank_stability(
    per_fold: pd.DataFrame, pooled: pd.DataFrame, out_dir: Path
) -> Path:
    """Model rank by fold."""
    order = model_order(pooled)
    by_fold = (
        per_fold.groupby(["model", "fold"], observed=True)["nmae"]
        .mean()
        .unstack("model")
    )
    ranks = by_fold.rank(axis=1).T.reindex(order)

    fig, ax = plt.subplots(
        figsize=(0.62 * ranks.shape[1] + 4.2, 0.46 * len(order) + 2.6)
    )
    # Sequential shades: darker means a worse rank.
    mesh = ax.imshow(
        ranks.to_numpy(), cmap="Blues", aspect="auto", vmin=0.5, vmax=len(order) + 0.5
    )

    for row in range(ranks.shape[0]):
        for col in range(ranks.shape[1]):
            value = ranks.iat[row, col]
            cell = mesh.cmap(mesh.norm(value))
            ax.text(
                col,
                row,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=9,
                color=_readable_ink(cell),
            )

    ax.set_xticks(range(ranks.shape[1]))
    ax.set_xticklabels([f"{f:.0f}" for f in ranks.columns])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label(m) for m in order])
    ax.set_xlabel("Fold (chronologisch, je 90 Testtage)")
    ax.set_title("Rang je Modell und Fold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.03)
    bar.set_label("Rang je Fold (1 = bester)", fontsize=9, color=INK_SOFT)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8)

    return _save(fig, out_dir, "03_rangfolge.png")


def plot_stratified(
    strata: pd.DataFrame, pooled: pd.DataFrame, cfg: dict, out_dir: Path
) -> Path:
    """Errors by season, cloud cover, and time of day."""
    order = model_order(pooled)
    focus = focus_models(pooled, cfg)
    styles = series_styles(order, focus)

    panels = (
        ("month", "nach Monat", [str(m) for m in range(1, 13)], MONTHS),
        ("kt_bin", "nach Bewölkung", list(KT_LABELS), KT_LABELS),
        ("hour", "nach Stunde (UTC)", [str(h) for h in range(24)], None),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (key, title, categories, ticks) in zip(axes, panels, strict=True):
        block = strata[strata["stratum_type"] == key]
        pivot = block.pivot_table(index="stratum", columns="model", values="nmae")
        pivot = pivot.reindex([c for c in categories if c in pivot.index])

        for name in order:
            if name not in pivot.columns:
                continue
            style = styles[name]
            ax.plot(
                range(len(pivot)),
                pivot[name],
                label=label(name),
                markersize=4 if name in focus else 3,
                markevery=1 if key != "hour" else 2,
                **style,
            )

        ax.set_xticks(range(len(pivot)))
        labels = list(ticks) if ticks is not None else list(pivot.index)
        ax.set_xticklabels(
            labels[: len(pivot)], rotation=45 if key != "hour" else 0, fontsize=8
        )
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("nMAE")
    handles, names = axes[0].get_legend_handles_labels()
    axes[1].legend(
        handles,
        names,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncols=min(4, len(names)),
        frameon=False,
        handlelength=2.6,
    )
    fig.suptitle(
        "Fehler nach Saison, Bewölkung und Tageszeit", y=1.02, fontsize=13, color=INK
    )
    fig.tight_layout()
    return _save(fig, out_dir, "04_fehler_stratifiziert.png")


def pick_example_days(predictions: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The clearest and cloudiest test days."""
    daylight = predictions[daylight_mask(predictions["sun_elevation"])]
    day = pd.to_datetime(daylight["time"], utc=True).dt.floor("D")
    per_day = (
        daylight.assign(day=day).groupby("day").agg(kt=("kt", "mean"), n=("kt", "size"))
    )

    summer = per_day[per_day["n"] >= per_day["n"].median()]
    if summer.empty:
        summer = per_day
    return summer["kt"].idxmax(), summer["kt"].idxmin()


def plot_example_days(
    predictions: pd.DataFrame, pooled: pd.DataFrame, cfg: dict, out_dir: Path
) -> Path:
    """A clear and cloudy day: model behavior."""
    order = model_order(pooled)
    focus = focus_models(pooled, cfg)
    styles = series_styles(order, focus)
    single = _first_seed(predictions)
    time = pd.to_datetime(single["time"], utc=True)

    clear_day, overcast_day = pick_example_days(predictions)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

    for ax, (chosen, title) in zip(
        axes, ((clear_day, "klarer Tag"), (overcast_day, "bedeckter Tag")), strict=True
    ):
        block = single[time.dt.floor("D") == chosen]
        hours = pd.to_datetime(block["time"], utc=True).dt.hour

        truth = block.drop_duplicates("time").sort_values("time")
        ax.fill_between(
            pd.to_datetime(truth["time"], utc=True).dt.hour,
            truth["y_true_mwh"] / 1000,
            color=INK,
            alpha=0.12,
            label="Ist-Einspeisung",
            zorder=1,
        )
        for name in order:
            rows = block[block["model"] == name].sort_values("time")
            if rows.empty:
                continue
            ax.plot(
                hours[rows.index],
                rows["y_pred_mwh"] / 1000,
                label=label(name),
                markersize=4.5 if name in focus else 0,
                markevery=3,
                **styles[name],
            )

        ax.set_title(f"{chosen:%d.%m.%Y}, {title}")
        ax.set_xlabel("Stunde (UTC)")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Einspeisung (GWh/h)")
    handles, names = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        names,
        loc="upper center",
        bbox_to_anchor=(1.05, -0.18),
        ncols=min(4, len(names)),
        frameon=False,
        handlelength=2.6,
    )
    fig.suptitle(
        "Beispieltage",
        y=1.02,
        fontsize=13,
        color=INK,
    )
    fig.tight_layout()
    return _save(fig, out_dir, "05_beispieltage.png")


def plot_significance(tests: pd.DataFrame, out_dir: Path) -> Path:
    df = tests.sort_values("mean_loss_diff", ascending=False).reset_index(drop=True)
    positions = np.arange(len(df))
    significant = (df["p_holm"] < 0.05).to_numpy()

    fig, ax = plt.subplots(figsize=(9, 0.50 * len(df) + 2.8))
    ax.axvline(0.0, color=INK_SOFT, lw=1.0, zorder=1)
    ax.errorbar(
        df["mean_loss_diff"],
        positions,
        xerr=1.96 * df["hac_se"],
        fmt="none",
        ecolor=INK_SOFT,
        elinewidth=1.2,
        capsize=3,
        zorder=2,
    )
    for mask, marker, face, name in (
        (significant, "o", ACCENTS[0], "nach Holm signifikant (5 %)"),
        (~significant, "o", SURFACE, "nicht signifikant"),
    ):
        if mask.any():
            ax.plot(
                df.loc[mask, "mean_loss_diff"],
                positions[mask],
                marker,
                markersize=8,
                markerfacecolor=face,
                markeredgecolor=ACCENTS[0],
                markeredgewidth=1.8,
                linestyle="none",
                label=name,
                zorder=3,
            )

    ax.set_yticks(positions)
    ax.set_yticklabels([label(m) for m in df["model"]])
    ax.set_xlabel(
        f"mittlere Differenz des Tagesverlusts gegen {label(df['reference'].iloc[0])}"
    )
    ax.set_title("Giacomini-White gegen die Referenz")
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    handles, names = ax.get_legend_handles_labels()
    _legend_below(ax, handles, names, ncols=2)

    _note(
        ax,
        "Tagesverluste, HAC-Varianz, 95-Prozent-Intervall.",
        inches=1.02,
    )
    return _save(fig, out_dir, "06_signifikanz.png")


def plot_bias(
    predictions: pd.DataFrame,
    strata: pd.DataFrame,
    pooled: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
) -> Path:
    order = model_order(pooled)
    focus = focus_models(pooled, cfg)
    styles = series_styles(order, focus)
    capacity_col = NORMALISERS[cfg["evaluation"]["normaliser"]]

    fig, (ax_bias, ax_cal) = plt.subplots(1, 2, figsize=(13, 5.2))

    block = strata[strata["stratum_type"] == "hour"]
    pivot = block.pivot_table(index="stratum", columns="model", values="nmbe")
    pivot.index = pivot.index.astype(int)
    pivot = pivot.sort_index()

    ax_bias.axhline(0.0, color=INK_SOFT, lw=1.0, zorder=1)
    for name in order:
        if name in pivot.columns:
            ax_bias.plot(
                pivot.index,
                pivot[name],
                label=label(name),
                markersize=4,
                **styles[name],
            )
    ax_bias.set_xlabel("Stunde (UTC)")
    ax_bias.set_ylabel("nMBE (positiv = Überschätzung)")
    ax_bias.set_title("Bias im Tagesgang")
    ax_bias.grid(alpha=0.6)
    ax_bias.set_axisbelow(True)

    daylight = predictions[daylight_mask(predictions["sun_elevation"])].copy()
    capacity = daylight[capacity_col]
    daylight["cf_pred"] = daylight["y_pred_mwh"] / capacity
    daylight["residual"] = (daylight["y_true_mwh"] - daylight["y_pred_mwh"]) / capacity

    ax_cal.axhline(0.0, color=INK_SOFT, lw=1.0, zorder=1)
    for name in order:
        rows = daylight[daylight["model"] == name]
        if rows.empty:
            continue
        bins = pd.qcut(rows["cf_pred"].rank(pct=True), 20, labels=False)
        grouped = rows.groupby(bins, observed=True)[["cf_pred", "residual"]].mean()
        ax_cal.plot(
            grouped["cf_pred"],
            grouped["residual"],
            label=label(name),
            markersize=4,
            **styles[name],
        )

    ax_cal.set_xlabel("prognostizierter Kapazitätsfaktor")
    ax_cal.set_ylabel("mittleres Residuum (Ist − Prognose)")
    ax_cal.set_title("Bias über das Ertragsniveau")
    ax_cal.grid(alpha=0.6)
    ax_cal.set_axisbelow(True)
    ax_cal.annotate(
        "unter null: Überschätzung",
        xy=(0.03, 0.05),
        xycoords="axes fraction",
        fontsize=8,
        color=INK_SOFT,
    )

    handles, names = ax_bias.get_legend_handles_labels()
    ax_bias.legend(
        handles,
        names,
        loc="upper center",
        bbox_to_anchor=(1.10, -0.14),
        ncols=min(4, len(names)),
        frameon=False,
        handlelength=2.6,
    )
    fig.suptitle("Systematischer Bias", y=1.0, fontsize=13, color=INK)
    fig.tight_layout()
    return _save(fig, out_dir, "07_systematischer_bias.png")


def largest_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one series per model: the one fitted on the largest context."""
    largest = frame.groupby("model", observed=True)["context_rows"].transform("max")
    return frame[frame["context_rows"] == largest]


def plot_context_curve(pooled: pd.DataFrame, out_dir: Path) -> Path:
    """nMAE over the size of the training context, one line per learning model."""
    block = pooled[
        (pooled["featureset"] != REFERENCE_FEATURESET) & (pooled["context_rows"] > 0)
    ]
    if block["context_rows"].nunique() < 2:
        raise ValueError("Kontextkurve braucht mindestens zwei Kontextgrößen")

    order = [m for m in model_order(pooled) if m in set(block["model"])]
    styles = series_styles(order, order[:3])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name in order:
        rows = block[block["model"] == name].sort_values("context_rows")
        ax.plot(rows["context_rows"], rows["nmae"], label=label(name), **styles[name])

    # The reference stays at the full window, so it is a horizontal guide.
    reference = pooled[pooled["featureset"] == REFERENCE_FEATURESET].sort_values("nmae")
    if not reference.empty:
        best = reference.iloc[0]
        ax.axhline(best["nmae"], color=CONTEXT_SOFT, linestyle=":", linewidth=1.2)
        ax.annotate(
            label(best["model"]),
            (block["context_rows"].max(), best["nmae"]),
            textcoords="offset points",
            xytext=(-4, 4),
            ha="right",
            fontsize=8,
            color=INK_SOFT,
        )

    # Log scale spreads the small contexts; the ticks stay the sizes we ran.
    sizes = sorted(block["context_rows"].unique())
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{size // 1000}k" for size in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Kontextzeilen im Trainingsfenster")
    ax.set_ylabel("nMAE")
    ax.set_title("Prognosegüte über der Kontextgröße")
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    handles, names = ax.get_legend_handles_labels()
    _legend_below(ax, handles, names)

    _note(ax, "Mittel über alle Folds, Tagstunden.", inches=1.02)
    return _save(fig, out_dir, "08_kontextkurve.png")


def make_all(
    out_dir: Path,
    cfg: dict,
    predictions: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    spans: pd.DataFrame,
) -> list[Path]:
    """Write all figures for a run under <run>/figures."""
    _style()
    figures_dir = Path(out_dir) / "figures"
    sweep = results["metrics_agg"]

    # Only the curve shows the sweep; the rest shows the largest context.
    if sweep["context_rows"].nunique() > 1:
        predictions = largest_context(predictions)
        results = {name: largest_context(f) for name, f in results.items()}
    pooled = results["metrics_agg"]

    jobs = [
        ("00_splits", lambda: plot_splits(spans, figures_dir)),
        ("01_skill", lambda: plot_skill(pooled, cfg, figures_dir)),
        (
            "02_fehlerverteilung",
            lambda: plot_error_distribution(predictions, pooled, cfg, figures_dir),
        ),
        (
            "03_rangfolge",
            lambda: plot_rank_stability(results["metrics_fold"], pooled, figures_dir),
        ),
        (
            "04_fehler_stratifiziert",
            lambda: plot_stratified(results["strata"], pooled, cfg, figures_dir),
        ),
        (
            "05_beispieltage",
            lambda: plot_example_days(predictions, pooled, cfg, figures_dir),
        ),
        (
            "07_systematischer_bias",
            lambda: plot_bias(predictions, results["strata"], pooled, cfg, figures_dir),
        ),
    ]
    if "tests" in results:
        jobs.insert(
            6,
            (
                "06_signifikanz",
                lambda: plot_significance(results["tests"], figures_dir),
            ),
        )
    jobs.append(("08_kontextkurve", lambda: plot_context_curve(sweep, figures_dir)))

    written = []
    for name, job in jobs:
        try:
            written.append(job())
        except Exception as error:
            logger.warning(f"Abbildung {name} übersprungen: {error}")

    logger.info(f"{len(written)} Abbildungen in {figures_dir}")
    return written
