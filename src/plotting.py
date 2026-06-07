from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data import StrategyConfig
from sampling import SamplingResult


LIGHT_BLUE = "#83CEE2"
GREEN = "#A6BC09"
ORANGE = "#FC9200"
PURPLE = "#65016B"
BLACK = "#000000"
GRID_COLOR = "#D8DDE6"

STYLE_MAP = {
    "sequential": {"label": "Sequential", "color": LIGHT_BLUE, "marker": ".", "linestyle": "-"},
    "sidebyside": {"label": "Side-by-side", "color": GREEN, "marker": ".", "linestyle": "-"},
    "manual": {"label": "Manual", "color": ORANGE, "marker": ".", "linestyle": "-"},
    "no-training": {"label": "No-training", "color": LIGHT_BLUE, "marker": ".", "linestyle": "-"},
    "training": {"label": "Training", "color": ORANGE, "marker": ".", "linestyle": "-"},
}


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.edgecolor": BLACK,
            "axes.linewidth": 1.0,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.7,
            "grid.linewidth": 0.8,
        }
    )


def _annotate_last(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray) -> None:
    if x_values.size == 0:
        return
    ax.annotate(
        f"{y_values[-1]:.3f}",
        xy=(float(x_values[-1]), float(y_values[-1])),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=10,
        color=BLACK,
    )


def _valid_xy(result: SamplingResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.asarray(result.votes_num, dtype=float)
    y_values = np.asarray(result.srocc, dtype=float)
    yerr = np.asarray(result.srocc_std, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(yerr)
    return x_values[mask], y_values[mask], yerr[mask]


def plot_convergence(
    results: dict[str, SamplingResult],
    strategies: tuple[StrategyConfig, ...],
    output_path: Path,
    title: str,
    show_title: bool,
) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    for strategy in strategies:
        if strategy.name not in results:
            continue
        style = STYLE_MAP.get(
            strategy.name,
            {"label": strategy.label, "color": PURPLE, "marker": ".", "linestyle": "-"},
        )
        x_values, y_values, yerr = _valid_xy(results[strategy.name])
        if x_values.size == 0:
            continue
        ax.errorbar(
            x_values,
            y_values,
            yerr=yerr,
            label=style.get("label", strategy.label),
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=6.5,
            elinewidth=1.2,
            capsize=3.5,
            capthick=1.2,
        )
        _annotate_last(ax, x_values, y_values)

    if show_title and title:
        ax.set_title(title)
    ax.set_xlabel("Number of votes")
    ax.set_ylabel("SROCC")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, axis="both")
    ax.margins(x=0.03)

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + 0.06 * (xmax - xmin))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
