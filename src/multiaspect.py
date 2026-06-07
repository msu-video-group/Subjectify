from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ASPECT_COLUMNS = [
    "fluency",
    "exposure",
    "contrast",
    "color",
    "sharpness",
    "noise",
    "compression_artifacts",
    "aesthetic",
]
ASPECT_LABELS = {
    "fluency": "Fluency",
    "exposure": "Exposure",
    "contrast": "Lighting",
    "color": "Color",
    "sharpness": "Sharpness",
    "noise": "Noise",
    "compression_artifacts": "Artifacts",
    "aesthetic": "Aesthetic",
}

PALETTE = [
    "#83CEE2",
    "#A6BC09",
    "#FC9200",
    "#8A6FB6",
    "#E06C75",
    "#4DB6AC",
    "#D4A017",
]
GRID_LINE = "#B9C8DB"
TEXT_COLOR = "#203040"
NEUTRAL_RING = "#7B93AC"
NEGATIVE_FILL = "#FFF1E8"
POSITIVE_FILL = "#EEF8F4"


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name)).strip("_")


def wrap_label(text: str, width: int = 22) -> str:
    words = str(text).split()
    if not words:
        return str(text)

    lines: list[str] = []
    line: list[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if line else 0)
        if current_len + extra > width and line:
            lines.append(" ".join(line))
            line = [word]
            current_len = len(word)
        else:
            line.append(word)
            current_len += extra
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def load_multiaspect_scores(path: Path, test_case: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"method", "test_case", "participant", *ASPECT_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} misses columns: {sorted(missing)}")

    if test_case:
        df = df.loc[df["test_case"].astype(str) == test_case].copy()
        if df.empty:
            raise ValueError(f"No rows found for test_case={test_case!r}.")

    for column in ASPECT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def aggregate_by_method(df: pd.DataFrame) -> pd.DataFrame:
    means = df.groupby("method", sort=True)[ASPECT_COLUMNS].mean()
    return means.sort_values(by=ASPECT_COLUMNS, ascending=False)


def _method_order_for_display(means_df: pd.DataFrame) -> list[str]:
    if means_df.empty:
        return []
    return means_df.mean(axis=1, skipna=True).sort_values(ascending=False).index.tolist()


def _impute_matrix(means_df: pd.DataFrame) -> np.ndarray:
    values = means_df.astype(float).to_numpy(copy=True)
    if values.size == 0:
        return values
    col_means = np.nanmean(values, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(col_means, inds[1])
    return values


def _select_diverse_methods(methods: list[str], values: np.ndarray, count: int) -> list[str]:
    if count <= 0 or count >= len(methods):
        return methods
    if not methods:
        return []

    centroid = values.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(values - centroid, axis=1)))
    selected = [first]
    min_dists = np.linalg.norm(values - values[first], axis=1)

    for _ in range(count - 1):
        idx = int(np.argmax(min_dists))
        selected.append(idx)
        min_dists = np.minimum(min_dists, np.linalg.norm(values - values[idx], axis=1))

    return [methods[index] for index in selected]


def select_methods(means_df: pd.DataFrame, max_methods: int, selection: str) -> list[str]:
    methods = _method_order_for_display(means_df)
    if max_methods <= 0 or max_methods >= len(methods):
        return methods

    if selection == "top":
        return methods[:max_methods]

    if selection == "diverse":
        chosen = set(_select_diverse_methods(methods, _impute_matrix(means_df.loc[methods]), max_methods))
        return [method for method in methods if method in chosen]

    if selection == "diverse-top":
        top = methods[0]
        if max_methods == 1:
            return [top]
        rest = methods[1:]
        chosen = {top, *_select_diverse_methods(rest, _impute_matrix(means_df.loc[rest]), max_methods - 1)}
        return [method for method in methods if method in chosen]

    raise ValueError(f"Unknown selection mode: {selection}")


def _get_angles(count: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.concatenate([angles, [angles[0]]])


def _close_values(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.concatenate([arr, [arr[0]]])


def _score_to_radius(values: list[float]) -> np.ndarray:
    return _close_values(values) + 1.0


def _shrink_polar_axis(ax: plt.Axes, scale: float = 0.74) -> None:
    pos = ax.get_position()
    center_x = pos.x0 + pos.width / 2.0
    center_y = pos.y0 + pos.height / 2.0
    new_w = pos.width * scale
    new_h = pos.height * scale
    ax.set_position([center_x - new_w / 2.0, center_y - new_h / 2.0, new_w, new_h])


def _apply_radar_style(ax: plt.Axes, labels: list[str]) -> None:
    count = len(labels)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_facecolor("#FFFFFF")
    ax.set_xticks(angles)
    ax.set_xticklabels([wrap_label(label) for label in labels], fontsize=20, color=TEXT_COLOR)
    ax.tick_params(axis="x", pad=42)

    tick_labels = ax.get_xticklabels()
    if tick_labels:
        x0, y0 = tick_labels[0].get_position()
        tick_labels[0].set_position((x0, y0 + 0.08))
        xb, yb = tick_labels[len(tick_labels) // 2].get_position()
        tick_labels[len(tick_labels) // 2].set_position((xb, yb + 0.08))

    ax.set_ylim(0, 2.0)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(["-1.0", "-0.5", "0.0", "+0.5", "+1.0"], fontsize=18, color="#506174")
    ax.set_rlabel_position(0)
    ax.tick_params(axis="y", pad=6)

    theta = np.linspace(0, 2 * np.pi, 512)
    ax.fill_between(theta, 0.0, 1.0, color=NEGATIVE_FILL, alpha=0.55, zorder=0)
    ax.fill_between(theta, 1.0, 2.0, color=POSITIVE_FILL, alpha=0.60, zorder=0)
    ax.plot(theta, np.ones_like(theta), color=NEUTRAL_RING, linewidth=1.6, linestyle="--", alpha=0.9, zorder=1)

    ax.grid(True, color=GRID_LINE, alpha=0.75, linewidth=1.0)
    ax.spines["polar"].set_color("#AFC1D6")
    ax.spines["polar"].set_linewidth(1.1)


def plot_combined_radar(
    means_df: pd.DataFrame,
    output_path: Path,
    max_methods: int = 0,
    selection: str = "diverse",
) -> None:
    if means_df.empty:
        raise ValueError("No multiaspect data to plot.")

    methods = select_methods(means_df, max_methods=max_methods, selection=selection)
    means_df = means_df.loc[methods, ASPECT_COLUMNS]
    angles = _get_angles(len(ASPECT_COLUMNS))
    labels = [ASPECT_LABELS[column] for column in ASPECT_COLUMNS]

    fig = plt.figure(figsize=(12.8, 12.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax = plt.subplot(111, polar=True)
    _apply_radar_style(ax, labels)
    _shrink_polar_axis(ax)

    for index, method in enumerate(methods):
        values = means_df.loc[method, ASPECT_COLUMNS].astype(float).fillna(0.0).tolist()
        radii = _score_to_radius(values)
        color = PALETTE[index % len(PALETTE)]
        ax.plot(angles, radii, linewidth=2.8, color=color, alpha=0.97)
        ax.fill(angles, radii, color=color, alpha=0.10)
        ax.scatter(angles[:-1], radii[:-1], s=28, color=color, zorder=4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
