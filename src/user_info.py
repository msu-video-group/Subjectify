from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


TITLE_FONTSIZE = 14
GRID_ALPHA = 0.25
BLUE = "#4CA0FF"
SLATE = "#6E8199"
CORAL = "#F26D5B"

PALETTE_BLUE = "#0182AC"
PALETTE_LIGHT_BLUE = "#83CEE2"
PALETTE_GREEN = "#A6BC09"
PALETTE_ORANGE = "#FC9200"
PALETTE_PURPLE = "#65016B"
PALETTE_RED = "#FD1B14"
PALETTE_DARK_BLUE = "#09357A"
PALETTE_YELLOW = "#FFD600"
PALETTE_BLACK = "#000000"
DONUT_COLORS = [
    PALETTE_BLUE,
    PALETTE_DARK_BLUE,
    PALETTE_GREEN,
    PALETTE_ORANGE,
    PALETTE_RED,
    PALETTE_PURPLE,
    PALETTE_LIGHT_BLUE,
    PALETTE_YELLOW,
    PALETTE_BLACK,
]

CODEC_LABELS = {
    "h264_avc": "H.264/AVC",
    "vp8": "VP8",
    "vp9": "VP9",
    "av1": "AV1",
    "hevc": "HEVC",
}

GENDER_ORDER = ["Male", "Female"]
GENDER_COLORS = {
    "Male": PALETTE_BLUE,
    "Female": PALETTE_GREEN,
}

VISION_ACUITY_ORDER = [
    "Good at near and far",
    "Normal with correction",
    "Difficulty near",
    "Difficulty far",
    "Severely reduced",
]
VISION_ACUITY_COLORS = {
    "Good at near and far": PALETTE_BLUE,
    "Normal with correction": PALETTE_GREEN,
    "Difficulty near": PALETTE_ORANGE,
    "Difficulty far": PALETTE_RED,
    "Severely reduced": PALETTE_PURPLE,
}

COLOR_VISION_ORDER = [
    "No difficulty",
    "Red/green difficulty",
    "Blue/yellow difficulty",
    "Do not know",
]
COLOR_VISION_COLORS = {
    "No difficulty": PALETTE_BLUE,
    "Red/green difficulty": PALETTE_ORANGE,
    "Blue/yellow difficulty": PALETTE_PURPLE,
    "Do not know": PALETTE_LIGHT_BLUE,
}


def load_user_info(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    payload_column = "additional" + "_info"
    if payload_column in df.columns:
        raise ValueError("User-info CSV must contain expanded user-info columns.")
    if "participant" not in df.columns:
        raise ValueError("User-info CSV must contain a participant column")
    return df


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.edgecolor": "#000000",
            "axes.linewidth": 1.0,
        }
    )


def save_fig(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)


def percent_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))


def as_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.astype(bool)
    mapped = series.map(
        lambda value: np.nan
        if pd.isna(value)
        else str(value).strip().lower() in {"true", "1", "yes", "on"}
    )
    return mapped.dropna().astype(bool)


def ordered_counts(values: pd.Series, order: list[str] | None = None) -> pd.Series:
    clean = values.dropna().astype(str)
    if order is None:
        return clean.value_counts()
    vc = clean.value_counts()
    return pd.Series({label: float(vc.get(label, 0.0)) for label in order}, dtype=float)


def short_label(label: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", str(label)).strip()


def collapse_ambient_lighting(series: pd.Series) -> pd.Series:
    collapsed: dict[str, float] = {}
    for label, value in series.items():
        label_str = str(label)
        if label_str.startswith("Daylight"):
            target = "Daylight"
        elif label_str.startswith("Room"):
            target = "Room"
        else:
            target = label_str
        collapsed[target] = collapsed.get(target, 0.0) + float(value)
    return pd.Series(collapsed).sort_values(ascending=False)


def donut_with_right_legend(
    ax: plt.Axes,
    title: str,
    series: pd.Series,
    colors_by_label: dict[str, str] | None = None,
    force_labels: list[str] | None = None,
    colors: list[str] | None = None,
    legend_size: str = "42%",
    legend_anchor_x: float = -0.22,
    title_fontsize: int = 11,
    legend_fontsize: float = 8.0,
    radius: float = 0.92,
    width: float = 0.40,
) -> None:
    series = series.astype(float)
    if force_labels is not None:
        series = pd.Series({label: float(series.get(label, 0.0)) for label in force_labels}, dtype=float)
    plotted = series[series > 0]

    ax.set_aspect("equal")
    ax.set_anchor("N")
    divider = make_axes_locatable(ax)
    ax_leg = divider.append_axes("right", size=legend_size, pad=0.0)
    ax_leg.axis("off")

    if plotted.empty:
        ax.set_axis_off()
        labels = force_labels or []
        proxies = [
            Patch(facecolor=(colors_by_label or {}).get(label, PALETTE_BLUE), edgecolor="white")
            for label in labels
        ]
        ax_leg.legend(
            proxies,
            [f"{short_label(label)} - 0%" for label in labels],
            loc="center left",
            bbox_to_anchor=(legend_anchor_x, 0.5),
            frameon=False,
            fontsize=legend_fontsize,
        )
        return

    labels = plotted.index.tolist()
    values = plotted.values.astype(float)
    if colors_by_label is not None:
        wedge_colors = [colors_by_label.get(label, PALETTE_BLUE) for label in labels]
    else:
        palette = colors or DONUT_COLORS
        wedge_colors = [palette[i % len(palette)] for i in range(len(labels))]

    wedges, _ = ax.pie(
        values / values.sum(),
        startangle=90,
        counterclock=False,
        labels=None,
        radius=radius,
        colors=wedge_colors,
        wedgeprops=dict(width=width, edgecolor="white"),
    )
    ax.set_axis_off()
    ax.text(
        0.5,
        0.90,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=title_fontsize,
        fontweight="normal",
    )

    legend_labels = force_labels or labels
    total = float(series.sum()) if float(series.sum()) > 0 else 1.0
    proxies: list[Patch] = []
    text_labels: list[str] = []
    for label in legend_labels:
        if label in labels:
            idx = labels.index(label)
            color = wedges[idx].get_facecolor()
        else:
            color = (colors_by_label or {}).get(label, PALETTE_BLUE)
        proxies.append(Patch(facecolor=color, edgecolor="white"))
        text_labels.append(f"{short_label(label)} - {float(series.get(label, 0.0)) / total * 100.0:.0f}%")

    ax_leg.legend(
        proxies,
        text_labels,
        loc="center left",
        bbox_to_anchor=(legend_anchor_x, 0.5),
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=1.0,
        handletextpad=0.55,
        borderaxespad=0.0,
        labelspacing=0.35,
    )


def grouped_percent_bar(
    ax: plt.Axes,
    labels: list[str],
    series: dict[str, np.ndarray],
    denominator: int,
    colors: list[str],
) -> None:
    x = np.arange(len(labels), dtype=float)
    keys = list(series.keys())
    width = 0.22 if len(keys) >= 3 else 0.30
    offsets = (np.arange(len(keys)) - (len(keys) - 1) / 2.0) * width
    for i, key in enumerate(keys):
        values = np.asarray(series[key], dtype=float) / max(1, denominator) * 100.0
        ax.bar(x + offsets[i], values, width=width, label=key, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right")
    ax.set_ylabel("Users (%)")
    ax.grid(True, axis="y", alpha=GRID_ALPHA)
    percent_axis(ax)
    ax.margins(x=0.08)


def codec_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    suffixes = ["power_efficient", "supported", "smooth", "8bit", "10bit", "sdr", "hdr"]
    rows: list[dict[str, object]] = []
    for column in df.columns:
        if not column.startswith("codec_"):
            continue
        rest = column[len("codec_") :]
        matched = None
        for suffix in suffixes:
            marker = f"_{suffix}"
            if rest.endswith(marker):
                matched = (rest[: -len(marker)], suffix)
                break
        if matched is None:
            continue
        cid, feature = matched
        count = int(as_bool_series(df[column]).sum())
        rows.append(
            {
                "codec_id": cid,
                "codec": CODEC_LABELS.get(cid, cid.upper()),
                "feature": feature,
                "count": count,
            }
        )
    return pd.DataFrame(rows)


def codec_counts(feature_df: pd.DataFrame, feature: str, codecs: list[str]) -> np.ndarray:
    if feature_df.empty:
        return np.zeros(len(codecs), dtype=float)
    subset = feature_df[feature_df["feature"] == feature].set_index("codec")["count"]
    return subset.reindex(codecs).fillna(0).astype(float).to_numpy()


def technical_codecs(df: pd.DataFrame) -> list[str]:
    feature_df = codec_feature_table(df)
    if feature_df.empty:
        return []
    supported = feature_df[feature_df["feature"] == "supported"].sort_values("count", ascending=False)
    return supported["codec"].drop_duplicates().tolist()


def plot_codec_capabilities(df: pd.DataFrame, output_dir: Path) -> None:
    feature_df = codec_feature_table(df)
    codecs = technical_codecs(df)
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    if codecs:
        grouped_percent_bar(
            ax,
            codecs,
            {
                "Supported": codec_counts(feature_df, "supported", codecs),
                "Smooth": codec_counts(feature_df, "smooth", codecs),
                "Power-efficient": codec_counts(feature_df, "power_efficient", codecs),
            },
            len(df),
            [BLUE, SLATE, CORAL],
        )
        ax.legend(loc="upper right", frameon=False)
        ax.set_title("Codec capabilities", fontsize=TITLE_FONTSIZE, pad=12)
    else:
        ax.text(0.5, 0.5, "No media capabilities data", ha="center", va="center")
        ax.set_axis_off()
    save_fig(fig, output_dir, "TechinicalStats_CodecCapabilities.png")


def plot_bit_depth_by_codec(df: pd.DataFrame, output_dir: Path) -> None:
    feature_df = codec_feature_table(df)
    codecs = technical_codecs(df)
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    if codecs:
        grouped_percent_bar(
            ax,
            codecs,
            {
                "8-bit": codec_counts(feature_df, "8bit", codecs),
                "10-bit": codec_counts(feature_df, "10bit", codecs),
            },
            len(df),
            [SLATE, BLUE],
        )
        ax.legend(loc="upper right", frameon=False)
        ax.set_title("Bit depth by codec", fontsize=TITLE_FONTSIZE, pad=12)
    else:
        ax.text(0.5, 0.5, "No media capabilities data", ha="center", va="center")
        ax.set_axis_off()
    save_fig(fig, output_dir, "TechinicalStats_BitDepthByCodec.png")


def plot_dynamic_range_by_codec(df: pd.DataFrame, output_dir: Path) -> None:
    feature_df = codec_feature_table(df)
    codecs = technical_codecs(df)
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    if codecs:
        grouped_percent_bar(
            ax,
            codecs,
            {
                "SDR": codec_counts(feature_df, "sdr", codecs),
                "HDR": codec_counts(feature_df, "hdr", codecs),
            },
            len(df),
            [SLATE, CORAL],
        )
        ax.legend(loc="upper right", frameon=False)
        ax.set_title("Dynamic range by codec", fontsize=TITLE_FONTSIZE, pad=12)
    else:
        ax.text(0.5, 0.5, "No media capabilities data", ha="center", va="center")
        ax.set_axis_off()
    save_fig(fig, output_dir, "TechinicalStats_DynamicRangeByCodec.png")


def plot_display_color_gamut(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    columns = [
        ("color_gamut_srgb", "sRGB"),
        ("color_gamut_p3", "P3"),
        ("color_gamut_rec2020", "Rec.2020"),
    ]
    counts = [int(as_bool_series(df[column]).sum()) if column in df else 0 for column, _ in columns]
    labels = [label for _, label in columns]
    pct = np.asarray(counts, dtype=float) / max(1, len(df)) * 100.0
    ax.bar(labels, pct, width=0.32, color=BLUE)
    ax.set_title("Display color gamut support", fontsize=TITLE_FONTSIZE, pad=12)
    ax.set_ylabel("Users (%)")
    ax.grid(True, axis="y", alpha=GRID_ALPHA)
    percent_axis(ax)
    save_fig(fig, output_dir, "TechinicalStats_DisplayColorGamutSupport.png")


def plot_binned_bar(
    values: pd.Series,
    bins: list[float],
    labels: list[str],
    title: str,
    xlabel: str,
    output_dir: Path,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if x.empty:
        ax.set_axis_off()
    else:
        binned = pd.cut(x, bins=bins, labels=labels, include_lowest=True, right=True)
        counts = binned.value_counts(dropna=True).reindex(labels).fillna(0)
        pct = counts / len(x) * 100.0
        pos = np.arange(len(labels))
        ax.bar(pos, pct.values, color=BLUE, alpha=0.55, width=0.64)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_title(title, fontsize=16, pad=12, fontweight="semibold")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Users (%)", fontsize=14)
        ax.grid(True, axis="y", alpha=GRID_ALPHA)
        percent_axis(ax)
        ax.tick_params(labelsize=12)
    save_fig(fig, output_dir, filename)


def plot_acuity_snellen(df: pd.DataFrame, output_dir: Path) -> None:
    plot_binned_bar(
        df["acuity_best_denominator"],
        [-np.inf, 12.0, 16.0, 20.0, 25.0, np.inf],
        ["10-12", "12-16", "16-20", "20-25", ">25"],
        "Acuity Snellen",
        "Best denominator",
        output_dir,
        "AcuitySnellen.png",
    )


def plot_color_vision_accuracy(df: pd.DataFrame, output_dir: Path) -> None:
    plot_binned_bar(
        df["color_vision_accuracy"] * 100.0,
        [75.0, 80.0, 85.0, 90.0, 95.0, 100.0],
        ["75-80%", "80-85%", "85-90%", "90-95%", "95-100%"],
        "Color vision",
        "Accuracy (%)",
        output_dir,
        "ColorVisionAccuracy.png",
    )


def hist_peak_candidates(values: pd.Series, bins: int = 48) -> list[tuple[float, float]]:
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    if x.size == 0:
        return []
    counts, edges = np.histogram(x, bins=bins)
    pct = counts / max(1, x.size) * 100.0
    centers = (edges[:-1] + edges[1:]) / 2.0
    if pct.size >= 3:
        kernel = np.array([1.0, 2.0, 1.0])
        smooth = np.convolve(pct, kernel / kernel.sum(), mode="same")
        peak_idx = [
            idx
            for idx in range(1, len(smooth) - 1)
            if smooth[idx] >= smooth[idx - 1] and smooth[idx] > smooth[idx + 1] and pct[idx] > 0
        ]
    else:
        smooth = pct
        peak_idx = [idx for idx, value in enumerate(pct) if value > 0]
    if len(peak_idx) < 2:
        peak_idx = list(np.argsort(smooth)[::-1][: min(2, len(smooth))])
    peak_idx = sorted(dict.fromkeys(int(i) for i in peak_idx))[:2]
    return [(float(centers[i]), float(pct[i])) for i in peak_idx]


def annotate_hist_peaks(ax: plt.Axes, values: pd.Series, bins: int = 48) -> None:
    peaks = hist_peak_candidates(values, bins=bins)
    if not peaks:
        return
    y_max = ax.get_ylim()[1]
    for i, (x_peak, y_peak) in enumerate(peaks):
        color = [SLATE, CORAL][i % 2]
        ax.axvline(x_peak, linestyle="--", linewidth=1.5, color=color, alpha=0.95)
        ax.text(
            x_peak,
            min(y_max * (0.92 - i * 0.10), y_peak + y_max * 0.08),
            f'{x_peak:.1f}"',
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=color,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, alpha=0.92),
        )


def plot_diagonal_hist(
    df: pd.DataFrame,
    column: str,
    title: str,
    output_dir: Path,
    filename: str,
) -> None:
    values = pd.to_numeric(df[column], errors="coerce").dropna().astype(float)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    if values.empty:
        ax.set_axis_off()
    else:
        sns.histplot(values, stat="percent", bins=48, ax=ax, color=BLUE, alpha=0.55)
        annotate_hist_peaks(ax, values, bins=48)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.set_title(title, fontweight="semibold", fontsize=14)
        ax.set_xlabel("Inches")
        ax.set_ylabel("Users (%)")
        ax.grid(True, axis="y", alpha=GRID_ALPHA)
    save_fig(fig, output_dir, filename)


def plot_question_vs_measured(df: pd.DataFrame, output_dir: Path) -> None:
    columns = ["screen_diagonal_inches_question", "screen_diagonal_inches_measured"]
    data = df.dropna(subset=columns).copy()
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    if data.empty:
        ax.set_axis_off()
    else:
        sns.kdeplot(
            data=data,
            x=columns[0],
            y=columns[1],
            fill=True,
            levels=25,
            thresh=0.03,
            ax=ax,
            color=BLUE,
            warn_singular=False,
        )
        sns.scatterplot(
            data=data,
            x=columns[0],
            y=columns[1],
            s=18,
            alpha=0.6,
            edgecolor="white",
            linewidth=0.4,
            ax=ax,
            color=BLUE,
        )
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, linestyle="--", linewidth=1.1, color="#3a4a5a")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title("Question vs measured", fontweight="semibold", fontsize=14)
        ax.set_xlabel("Question (inches)")
        ax.set_ylabel("Measured (inches)")
        ax.grid(True, alpha=0.18)
    save_fig(fig, output_dir, "ScreenSizes_QuestionVsMeasured.png")


def plot_resolution_reference_lines(ax: plt.Axes, data: pd.DataFrame) -> None:
    standards = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "2160p": (3840, 2160),
    }
    x_data = pd.to_numeric(data["screen_width"], errors="coerce").dropna()
    y_data = pd.to_numeric(data["screen_height"], errors="coerce").dropna()
    x_min = float(min([x_data.min()] + [value[0] for value in standards.values()]))
    x_max = float(max([x_data.max()] + [value[0] for value in standards.values()]))
    y_min = float(min([y_data.min()] + [value[1] for value in standards.values()]))
    y_max = float(max([y_data.max()] + [value[1] for value in standards.values()]))
    x_pad = max(40.0, (x_max - x_min) * 0.05)
    y_pad = max(30.0, (y_max - y_min) * 0.06)
    ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
    label_dx = max(16.0, x_pad * 0.08)
    label_dy = max(12.0, y_pad * 0.10)
    for label, (width, height) in standards.items():
        ax.axvline(width, linestyle="--", linewidth=1.0, color=SLATE, alpha=0.55, zorder=1)
        ax.axhline(height, linestyle="--", linewidth=1.0, color=SLATE, alpha=0.55, zorder=1)
        ax.scatter(width, height, s=34, color=CORAL, edgecolor="white", linewidth=0.5, zorder=5)
        ax.text(
            width + label_dx,
            height - label_dy,
            f"{label} ({width}x{height})",
            fontsize=9.5,
            color=CORAL,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=CORAL, alpha=0.90),
            zorder=6,
        )


def plot_screen_width_height(df: pd.DataFrame, output_dir: Path) -> None:
    data = df.dropna(subset=["screen_width", "screen_height"]).copy()
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    if data.empty:
        ax.set_axis_off()
    else:
        sns.kdeplot(
            data=data,
            x="screen_width",
            y="screen_height",
            fill=True,
            levels=25,
            thresh=0.03,
            ax=ax,
            color=BLUE,
            warn_singular=False,
        )
        sns.scatterplot(
            data=data,
            x="screen_width",
            y="screen_height",
            s=18,
            alpha=0.6,
            edgecolor="white",
            linewidth=0.4,
            ax=ax,
            color=BLUE,
        )
        plot_resolution_reference_lines(ax, data)
        ax.set_title("Screen width vs height", fontweight="semibold", fontsize=14)
        ax.set_xlabel("Screen width (px)")
        ax.set_ylabel("Screen height (px)")
        ax.grid(True, alpha=0.18)
    save_fig(fig, output_dir, "ScreenSizes_ScreenWidthHeight.png")


def plot_screen_sizes(df: pd.DataFrame, output_dir: Path) -> None:
    plot_diagonal_hist(
        df,
        "screen_diagonal_inches_question",
        "Diagonal (question)",
        output_dir,
        "ScreenSizes_DiagonalQuestion.png",
    )
    plot_diagonal_hist(
        df,
        "screen_diagonal_inches_measured",
        "Diagonal (measured)",
        output_dir,
        "ScreenSizes_DiagonalMeasured.png",
    )
    plot_question_vs_measured(df, output_dir)
    plot_screen_width_height(df, output_dir)


def plot_screen_setting(
    df: pd.DataFrame,
    column: str,
    title: str,
    output_dir: Path,
    filename: str,
    collapse: Callable[[pd.Series], pd.Series] | None = None,
) -> None:
    series = ordered_counts(df[column])
    if collapse is not None:
        series = collapse(series)
    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    donut_with_right_legend(
        ax,
        title,
        series,
        colors=DONUT_COLORS,
        legend_size="38%",
        legend_anchor_x=-0.22,
        title_fontsize=12,
        legend_fontsize=8.5,
    )
    save_fig(fig, output_dir, filename)


def plot_screen_settings(df: pd.DataFrame, output_dir: Path) -> None:
    plot_screen_setting(df, "auto_brightness", "Auto brightness", output_dir, "ScreenSettings_AutoBrightness.png")
    plot_screen_setting(
        df,
        "blue_light_reduction",
        "Blue light reduction",
        output_dir,
        "ScreenSettings_BlueLightReduction.png",
    )
    plot_screen_setting(
        df,
        "ambient_lighting",
        "Ambient lighting",
        output_dir,
        "ScreenSettings_AmbientLighting.png",
        collapse=collapse_ambient_lighting,
    )
    plot_screen_setting(
        df,
        "screen_technology",
        "Screen technology",
        output_dir,
        "ScreenSettings_ScreenTechnology.png",
    )


def plot_histogram(
    values: pd.Series,
    title: str,
    output_dir: Path,
    filename: str,
    bins: int = 28,
    xlim: tuple[float, float] | None = None,
    major_locator: float = 10,
    minor_locator: float = 5,
) -> None:
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    if x.empty:
        ax.set_axis_off()
    else:
        ax.hist(
            x,
            bins=bins,
            weights=np.ones(len(x)) * 100.0 / max(1, len(x)),
            color=BLUE,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.9,
        )
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12, fontweight="normal")
        ax.set_ylabel("Users (%)", fontsize=11)
        ax.set_xlabel("")
        ax.grid(True, axis="y", alpha=GRID_ALPHA)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.tick_params(labelsize=11)
        ax.xaxis.set_major_locator(MultipleLocator(major_locator))
        ax.xaxis.set_minor_locator(MultipleLocator(minor_locator))
        if xlim is not None:
            ax.set_xlim(*xlim)
    save_fig(fig, output_dir, filename)


def plot_age_height(df: pd.DataFrame, output_dir: Path) -> None:
    plot_histogram(df["age"], "Age", output_dir, "AgeHeight_Age.png")
    plot_histogram(df["height_cm"], "Height", output_dir, "AgeHeight_Height.png", xlim=(140, 210))


def plot_demographics_donut(
    df: pd.DataFrame,
    column: str,
    title: str,
    order: list[str],
    colors_by_label: dict[str, str],
    output_dir: Path,
    filename: str,
) -> None:
    series = ordered_counts(df[column], order=order)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    donut_with_right_legend(
        ax,
        title,
        series,
        colors_by_label=colors_by_label,
        force_labels=order,
        legend_size="45%",
        legend_anchor_x=-0.20,
        title_fontsize=11,
        legend_fontsize=8.0,
    )
    save_fig(fig, output_dir, filename)


def plot_demographics_donuts(df: pd.DataFrame, output_dir: Path) -> None:
    plot_demographics_donut(
        df,
        "gender",
        "Gender",
        GENDER_ORDER,
        GENDER_COLORS,
        output_dir,
        "DemographicsDonuts_Gender.png",
    )
    plot_demographics_donut(
        df,
        "self_reported_visual_acuity",
        "Self-reported visual acuity",
        VISION_ACUITY_ORDER,
        VISION_ACUITY_COLORS,
        output_dir,
        "DemographicsDonuts_VisualAcuity.png",
    )
    plot_demographics_donut(
        df,
        "self_reported_color_vision",
        "Self-reported color vision",
        COLOR_VISION_ORDER,
        COLOR_VISION_COLORS,
        output_dir,
        "DemographicsDonuts_ColorVision.png",
    )


def plot_all_user_info(user_info_csv: Path, output_dir: Path) -> list[Path]:
    set_plot_style()
    df = load_user_info(user_info_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_codec_capabilities(df, output_dir)
    plot_bit_depth_by_codec(df, output_dir)
    plot_dynamic_range_by_codec(df, output_dir)
    plot_display_color_gamut(df, output_dir)
    plot_acuity_snellen(df, output_dir)
    plot_color_vision_accuracy(df, output_dir)
    plot_screen_sizes(df, output_dir)
    plot_screen_settings(df, output_dir)
    plot_age_height(df, output_dir)
    plot_demographics_donuts(df, output_dir)

    return sorted(output_dir.glob("*.png"))
