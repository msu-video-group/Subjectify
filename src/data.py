from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    file: Path
    label: str
    max_votes_per_pair: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    title: str
    show_title: bool
    output_file: str
    score_file: Path
    strategies: tuple[StrategyConfig, ...]
    require_complete_scores: bool = True


@dataclass(frozen=True)
class DatasetScores:
    scores: pd.DataFrame
    sequences: tuple[str, ...]
    presets: dict[str, list[str]]
    sequence_weight: dict[str, int]


def load_experiment_config(experiment_dir: Path) -> ExperimentConfig:
    config_path = experiment_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    strategies = tuple(
        StrategyConfig(
            name=item["name"],
            file=(experiment_dir / item["file"]).resolve(),
            label=item.get("label", item["name"]),
            max_votes_per_pair=item.get("max_votes_per_pair"),
        )
        for item in raw["strategies"]
    )
    return ExperimentConfig(
        dataset=raw["dataset"],
        title=raw.get("title", ""),
        show_title=bool(raw.get("show_title", True)),
        output_file=raw["output_file"],
        score_file=(experiment_dir / raw["score_file"]).resolve(),
        strategies=strategies,
        require_complete_scores=bool(raw.get("require_complete_scores", True)),
    )


def load_scores(score_file: Path) -> DatasetScores:
    scores = pd.read_csv(score_file)
    required = {"sequence", "preset", "score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"{score_file} misses columns: {sorted(missing)}")

    scores = scores.loc[:, ["sequence", "preset", "score"]].copy()
    scores["sequence"] = scores["sequence"].astype(str)
    scores["preset"] = scores["preset"].astype(str)
    scores["score"] = scores["score"].astype(float)

    sequences = tuple(dict.fromkeys(scores["sequence"].tolist()))
    presets: dict[str, list[str]] = {}
    for sequence in sequences:
        seq_scores = scores.loc[scores["sequence"] == sequence, "preset"]
        presets[sequence] = list(dict.fromkeys(seq_scores.tolist()))

    sequence_weight = {sequence: len(seq_presets) for sequence, seq_presets in presets.items()}
    return DatasetScores(
        scores=scores,
        sequences=sequences,
        presets=presets,
        sequence_weight=sequence_weight,
    )


def load_votes(strategies: tuple[StrategyConfig, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for strategy in strategies:
        df = pd.read_csv(strategy.file)
        required = {"answer", "left_method", "participant", "right_method", "test_case"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{strategy.file} misses columns: {sorted(missing)}")

        df = df.loc[:, ["answer", "left_method", "participant", "right_method", "test_case"]].copy()
        df["strategy"] = strategy.name
        frames.append(df)

    if not frames:
        raise ValueError("Experiment has no strategy CSV files.")
    return pd.concat(frames, ignore_index=True)


def sequence_from_test_case(test_case: str) -> str:
    return str(test_case).split("@", 1)[0]
