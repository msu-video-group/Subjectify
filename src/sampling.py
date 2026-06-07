from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from data import DatasetScores, sequence_from_test_case
from metrics import strategy_correlation
from models import scores_by_sequence


@dataclass(frozen=True)
class SamplingResult:
    votes_num: list[int]
    srocc: list[float]
    srocc_std: list[float]


def percent_to_total_votes(dataset_scores: DatasetScores, percent: int, max_votes_per_pair: int) -> int:
    total = 0
    for sequence in dataset_scores.sequences:
        weight = dataset_scores.sequence_weight[sequence]
        total += int(percent * max_votes_per_pair * weight * (weight - 1) / 200)
    return total


def _sample_strategy_votes(
    votes: pd.DataFrame,
    dataset_scores: DatasetScores,
    strategy: str,
    percent: int,
    max_votes_per_pair: int,
    random_seed: int,
) -> pd.DataFrame:
    strategy_votes = votes.loc[votes["strategy"] == strategy].copy()
    frames: list[pd.DataFrame] = []

    for index, sequence in enumerate(dataset_scores.sequences):
        weight = dataset_scores.sequence_weight[sequence]
        n_votes = int(percent * max_votes_per_pair * weight * (weight - 1) / 200)
        seq_votes = strategy_votes.loc[
            strategy_votes["test_case"].map(sequence_from_test_case) == sequence
        ]
        if seq_votes.empty or n_votes <= 0:
            continue

        frames.append(
            seq_votes.sample(
                n=n_votes,
                replace=len(seq_votes) < n_votes,
            )
        )

    if not frames:
        return strategy_votes.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def _one_bootstrap(
    votes: pd.DataFrame,
    dataset_scores: DatasetScores,
    strategy: str,
    percent: int,
    max_votes_per_pair: int,
    random_seed: int,
    model_name: str,
    require_complete_scores: bool,
) -> float:
    sampled = _sample_strategy_votes(
        votes=votes,
        dataset_scores=dataset_scores,
        strategy=strategy,
        percent=percent,
        max_votes_per_pair=max_votes_per_pair,
        random_seed=random_seed,
    )
    if sampled.empty:
        return float("nan")

    subjective = scores_by_sequence(
        sampled,
        sequences=dataset_scores.sequences,
        presets=dataset_scores.presets,
        model_name=model_name,
    )
    return strategy_correlation(
        subjective,
        dataset_scores,
        strategy,
        "srocc",
        require_complete_scores=require_complete_scores,
    )


def sample_strategy(
    votes: pd.DataFrame,
    dataset_scores: DatasetScores,
    strategy: str,
    percents: Iterable[int],
    sample_count: int,
    workers: int,
    random_key: int,
    max_votes_per_pair: int,
    model_name: str = "bradley-terry",
    require_complete_scores: bool = True,
) -> SamplingResult:
    x_values: list[int] = []
    means: list[float] = []
    stds: list[float] = []

    for percent in percents:
        seeds = [random_key + percent * 1_000_003 + i * 97 for i in range(sample_count)]
        values = Parallel(n_jobs=workers)(
            delayed(_one_bootstrap)(
                votes,
                dataset_scores,
                strategy,
                percent,
                max_votes_per_pair,
                seed,
                model_name,
                require_complete_scores,
            )
            for seed in seeds
        )
        arr = np.asarray(values, dtype=float)
        x_values.append(percent_to_total_votes(dataset_scores, percent, max_votes_per_pair))
        means.append(float(np.nanmean(arr)))
        stds.append(float(np.nanstd(arr)))

    return SamplingResult(votes_num=x_values, srocc=means, srocc_std=stds)


def sample_experiment(
    votes: pd.DataFrame,
    dataset_scores: DatasetScores,
    strategies: Iterable[str],
    sample_count: int,
    workers: int,
    random_key: int,
    max_votes_per_pair: int,
    percents: Iterable[int] = range(10, 100, 10),
    model_name: str = "bradley-terry",
    require_complete_scores: bool = True,
    strategy_max_votes_per_pair: dict[str, int] | None = None,
) -> dict[str, SamplingResult]:
    return {
        strategy: sample_strategy(
            votes=votes,
            dataset_scores=dataset_scores,
            strategy=strategy,
            percents=percents,
            sample_count=sample_count,
            workers=workers,
            random_key=random_key,
            max_votes_per_pair=(
                strategy_max_votes_per_pair.get(strategy, max_votes_per_pair)
                if strategy_max_votes_per_pair
                else max_votes_per_pair
            ),
            model_name=model_name,
            require_complete_scores=require_complete_scores,
        )
        for strategy in strategies
    }
