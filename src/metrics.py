from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from data import DatasetScores


def _weighted_fisher_mean(correlations: list[float], weights: list[int]) -> float:
    corr = np.asarray(correlations, dtype=float)
    weight = np.asarray(weights, dtype=float)
    valid = np.isfinite(corr) & np.isfinite(weight) & (weight > 0)
    if not np.any(valid):
        return float("nan")

    corr = corr[valid]
    weight = np.maximum(weight[valid] - 3.0, 1.0)
    with np.errstate(divide="ignore"):
        z = np.arctanh(corr)
    z[z == np.inf] = np.arctanh(0.99)
    z[z == -np.inf] = -np.arctanh(0.99)
    return float(np.tanh(np.average(z, weights=weight)))


def _sequence_correlation(
    subjective_scores: pd.DataFrame,
    reference_scores: pd.DataFrame,
    expected_len: int,
    metric: str,
    require_complete_scores: bool,
) -> tuple[float, int]:
    merged = subjective_scores.merge(reference_scores, on="preset", how="inner")
    if require_complete_scores and len(merged) != expected_len:
        return float("nan"), expected_len
    if len(merged) < 2:
        return float("nan"), len(merged)

    metric = metric.lower()
    if metric == "srocc":
        value = stats.spearmanr(merged["subjective_score"], merged["score"]).statistic
    elif metric == "plcc":
        value = stats.pearsonr(merged["subjective_score"], merged["score"]).statistic
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return float(abs(value)), len(merged)


def strategy_correlation(
    subjective_by_sequence: dict[str, pd.DataFrame],
    dataset_scores: DatasetScores,
    strategy: str,
    metric: str,
    require_complete_scores: bool = True,
) -> float:
    correlations: list[float] = []
    weights: list[int] = []

    for sequence in dataset_scores.sequences:
        key = f"{sequence}@display={strategy}"
        reference = dataset_scores.scores.loc[
            dataset_scores.scores["sequence"] == sequence,
            ["preset", "score"],
        ].copy()
        corr, weight = _sequence_correlation(
            subjective_scores=subjective_by_sequence.get(key, pd.DataFrame()),
            reference_scores=reference,
            expected_len=dataset_scores.sequence_weight[sequence],
            metric=metric,
            require_complete_scores=require_complete_scores,
        )
        correlations.append(corr)
        weights.append(weight)

    return _weighted_fisher_mean(correlations, weights)


def strategy_srocc(
    subjective_by_sequence: dict[str, pd.DataFrame],
    dataset_scores: DatasetScores,
    strategy: str,
) -> float:
    return strategy_correlation(subjective_by_sequence, dataset_scores, strategy, "srocc")
