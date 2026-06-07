#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import load_experiment_config, load_scores, load_votes, sequence_from_test_case
from metrics import strategy_correlation
from models import MODEL_FUNCTIONS, scores_by_sequence
from sampling import percent_to_total_votes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute model correlations for one experiment directory.")
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--model", choices=sorted(MODEL_FUNCTIONS), default="bradley-terry")
    parser.add_argument("--metrics", nargs="+", choices=["srocc", "plcc"], default=["srocc"])
    parser.add_argument("--percent", nargs="+", type=int, default=[100])
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--random-key", type=int, default=42)
    parser.add_argument("--max-votes-per-pair", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path for the computed table.")
    return parser.parse_args()


def sample_strategy_votes(
    votes: pd.DataFrame,
    dataset_scores,
    strategy: str,
    percent: int,
    max_votes_per_pair: int,
    random_seed: int,
) -> pd.DataFrame:
    strategy_votes = votes.loc[votes["strategy"] == strategy].copy()
    if percent >= 100:
        return strategy_votes

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
                random_state=random_seed + index * 10_007,
            )
        )

    if not frames:
        return strategy_votes.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def compute_one(
    votes: pd.DataFrame,
    dataset_scores,
    strategy: str,
    metric: str,
    percent: int,
    max_votes_per_pair: int,
    random_seed: int,
    require_complete_scores: bool,
    model_name: str,
) -> float:
    sampled = sample_strategy_votes(
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
        metric,
        require_complete_scores=require_complete_scores,
    )


def compute_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    experiment_dir = args.experiment_dir.resolve()
    config = load_experiment_config(experiment_dir)
    dataset_scores = load_scores(config.score_file)
    votes = load_votes(config.strategies)

    rows: list[dict[str, object]] = []
    for strategy in config.strategies:
        max_votes_per_pair = strategy.max_votes_per_pair or args.max_votes_per_pair
        for percent in args.percent:
            if percent <= 0 or percent > 100:
                raise ValueError("--percent values must be in the 1..100 range.")

            sample_count = 1 if percent == 100 else max(1, args.sample_count)
            seeds = [
                args.random_key + percent * 1_000_003 + index * 97
                for index in range(sample_count)
            ]
            for metric in args.metrics:
                values = Parallel(n_jobs=args.workers)(
                    delayed(compute_one)(
                        votes,
                        dataset_scores,
                        strategy.name,
                        metric,
                        percent,
                        max_votes_per_pair,
                        seed,
                        config.require_complete_scores,
                        args.model,
                    )
                    for seed in seeds
                )
                arr = np.asarray(values, dtype=float)
                rows.append(
                    {
                        "strategy": strategy.name,
                        "model": args.model,
                        "metric": metric,
                        "percent": percent,
                        "votes": percent_to_total_votes(
                            dataset_scores,
                            min(percent, 100),
                            max_votes_per_pair,
                        ),
                        "mean": float(np.nanmean(arr)),
                        "std": float(np.nanstd(arr)),
                        "samples": sample_count,
                    }
                )

    return rows


def main() -> None:
    args = parse_args()
    rows = compute_rows(args)
    df = pd.DataFrame(rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)

    print(df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
