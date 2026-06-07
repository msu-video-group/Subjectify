#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / ".matplotlib-cache"
MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import load_experiment_config, load_scores, load_votes
from models import MODEL_FUNCTIONS
from plotting import plot_convergence
from sampling import sample_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one convergence plot from one experiment directory.")
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--model", choices=sorted(MODEL_FUNCTIONS), default="bradley-terry")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--random-key", type=int, default=42)
    parser.add_argument("--max-votes-per-pair", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    config = load_experiment_config(experiment_dir)
    scores = load_scores(config.score_file)
    votes = load_votes(config.strategies)

    results = sample_experiment(
        votes=votes,
        dataset_scores=scores,
        strategies=[strategy.name for strategy in config.strategies],
        sample_count=args.sample_count,
        workers=args.workers,
        random_key=args.random_key,
        max_votes_per_pair=args.max_votes_per_pair,
        model_name=args.model,
        require_complete_scores=config.require_complete_scores,
        strategy_max_votes_per_pair={
            strategy.name: strategy.max_votes_per_pair
            for strategy in config.strategies
            if strategy.max_votes_per_pair is not None
        },
    )
    output_path = args.output_dir / config.output_file
    plot_convergence(
        results=results,
        strategies=config.strategies,
        output_path=output_path,
        title=config.title,
        show_title=config.show_title,
    )
    for strategy in config.strategies:
        result = results[strategy.name]
        print(f"{strategy.name}: final_srocc={result.srocc[-1]:.3f}")
    print(output_path)


if __name__ == "__main__":
    main()
