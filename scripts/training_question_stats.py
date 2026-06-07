#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"participant", "test_case", "passed_first_try", "replay_count"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print statistics for training-question tables.")
    parser.add_argument("training_questions_csv", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path for the statistics table.")
    return parser.parse_args()


def load_training_questions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} misses columns: {sorted(missing)}")
    df = df.loc[:, ["participant", "test_case", "passed_first_try", "replay_count"]].copy()
    df["participant"] = df["participant"].astype(str)
    df["test_case"] = df["test_case"].astype(str)
    df["passed_first_try"] = df["passed_first_try"].astype(bool)
    df["replay_count"] = pd.to_numeric(df["replay_count"], errors="coerce").fillna(0).astype(int)
    return df


def compute_stats(path: Path) -> pd.DataFrame:
    df = load_training_questions(path)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "test_case",
                "rows",
                "participants",
                "passed_first_try_percent",
                "mean_replay_count",
                "max_replay_count",
            ]
        )

    stats = (
        df.groupby("test_case", as_index=False)
        .agg(
            rows=("participant", "size"),
            participants=("participant", "nunique"),
            passed_first_try_percent=("passed_first_try", lambda s: float(s.mean() * 100.0)),
            mean_replay_count=("replay_count", "mean"),
            max_replay_count=("replay_count", "max"),
        )
        .sort_values("test_case", key=lambda s: s.str.extract(r"(\d+)", expand=False).astype(int))
        .reset_index(drop=True)
    )
    stats.insert(0, "source", path.parent.name)
    return stats


def main() -> None:
    args = parse_args()
    frames = [compute_stats(path.resolve()) for path in args.training_questions_csv]
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)

    if result.empty:
        print("No training-question rows found.")
        return

    print(
        result.to_string(
            index=False,
            formatters={
                "passed_first_try_percent": lambda value: f"{value:.2f}",
                "mean_replay_count": lambda value: f"{value:.2f}",
            },
        )
    )


if __name__ == "__main__":
    main()
