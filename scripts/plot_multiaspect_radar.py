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

from multiaspect import aggregate_by_method, load_multiaspect_scores, plot_combined_radar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a radar chart from a multiaspect score table.")
    parser.add_argument("multiaspect_csv", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "multiaspect_radar_Maxwell.png")
    parser.add_argument("--test-case", type=str, default=None)
    parser.add_argument("--max-methods", type=int, default=0, help="Maximum methods to draw; 0 draws all.")
    parser.add_argument(
        "--selection",
        choices=["top", "diverse", "diverse-top"],
        default="diverse",
        help="Method selection policy when --max-methods is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_multiaspect_scores(args.multiaspect_csv.resolve(), test_case=args.test_case)
    means = aggregate_by_method(df)
    plot_combined_radar(
        means_df=means,
        output_path=args.output.resolve(),
        max_methods=args.max_methods,
        selection=args.selection,
    )
    print(args.output)


if __name__ == "__main__":
    main()
