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

from user_info import plot_all_user_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build user-info figures from an assessor table.")
    parser.add_argument("user_info_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "user_info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = plot_all_user_info(args.user_info_csv.resolve(), args.output_dir.resolve())
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
