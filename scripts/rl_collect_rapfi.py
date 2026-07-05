from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rapfi import RapfiConfig
from renju_benchmark.rl.datasets import collect_rapfi_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Rapfi best-move imitation examples.")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rapfi-path", type=Path)
    parser.add_argument("--rapfi-cwd", type=Path)
    parser.add_argument("--move-timeout", type=float, default=10.0)
    args = parser.parse_args()

    config = None
    if args.rapfi_path is not None:
        config = RapfiConfig(executable=args.rapfi_path, cwd=args.rapfi_cwd, move_timeout=args.move_timeout)
    collect_rapfi_examples(count=args.count, output=args.output, config=config, seed=args.seed)
    print(f"wrote {args.count} Rapfi examples to {args.output}")


if __name__ == "__main__":
    main()

