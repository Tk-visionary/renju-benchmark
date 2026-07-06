from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rl.self_play import SelfPlayConfig, random_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a random HRM checkpoint for masked legal self-play.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--hrm-cycles", type=int, default=1)
    parser.add_argument("--hrm-low-steps", type=int, default=1)
    args = parser.parse_args()

    config = SelfPlayConfig(
        channels=args.channels,
        hrm_cycles=args.hrm_cycles,
        hrm_low_steps=args.hrm_low_steps,
        seed=args.seed,
    )
    random_checkpoint(args.output, config)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()[:16]
    print(f"wrote random HRM checkpoint to {args.output}")
    print(f"checkpoint_hash={digest}")


if __name__ == "__main__":
    main()
