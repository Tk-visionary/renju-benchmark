from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rl.self_play import SelfPlayConfig, collect_self_play


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect masked-legal HRM self-play examples.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--hrm-cycles", type=int, default=1)
    parser.add_argument("--hrm-low-steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--max-plies", type=int, default=225)
    parser.add_argument("--forbidden-depth", type=int, default=2)
    parser.add_argument("--local-legal-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parallel-games", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = SelfPlayConfig(
        channels=args.channels,
        hrm_cycles=args.hrm_cycles,
        hrm_low_steps=args.hrm_low_steps,
        temperature=args.temperature,
        epsilon=args.epsilon,
        max_plies=args.max_plies,
        forbidden_depth=args.forbidden_depth,
        local_legal_only=args.local_legal_only,
        parallel_games=args.parallel_games,
        seed=args.seed,
    )
    rows = collect_self_play(args.checkpoint, args.output, args.games, config, device=args.device)
    results: dict[str, int] = {}
    for row in rows:
        results[row["game_result"]] = results.get(row["game_result"], 0) + 1
    print(json.dumps({
        "games": args.games,
        "positions": len(rows),
        "output": str(args.output),
        "position_results": results,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
