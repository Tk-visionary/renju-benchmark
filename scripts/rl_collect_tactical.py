from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rules import BLACK, format_coord
from renju_benchmark.rl.board_encoding import coord_to_index
from renju_benchmark.rl.datasets import random_reachable_position
from renju_benchmark.rl.search import tactical_heuristic_move


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect tactical-heuristic imitation examples.")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=30)
    parser.add_argument("--candidate-limit", type=int, default=32)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    attempts = 0
    max_attempts = max(args.count * 10, args.count)
    while len(rows) < args.count and attempts < max_attempts:
        attempts += 1
        board, turn, _history = random_reachable_position(rng, rng.randint(args.min_plies, args.max_plies))
        try:
            move = tactical_heuristic_move(board, turn, limit=args.candidate_limit)
        except ValueError:
            continue
        coord = format_coord(*move)
        rows.append(json.dumps({
            "board": board.to_text(),
            "side": "black" if turn == BLACK else "white",
            "best_move": coord,
            "tactical_best": coord,
            "policy_index": coord_to_index(coord),
            "source": "tactical_heuristic_move",
        }))

    if len(rows) < args.count:
        raise RuntimeError(f"collected {len(rows)} examples after {attempts} attempts; requested {args.count}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + "\n")
    print(f"wrote {args.count} tactical examples to {args.output}")


if __name__ == "__main__":
    main()
