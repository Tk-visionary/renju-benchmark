from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.agents import heuristic_move
from renju_benchmark.rapfi import RapfiConfig
from renju_benchmark.rules import BLACK, WHITE
from renju_benchmark.rl.rapfi_env import play_vs_rapfi


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a baseline move function against Rapfi.")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--rapfi-path", type=Path)
    parser.add_argument("--rapfi-cwd", type=Path)
    parser.add_argument("--move-timeout", type=float, default=10.0)
    parser.add_argument("--timeout-turn-ms", type=int, default=100)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--max-node", type=int, default=1000)
    args = parser.parse_args()

    config = None
    if args.rapfi_path is not None:
        config = RapfiConfig(
            executable=args.rapfi_path,
            cwd=args.rapfi_cwd,
            move_timeout=args.move_timeout,
            timeout_turn_ms=args.timeout_turn_ms,
            max_depth=args.max_depth,
            max_node=args.max_node,
        )

    rows = []
    for game_index in range(args.games):
        color = BLACK if game_index % 2 == 0 else WHITE
        result = play_vs_rapfi(
            lambda board, side: heuristic_move(board, side),
            model_color=color,
            config=config,
            max_plies=args.max_plies,
        )
        rows.append({
            "game": game_index,
            "model_color": "black" if color == BLACK else "white",
            "result": result.result,
            "winner": result.winner,
            "moves": result.moves,
        })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
