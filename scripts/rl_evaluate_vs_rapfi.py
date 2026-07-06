from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.agents import heuristic_move
from renju_benchmark.rapfi import RapfiConfig
from renju_benchmark.rules import BLACK, WHITE
from renju_benchmark.rl.inference import PolicyValueAgent
from renju_benchmark.rl.rapfi_env import play_vs_rapfi, summarize_game_rows
from renju_benchmark.rl.search import tactical_heuristic_move


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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tactical", action="store_true")
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--reuse-rapfi-process", action="store_true")
    parser.add_argument("--games-only", action="store_true")
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
    agent = PolicyValueAgent(args.checkpoint, device=args.device) if args.checkpoint else None

    rows = []
    for game_index in range(args.games):
        color = BLACK if game_index % 2 == 0 else WHITE
        if agent is None and args.tactical:
            def move_fn(board, side):
                return tactical_heuristic_move(board, side, limit=args.candidate_limit)
        elif agent is None:
            move_fn = heuristic_move
        elif args.tactical:
            def move_fn(board, side):
                return agent.tactical_move(board, side, limit=args.candidate_limit)
        else:
            move_fn = agent.move
        result = play_vs_rapfi(
            move_fn,
            model_color=color,
            config=config,
            max_plies=args.max_plies,
            fresh_rapfi_per_move=not args.reuse_rapfi_process,
        )
        rows.append({
            "game": game_index,
            "agent": agent_name(agent is not None, args.tactical),
            "model_color": "black" if color == BLACK else "white",
            "result": result.result,
            "winner": result.winner,
            "moves": result.moves,
        })
    output = rows if args.games_only else {"summary": summarize_game_rows(rows), "games": rows}
    print(json.dumps(output, indent=2, sort_keys=not args.games_only))


def agent_name(has_checkpoint: bool, tactical: bool) -> str:
    if has_checkpoint and tactical:
        return "policy_value_tactical"
    if has_checkpoint:
        return "policy_value"
    if tactical:
        return "tactical_heuristic"
    return "heuristic"


if __name__ == "__main__":
    main()
