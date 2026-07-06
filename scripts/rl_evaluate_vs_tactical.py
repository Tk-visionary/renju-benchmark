from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rules import BLACK, WHITE
from renju_benchmark.rl.inference import PolicyValueAgent
from renju_benchmark.rl.rapfi_env import play_match, summarize_game_rows
from renju_benchmark.rl.search import tactical_heuristic_move


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a policy checkpoint against the tactical heuristic baseline.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--policy-tactical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--games-only", action="store_true")
    args = parser.parse_args()

    agent = PolicyValueAgent(args.checkpoint, device=args.device)

    def policy_move(board, side):
        if args.policy_tactical:
            return agent.tactical_move(board, side, limit=args.candidate_limit)
        return agent.move(board, side)

    def baseline_move(board, side):
        return tactical_heuristic_move(board, side, limit=args.candidate_limit)

    rows = []
    for game_index in range(args.games):
        model_color = BLACK if game_index % 2 == 0 else WHITE
        if model_color == BLACK:
            result = play_match(policy_move, baseline_move, max_plies=args.max_plies)
        else:
            result = play_match(baseline_move, policy_move, max_plies=args.max_plies)
        rows.append({
            "game": game_index,
            "agent": "policy_value_tactical" if args.policy_tactical else "policy_value",
            "opponent": "tactical_heuristic",
            "model_color": "black" if model_color == BLACK else "white",
            "result": result.result,
            "winner": result.winner,
            "moves": result.moves,
        })

    output = rows if args.games_only else {"summary": summarize_game_rows(rows), "games": rows}
    print(json.dumps(output, indent=2, sort_keys=not args.games_only))


if __name__ == "__main__":
    main()
