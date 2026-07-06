from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rules import Board, format_coord
from renju_benchmark.rl.inference import PolicyValueAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose one move with a trained policy/value checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("board", type=Path)
    parser.add_argument("--side", choices=["black", "white"], default="black")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tactical", action="store_true")
    parser.add_argument("--candidate-limit", type=int, default=32)
    args = parser.parse_args()

    board = Board.from_text(args.board.read_text())
    agent = PolicyValueAgent(args.checkpoint, device=args.device)
    move = (
        agent.tactical_move(board, args.side, limit=args.candidate_limit)
        if args.tactical
        else agent.move(board, args.side)
    )
    print(format_coord(*move))


if __name__ == "__main__":
    main()
