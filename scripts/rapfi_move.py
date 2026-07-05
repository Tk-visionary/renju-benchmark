from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rapfi import RapfiConfig, rapfi_move
from renju_benchmark.rules import BLACK, WHITE, Board, format_coord


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("board", type=Path, help="15-line board text file using '.', 'X', 'O'")
    parser.add_argument("--side", choices=["black", "white"], default="black")
    parser.add_argument("--rapfi", default=None, help="Path to Rapfi-compatible executable. Defaults to RAPFI_PATH.")
    parser.add_argument("--cwd", default=None, help="Working directory for Rapfi, useful for config/weights.")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    board = Board.from_text(args.board.read_text())
    color = BLACK if args.side == "black" else WHITE
    config = RapfiConfig.from_env() if args.rapfi is None else RapfiConfig(args.rapfi, cwd=args.cwd, move_timeout=args.timeout)
    row, col = rapfi_move(board, color, config)
    print(format_coord(row, col))


if __name__ == "__main__":
    main()

