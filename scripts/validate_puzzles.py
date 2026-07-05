from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.tasks import _immediate_winning_moves, parse_coord_relaxed
from renju_benchmark.rules import BLACK, WHITE, Board, MoveResult, RenjuGame, RuleMode

VALID_MODES = {"fast", "strict", "puzzle"}


def color_from_side(side: str) -> str:
    return BLACK if side.lower() in {"black", "x"} else WHITE


def classify_result(result: MoveResult) -> str:
    if result in (MoveResult.BLACK_WIN, MoveResult.WHITE_WIN):
        return "win"
    if result == MoveResult.BLACK_FORBIDDEN:
        return "forbidden"
    if result == MoveResult.ILLEGAL_OCCUPIED:
        return "occupied"
    if result == MoveResult.ILLEGAL_OFF_BOARD:
        return "off_board"
    return "legal"


def play_strict(board: Board, side: str, coord: str) -> tuple[MoveResult, Board]:
    row, col = parse_coord_relaxed(coord)
    game = RenjuGame.from_board(board, turn=color_from_side(side), mode=RuleMode.STRICT)
    result = game.play(row, col)
    return result, game.board


def validate_record(record: dict) -> None:
    board = Board.from_text(record["board"])
    side = color_from_side(record["side"])
    tags = set(record.get("tags", []))
    mode = record.get("mode")
    if mode is not None and str(mode).lower() not in VALID_MODES:
        raise ValueError(f"{record['id']}: invalid mode {mode}")

    if record.get("track") == "rule_classification":
        result, _ = play_strict(board, record["side"], record["move"])
        expected = record["expected"].replace("-", "_")
        actual = classify_result(result)
        if actual != expected:
            raise ValueError(f"{record['id']}: expected {expected}, got {actual}")
        if "off_board" in tags and actual != "off_board":
            raise ValueError(f"{record['id']}: off_board tag expected off_board, got {actual}")
        if "occupied" in tags and actual != "occupied":
            raise ValueError(f"{record['id']}: occupied tag expected occupied, got {actual}")
        if "win" in tags and actual != "win":
            raise ValueError(f"{record['id']}: win tag expected win, got {actual}")
        if "forbidden" in tags and actual != "forbidden":
            raise ValueError(f"{record['id']}: forbidden tag expected forbidden, got {actual}")
        return

    if "must_block" in tags:
        opponent = WHITE if side == BLACK else BLACK
        if not _immediate_winning_moves(board, opponent):
            raise ValueError(f"{record['id']}: must_block tagged but opponent had no immediate wins")

    for field in ("best_moves", "good_moves", "blocking_moves", "forbidden_moves"):
        for coord in record.get(field, []):
            result, next_board = play_strict(board, record["side"], coord)
            if field == "forbidden_moves" and result != MoveResult.BLACK_FORBIDDEN:
                raise ValueError(f"{record['id']}: {coord} expected forbidden, got {result.value}")
            if field != "forbidden_moves" and result in (
                MoveResult.ILLEGAL_OCCUPIED,
                MoveResult.ILLEGAL_OFF_BOARD,
                MoveResult.BLACK_FORBIDDEN,
            ):
                raise ValueError(f"{record['id']}: {coord} expected playable, got {result.value}")
            if "win" in tags and field == "best_moves" and result not in (MoveResult.BLACK_WIN, MoveResult.WHITE_WIN):
                raise ValueError(f"{record['id']}: {coord} tagged win but got {result.value}")
            if "must_block" in tags and field in {"best_moves", "blocking_moves"}:
                opponent = WHITE if side == BLACK else BLACK
                if _immediate_winning_moves(next_board, opponent):
                    raise ValueError(f"{record['id']}: {coord} did not remove immediate opponent wins")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    count = 0
    for line_no, line in enumerate(args.path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            validate_record(json.loads(line))
        except Exception as exc:
            raise SystemExit(f"{args.path}:{line_no}: {exc}") from exc
        count += 1
    print(f"validated {count} puzzles")


if __name__ == "__main__":
    main()
