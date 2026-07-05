from __future__ import annotations

import random

from .rules import BLACK, WHITE, Board, black_forbidden_report, format_coord, would_make_exact_five


def candidate_moves(board: Board, radius: int = 2) -> list[tuple[int, int]]:
    return board.neighbor_empty_points(radius)


def legal_moves(
    board: Board,
    color: str,
    local_only: bool = False,
    forbidden_depth: int = 0,
) -> list[tuple[int, int]]:
    moves = []
    points = candidate_moves(board) if local_only else board.empty_points()
    for row, col in points:
        if color == BLACK and black_forbidden_report(board, row, col, _depth=forbidden_depth).forbidden:
            continue
        moves.append((row, col))
    return moves


def heuristic_move(board: Board, color: str) -> tuple[int, int]:
    moves = legal_moves(board, color, local_only=True, forbidden_depth=0)
    if not moves:
        raise ValueError("no legal moves")
    opponent = WHITE if color == BLACK else BLACK

    for row, col in moves:
        if would_make_exact_five(board, row, col, color):
            return row, col
    for row, col in moves:
        if opponent == BLACK and black_forbidden_report(board, row, col, _depth=0).forbidden:
            continue
        if would_make_exact_five(board, row, col, opponent):
            return row, col

    center = (7, 7)
    return min(moves, key=lambda move: abs(move[0] - center[0]) + abs(move[1] - center[1]))


def random_move(board: Board, color: str, seed: int | None = None) -> tuple[int, int]:
    rng = random.Random(seed)
    moves = legal_moves(board, color, local_only=True, forbidden_depth=0)
    if not moves:
        raise ValueError("no legal moves")
    return rng.choice(moves)


def move_to_text(move: tuple[int, int]) -> str:
    return format_coord(*move)
