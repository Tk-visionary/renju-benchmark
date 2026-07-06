from __future__ import annotations

from dataclasses import dataclass

from renju_benchmark.agents import legal_moves
from renju_benchmark.rules import BLACK, WHITE, Board, black_forbidden_report, would_make_exact_five


@dataclass(frozen=True)
class TacticalMove:
    move: tuple[int, int]
    role: str


def opponent(color: str) -> str:
    return WHITE if color == BLACK else BLACK


def winning_moves(board: Board, color: str, forbidden_depth: int = 2) -> list[tuple[int, int]]:
    return [
        move
        for move in legal_moves(board, color, local_only=True, forbidden_depth=forbidden_depth)
        if would_make_exact_five(board, *move, color)
    ]


def opponent_winning_replies(board: Board, color: str, move: tuple[int, int]) -> list[tuple[int, int]]:
    next_board = board.place(*move, color)
    return winning_moves(next_board, opponent(color), forbidden_depth=0)


def tactical_candidates(board: Board, color: str, radius: int = 2, limit: int = 32) -> list[tuple[int, int]]:
    moves = legal_moves(board, color, local_only=True, forbidden_depth=2)
    other = opponent(color)
    wins = [move for move in moves if would_make_exact_five(board, *move, color)]
    if wins:
        return wins[:limit]

    blocks = []
    for row, col in moves:
        if other == BLACK and black_forbidden_report(board, row, col, _depth=0).forbidden:
            continue
        if would_make_exact_five(board, row, col, other):
            blocks.append((row, col))

    center = (7, 7)
    ordered = blocks + sorted(
        board.neighbor_empty_points(radius),
        key=lambda move: abs(move[0] - center[0]) + abs(move[1] - center[1]),
    )
    seen = set()
    result = []
    legal = set(moves)
    for move in ordered:
        if move in seen or move not in legal:
            continue
        seen.add(move)
        result.append(move)
        if len(result) >= limit:
            break
    return result


def tactical_candidates_with_roles(
    board: Board,
    color: str,
    radius: int = 2,
    limit: int = 32,
) -> list[TacticalMove]:
    moves = tactical_candidates(board, color, radius=radius, limit=max(limit * 2, limit))
    immediate_wins = set(winning_moves(board, color))
    if immediate_wins:
        return [TacticalMove(move, "win") for move in moves if move in immediate_wins][:limit]

    other = opponent(color)
    blocks = set()
    for move in moves:
        if other == BLACK and black_forbidden_report(board, *move, _depth=0).forbidden:
            continue
        if would_make_exact_five(board, *move, other):
            blocks.add(move)

    tactical: list[TacticalMove] = []
    fallback: list[TacticalMove] = []
    for move in moves:
        role = "block" if move in blocks else "safe"
        if opponent_winning_replies(board, color, move):
            role = "unsafe"
        item = TacticalMove(move, role)
        if role == "unsafe":
            fallback.append(item)
        else:
            tactical.append(item)
        if len(tactical) >= limit:
            return tactical

    return (tactical + fallback)[:limit]
