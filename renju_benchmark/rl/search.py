from __future__ import annotations

from renju_benchmark.agents import legal_moves
from renju_benchmark.rules import BLACK, WHITE, Board, black_forbidden_report, would_make_exact_five


def tactical_candidates(board: Board, color: str, radius: int = 2, limit: int = 32) -> list[tuple[int, int]]:
    moves = legal_moves(board, color, local_only=True, forbidden_depth=2)
    opponent = WHITE if color == BLACK else BLACK
    wins = [move for move in moves if would_make_exact_five(board, *move, color)]
    if wins:
        return wins[:limit]

    blocks = []
    for row, col in moves:
        if opponent == BLACK and black_forbidden_report(board, row, col, _depth=0).forbidden:
            continue
        if would_make_exact_five(board, row, col, opponent):
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
