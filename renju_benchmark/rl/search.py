from __future__ import annotations

from dataclasses import dataclass

from renju_benchmark.agents import legal_moves
from renju_benchmark.rules import BLACK, WHITE, Board, black_forbidden_report, would_make_exact_five


@dataclass(frozen=True)
class TacticalMove:
    move: tuple[int, int]
    role: str


ROLE_ORDER = ("win", "block", "force_win", "threat", "safe", "unsafe")


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


def winning_threat_count(board: Board, color: str, move: tuple[int, int], forbidden_depth: int = 2) -> int:
    next_board = board.place(*move, color)
    return len(winning_moves(next_board, color, forbidden_depth=forbidden_depth))


def opponent_force_win_replies(
    board: Board,
    color: str,
    move: tuple[int, int],
    limit: int = 64,
) -> list[tuple[int, int]]:
    next_board = board.place(*move, color)
    other = opponent(color)
    replies = []
    center = (7, 7)
    candidate_replies = sorted(
        legal_moves(next_board, other, local_only=True, forbidden_depth=0),
        key=lambda item: abs(item[0] - center[0]) + abs(item[1] - center[1]),
    )
    for reply in candidate_replies[:limit]:
        if winning_threat_count(next_board, other, reply, forbidden_depth=0) >= 2:
            replies.append(reply)
    return replies


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
    force_reply_limit: int = 16,
    threat_forbidden_depth: int = 2,
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

    by_role: dict[str, list[TacticalMove]] = {
        "block": [],
        "force_win": [],
        "threat": [],
        "safe": [],
        "unsafe": [],
    }
    for move in moves:
        role = "block" if move in blocks else "safe"
        if role != "block":
            if opponent_winning_replies(board, color, move) or opponent_force_win_replies(
                board,
                color,
                move,
                limit=force_reply_limit,
            ):
                role = "unsafe"
            else:
                threats = winning_threat_count(board, color, move, forbidden_depth=threat_forbidden_depth)
                if threats >= 2:
                    role = "force_win"
                elif threats == 1:
                    role = "threat"
        item = TacticalMove(move, role)
        by_role[role].append(item)

    ordered = by_role["block"] + by_role["force_win"] + by_role["threat"] + by_role["safe"] + by_role["unsafe"]
    return ordered[:limit]


def tactical_heuristic_move(
    board: Board,
    color: str,
    limit: int = 32,
    force_reply_limit: int = 16,
    threat_forbidden_depth: int = 2,
) -> tuple[int, int]:
    candidates = tactical_candidates_with_roles(
        board,
        color,
        limit=limit,
        force_reply_limit=force_reply_limit,
        threat_forbidden_depth=threat_forbidden_depth,
    )
    if not candidates:
        raise ValueError("no tactical candidates")
    center = (7, 7)
    for role in ROLE_ORDER:
        role_moves = [item.move for item in candidates if item.role == role]
        if role_moves:
            return min(role_moves, key=lambda move: abs(move[0] - center[0]) + abs(move[1] - center[1]))
    return candidates[0].move
