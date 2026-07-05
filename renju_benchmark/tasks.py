from __future__ import annotations

import json
import re

from .agents import heuristic_move
from .rules import (
    BLACK,
    WHITE,
    Board,
    MoveResult,
    RenjuGame,
    RuleMode,
    format_coord,
    parse_coord,
    would_make_exact_five,
    would_make_overline,
)

COORD_RE = re.compile(r"\b([A-Oa-o](?:1[0-5]|[1-9]))\b")
ANY_COORD_RE = re.compile(r"\b([A-Za-z])([0-9]{1,2})\b")
LABELS = {"legal", "win", "forbidden", "occupied", "off_board"}


def parse_coord_relaxed(coord: str) -> tuple[int, int]:
    match = ANY_COORD_RE.fullmatch(coord.strip())
    if not match:
        raise ValueError(f"bad coordinate: {coord}")
    col = ord(match.group(1).lower()) - ord("a")
    row = int(match.group(2)) - 1
    return row, col


def extract_first_coord(text: str) -> tuple[int, int]:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and isinstance(payload.get("move"), str):
            return parse_coord_relaxed(payload["move"])
    except json.JSONDecodeError:
        pass
    match = ANY_COORD_RE.search(text)
    if not match:
        raise ValueError("model response did not contain a coordinate")
    return parse_coord_relaxed(match.group(0))


def extract_label(text: str) -> str:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and isinstance(payload.get("class"), str):
            label = payload["class"].strip().lower().replace("-", "_")
            if label in LABELS:
                return label
    except json.JSONDecodeError:
        pass
    normalized = text.strip().lower().replace("-", "_")
    previous = ""
    for token in re.findall(r"[a-z_]+", normalized):
        if token in LABELS:
            if previous in {"not", "non", "no"}:
                previous = token
                continue
            return token
        previous = token
    raise ValueError("model response did not contain a valid class label")


def _opponent(color: str) -> str:
    return WHITE if color == BLACK else BLACK


def _immediate_winning_moves(board: Board, color: str) -> set[tuple[int, int]]:
    wins = set()
    for row, col in board.neighbor_empty_points(radius=2):
        if would_make_exact_five(board, row, col, color):
            wins.add((row, col))
        elif color == WHITE and would_make_overline(board, row, col, WHITE):
            wins.add((row, col))
    return wins


def score_candidate_move(
    board: Board,
    color: str,
    move: tuple[int, int],
    best_moves: set[tuple[int, int]],
    good_moves: set[tuple[int, int]] | None = None,
    blocking_moves: set[tuple[int, int]] | None = None,
    forbidden_moves: set[tuple[int, int]] | None = None,
    mode: RuleMode | str = RuleMode.STRICT,
) -> float:
    good_moves = good_moves or set()
    blocking_moves = blocking_moves or set()
    forbidden_moves = forbidden_moves or set()
    if move in forbidden_moves:
        return 0.0
    game = RenjuGame.from_board(board=board, turn=color, mode=mode)
    result = game.play(*move)
    if result in (MoveResult.ILLEGAL_OCCUPIED, MoveResult.ILLEGAL_OFF_BOARD, MoveResult.BLACK_FORBIDDEN):
        return 0.0
    if result in (MoveResult.BLACK_WIN, MoveResult.WHITE_WIN):
        return 1.0 if not best_moves or move in best_moves else 0.75
    if move in best_moves:
        return 1.0
    if move in good_moves:
        return 0.80
    if move in blocking_moves:
        return 0.75

    opponent = _opponent(color)
    before_opponent_wins = _immediate_winning_moves(board, opponent)
    after_opponent_wins = _immediate_winning_moves(game.board, opponent)
    creates_threat = bool(_immediate_winning_moves(game.board, color))

    if after_opponent_wins:
        return 0.20
    if before_opponent_wins or creates_threat:
        return 0.75

    reply = heuristic_move(game.board, opponent)
    reply_game = RenjuGame.from_board(board=game.board, turn=opponent, mode=RuleMode.FAST)
    reply_result = reply_game.play(*reply)
    if reply_result in (MoveResult.BLACK_WIN, MoveResult.WHITE_WIN):
        return 0.20
    return 0.50


try:
    import kaggle_benchmarks as kbench
except Exception:  # pragma: no cover - Kaggle-only dependency
    kbench = None


if kbench is not None:

    @kbench.task(name="renju_next_move")
    def renju_next_move(
        llm,
        board_text: str,
        side: str,
        best_moves: list[str],
        good_moves: list[str] | None = None,
        blocking_moves: list[str] | None = None,
        forbidden_moves: list[str] | None = None,
        mode: str = "strict",
    ) -> float:
        board = Board.from_text(board_text)
        color = BLACK if side.lower() in ("black", "x") else WHITE
        prompt = (
            "You are playing Renju on a 15x15 board. Coordinates are A1 through O15. "
            "X is black and O is white. Black loses by forbidden overline, double-four, "
            "or forbidden double-three unless black simultaneously makes exactly five. "
            f"It is {side}'s turn. Return exactly one JSON object like {{\"move\":\"H8\"}}.\n\n{board.to_text()}"
        )
        response = llm.prompt(prompt)
        try:
            move = extract_first_coord(response)
        except ValueError:
            return 0.0
        answers = {parse_coord(coord) for coord in best_moves}
        good = {parse_coord(coord) for coord in (good_moves or [])}
        blocking = {parse_coord(coord) for coord in (blocking_moves or [])}
        forbidden = {parse_coord_relaxed(coord) for coord in (forbidden_moves or [])}
        return score_candidate_move(board, color, move, answers, good, blocking, forbidden, mode=mode)


    @kbench.task(name="renju_rule_classification")
    def renju_rule_classification(llm, board_text: str, side: str, move: str, expected: str) -> bool:
        board = Board.from_text(board_text)
        prompt = (
            "Classify this Renju move as one of: legal, win, forbidden, occupied, off_board. "
            "Use RIF rules: white overline wins, black overline/double-four/forbidden double-three lose "
            "unless black simultaneously makes exactly five.\n"
            f"Side: {side}\nMove: {move}\nBoard:\n{board.to_text()}\n"
            "Return exactly one JSON object like {\"class\":\"legal\"}."
        )
        response = llm.prompt(prompt)
        try:
            label = extract_label(response)
        except ValueError:
            return False
        return label == expected.lower().replace("-", "_")


def describe_move(board_text: str, side: str, move: str) -> str:
    board = Board.from_text(board_text)
    color = BLACK if side.lower() in ("black", "x") else WHITE
    row, col = parse_coord_relaxed(move)
    game = RenjuGame(board=board, turn=color)
    result = game.play(row, col)
    coord = format_coord(row, col) if board.in_bounds(row, col) else move.strip().upper()
    return f"{coord}: {result.value}"
