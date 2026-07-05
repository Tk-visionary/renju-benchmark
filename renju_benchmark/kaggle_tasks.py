from __future__ import annotations

import json

from .rules import BLACK, WHITE, Board, parse_coord
from .tasks import extract_first_coord, extract_label, parse_coord_relaxed, score_candidate_move

try:
    import kaggle_benchmarks as kbench
except Exception:  # pragma: no cover - Kaggle-only dependency
    kbench = None


def _side_color(side: str) -> str:
    return BLACK if side.lower() in ("black", "x") else WHITE


def _next_move_prompt(board: Board, side: str) -> str:
    return (
        "You are playing Renju on a 15x15 board. Coordinates are A1 through O15. "
        "X is black and O is white. Black loses by forbidden overline, double-four, "
        "or forbidden double-three unless black simultaneously makes exactly five. "
        f"It is {side}'s turn. Return exactly one JSON object like {{\"move\":\"H8\"}}.\n\n{board.to_text()}"
    )


def _rule_prompt(board: Board, side: str, move: str) -> str:
    return (
        "Classify this Renju move as one of: legal, win, forbidden, occupied, off_board. "
        "Use RIF rules: white overline wins, black overline/double-four/forbidden double-three lose "
        "unless black simultaneously makes exactly five.\n"
        f"Side: {side}\nMove: {move}\nBoard:\n{board.to_text()}\n"
        'Return exactly one JSON object like {"class":"legal"}.'
    )


def score_next_move_response(
    response: str,
    board_text: str,
    side: str,
    best_moves: list,
    good_moves: list | None = None,
    blocking_moves: list | None = None,
    forbidden_moves: list | None = None,
    mode: str = "strict",
) -> float:
    board = Board.from_text(board_text)
    try:
        move = extract_first_coord(response)
    except ValueError:
        return 0.0
    return score_candidate_move(
        board,
        _side_color(side),
        move,
        {parse_coord(coord) for coord in best_moves},
        {parse_coord(coord) for coord in (good_moves or [])},
        {parse_coord(coord) for coord in (blocking_moves or [])},
        {parse_coord_relaxed(coord) for coord in (forbidden_moves or [])},
        mode=mode,
    )


def score_rule_response(response: str, expected: str) -> bool:
    try:
        label = extract_label(response)
    except ValueError:
        return False
    return label == expected.lower().replace("-", "_")


if kbench is not None:

    @kbench.task(name="renju_next_move")
    def renju_next_move(
        llm,
        board_text: str,
        side: str,
        best_moves: list,
        good_moves: list = None,
        blocking_moves: list = None,
        forbidden_moves: list = None,
        mode: str = "strict",
    ) -> float:
        board = Board.from_text(board_text)
        response = llm.prompt(_next_move_prompt(board, side))
        return score_next_move_response(
            response,
            board_text,
            side,
            best_moves,
            good_moves,
            blocking_moves,
            forbidden_moves,
            mode,
        )

    @kbench.task(name="renju_next_move_record")
    def renju_next_move_record(llm, record_json: str) -> float:
        record = json.loads(record_json)
        return renju_next_move(
            llm,
            record["board"],
            record["side"],
            record.get("best_moves", []),
            record.get("good_moves", []),
            record.get("blocking_moves", []),
            record.get("forbidden_moves", []),
            record.get("mode", "strict"),
        )

    @kbench.task(name="renju_rule_classification")
    def renju_rule_classification(llm, board_text: str, side: str, move: str, expected: str) -> bool:
        board = Board.from_text(board_text)
        response = llm.prompt(_rule_prompt(board, side, move))
        return score_rule_response(response, expected)

    @kbench.task(name="renju_rule_classification_record")
    def renju_rule_classification_record(llm, record_json: str) -> bool:
        record = json.loads(record_json)
        return renju_rule_classification(
            llm,
            record["board"],
            record["side"],
            record["move"],
            record["expected"],
        )

