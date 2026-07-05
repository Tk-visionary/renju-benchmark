# %%
import json
import re

import kaggle_benchmarks as kbench
import pandas as pd


# %%
BOARD_SIZE = 15
EMPTY = "."
BLACK = "X"
WHITE = "O"
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
COORD_RE = re.compile(r"\b([A-Za-z])([0-9]{1,2})\b")

OPENING_BOARD = "\n".join([
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
    ".......X.......",
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
])

EVALUATION_DATA = pd.DataFrame([
    {
        "match_id": "gemini-vs-claude-black",
        "black_model": "google/gemini-3-flash-preview",
        "white_model": "anthropic/claude-haiku-4-5@20251001",
        "board_text": OPENING_BOARD,
        "max_plies": 12,
    },
    {
        "match_id": "gemini-vs-claude-white",
        "black_model": "anthropic/claude-haiku-4-5@20251001",
        "white_model": "google/gemini-3-flash-preview",
        "board_text": OPENING_BOARD,
        "max_plies": 12,
    },
])


# %%
def parse_coord(text: str) -> tuple[int, int] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("move"), str):
        text = payload["move"]

    match = COORD_RE.search(text)
    if not match:
        return None
    col = ord(match.group(1).lower()) - ord("a")
    row = int(match.group(2)) - 1
    return row, col


def format_coord(row: int, col: int) -> str:
    return f"{chr(ord('A') + col)}{row + 1}"


def board_from_text(text: str) -> list[list[str]]:
    return [list(line.strip()) for line in text.strip().splitlines()]


def board_to_text(board: list[list[str]]) -> str:
    return "\n".join("".join(row) for row in board)


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def run_length(board: list[list[str]], row: int, col: int, color: str, dx: int, dy: int) -> int:
    total = 1
    for sign in (-1, 1):
        step = 1
        while True:
            rr = row + sign * dy * step
            cc = col + sign * dx * step
            if not in_bounds(rr, cc) or board[rr][cc] != color:
                break
            total += 1
            step += 1
    return total


def move_result(board: list[list[str]], row: int, col: int, color: str) -> str:
    if not in_bounds(row, col):
        return "illegal_off_board"
    if board[row][col] != EMPTY:
        return "illegal_occupied"

    board[row][col] = color
    lengths = [run_length(board, row, col, color, dx, dy) for dx, dy in DIRECTIONS]
    exact_five = any(length == 5 for length in lengths)
    overline = any(length >= 6 for length in lengths)
    if color == BLACK and overline and not exact_five:
        return "black_forbidden"
    if exact_five or (color == WHITE and overline):
        return "black_win" if color == BLACK else "white_win"
    if all(cell != EMPTY for board_row in board for cell in board_row):
        return "draw"
    return "ok"


def prompt_for_move(board: list[list[str]], side: str, match_id: str, ply: int) -> str:
    return (
        "You are playing Renju on a 15x15 board. Coordinates are A1 through O15. "
        "X is black and O is white. Black loses by forbidden overline; exact five wins. "
        "Return exactly one JSON object like {\"move\":\"H8\"} and no extra text.\n"
        f"Match: {match_id}\nPly: {ply}\nSide to move: {side}\nBoard:\n{board_to_text(board)}"
    )


def winner_from_illegal(color: str) -> str:
    return "white_win" if color == BLACK else "black_win"


def score_for_black(result: str) -> float:
    if result == "draw":
        return 0.5
    if result == "black_win":
        return 1.0
    if result == "white_win":
        return 0.0
    return 0.5


# %%
@kbench.task(name="renju-model-arena-public")
def renju_model_arena_public(
    llm,
    match_id: str,
    black_model: str,
    white_model: str,
    board_text: str,
    max_plies: int,
) -> dict:
    del llm
    players = {
        "black": kbench.llms[black_model],
        "white": kbench.llms[white_model],
    }
    board = board_from_text(board_text)
    moves = []
    result = "draw"

    for ply in range(max_plies):
        color = BLACK if ply % 2 == 0 else WHITE
        side = "black" if color == BLACK else "white"
        player = players[side]
        with kbench.chats.new(f"{match_id}-{side}-{ply}"):
            response = player.prompt(prompt_for_move(board, side, match_id, ply))
        move = parse_coord(response)
        if move is None:
            result = winner_from_illegal(color)
            moves.append({"ply": ply, "side": side, "move": None, "result": "parse_failure"})
            break

        row, col = move
        coord = format_coord(row, col) if in_bounds(row, col) else f"{row},{col}"
        played = move_result(board, row, col, color)
        moves.append({"ply": ply, "side": side, "move": coord, "result": played})
        if played in {"illegal_off_board", "illegal_occupied", "black_forbidden"}:
            result = winner_from_illegal(color)
            break
        if played in {"black_win", "white_win", "draw"}:
            result = played
            break

    return {
        "match_id": match_id,
        "black": black_model,
        "white": white_model,
        "result": result,
        "black_score": score_for_black(result),
        "white_score": 1.0 - score_for_black(result),
        "moves": moves,
    }


# %%
renju_model_arena_public.evaluate(
    llm=[kbench.llm],
    evaluation_data=EVALUATION_DATA,
)
