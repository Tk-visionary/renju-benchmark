from __future__ import annotations

from dataclasses import dataclass

from renju_benchmark.agents import legal_moves
from renju_benchmark.rules import BLACK, WHITE, BOARD_SIZE, Board, format_coord, parse_coord

CHANNELS = (
    "black_stones",
    "white_stones",
    "side_to_move",
    "legal_mask",
    "last_move",
)


@dataclass(frozen=True)
class EncodedPosition:
    board_text: str
    side: str
    planes: list[list[list[float]]]
    legal_policy_mask: list[float]


def side_to_color(side: str) -> str:
    return BLACK if side.lower() in {"black", "x"} else WHITE


def move_to_index(move: tuple[int, int]) -> int:
    row, col = move
    return row * BOARD_SIZE + col


def index_to_move(index: int) -> tuple[int, int]:
    return divmod(index, BOARD_SIZE)


def coord_to_index(coord: str) -> int:
    return move_to_index(parse_coord(coord))


def index_to_coord(index: int) -> str:
    return format_coord(*index_to_move(index))


def legal_policy_mask(board: Board, color: str, local_only: bool = False, forbidden_depth: int = 2) -> list[float]:
    mask = [0.0] * (BOARD_SIZE * BOARD_SIZE)
    if color == WHITE:
        points = board.neighbor_empty_points() if local_only else board.empty_points()
        for move in points:
            mask[move_to_index(move)] = 1.0
        return mask
    for move in legal_moves(board, color, local_only=local_only, forbidden_depth=forbidden_depth):
        mask[move_to_index(move)] = 1.0
    return mask


def encode_position(
    board: Board,
    side: str,
    last_move: tuple[int, int] | None = None,
    forbidden_depth: int = 2,
    local_only: bool = False,
) -> EncodedPosition:
    color = side_to_color(side)
    planes = [
        [[0.0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for _ in CHANNELS
    ]
    for index, stone in enumerate(board.cells):
        row, col = index_to_move(index)
        if stone == BLACK:
            planes[0][row][col] = 1.0
        elif stone == WHITE:
            planes[1][row][col] = 1.0
    side_value = 1.0 if color == BLACK else 0.0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            planes[2][row][col] = side_value

    mask = legal_policy_mask(board, color, local_only=local_only, forbidden_depth=forbidden_depth)
    for index, value in enumerate(mask):
        row, col = index_to_move(index)
        planes[3][row][col] = value

    if last_move is not None:
        row, col = last_move
        if board.in_bounds(row, col):
            planes[4][row][col] = 1.0

    return EncodedPosition(
        board_text=board.to_text(),
        side="black" if color == BLACK else "white",
        planes=planes,
        legal_policy_mask=mask,
    )
