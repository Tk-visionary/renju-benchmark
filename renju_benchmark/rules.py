from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Iterable

EMPTY = "."
BLACK = "X"
WHITE = "O"
OUT = "#"
BOARD_SIZE = 15
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


class MoveResult(str, Enum):
    OK = "ok"
    BLACK_WIN = "black_win"
    WHITE_WIN = "white_win"
    DRAW = "draw"
    ILLEGAL_OCCUPIED = "illegal_occupied"
    ILLEGAL_OFF_BOARD = "illegal_off_board"
    BLACK_FORBIDDEN = "black_forbidden"


class RuleMode(str, Enum):
    FAST = "fast"
    STRICT = "strict"
    PUZZLE = "puzzle"


def forbidden_depth_for_mode(mode: RuleMode | str | int) -> int:
    if isinstance(mode, int):
        return mode
    rule_mode = RuleMode(mode)
    if rule_mode == RuleMode.FAST:
        return 0
    return 2


@dataclass(frozen=True)
class ForbiddenReport:
    forbidden: bool
    overline: bool
    double_four: bool
    double_three: bool
    fours: int
    threes: int


@lru_cache(maxsize=100_000)
def _neighbor_empty_indices(cells: str, radius: int) -> tuple[int, ...]:
    occupied = [
        divmod(index, BOARD_SIZE)
        for index, cell in enumerate(cells)
        if cell != EMPTY
    ]
    if not occupied:
        return (BOARD_SIZE // 2 * BOARD_SIZE + BOARD_SIZE // 2,)
    points = set()
    for row, col in occupied:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = row + dr
                nc = col + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    index = nr * BOARD_SIZE + nc
                    if cells[index] == EMPTY:
                        points.add(index)
    center_row = BOARD_SIZE // 2
    center_col = BOARD_SIZE // 2
    return tuple(
        sorted(
            points,
            key=lambda index: abs(index // BOARD_SIZE - center_row) + abs(index % BOARD_SIZE - center_col),
        )
    )


@dataclass(frozen=True)
class Board:
    cells: str

    @classmethod
    def empty(cls) -> "Board":
        return cls(EMPTY * (BOARD_SIZE * BOARD_SIZE))

    @classmethod
    def from_text(cls, text: str) -> "Board":
        rows = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if len(rows) != BOARD_SIZE:
            raise ValueError(f"expected {BOARD_SIZE} rows, got {len(rows)}")
        parsed = []
        for row in rows:
            if len(row) != BOARD_SIZE:
                raise ValueError(f"expected row length {BOARD_SIZE}, got {len(row)}")
            if any(ch not in (EMPTY, BLACK, WHITE) for ch in row):
                raise ValueError("board text may only contain '.', 'X', and 'O'")
            parsed.append(row)
        return cls("".join(parsed))

    def to_text(self) -> str:
        return "\n".join(
            self.cells[row * BOARD_SIZE : (row + 1) * BOARD_SIZE]
            for row in range(BOARD_SIZE)
        )

    def idx(self, row: int, col: int) -> int:
        return row * BOARD_SIZE + col

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def get(self, row: int, col: int) -> str:
        return self.cells[self.idx(row, col)]

    def is_empty(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.get(row, col) == EMPTY

    def place(self, row: int, col: int, color: str) -> "Board":
        if color not in (BLACK, WHITE):
            raise ValueError(f"unknown color: {color}")
        if not self.in_bounds(row, col):
            raise ValueError("move is off board")
        index = self.idx(row, col)
        if self.cells[index] != EMPTY:
            raise ValueError("intersection is occupied")
        return Board(self.cells[:index] + color + self.cells[index + 1 :])

    def empty_points(self) -> Iterable[tuple[int, int]]:
        for index, cell in enumerate(self.cells):
            if cell == EMPTY:
                yield divmod(index, BOARD_SIZE)

    def neighbor_empty_points(self, radius: int = 2) -> list[tuple[int, int]]:
        return [divmod(index, BOARD_SIZE) for index in _neighbor_empty_indices(self.cells, radius)]

    def is_full(self) -> bool:
        return EMPTY not in self.cells


def parse_coord(coord: str) -> tuple[int, int]:
    coord = coord.strip().lower()
    if len(coord) < 2:
        raise ValueError(f"bad coordinate: {coord}")
    col_ch = coord[0]
    if not ("a" <= col_ch <= "o"):
        raise ValueError(f"bad column: {coord}")
    row = int(coord[1:]) - 1
    col = ord(col_ch) - ord("a")
    if not (0 <= row < BOARD_SIZE):
        raise ValueError(f"bad row: {coord}")
    return row, col


def format_coord(row: int, col: int) -> str:
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError("coordinate off board")
    return f"{chr(ord('A') + col)}{row + 1}"


def _cell_with_overlay(
    board: Board,
    row: int,
    col: int,
    overlay: tuple[int, int, str] | None = None,
) -> str:
    if not board.in_bounds(row, col):
        return OUT
    if overlay is not None and (row, col) == (overlay[0], overlay[1]):
        return overlay[2]
    return board.get(row, col)


def _line_window(
    board: Board,
    row: int,
    col: int,
    dx: int,
    dy: int,
    radius: int = 5,
    overlay_color: str | None = None,
) -> str:
    chars = []
    for offset in range(-radius, radius + 1):
        r = row + dy * offset
        c = col + dx * offset
        if not board.in_bounds(r, c):
            chars.append(OUT)
        elif offset == 0 and overlay_color is not None:
            chars.append(overlay_color)
        else:
            chars.append(board.get(r, c))
    return "".join(chars)


def _run_length(board: Board, row: int, col: int, color: str, dx: int, dy: int) -> int:
    return _run_length_with_overlay(board, row, col, color, dx, dy)


def _run_length_with_overlay(
    board: Board,
    row: int,
    col: int,
    color: str,
    dx: int,
    dy: int,
    overlay: tuple[int, int, str] | None = None,
) -> int:
    total = 1
    for sign in (-1, 1):
        step = 1
        while True:
            r = row + sign * dy * step
            c = col + sign * dx * step
            if _cell_with_overlay(board, r, c, overlay) != color:
                break
            total += 1
            step += 1
    return total


def has_exact_five(board: Board, row: int, col: int, color: str) -> bool:
    return any(_run_length(board, row, col, color, dx, dy) == 5 for dx, dy in DIRECTIONS)


def has_overline(board: Board, row: int, col: int, color: str) -> bool:
    return any(_run_length(board, row, col, color, dx, dy) >= 6 for dx, dy in DIRECTIONS)


def would_make_exact_five(board: Board, row: int, col: int, color: str) -> bool:
    if not board.is_empty(row, col):
        return False
    overlay = (row, col, color)
    return any(
        _run_length_with_overlay(board, row, col, color, dx, dy, overlay) == 5
        for dx, dy in DIRECTIONS
    )


def would_make_overline(board: Board, row: int, col: int, color: str) -> bool:
    if not board.is_empty(row, col):
        return False
    overlay = (row, col, color)
    return any(
        _run_length_with_overlay(board, row, col, color, dx, dy, overlay) >= 6
        for dx, dy in DIRECTIONS
    )


def _direction_empty_points(
    board: Board,
    row: int,
    col: int,
    dx: int,
    dy: int,
    radius: int = 5,
) -> Iterable[tuple[int, int]]:
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        r = row + dy * offset
        c = col + dx * offset
        if board.in_bounds(r, c) and board.is_empty(r, c):
            yield r, c


@lru_cache(maxsize=500_000)
def _straight_four_count_line(line: str, center: int, color: str) -> int:
    if line[center] != color:
        raise ValueError("line center must contain the virtual stone")
    winning_extensions = 0
    for index, cell in enumerate(line):
        if index == center or cell != EMPTY:
            continue
        test = line[:index] + color + line[index + 1 :]
        left = index
        while left > 0 and test[left - 1] == color:
            left -= 1
        right = index
        while right + 1 < len(test) and test[right + 1] == color:
            right += 1
        if right - left + 1 == 5:
            winning_extensions += 1
    return winning_extensions


def _straight_four_count_after_virtual_move(
    board: Board,
    row: int,
    col: int,
    color: str,
    dx: int,
    dy: int,
) -> int:
    radius = 5
    line = _line_window(board, row, col, dx, dy, radius=radius, overlay_color=color)
    return _straight_four_count_line(min(line, line[::-1]), radius, color)


def _four_count(board: Board, row: int, col: int, color: str, stop_at: int = 2) -> int:
    count = 0
    for dx, dy in DIRECTIONS:
        if _straight_four_count_after_virtual_move(board, row, col, color, dx, dy) >= 1:
            count += 1
            if count >= stop_at:
                return count
    return count


def _three_extension_is_allowed(board: Board, row: int, col: int, color: str, depth: int) -> bool:
    if not board.is_empty(row, col):
        return False
    if would_make_exact_five(board, row, col, color):
        return False
    if color == BLACK and depth > 0:
        report = black_forbidden_report(board, row, col, _depth=depth - 1)
        if report.overline or report.double_four or report.double_three:
            return False
    return any(
        _straight_four_count_after_virtual_move(board, row, col, color, dx, dy) >= 2
        for dx, dy in DIRECTIONS
    )


def _three_count(
    board: Board,
    row: int,
    col: int,
    color: str,
    depth: int = 2,
    stop_at: int = 2,
) -> int:
    placed = board.place(row, col, color)
    count = 0
    for dx, dy in DIRECTIONS:
        line = _line_window(board, row, col, dx, dy, radius=4, overlay_color=color)
        if line.count(color) < 3:
            continue
        viable = False
        for nr, nc in _direction_empty_points(placed, row, col, dx, dy, radius=4):
            if _three_extension_is_allowed(placed, nr, nc, color, depth):
                viable = True
                break
        if viable:
            count += 1
            if count >= stop_at:
                return count
    return count


@lru_cache(maxsize=200_000)
def black_forbidden_report(board: Board, row: int, col: int, _depth: int = 2) -> ForbiddenReport:
    if not board.is_empty(row, col):
        raise ValueError("forbidden report requires an empty point")
    if would_make_exact_five(board, row, col, BLACK):
        return ForbiddenReport(False, False, False, False, 0, 0)

    overline = would_make_overline(board, row, col, BLACK)
    fours = _four_count(board, row, col, BLACK, stop_at=2)
    threes = _three_count(board, row, col, BLACK, depth=_depth, stop_at=2) if _depth > 0 else 0
    double_four = fours > 1
    double_three = threes > 1
    return ForbiddenReport(
        forbidden=overline or double_four or double_three,
        overline=overline,
        double_four=double_four,
        double_three=double_three,
        fours=fours,
        threes=threes,
    )


@dataclass
class RenjuGame:
    board: Board
    turn: str = BLACK
    winner: str | None = None
    result: MoveResult | None = None
    forbidden_depth: int = 2

    @classmethod
    def new(cls, mode: RuleMode | str | int = RuleMode.STRICT) -> "RenjuGame":
        return cls(Board.empty(), BLACK, forbidden_depth=forbidden_depth_for_mode(mode))

    @classmethod
    def from_board(
        cls,
        board: Board,
        turn: str = BLACK,
        mode: RuleMode | str | int = RuleMode.STRICT,
    ) -> "RenjuGame":
        return cls(board=board, turn=turn, forbidden_depth=forbidden_depth_for_mode(mode))

    def legal_moves(self, local_only: bool = False, radius: int = 2) -> list[tuple[int, int]]:
        moves = []
        points = self.board.neighbor_empty_points(radius) if local_only else self.board.empty_points()
        for row, col in points:
            if self.turn == BLACK and black_forbidden_report(self.board, row, col, _depth=self.forbidden_depth).forbidden:
                continue
            moves.append((row, col))
        return moves

    def play(self, row: int, col: int) -> MoveResult:
        if self.winner is not None or self.result == MoveResult.DRAW:
            raise ValueError("game is already over")
        if not self.board.in_bounds(row, col):
            self.result = MoveResult.ILLEGAL_OFF_BOARD
            return self.result
        if not self.board.is_empty(row, col):
            self.result = MoveResult.ILLEGAL_OCCUPIED
            return self.result
        if self.turn == BLACK:
            if would_make_exact_five(self.board, row, col, BLACK):
                self.board = self.board.place(row, col, BLACK)
                self.winner = BLACK
                self.result = MoveResult.BLACK_WIN
                return self.result
            report = black_forbidden_report(self.board, row, col, _depth=self.forbidden_depth)
            if report.forbidden:
                self.board = self.board.place(row, col, BLACK)
                self.winner = WHITE
                self.result = MoveResult.BLACK_FORBIDDEN
                return self.result
        else:
            white_wins = would_make_exact_five(self.board, row, col, WHITE) or would_make_overline(self.board, row, col, WHITE)

        color = self.turn
        self.board = self.board.place(row, col, color)
        if color == WHITE and white_wins:
            self.winner = WHITE
            self.result = MoveResult.WHITE_WIN
            return self.result
        if self.board.is_full():
            self.result = MoveResult.DRAW
            return self.result
        self.turn = WHITE if self.turn == BLACK else BLACK
        self.result = MoveResult.OK
        return self.result
