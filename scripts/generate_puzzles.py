from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rules import BLACK, WHITE, BOARD_SIZE, Board, MoveResult, RenjuGame, RuleMode, format_coord

ANY_COORD_RE = re.compile(r"\b([A-Za-z])([0-9]{1,2})\b")


def parse_coord_relaxed(coord: str) -> tuple[int, int]:
    match = ANY_COORD_RE.fullmatch(coord.strip())
    if not match:
        raise ValueError(f"bad coordinate: {coord}")
    return int(match.group(2)) - 1, ord(match.group(1).lower()) - ord("a")


def empty_rows() -> list[list[str]]:
    return [list("." * BOARD_SIZE) for _ in range(BOARD_SIZE)]


def board_text(points: list[tuple[int, int, str]]) -> str:
    rows = empty_rows()
    for row, col, color in points:
        rows[row][col] = color
    return "\n".join("".join(row) for row in rows)


def side_name(color: str) -> str:
    return "black" if color == BLACK else "white"


def rule_record(
    record_id: str,
    side: str,
    board: str,
    move: str,
    expected: str,
    tags: list[str],
    difficulty: str,
) -> dict:
    return {
        "id": record_id,
        "track": "rule_classification",
        "side": side,
        "board": board,
        "move": move,
        "expected": expected,
        "tags": tags,
        "difficulty": difficulty,
    }


def next_record(
    record_id: str,
    side: str,
    board: str,
    best_moves: list[str],
    tags: list[str],
    good_moves: list[str] | None = None,
    blocking_moves: list[str] | None = None,
    forbidden_moves: list[str] | None = None,
    difficulty: str = "medium",
) -> dict:
    return {
        "id": record_id,
        "track": "next_move",
        "side": side,
        "board": board,
        "best_moves": best_moves,
        "good_moves": good_moves or [],
        "blocking_moves": blocking_moves or [],
        "forbidden_moves": forbidden_moves or [],
        "tags": tags,
        "difficulty": difficulty,
    }


def make_exact_five(seed: int, index: int, color: str) -> list[dict]:
    rng = random.Random(seed * 10_000 + index)
    row = rng.randrange(2, 13)
    col = rng.randrange(2, 8)
    board = board_text([(row, col + offset, color) for offset in range(4)])
    side = side_name(color)
    moves = [format_coord(row, col - 1), format_coord(row, col + 4)]
    tags = [side, "win", "five"]
    return [
        next_record(f"next_exact_five_seed{seed}_{index:05d}", side, board, moves, tags, difficulty="easy"),
        rule_record(f"rule_exact_five_seed{seed}_{index:05d}", side, board, moves[0], "win", tags, "easy"),
    ]


def make_white_overline(seed: int, index: int) -> list[dict]:
    rng = random.Random(seed * 20_000 + index)
    row = rng.randrange(2, 13)
    col = rng.randrange(2, 7)
    board = board_text([(row, col + offset, WHITE) for offset in range(5)])
    moves = [format_coord(row, col - 1), format_coord(row, col + 5)]
    tags = ["white", "overline", "win"]
    return [
        next_record(f"next_white_overline_seed{seed}_{index:05d}", "white", board, moves, tags, difficulty="medium"),
        rule_record(f"rule_white_overline_seed{seed}_{index:05d}", "white", board, moves[0], "win", tags, "medium"),
    ]


def make_black_overline(seed: int, index: int) -> list[dict]:
    rng = random.Random(seed * 30_000 + index)
    row = rng.randrange(2, 13)
    col = rng.randrange(2, 7)
    board = board_text([(row, col + offset, BLACK) for offset in range(5)])
    moves = [format_coord(row, col - 1), format_coord(row, col + 5)]
    tags = ["black", "overline", "forbidden", "rule"]
    return [
        next_record(
            f"next_black_overline_seed{seed}_{index:05d}",
            "black",
            board,
            [],
            tags,
            forbidden_moves=moves,
            difficulty="medium",
        ),
        rule_record(f"rule_black_overline_seed{seed}_{index:05d}", "black", board, moves[0], "forbidden", tags, "medium"),
    ]


def make_double_four(seed: int, index: int) -> list[dict]:
    row = 7 + (index % 3) - 1
    col = 7 + (index % 5) - 2
    points = [
        (row, col - 3, BLACK), (row, col - 2, BLACK), (row, col - 1, BLACK),
        (row - 3, col, BLACK), (row - 2, col, BLACK), (row - 1, col, BLACK),
    ]
    board = board_text(points)
    move = format_coord(row, col)
    tags = ["black", "double_four", "forbidden", "rule"]
    return [
        next_record(
            f"next_double_four_seed{seed}_{index:05d}",
            "black",
            board,
            [],
            tags,
            forbidden_moves=[move],
            difficulty="hard",
        ),
        rule_record(f"rule_double_four_seed{seed}_{index:05d}", "black", board, move, "forbidden", tags, "hard"),
    ]


def make_double_three(seed: int, index: int) -> list[dict]:
    row = 7 + (index % 3) - 1
    col = 7 + (index % 5) - 2
    points = [
        (row, col - 2, BLACK), (row, col - 1, BLACK),
        (row - 2, col, BLACK), (row - 1, col, BLACK),
    ]
    board = board_text(points)
    move = format_coord(row, col)
    tags = ["black", "double_three", "forbidden", "strict", "rule"]
    return [
        next_record(
            f"next_double_three_seed{seed}_{index:05d}",
            "black",
            board,
            [],
            tags,
            forbidden_moves=[move],
            difficulty="hard",
        ),
        rule_record(f"rule_double_three_seed{seed}_{index:05d}", "black", board, move, "forbidden", tags, "hard"),
    ]


def make_must_block(seed: int, index: int) -> list[dict]:
    rng = random.Random(seed * 40_000 + index)
    row = rng.randrange(2, 13)
    col = rng.randrange(2, 8)
    board = board_text([(row, col - 1, BLACK)] + [(row, col + offset, WHITE) for offset in range(4)])
    block = format_coord(row, col + 4)
    tags = ["black", "defense", "must_block"]
    return [
        next_record(
            f"next_must_block_seed{seed}_{index:05d}",
            "black",
            board,
            [block],
            tags,
            blocking_moves=[block],
            difficulty="medium",
        ),
        rule_record(f"rule_must_block_seed{seed}_{index:05d}", "black", board, block, "legal", tags, "medium"),
    ]


def make_occupied(seed: int, index: int) -> dict:
    rng = random.Random(seed * 50_000 + index)
    row = rng.randrange(0, BOARD_SIZE)
    col = rng.randrange(0, BOARD_SIZE)
    board = board_text([(row, col, BLACK)])
    return rule_record(
        f"rule_occupied_seed{seed}_{index:05d}",
        "white",
        board,
        format_coord(row, col),
        "occupied",
        ["occupied", "rule"],
        "easy",
    )


def make_tempting_occupied(seed: int, index: int) -> dict:
    rng = random.Random(seed * 55_000 + index)
    row = rng.randrange(2, 13)
    col = rng.randrange(2, 8)
    points = [(row, col + offset, BLACK) for offset in range(4)]
    points.append((row, col + 4, WHITE))
    board = board_text(points)
    return rule_record(
        f"rule_tempting_occupied_seed{seed}_{index:05d}",
        "black",
        board,
        format_coord(row, col + 4),
        "occupied",
        ["occupied", "tempting_win", "rule"],
        "medium",
    )


def make_off_board(seed: int, index: int) -> dict:
    moves = ["P8", "A0", "H16", "Z9"]
    move = moves[index % len(moves)]
    side = "black" if index % 2 == 0 else "white"
    return rule_record(
        f"rule_off_board_seed{seed}_{index:05d}",
        side,
        board_text([]),
        move,
        "off_board",
        ["off_board", "rule"],
        "easy",
    )


def make_overline_color_contrast(seed: int, index: int) -> list[dict]:
    rng = random.Random(seed * 60_000 + index)
    row = rng.randrange(2, 13)
    col = rng.randrange(2, 7)
    black_board = board_text([(row, col + offset, BLACK) for offset in range(5)])
    white_board = board_text([(row, col + offset, WHITE) for offset in range(5)])
    move = format_coord(row, col - 1)
    return [
        rule_record(
            f"rule_contrast_black_overline_seed{seed}_{index:05d}",
            "black",
            black_board,
            move,
            "forbidden",
            ["black", "overline", "forbidden", "color_contrast", "rule"],
            "hard",
        ),
        rule_record(
            f"rule_contrast_white_overline_seed{seed}_{index:05d}",
            "white",
            white_board,
            move,
            "win",
            ["white", "overline", "win", "color_contrast", "rule"],
            "hard",
        ),
    ]


def make_exact_five_exception(seed: int, index: int) -> list[dict]:
    row = 7 + (index % 3) - 1
    col = 7 + (index % 5) - 2
    points = [
        (row, col - 4, BLACK), (row, col - 3, BLACK), (row, col - 2, BLACK), (row, col - 1, BLACK),
        (row - 3, col, BLACK), (row - 2, col, BLACK), (row - 1, col, BLACK),
    ]
    board = board_text(points)
    move = format_coord(row, col)
    tags = ["black", "exact_five_exception", "win", "rule"]
    return [
        next_record(f"next_exact_five_exception_seed{seed}_{index:05d}", "black", board, [move], tags, difficulty="hard"),
        rule_record(f"rule_exact_five_exception_seed{seed}_{index:05d}", "black", board, move, "win", tags, "hard"),
    ]


def classify_move(board_text_value: str, side: str, move: str) -> str:
    board = Board.from_text(board_text_value)
    row, col = parse_coord_relaxed(move)
    game = RenjuGame.from_board(board, turn=BLACK if side == "black" else WHITE, mode=RuleMode.STRICT)
    result = game.play(row, col)
    if result in (MoveResult.BLACK_WIN, MoveResult.WHITE_WIN):
        return "win"
    if result == MoveResult.BLACK_FORBIDDEN:
        return "forbidden"
    if result == MoveResult.ILLEGAL_OCCUPIED:
        return "occupied"
    if result == MoveResult.ILLEGAL_OFF_BOARD:
        return "off_board"
    return "legal"


def generate(seed: int, count_per_family: int) -> list[dict]:
    records = []
    for index in range(count_per_family):
        records.extend(make_exact_five(seed, index, BLACK))
        records.extend(make_exact_five(seed + 1, index, WHITE))
        records.extend(make_white_overline(seed, index))
        records.extend(make_black_overline(seed, index))
        records.extend(make_double_four(seed, index))
        records.extend(make_double_three(seed, index))
        records.extend(make_must_block(seed, index))
        records.extend(make_exact_five_exception(seed, index))
        records.extend(make_overline_color_contrast(seed, index))
        records.append(make_occupied(seed, index))
        records.append(make_tempting_occupied(seed, index))
        records.append(make_off_board(seed, index))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count-per-family", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("data/generated/validation_public.jsonl"))
    args = parser.parse_args()

    records = generate(args.seed, args.count_per_family)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    print(f"wrote {len(records)} puzzles to {args.output}")


if __name__ == "__main__":
    main()
