from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rules import BLACK, WHITE, Board, MoveResult, RenjuGame, parse_coord
from renju_benchmark.tasks import extract_first_coord, extract_label, parse_coord_relaxed, score_candidate_move

RESULT_TYPES = (
    "parse_failure",
    "correct",
    "incorrect",
    "win",
    "legal",
    "occupied",
    "off_board",
    "black_forbidden",
)
VALID_MODES = {"fast", "strict", "puzzle"}


def color_from_side(side: str) -> str:
    return BLACK if side.lower() in {"black", "x"} else WHITE


def coord_set(coords: list[str], relaxed: bool = False) -> set[tuple[int, int]]:
    parser = parse_coord_relaxed if relaxed else parse_coord
    return {parser(coord) for coord in coords}


def mode_for_record(record: dict) -> str:
    if record.get("mode"):
        mode = str(record["mode"]).lower()
        if mode not in VALID_MODES:
            raise ValueError(f"{record['id']}: invalid mode {mode}")
        return mode
    tags = set(record.get("tags", []))
    if "strict" in tags or "double_three" in tags:
        return "strict"
    return "fast"


def classify_move_result(board: Board, side: str, move: tuple[int, int], mode: str) -> str:
    color = color_from_side(side)
    game = RenjuGame.from_board(board, turn=color, mode=mode)
    result = game.play(*move)
    if result in (MoveResult.BLACK_WIN, MoveResult.WHITE_WIN):
        return "win"
    if result == MoveResult.BLACK_FORBIDDEN:
        return "black_forbidden"
    if result == MoveResult.ILLEGAL_OFF_BOARD:
        return "off_board"
    if result == MoveResult.ILLEGAL_OCCUPIED:
        return "occupied"
    return "legal"


def score_record(record: dict, response: str) -> dict:
    track = record.get("track", "next_move")
    if track == "rule_classification":
        try:
            label = extract_label(response)
        except ValueError:
            return {"score": 0.0, "parsed": False, "result": "parse_failure"}
        expected = record["expected"].lower().replace("-", "_")
        return {
            "score": 1.0 if label == expected else 0.0,
            "parsed": True,
            "result": "correct" if label == expected else "incorrect",
        }

    board = Board.from_text(record["board"])
    color = color_from_side(record["side"])
    mode = mode_for_record(record)
    try:
        move = extract_first_coord(response)
    except ValueError:
        return {"score": 0.0, "parsed": False, "result": "parse_failure"}
    score = score_candidate_move(
        board,
        color,
        move,
        coord_set(record.get("best_moves", [])),
        coord_set(record.get("good_moves", [])),
        coord_set(record.get("blocking_moves", [])),
        coord_set(record.get("forbidden_moves", []), relaxed=True),
        mode=mode,
    )
    return {
        "score": score,
        "parsed": True,
        "result": classify_move_result(board, record["side"], move, mode),
    }


def load_predictions(path: Path) -> dict[str, str]:
    predictions = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        predictions[row["id"]] = row["response"]
    return predictions


def add_metric(metrics: dict[str, list[float]], key: str, value: float) -> None:
    metrics[key].append(value)


def add_result_rates(metrics: dict[str, list[float]], track: str, result: str) -> None:
    for result_type in RESULT_TYPES:
        value = 1.0 if result == result_type else 0.0
        add_metric(metrics, f"{track}/result:{result_type}_rate", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    metrics: dict[str, list[float]] = defaultdict(list)
    parse_success: list[float] = []

    for line in args.records.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        response = predictions.get(record["id"], "")
        detail = score_record(record, response)
        score = float(detail["score"])
        parsed = bool(detail["parsed"])
        result = str(detail["result"])
        track = record.get("track", "next_move")
        parse_success.append(1.0 if parsed else 0.0)
        add_metric(metrics, "overall", score)
        add_metric(metrics, track, score)
        add_result_rates(metrics, track, result)
        for tag in record.get("tags", []):
            add_metric(metrics, f"tag:{tag}", score)
            add_metric(metrics, f"{track}/tag:{tag}", score)
        family = record.get("family")
        if family:
            add_metric(metrics, f"family:{family}", score)
            add_metric(metrics, f"{track}/family:{family}", score)
        difficulty = record.get("difficulty")
        if difficulty:
            add_metric(metrics, f"difficulty:{difficulty}", score)
            add_metric(metrics, f"{track}/difficulty:{difficulty}", score)

    output = {
        "parse_success_rate": sum(parse_success) / len(parse_success) if parse_success else 0.0,
        **{
            key: sum(values) / len(values)
            for key, values in sorted(metrics.items())
            if values
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
