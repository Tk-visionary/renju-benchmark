from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rules import BLACK, WHITE, Board, parse_coord
from renju_benchmark.tasks import extract_first_coord, extract_label, parse_coord_relaxed, score_candidate_move


def color_from_side(side: str) -> str:
    return BLACK if side.lower() in {"black", "x"} else WHITE


def coord_set(coords: list[str], relaxed: bool = False) -> set[tuple[int, int]]:
    parser = parse_coord_relaxed if relaxed else parse_coord
    return {parser(coord) for coord in coords}


def score_record(record: dict, response: str) -> tuple[float, bool]:
    track = record.get("track", "next_move")
    if track == "rule_classification":
        try:
            label = extract_label(response)
        except ValueError:
            return 0.0, False
        expected = record["expected"].lower().replace("-", "_")
        return (1.0 if label == expected else 0.0), True

    board = Board.from_text(record["board"])
    color = color_from_side(record["side"])
    try:
        move = extract_first_coord(response)
    except ValueError:
        return 0.0, False
    score = score_candidate_move(
        board,
        color,
        move,
        coord_set(record.get("best_moves", [])),
        coord_set(record.get("good_moves", [])),
        coord_set(record.get("blocking_moves", [])),
        coord_set(record.get("forbidden_moves", []), relaxed=True),
        mode="fast",
    )
    return score, True


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
        score, parsed = score_record(record, response)
        parse_success.append(1.0 if parsed else 0.0)
        add_metric(metrics, "overall", score)
        add_metric(metrics, record.get("track", "next_move"), score)
        for tag in record.get("tags", []):
            add_metric(metrics, tag, score)

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

