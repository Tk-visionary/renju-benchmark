from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rating import RatingTable

GAME_RESULTS = {"black_win", "white_win", "draw"}


def iter_json_values(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_json_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_json_values(child)


def iter_result_dicts(path: Path):
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    seen: set[tuple[str, str, str, str]] = set()
    for file_path in files:
        try:
            payload = json.loads(file_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for value in iter_json_values(payload):
            if not isinstance(value, dict):
                continue
            result_value = value.get("result")
            if not isinstance(result_value, str) or result_value not in GAME_RESULTS:
                continue
            black = value.get("black")
            white = value.get("white")
            if not isinstance(black, str) or not isinstance(white, str):
                continue
            key = (str(value.get("match_id", "")), black, white, result_value)
            if key in seen:
                continue
            seen.add(key)
            yield value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Downloaded Kaggle Benchmark result file or directory")
    parser.add_argument("--default-rating", type=float, default=1900.0)
    parser.add_argument("--k-factor", type=float, default=32.0)
    args = parser.parse_args()

    table = RatingTable(default_rating=args.default_rating, k_factor=args.k_factor)
    games = []
    for result in iter_result_dicts(args.path):
        table.record_game(result["black"], result["white"], result["result"])
        games.append({
            "match_id": result.get("match_id"),
            "black": result["black"],
            "white": result["white"],
            "result": result["result"],
            "candidate_score": result.get("candidate_score"),
        })

    output = {
        "games": games,
        "leaderboard": [
            {"model": model, "rating": round(rating, 2), "games": count}
            for model, rating, count in table.leaderboard()
        ],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
