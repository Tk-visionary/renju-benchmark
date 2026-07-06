from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rl.board_encoding import coord_to_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert benchmark next-move records into RL imitation examples.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-good-moves", action="store_true")
    args = parser.parse_args()

    rows = []
    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("track") != "next_move":
            continue
        moves = list(record.get("best_moves", []))
        if args.include_good_moves:
            moves.extend(record.get("good_moves", []))
        if not moves:
            continue
        move = moves[0]
        rows.append(json.dumps({
            "id": record.get("id"),
            "family": record.get("family"),
            "board": record["board"],
            "side": record["side"],
            "best_move": move,
            "policy_index": coord_to_index(move),
            "source": "benchmark_next_move",
        }))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} examples to {args.output}")


if __name__ == "__main__":
    main()
