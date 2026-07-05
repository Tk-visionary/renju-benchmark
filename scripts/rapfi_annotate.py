from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renju_benchmark.rapfi import RapfiConfig, RapfiEngine, RapfiError
from renju_benchmark.rules import BLACK, WHITE, Board, format_coord


def color_from_side(side: str) -> str:
    return BLACK if side.lower() in {"black", "x"} else WHITE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rapfi", default=None, help="Path to Rapfi-compatible executable. Defaults to RAPFI_PATH.")
    parser.add_argument("--cwd", default=None, help="Working directory for Rapfi config/weights.")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--track", default="next_move", help="Only annotate records with this track.")
    args = parser.parse_args()

    config = RapfiConfig.from_env() if args.rapfi is None else RapfiConfig(args.rapfi, cwd=args.cwd, move_timeout=args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with RapfiEngine(config) as engine, args.output.open("w") as out:
        for line in args.records.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("track", "next_move") != args.track:
                continue
            try:
                board = Board.from_text(record["board"])
                row, col = engine.best_move(board, color_from_side(record["side"]))
                annotation = {
                    "id": record["id"],
                    "rapfi_move": format_coord(row, col),
                    "rapfi_role": "candidate",
                }
            except RapfiError as exc:
                annotation = {
                    "id": record["id"],
                    "rapfi_error": str(exc),
                    "rapfi_role": "candidate",
                }
            out.write(json.dumps(annotation) + "\n")
            written += 1
    print(f"wrote {written} annotations to {args.output}")


if __name__ == "__main__":
    main()

