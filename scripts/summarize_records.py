from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def increment(counter: Counter[str], key: str | None) -> None:
    counter[str(key or "missing")] += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    args = parser.parse_args()

    track = Counter()
    difficulty = Counter()
    mode = Counter()
    tags = Counter()
    track_difficulty = Counter()
    track_mode = Counter()

    total = 0
    for line in args.records.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        total += 1
        record_track = record.get("track", "next_move")
        record_difficulty = record.get("difficulty")
        record_mode = record.get("mode")
        increment(track, record_track)
        increment(difficulty, record_difficulty)
        increment(mode, record_mode)
        increment(track_difficulty, f"{record_track}/{record_difficulty or 'missing'}")
        increment(track_mode, f"{record_track}/{record_mode or 'missing'}")
        for tag in record.get("tags", []):
            increment(tags, tag)

    summary = {
        "total": total,
        "track": dict(sorted(track.items())),
        "difficulty": dict(sorted(difficulty.items())),
        "mode": dict(sorted(mode.items())),
        "track_difficulty": dict(sorted(track_difficulty.items())),
        "track_mode": dict(sorted(track_mode.items())),
        "tags": dict(sorted(tags.items())),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

