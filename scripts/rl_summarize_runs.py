from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def metrics_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("metrics.json"))


def get_number(payload: dict[str, Any], section: str, key: str) -> float | int | None:
    value = payload.get(section, {}).get(key)
    return value if isinstance(value, int | float) else None


def summarize_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    config = payload.get("config", {})
    run_dir = config.get("run_dir") if isinstance(config, dict) else None
    return {
        "run": str(run_dir or path.parent),
        "metrics": str(path),
        "count": config.get("count") if isinstance(config, dict) else None,
        "tactical_count": config.get("tactical_count") if isinstance(config, dict) else None,
        "epochs": config.get("epochs") if isinstance(config, dict) else None,
        "channels": config.get("channels") if isinstance(config, dict) else None,
        "resblocks": config.get("resblocks") if isinstance(config, dict) else None,
        "rapfi_max_depth": config.get("max_depth") if isinstance(config, dict) else None,
        "rapfi_max_node": config.get("max_node") if isinstance(config, dict) else None,
        "top1_accuracy": get_number(payload, "imitation", "top1_accuracy"),
        "top5_accuracy": _topk_accuracy(payload.get("imitation", {})),
        "tactical_score": get_number(payload, "tactical_match", "score"),
        "tactical_win_rate": get_number(payload, "tactical_match", "win_rate"),
        "tactical_illegal_rate": get_number(payload, "tactical_match", "illegal_rate"),
        "rapfi_score": get_number(payload, "match", "score"),
        "rapfi_win_rate": get_number(payload, "match", "win_rate"),
        "rapfi_illegal_rate": get_number(payload, "match", "illegal_rate"),
        "rapfi_games": get_number(payload, "match", "games"),
        "tactical_games": get_number(payload, "tactical_match", "games"),
    }


def _topk_accuracy(imitation: Any) -> float | None:
    if not isinstance(imitation, dict):
        return None
    for key, value in sorted(imitation.items()):
        if key.startswith("top") and key.endswith("_accuracy") and key != "top1_accuracy":
            return value if isinstance(value, int | float) else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RL experiment metrics.json files.")
    parser.add_argument("paths", nargs="+", type=Path, help="Run directories or metrics.json files")
    parser.add_argument("--min-rapfi-score", type=float, help="Exit nonzero if every run is below this score")
    args = parser.parse_args()

    rows = []
    for path in args.paths:
        for metrics_path in metrics_files(path):
            rows.append(summarize_metrics(metrics_path))
    rows.sort(key=lambda row: (row.get("rapfi_score") is None, -(row.get("rapfi_score") or 0.0), row["run"]))

    output = {"runs": rows}
    print(json.dumps(output, indent=2, sort_keys=True))

    if args.min_rapfi_score is not None:
        best = max((row.get("rapfi_score") or 0.0 for row in rows), default=0.0)
        if best < args.min_rapfi_score:
            print(
                f"best rapfi_score {best:.3f} is below required {args.min_rapfi_score:.3f}",
                file=sys.stderr,
            )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
